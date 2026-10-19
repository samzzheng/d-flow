"""CPU-only integrator correctness and last-interval sensitivity audit.

No inverse solve, model training, or CUDA use. The terminal experiment keeps
the RK4-100 trajectory through t=.99 fixed and changes only the final interval.
It must not be interpreted as a reconstruction comparison after reoptimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
from sampler.ode_samplers import ode_integrate


def terminal_step(model, x, t, h, substeps=1, endpoint_epsilon=0.0):
    hh = h / substeps
    for j in range(substeps):
        tt = t + j * hh
        k1 = bench.velocity(model, tt, x)
        k2 = bench.velocity(model, tt + hh / 2, x + hh * k1 / 2)
        k3 = bench.velocity(model, tt + hh / 2, x + hh * k2 / 2)
        k4time = tt + hh - (endpoint_epsilon if j == substeps - 1 else 0.0)
        k4 = bench.velocity(model, k4time, x + hh * k3)
        x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class AnalyticVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.times = []

    def forward(self, x, t):
        self.times.append(float(t[0]))
        return x * t.reshape(-1, 1, 1, 1)


def analytic_checks():
    rows = {}
    for method in ("forward_euler", "rk4"):
        errors = []
        for n in (10, 20, 40, 80):
            m = AnalyticVelocity()
            z = torch.ones(1, 1, 1, 1, dtype=torch.float64, requires_grad=True)
            x = bench.solve_flow(m, z, n, method)
            g, = torch.autograd.grad(x.sum(), z)
            err = abs(float(x.detach()) - math.exp(0.5))
            errors.append(err)
            rows[f"{method}{n}"] = dict(error=err, gradient_error=abs(float(g) - math.exp(.5)),
                                          evaluations=len(m.times), first_time=min(m.times),
                                          last_time=max(m.times))
        rows[f"{method}_observed_orders"] = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert rows["rk4_observed_orders"][-1] > 3.8
    assert .95 < rows["forward_euler_observed_orders"][-1] < 1.05
    for method in ("euler", "rk4"):
        times = []
        def f(t, x):
            times.append(float(t))
            return x * t
        ode_integrate(f, torch.ones(1, 1, 1, 1), sampler=method,
                      ode_opts={"options": {"step_size": .01}})
        rows[f"generic_{method}100_time_grid"] = dict(evaluations=len(times),
                                                       first_time=min(times), last_time=max(times))
    previous = json.loads((ROOT / "outputs/rk4_accuracy_2026_09_27/refinement/results.json").read_text())
    ds = previous["successive_differences"]
    levels = (25, 50, 100, 200, 400, 800)
    diffs = [ds[f"rk4_{a}_vs_{b}"] for a, b in zip(levels, levels[1:])]
    rows["existing_model_refinement_orders"] = [math.log2(a / b) for a, b in zip(diffs, diffs[1:])]
    return rows


def add_measurement_metrics(report, outputs, truth):
    angles = torch.linspace(0, math.pi, 16)[:-1]
    clean = bench.sparse_radon(truth, angles)
    observed, noise, _ = bench.add_relative_noise(
        clean, .01, torch.Generator().manual_seed(20260807 + 250 * 1000 + 1))
    report["measurement"] = {"views": 15, "relative_noise": .01,
                             "noise_seed": 20260807 + 250 * 1000 + 1,
                             "noise_norm": float(noise.norm())}
    with torch.no_grad():
        for name, value in outputs.items():
            predicted = bench.sparse_radon(value, angles)
            predicted_clamped = bench.sparse_radon(value.clamp(0, 1), angles)
            report["terminal_results"][name].update(
                raw_residual_over_noise=float((predicted-observed).norm()/noise.norm()),
                clamped_residual_over_noise=float((predicted_clamped-observed).norm()/noise.norm()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/rk4_integrator_audit_2026_09_27")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip-model", action="store_true")
    args = p.parse_args()
    if not 1 <= args.threads <= 4:
        p.error("Use 1 to 4 CPU threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"analytic": analytic_checks(), "device": "cpu", "threads": args.threads,
              "torch": str(torch.__version__), "purpose": "fixed-source terminal sensitivity, not reoptimized accuracy"}
    def save():
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    if args.skip_model:
        return
    checkpoint = ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    sourcefile = ROOT / "outputs/rk4_noise_floor_2026_09_27/sample1_rk4/accepted/state_00094.pt"
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report["source_file"] = str(sourcefile.relative_to(ROOT))
    source = torch.load(sourcefile, map_location="cpu", weights_only=False)["source"]
    report["source_sha256"] = hashlib.sha256(source.numpy().tobytes()).hexdigest()
    model = bench.load_model(250, 100, torch.device("cpu"),
                             "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
    assert not model.mid_attn and not any(model.att_channels)
    truth = bench.normalize_images(torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu",
                                               weights_only=True, mmap=True)["test"][1:2])
    start = time.perf_counter()
    outputs = {}
    with torch.no_grad():
        x = source.clone()
        for i in range(99):
            x = terminal_step(model, x, i * .01, .01)
        report["prefix_seconds"] = time.perf_counter() - start
        print(json.dumps({"prefix_seconds": report["prefix_seconds"]}), flush=True)
        outputs["last_euler1"] = x + .01 * bench.velocity(model, .99, x)
        for n in (1, 4, 16, 64):
            outputs[f"last_rk4_{n}"] = terminal_step(model, x, .99, .01, n)
            print(json.dumps({"finished": f"last_rk4_{n}", "elapsed": time.perf_counter() - start}), flush=True)
        for eps in (1e-6, 1e-4, 1e-3):
            outputs[f"last_rk4_1_endpoint_minus_{eps}"] = terminal_step(model, x, .99, .01, endpoint_epsilon=eps)
        # Time-only sensitivity at a fixed spatial state (not a trajectory).
        vt1 = bench.velocity(model, 1.0, x)
        report["fixed_state_time_sensitivity"] = {}
        for eps in (1e-6, 1e-4, 1e-3, 1e-2):
            v = bench.velocity(model, 1-eps, x)
            report["fixed_state_time_sensitivity"][str(eps)] = float((v-vt1).norm()/vt1.norm())
    ref = outputs["last_rk4_64"]
    base = outputs["last_rk4_1"]
    report["terminal_results"] = {name: {
        "relative_to_local_rk4_64": float((value-ref).norm()/ref.norm()),
        "relative_to_original_rk4_100": float((value-base).norm()/base.norm()),
        "image_relative_l2_clamped": float((value.clamp(0,1)-truth).norm()/truth.norm()),
        "image_relative_l2_raw": float((value-truth).norm()/truth.norm())}
        for name, value in outputs.items()}
    saved = torch.load(sourcefile, map_location="cpu", weights_only=False)["raw_reconstruction"]
    report["recomputed_baseline_relative_difference"] = float((base-saved).norm()/saved.norm())
    add_measurement_metrics(report, outputs, truth)
    report["elapsed_seconds"] = time.perf_counter()-start
    torch.save({"source": source, "prefix_t099": x, "outputs": outputs}, args.out / "images.pt")
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

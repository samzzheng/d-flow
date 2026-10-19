"""CPU local line-search fork for fast DLR at the empty-history sample.

Compute fresh fast gradients before and after the original tiny first step.
From that second source, test the current direction capped at 0.01 sqrt(d).
Pass --include-legacy to also reproduce the historical repeated tiny trial.
This is a local diagnostic, not a full solve.
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
import test_parallel_bug_integrator as helper
from dlr_dflow import dlr_dflow as core
from dlr_dflow import enriched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--include-legacy", action="store_true",
                        help="Explicitly compare the historical repeated-L1 trial")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/optimizer_settings_2026_09_27/first_iteration_only/strict_progress/fast_local_probe.json")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("Use 1–4 CPU threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    model, tag = helper.load(250, device)
    assert not model.mid_attn and not any(model.att_channels)
    data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
    angles = torch.linspace(0, math.pi, 16)[:-1]
    z0, target = helper.make_problem(250, args.sample, data, angles, device)
    truth = helper.bench.normalize_images(data[args.sample:args.sample + 1])
    core.SHARE_LINEARIZATION = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    checkpoint = ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"protocol": {"device": "cpu", "threads": args.threads, "sample": args.sample,
        "resolution": 250, "model": tag, "checkpoint_sha256": digest,
        "base_seed": helper.SEED, "noise": helper.NOISE, "flow_steps": 100,
        "rank": 8, "trace_probes": 4, "enrichment_probes": 0,
        "initialization_oversampling": 8, "terminal_alpha": 0,
        "factor_refresh": "every gradient", "source_prior": "none",
        "objective": "relative_scaled", "c1": 1e-4, "step_cap": 2.5,
        "armijo_initial_step_policy": helper.bench.ArmijoLBFGS.initial_step_policy,
        "armijo_progress_policy": helper.bench.ArmijoLBFGS.progress_policy,
        "include_legacy_comparison": args.include_legacy,
        "purpose": "local first-step/second-step audit, not a complete optimization"}, "trials": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def emit(event, **details):
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": event, **details}), flush=True)

    def evaluate(z):
        with torch.no_grad():
            x = helper.bench.solve_flow(model, z, 100, "forward_euler")
            f, relative_sq = helper.bench.data_objective(helper.bench.sparse_radon(x, angles), target, 250, "relative_scaled")
            metrics = {"objective": float(f), "residual": math.sqrt(float(relative_sq)),
                "image_relative_l2": float(helper.bench.relative_l2(x.clamp(0, 1), truth)),
                "raw_image_relative_l2": float(helper.bench.relative_l2(x, truth))}
        return x, metrics

    def gradient(z, x):
        start = time.perf_counter()
        _, factors, _ = core.sensitivity_factors(model, z, 100, rank=8,
            trace_probes=4, enrichment_probes=0, initialization_oversampling=8,
            factor_stride=1, zero_terminal_baseline=True)
        terminal = helper.terminal_gradient(x, target, 250, angles)
        g = core.approx_source_gradient(factors, terminal, 0.0).reshape_as(z).detach()
        return g, time.perf_counter() - start

    x0, m0 = evaluate(z0)
    report["initial"] = m0
    emit("initial", **m0)
    g0, seconds0 = gradient(z0, x0)
    t0 = min(1.0, 1.0 / float(g0.abs().sum()))
    s0 = -t0 * g0
    z1 = z0 + s0
    x1, m1 = evaluate(z1)
    rhs0 = m0["objective"] - 1e-4 * t0 * float(g0.square().sum())
    report["first_step"] = dict(m1, step_norm=float(s0.norm()), armijo_rhs=rhs0,
        accepted=(not torch.equal(z1, z0) and m1["objective"] < m0["objective"]
                  and m1["objective"] - m0["objective"] <= -1e-4 * t0 * float(g0.square().sum())),
        gradient_seconds=seconds0)
    emit("first_step", **report["first_step"])
    if not report["first_step"]["accepted"]:
        raise RuntimeError("Tiny first step did not pass Armijo; second-step fork is inapplicable")
    g1, seconds1 = gradient(z1, x1)
    curvature = float(((g1 - g0) * s0).sum())
    report["curvature"] = {"y_dot_s": curvature, "pair_stored_by_current_optimizer": curvature > 1e-10,
        "second_gradient_seconds": seconds1, "gradient_norm": float(g1.norm())}
    emit("curvature", **report["curvature"])
    policies = [("once_scaling_then_cap", min(1.0, 2.5 / float(g1.norm())))]
    if args.include_legacy:
        policies.insert(0, ("repeat_l1_scaling", min(1.0, 1.0 / float(g1.abs().sum()))))
    for policy, step_size in policies:
        _, metrics = evaluate(z1 - step_size * g1)
        rhs = m1["objective"] - 1e-4 * step_size * float(g1.square().sum())
        row = dict(metrics, policy=policy, step_norm=step_size * float(g1.norm()),
            armijo_rhs=rhs,
            armijo_progress_policy=("legacy_rounded_rhs" if policy == "repeat_l1_scaling"
                                    else helper.bench.ArmijoLBFGS.progress_policy),
            accepted=(metrics["objective"] <= rhs if policy == "repeat_l1_scaling" else
                      not torch.equal(z1 - step_size * g1, z1) and metrics["objective"] < m1["objective"]
                      and metrics["objective"] - m1["objective"] <= -1e-4 * step_size * float(g1.square().sum())),
            relative_objective_drop=(m1["objective"] - metrics["objective"]) / m1["objective"])
        report["trials"].append(row)
        emit("trial", **row)
    torch.save({"z0": z0, "z1": z1, "g0": g0, "g1": g1, "protocol": report["protocol"]}, args.out.with_suffix(".pt"))
    emit("complete", output=str(args.out))


if __name__ == "__main__":
    main()

"""One exact CPU Euler gradient at a completed DLR solve's saved endpoint.

Reads the cached approximate gradient and optimizer history without rerunning
DLR. The exact forward pass supplies the image/objective validation and the
single source-gradient evaluation. An optional PID guard prevents this job
from starting while the solve whose cores it will use is still running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import test_parallel_bug_integrator as helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/fast_accuracy_recovery_2026_09_27/fresh_sketch32_s0")
    parser.add_argument("--profile", default="sketch32")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--worker-pid", type=int)
    parser.add_argument("--resume-snapshot", type=Path)
    parser.add_argument("--allow-in-progress", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/fast_accuracy_recovery_2026_09_27/endpoint_sketch32")
    args = parser.parse_args()
    if args.worker_pid is not None and Path(f"/proc/{args.worker_pid}").exists():
        parser.error("The monitored solve worker is still running; do not reuse its cores yet")
    if not 1 <= args.threads <= 8:
        parser.error("Use one to eight CPU threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    original = json.loads((args.run_dir / "results.json").read_text())
    run, = [r for r in original["runs"] if r["profile"] == args.profile]
    if not args.allow_in_progress and ("stop_reason" not in run or "final" not in run):
        parser.error("The selected solve has not recorded a completed endpoint")
    snapshot = args.resume_snapshot or args.run_dir / f"{args.profile}_resume.pt"
    saved = torch.load(snapshot, map_location="cpu", weights_only=True)
    protocol = saved["protocol"]
    expected = {"device": "cpu", "resolution": 250, "flow_steps": 100,
                "integrator": "Euler", "noise": helper.NOISE, "base_seed": helper.SEED,
                "angles": 15, "source_prior": "none", "objective": "relative_scaled"}
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Endpoint protocol does not match the exact-gradient diagnostic")
    if saved["settings"]["threads"] != args.threads:
        raise ValueError("Match the solve's CPU thread count")
    if ((not args.allow_in_progress and saved["step"] != run["final"]["step"])
            or saved["optimizer"]["step_attempts"] != saved["step"]):
        raise ValueError("Endpoint report and optimizer checkpoint disagree")
    if not math.isclose(saved["optimizer"]["f"], saved["metrics"]["objective"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Cached optimizer objective does not match the accepted source metrics")
    sample = saved["settings"]["sample"]
    source = saved["source"].detach().clone().requires_grad_(True)
    approximate = saved["optimizer"]["g"].detach().reshape_as(source)
    if not torch.isfinite(approximate).all():
        raise ValueError("Cached approximate gradient is nonfinite")
    checkpoint = ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if checkpoint_hash != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint digest changed")
    model, tag = helper.load(250, torch.device("cpu"))
    assert not model.mid_attn and not any(model.att_channels)
    data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
    truth = helper.bench.normalize_images(data[sample:sample + 1])
    angles = torch.linspace(0, math.pi, 16)[:-1]
    clean = helper.bench.sparse_radon(truth, angles)
    _, target = helper.make_problem(250, sample, data, angles, torch.device("cpu"))
    noise_norm = float((target - clean).norm())
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "exact_gradient_start", "sample": sample,
                      "step": saved["step"], "threads": args.threads,
                      "cpu_affinity": sorted(os.sched_getaffinity(0))}), flush=True)
    started = time.perf_counter()
    raw_image = helper.bench.solve_flow(model, source, 100, "forward_euler", False)
    prediction = helper.bench.sparse_radon(raw_image, angles)
    objective, residual_sq = helper.bench.data_objective(prediction, target, 250, "relative_scaled")
    exact, = torch.autograd.grad(objective, source)
    seconds = time.perf_counter() - started
    exact, raw_image = exact.detach(), raw_image.detach()
    measured = {"objective": float(objective.detach()), "residual": math.sqrt(float(residual_sq.detach())),
                "residual_over_noise": float((prediction.detach() - target).norm()) / noise_norm,
                "image_relative_l2": float(helper.bench.relative_l2(raw_image.clamp(0, 1), truth)),
                "raw_image_relative_l2": float(helper.bench.relative_l2(raw_image, truth))}
    torch.testing.assert_close(raw_image, saved["raw_image"], rtol=1e-5, atol=1e-6)
    for key, value in measured.items():
        if not math.isclose(value, saved["metrics"][key], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"Recomputed accepted-source metric disagrees: {key}")
    reference, estimate = exact.flatten().double(), approximate.flatten().double()
    exact_norm, approximate_norm = reference.norm(), estimate.norm()
    dot = reference @ estimate
    comparison = {"exact_gradient_norm": float(exact_norm),
                  "approximate_gradient_norm": float(approximate_norm),
                  "relative_error": float((estimate - reference).norm() / exact_norm) if exact_norm > 0 else None,
                  "cosine": float(dot / (exact_norm * approximate_norm)) if exact_norm * approximate_norm > 0 else None,
                  "norm_ratio": float(approximate_norm / exact_norm) if exact_norm > 0 else None,
                  "exact_dot_approximate": float(dot),
                  "true_derivative_along_negative_approximate": float(-dot),
                  "predicted_derivative_along_negative_approximate": float(-estimate.square().sum()),
                  "beta": float(dot / estimate.square().sum()) if approximate_norm > 0 else None,
                  "negative_approximate_is_true_descent": bool(dot > 0)}
    # Pure saved-state algebra: diagnose the next L-BFGS direction as well.
    # This does not call a closure or evaluate any additional model state.
    cached = saved["optimizer"]
    parameter = source.detach().clone().requires_grad_(True)
    optimizer = helper.bench.ArmijoLBFGS(parameter, lambda _: None)
    optimizer.S, optimizer.Y = cached["S"], cached["Y"]
    direction = optimizer._direction(approximate.flatten())
    fallback = not math.isfinite(float(approximate.flatten() @ direction)) or float(approximate.flatten() @ direction) >= 0
    if fallback:
        direction = -approximate.flatten()
    direction64 = direction.double()
    initial_step_size = 1.0
    if cached["step_attempts"] == 0:
        initial_step_size = min(1.0, 1.0 / float(approximate.abs().sum()))
    step_cap = 0.01 * math.sqrt(source.numel())
    if direction64.norm() > 0:
        initial_step_size = min(initial_step_size, step_cap / float(direction64.norm()))
    next_direction = {"non_descent_fallback": fallback, "history_size": len(optimizer.S),
                      "norm": float(direction64.norm()),
                      "true_directional_derivative": float(reference @ direction64),
                      "approximate_directional_derivative": float(estimate @ direction64),
                      "is_true_descent": bool(reference @ direction64 < 0),
                      "configured_lr": 1.0, "step_cap": step_cap,
                      "capped_initial_step_size": initial_step_size,
                      "capped_initial_source_step_norm": initial_step_size * float(direction64.norm()),
                      "true_linear_change_at_capped_initial_trial": initial_step_size * float(reference @ direction64)}
    validation = {"native_image_relative_difference": float((raw_image - saved["raw_image"]).norm() / saved["raw_image"].norm()),
                  "native_image_max_absolute_difference": float((raw_image - saved["raw_image"]).abs().max()),
                  "metric_differences": {key: value - saved["metrics"][key] for key, value in measured.items()}}
    report = {"protocol": {**protocol, "model": tag, "sample": sample, "threads": args.threads,
                           "cpu_affinity": sorted(os.sched_getaffinity(0)),
                           "exact_gradient_evaluations": 1, "dlr_evaluations": 0,
                           "exact_activation_checkpointing": False,
                           "gradient_comparison": "native exact Euler derivative versus cached accepted-source optimizer.g"},
              "source_run": str(args.run_dir), "source_step": saved["step"],
              "stop_reason": run.get("stop_reason", "in_progress_snapshot"), "metrics": measured,
              "validation": validation, "gradient": comparison,
              "cached_next_lbfgs_direction": next_direction,
              "exact_gradient_seconds": seconds, "status": "complete"}
    torch.save({"protocol": report["protocol"], "source": source.detach(),
                "exact_gradient": exact, "approximate_gradient": approximate,
                "raw_image": raw_image, "target": target,
                "cached_next_lbfgs_direction": direction,
                "metrics": measured}, args.out / "gradients.pt")
    torch.save(saved, args.out / "optimizer_snapshot.pt")
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()

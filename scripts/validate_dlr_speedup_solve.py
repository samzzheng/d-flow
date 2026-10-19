"""Short paired DLR solves with the same true objective and capped Armijo search.

  .venv/bin/python scripts/validate_dlr_speedup_solve.py --device cuda:0 \
      --resolutions 64 250 --samples 0 1 --outer-steps 20

Reference: rank 8, eight trace probes, four enrichment probes, old exact flags.
Fast: rank 8, four trace probes, zero enrichment, shared tape/full-step graphs.
Add --methods reference optimized fast exact to include unchanged-budget shared
implementation and exact-gradient controls. Capture is included in solve time.
There is no residual-based early stop; only two failed Armijo searches stop a run.
"""
from __future__ import annotations

import argparse
import gc
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
from dlr_dflow.graphed_products import GraphedAProducts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 250])
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--outer-steps", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=100)
    parser.add_argument("--methods", nargs="+", choices=["reference", "optimized", "fast", "exact"],
                        default=["reference", "fast"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-cap", type=float, default=.01,
                        help="Maximum source step divided by sqrt(image dimension)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/dlr_speedup_solve")
    args = parser.parse_args()
    if min(args.outer_steps, args.flow_steps, args.threads) < 1 or min(args.samples) < 0:
        parser.error("Step/thread counts must be positive and sample indices nonnegative")
    if args.step_cap <= 0:
        parser.error("--step-cap must be positive")
    return args


def finite_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Select a CUDA device for the captured implementation comparison")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    helper.STEPS = args.flow_steps
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "results.json"
    report = {"settings": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "protocol": {"base_seed": helper.SEED, "noise": helper.NOISE, "angles": 15,
                           "rank": 8, "initialization_oversampling": 8,
                           "source_prior": "none", "terminal_alpha": 0,
                           "objective": "relative_scaled",
                           "optimizer": "ArmijoLBFGS", "lr": 1.0, "history": 100,
                           "armijo_initial_step_policy": helper.bench.ArmijoLBFGS.initial_step_policy,
                           "armijo_progress_policy": helper.bench.ArmijoLBFGS.progress_policy,
                           "max_ls": 30, "c1": 1e-4,
                           "terminal_gradient_state": "cached exact forward at accepted source",
                           "floating_point_control": "all arms share the native forward objective and terminal-gradient state; existing benchmark uses the DLR primal state",
                           "timing": "initial objective, graph capture, gradients, trials and diagnostics",
                           "best_selection": "minimum true objective over accepted sources",
                           "early_stop": "two consecutive failed Armijo searches only"},
              "environment": {"torch": str(torch.__version__),
                              "gpu": torch.cuda.get_device_name(device),
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
              "runs": []}

    def progress(event, **details):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        print(json.dumps(finite_json({"event": event, **details}), allow_nan=False), flush=True)

    old_share, old_skip = core.SHARE_LINEARIZATION, enriched.SKIP_UNTRUNCATED_SVD
    try:
        pair_index = 0
        for resolution in args.resolutions:
            model, tag = helper.load(resolution, device)
            checkpoint = ROOT / "outputs/fm_unet" / (
                "circles-64/n1/checkpoints/epoch_0050.pt" if resolution == 64 else
                f"circles-{resolution}-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt")
            with checkpoint.open("rb") as handle:
                checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            data = torch.load(ROOT / f"data/circles-{resolution}-n1.pt", map_location="cpu",
                              weights_only=True, mmap=True)["test"]
            angles = torch.linspace(0, math.pi, 16, device=device)[:-1]
            for sample in args.samples:
                if sample >= len(data):
                    raise ValueError(f"Sample {sample} outside dataset with {len(data)} entries")
                z0, y = helper.make_problem(resolution, sample, data, angles, device)
                truth = helper.bench.normalize_images(data[sample:sample + 1]).to(device)
                # Rotate complete-solve order to reduce systematic first-run bias.
                offset = pair_index % len(args.methods)
                order = args.methods[offset:] + args.methods[:offset]
                pair_index += 1
                for method in order:
                    core.SHARE_LINEARIZATION = method in ("optimized", "fast")
                    enriched.SKIP_UNTRUNCATED_SVD = method in ("optimized", "fast")
                    probes, enrichment = (4, 0) if method == "fast" else (8, 4)
                    source = z0.clone().requires_grad_(True)
                    run = {"resolution": resolution, "sample": sample, "method": method,
                           "model": tag, "checkpoint_sha256": checkpoint_hash,
                           "solve_order": order,
                           "trace_probes": None if method == "exact" else probes,
                           "enrichment_probes": None if method == "exact" else enrichment,
                           "shared_linearization": core.SHARE_LINEARIZATION,
                           "skip_untruncated_svd": enriched.SKIP_UNTRUNCATED_SVD,
                           "full_step_graphs": method == "fast" and args.cuda_graphs,
                           "steps": []}
                    report["runs"].append(run)
                    products = None
                    counts = {"gradient_evaluations": 0, "true_forward_evaluations": 0}
                    last_trial = None
                    accepted = None
                    best = None
                    best_image = None
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    if method != "exact" and args.cuda_graphs:
                        products = GraphedAProducts(model, z0,
                            share_linearization=core.SHARE_LINEARIZATION,
                            full_step_graphs=method == "fast")

                    def true_objective(candidate):
                        nonlocal last_trial
                        with torch.no_grad():
                            x = helper.bench.solve_flow(model, candidate, args.flow_steps,
                                                        "forward_euler")
                            value, relative_squared = helper.bench.data_objective(
                                helper.bench.sparse_radon(x, angles), y, resolution, "relative_scaled")
                        counts["true_forward_evaluations"] += 1
                        last_trial = {"source": candidate.detach().clone(), "x": x.detach(),
                                      "objective": float(value),
                                      "residual": math.sqrt(float(relative_squared))}
                        return last_trial["objective"]

                    optimizer = helper.bench.ArmijoLBFGS(
                        source, true_objective, lr=1.0, history=100, max_ls=30, c1=1e-4,
                        step_cap=args.step_cap * math.sqrt(source.numel()))

                    def closure():
                        nonlocal accepted
                        optimizer.zero_grad()
                        if last_trial is None or not torch.equal(last_trial["source"], source.detach()):
                            true_objective(source.detach())
                        accepted = last_trial
                        if method == "exact":
                            gradient, _, _ = helper.exact_gradient(model, source.detach(), y,
                                                                   resolution, angles)
                        else:
                            _, factors, _ = core.sensitivity_factors(
                                model, source.detach(), args.flow_steps, rank=8,
                                trace_probes=probes, initialization_oversampling=8,
                                enrichment_probes=enrichment, factor_stride=1,
                                products=products, zero_terminal_baseline=True)
                            gx = helper.terminal_gradient(accepted["x"], y, resolution, angles)
                            gradient = core.approx_source_gradient(factors, gx, 0.0)
                        source.grad = gradient.reshape_as(source).detach()
                        counts["gradient_evaluations"] += 1
                        return accepted["objective"]

                    def record(step):
                        nonlocal best, best_image
                        with torch.no_grad():
                            raw_error = helper.bench.relative_l2(accepted["x"], truth).item()
                            image_error = helper.bench.relative_l2(accepted["x"].clamp(0, 1), truth).item()
                        torch.cuda.synchronize(device)
                        row = {"step": step, "true_objective": accepted["objective"],
                               "residual": accepted["residual"],
                               "source_norm_over_sqrt_d": source.norm().item() / math.sqrt(source.numel()),
                               "image_relative_l2": image_error, "raw_image_relative_l2": raw_error,
                               **counts,
                               "line_search_function_evaluations": optimizer.total_function_evaluations,
                               "last_line_search_function_evaluations": optimizer.last_function_evaluations,
                               "failed_searches": optimizer.failed_searches,
                               "consecutive_failed_searches": optimizer.consecutive_failures,
                               "elapsed_s": time.perf_counter() - started}
                        run["steps"].append(row)
                        if best is None or row["true_objective"] < best["true_objective"]:
                            best = dict(row)
                        if best_image is None or image_error < best_image["image_relative_l2"]:
                            best_image = dict(row)
                        progress("step", resolution=resolution, sample=sample, method=method, **row)

                    true_objective(source.detach())
                    accepted = last_trial
                    record(0)
                    for step in range(1, args.outer_steps + 1):
                        optimizer.step(closure)
                        record(step)
                        if optimizer.stalled:
                            break
                    torch.cuda.synchronize(device)
                    run.update(elapsed_s=time.perf_counter() - started,
                               stop_reason="two_failed_line_searches" if optimizer.stalled else "step_budget",
                               steps_run=len(run["steps"]) - 1, final=dict(run["steps"][-1]),
                               best_by_objective=best, best_image_among_accepted=best_image,
                               capped_steps=optimizer.capped_steps, **counts,
                               line_search_function_evaluations=optimizer.total_function_evaluations)
                    progress("solve_complete", resolution=resolution, sample=sample,
                             method=method, elapsed_s=run["elapsed_s"], stop_reason=run["stop_reason"],
                             final=run["final"], best_by_objective=best)
                    del products, optimizer, source, last_trial, accepted
                    gc.collect()
                    torch.cuda.empty_cache()
            del model, data
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        core.SHARE_LINEARIZATION, enriched.SKIP_UNTRUNCATED_SVD = old_share, old_skip
    progress("complete", output=str(path))


if __name__ == "__main__":
    main()

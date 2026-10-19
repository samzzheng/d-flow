"""CPU-only paired reconstruction check: exact Euler versus fresh fast DLR.

Start both arms at the same cached exact-Armijo source. Optimizer history is
reset for each arm, so nonzero --start-iterate is a local continuation check,
not a continuation of the original optimizer. The true native Euler objective
and its terminal loss gradient are shared by both arms; every fast gradient
rebuilds rank-8, four-probe, zero-enrichment factors over all 100 flow steps.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", type=int, default=[0])
    parser.add_argument("--start-iterate", type=int, default=10)
    parser.add_argument("--outer-steps", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=("exact", "fast"), default=["exact", "fast"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--points-cache", type=Path,
        default=ROOT / "outputs/dlr_speed_2026_09_27/initial/points")
    parser.add_argument("--out", type=Path,
        default=ROOT / "outputs/accuracy_fast_euler_2026_09_27/first_iteration_only/strict_progress/continuation")
    args = parser.parse_args()
    if not (0 < args.threads <= 4) or args.outer_steps < 1 or args.start_iterate < 0:
        parser.error("Use 1–4 CPU threads, positive outer steps and nonnegative start iterate")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    resolution, flow_steps = 250, 100
    device = torch.device("cpu")
    model, tag = helper.load(resolution, device)
    checkpoint = ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu",
                      weights_only=True, mmap=True)["test"]
    angles = torch.linspace(0, math.pi, 16)[:-1]
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"settings": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "protocol": {"device": "cpu", "resolution": resolution, "model": tag,
                           "checkpoint_sha256": checkpoint_hash, "flow_steps": flow_steps,
                           "rank": 8, "trace_probes": 4, "enrichment_probes": 0,
                           "initialization_oversampling": 8, "terminal_alpha": 0,
                           "factor_refresh": "every gradient", "noise": helper.NOISE,
                           "base_seed": helper.SEED, "angles": 15, "source_prior": "none",
                           "objective": "relative_scaled", "optimizer": "ArmijoLBFGS",
                           "armijo_initial_step_policy": helper.bench.ArmijoLBFGS.initial_step_policy,
                           "armijo_progress_policy": helper.bench.ArmijoLBFGS.progress_policy,
                           "lr": 1.0, "history": 100, "max_ls": 30, "c1": 1e-4,
                           "step_cap_over_sqrt_d": .01, "reset_optimizer_history": True,
                           "native_forward_for_objective_and_terminal_gradient": True,
                           "exact_activation_checkpointing": False,
                           "timing_includes": "gradients, true-forward trials and reporting"},
              "runs": []}

    def progress(event, **details):
        temporary = args.out / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.out / "results.json")
        print(json.dumps({"event": event, **details}), flush=True)

    for sample in args.samples:
        cache = torch.load(args.points_cache / f"res250_sample{sample}.pt",
                           map_location="cpu", weights_only=True)
        metadata = cache["metadata"]
        if (metadata["checkpoint_sha256"] != checkpoint_hash or metadata["steps"] != flow_steps
                or metadata["noise"] != helper.NOISE or metadata["base_seed"] != helper.SEED):
            raise ValueError("Point cache does not match the model and measurement protocol")
        start = cache["points"][args.start_iterate]
        _, target = helper.make_problem(resolution, sample, data, angles, device)
        truth = helper.bench.normalize_images(data[sample:sample + 1])
        for method in args.methods:
            core.SHARE_LINEARIZATION = method == "fast"
            enriched.SKIP_UNTRUNCATED_SVD = method == "fast"
            source = start.clone().requires_grad_(True)
            run = {"sample": sample, "method": method, "steps": [],
                   "starting_point_progress_policy": (
                       "initial_random_source" if args.start_iterate == 0
                       else metadata.get("armijo_progress_policy", "legacy_rounded_rhs")),
                   "starting_point_optimizer_policy": (
                       "initial_random_source" if args.start_iterate == 0
                       else metadata.get("armijo_initial_step_policy", "legacy_empty_history"))}
            report["runs"].append(run)
            states = {0: source.detach().clone()}
            counts = {"gradient_evaluations": 0, "true_forward_evaluations": 0}
            last_trial, accepted = None, None
            started = time.perf_counter()

            def objective(candidate):
                nonlocal last_trial
                with torch.no_grad():
                    x = helper.bench.solve_flow(model, candidate, flow_steps, "forward_euler")
                    value, relative_squared = helper.bench.data_objective(
                        helper.bench.sparse_radon(x, angles), target, resolution, "relative_scaled")
                counts["true_forward_evaluations"] += 1
                last_trial = {"source": candidate.detach().clone(), "x": x.detach(),
                              "objective": float(value), "residual": math.sqrt(float(relative_squared))}
                return last_trial["objective"]

            optimizer = helper.bench.ArmijoLBFGS(source, objective, lr=1.0, history=100,
                                                max_ls=30, c1=1e-4, step_cap=2.5)

            def closure():
                nonlocal accepted
                optimizer.zero_grad()
                if last_trial is None or not torch.equal(last_trial["source"], source.detach()):
                    objective(source.detach())
                accepted = last_trial
                if method == "exact":
                    x = helper.bench.solve_flow(model, source, flow_steps, "forward_euler", False)
                    loss, _ = helper.bench.data_objective(
                        helper.bench.sparse_radon(x, angles), target, resolution, "relative_scaled")
                    gradient, = torch.autograd.grad(loss, source)
                else:
                    _, factors, _ = core.sensitivity_factors(model, source.detach(), flow_steps,
                        rank=8, trace_probes=4, enrichment_probes=0,
                        initialization_oversampling=8, factor_stride=1, zero_terminal_baseline=True)
                    terminal = helper.terminal_gradient(accepted["x"], target, resolution, angles)
                    gradient = core.approx_source_gradient(factors, terminal, 0.0).reshape_as(source)
                source.grad = gradient.detach()
                counts["gradient_evaluations"] += 1
                return accepted["objective"]

            def record(step):
                x = accepted["x"]
                row = {"step": step, "true_objective": accepted["objective"],
                       "residual": accepted["residual"],
                       "image_relative_l2": float(helper.bench.relative_l2(x.clamp(0, 1), truth)),
                       "raw_image_relative_l2": float(helper.bench.relative_l2(x, truth)),
                       "source_norm_over_sqrt_d": float(source.detach().norm()) / 250,
                       "elapsed_s": time.perf_counter() - started,
                       "function_evaluations": optimizer.total_function_evaluations,
                       "failed_searches": optimizer.failed_searches,
                       "capped_steps": optimizer.capped_steps, **counts}
                run["steps"].append(row)
                states[step] = source.detach().clone()
                torch.save({"protocol": report["protocol"], "sample": sample, "method": method,
                            "start_exact_iterate": args.start_iterate, "sources": states},
                           args.out / f"sources_sample{sample}_{method}.pt")
                progress("step", sample=sample, method=method, **row)

            objective(source.detach())
            accepted = last_trial
            record(0)
            for step in range(1, args.outer_steps + 1):
                optimizer.step(closure)
                record(step)
                if optimizer.stalled:
                    break
            run.update(final=dict(run["steps"][-1]),
                       best_by_objective=dict(min(run["steps"], key=lambda row: row["true_objective"])),
                       stop_reason="two_failed_searches" if optimizer.stalled else "step_budget")
            progress("solve_complete", sample=sample, method=method, final=run["final"])
            del optimizer, source, last_trial, accepted, states
            gc.collect()
    progress("complete", output=str(args.out / "results.json"))


if __name__ == "__main__":
    main()

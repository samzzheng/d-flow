"""Paired D-Flow/DLR-D-Flow sparse-CT benchmark on circle data.

Each method starts from the same source and uses fixed-step LBFGS until its
measurement relative L2 residual reaches the target or the outer-step budget
is exhausted.  Per-iteration timing and the best iterate are recorded.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dlr_dflow.dlr_dflow as dd
import dlr_dflow.tangent_unet as tangent
from dlr_dflow.graphed_products import GraphedAProducts


FLOW_STEPS = 100
RANK = 8
TRACE_PROBES = 32
INITIALIZATION_OVERSAMPLING = 8
ENRICHMENT_PROBES = 4
N_ANGLES = 15
LBFGS_LR = 1.0
LBFGS_HISTORY_SIZE = 100
TARGET_RELATIVE_RESIDUAL = 0.01
MAX_OUTER_STEPS = 200
EPOCH = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolutions", default="64,96,128,160,200,250")
    parser.add_argument("--samples", default="0,1,2,3,4")
    parser.add_argument("--algorithms", default="D-Flow,DLR-D-Flow")
    parser.add_argument("--jobs", help="optional comma-separated resolution:sample pairs")
    parser.add_argument("--max-outer-steps", type=int, default=MAX_OUTER_STEPS)
    parser.add_argument("--target", type=float, default=TARGET_RELATIVE_RESIDUAL)
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--run-dir-template",
                        help="model run dir, {res} substituted; default is the "
                             "conventional outputs/fm_unet/circles-{res}/n1")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="checkpoint epoch to load (default 50)")
    parser.add_argument("--attention-jvp-mode", choices=["moments", "chunked"],
                        help="override the resolution-based attention JVP choice")
    parser.add_argument("--attention-score-mb", type=int,
                        help="override ATTENTION_SCORE_BYTES (chunked query-chunk cap)")
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "outputs/dflow_vs_dlr_sparse_ct_100step_lbfgs_1pct",
    )
    return parser.parse_args()


def sparse_ct(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    count = angles.numel()
    repeated = x[:, None].expand(batch, count, channels, height, width)
    repeated = repeated.reshape(batch * count, channels, height, width)
    batch_angles = angles.repeat(batch)
    cosine, sine = torch.cos(batch_angles), torch.sin(batch_angles)
    theta = torch.zeros(batch * count, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    grid = F.affine_grid(theta, repeated.shape, align_corners=False)
    rotated = F.grid_sample(
        repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rotated.mean(-2).reshape(batch, count, channels, width).permute(0, 2, 1, 3)


def checkpointed_flow(model: torch.nn.Module, source: torch.Tensor) -> torch.Tensor:
    step_size = 1.0 / FLOW_STEPS
    value = source
    for index in range(FLOW_STEPS):
        t = index * step_size
        value = checkpoint(
            lambda current, time_value=t: (
                current + step_size * dd.velocity(model, time_value, current)),
            value, use_reentrant=False, preserve_rng_state=False,
        )
    return value


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_lbfgs(source: torch.Tensor) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(
        [source], lr=LBFGS_LR, max_iter=1, history_size=LBFGS_HISTORY_SIZE,
        line_search_fn=None, tolerance_grad=1e-7, tolerance_change=1e-9,
    )


ATTENTION_JVP_OVERRIDE: str | None = None
ATTENTION_SCORE_BYTES_OVERRIDE: int | None = None


def configure_kernels(resolution: int) -> None:
    dd.HAND_JVP = True
    dd.HAND_VJP = True
    tangent.ATTENTION_MODE = "moments"
    tangent.ATTENTION_JVP_MODE = (
        ATTENTION_JVP_OVERRIDE if ATTENTION_JVP_OVERRIDE
        else ("chunked" if resolution <= 128 else "moments"))
    tangent.ATTENTION_VJP_MODE = "chunked"
    tangent.GN_STATS_MODE = "native" if resolution <= 96 else "manual"
    if ATTENTION_SCORE_BYTES_OVERRIDE is not None:
        tangent.ATTENTION_SCORE_BYTES = ATTENTION_SCORE_BYTES_OVERRIDE


def run_method(
    algorithm: str,
    model: torch.nn.Module,
    source0: torch.Tensor,
    truth: torch.Tensor,
    operator,
    observation: torch.Tensor,
    device: torch.device,
    max_outer_steps: int,
    target: float,
    use_cuda_graphs: bool,
) -> tuple[dict, list[dict], dict[str, torch.Tensor]]:
    source = source0.detach().clone().requires_grad_(True)
    optimizer = make_lbfgs(source)
    observation_norm = observation.norm().item()
    checkpointed = algorithm == "D-Flow" and truth.shape[-1] >= 160
    products = (
        GraphedAProducts(model, source.detach())
        if algorithm == "DLR-D-Flow" and use_cuda_graphs else None
    )
    rows: list[dict] = []
    best = {"relative_residual": float("inf"), "step": 0, "source": None,
            "reconstruction": None, "image_relative_l2": float("inf")}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, max_outer_steps + 1):
        source_evaluated = source.detach().clone()
        closure_data: dict = {}
        closure_evaluations = 0

        synchronize(device)
        started = time.perf_counter()

        def closure() -> torch.Tensor:
            nonlocal closure_evaluations
            closure_evaluations += 1
            optimizer.zero_grad(set_to_none=True)
            if algorithm == "D-Flow":
                terminal = (
                    checkpointed_flow(model, source)
                    if checkpointed else dd.euler_flow(model, source, FLOW_STEPS)
                )
                residual = operator(terminal) - observation
                loss = 0.5 * residual.square().sum()
                loss.backward()
            else:
                terminal, factors, alpha = dd.sensitivity_factors(
                    model, source.detach(), FLOW_STEPS, rank=RANK,
                    method="enriched", trace_probes=TRACE_PROBES,
                    initialization_oversampling=INITIALIZATION_OVERSAMPLING,
                    enrichment_probes=ENRICHMENT_PROBES, factor_stride=1,
                    products=products,
                )
                detached_terminal = terminal.detach().requires_grad_(True)
                residual = operator(detached_terminal) - observation
                loss = 0.5 * residual.square().sum()
                terminal_gradient, = torch.autograd.grad(loss, detached_terminal)
                source.grad = dd.approx_source_gradient(
                    factors, terminal_gradient, alpha).reshape_as(source)
            closure_data["terminal"] = terminal.detach()
            closure_data["relative_residual"] = (
                residual.detach().norm().item() / observation_norm)
            return loss.detach() if algorithm == "DLR-D-Flow" else loss

        optimizer.step(closure)
        synchronize(device)
        elapsed = time.perf_counter() - started

        terminal = closure_data["terminal"]
        relative_residual = closure_data["relative_residual"]
        image_relative_l2 = (
            (terminal.clamp(0, 1) - truth).norm() / truth.norm()).item()
        if relative_residual < best["relative_residual"]:
            best.update(
                relative_residual=relative_residual,
                step=step,
                source=source_evaluated.cpu(),
                reconstruction=terminal.clamp(0, 1).cpu(),
                image_relative_l2=image_relative_l2,
            )
        rows.append({
            "step": step,
            "seconds": elapsed,
            "relative_measurement_residual": relative_residual,
            "image_relative_l2": image_relative_l2,
            "closure_evaluations": closure_evaluations,
            "is_best_so_far": step == best["step"],
        })
        print(
            f"  {algorithm:10s} step {step:3d}: residual "
            f"{100 * relative_residual:7.3f}%  {elapsed:8.3f} s",
            flush=True,
        )
        if relative_residual <= target:
            break

    times = [row["seconds"] for row in rows]
    summary = {
        "algorithm": algorithm,
        "steps_run": len(rows),
        "reached_target": best["relative_residual"] <= target,
        "best_step": best["step"],
        "best_relative_measurement_residual": best["relative_residual"],
        "best_image_relative_l2": best["image_relative_l2"],
        "total_seconds": sum(times),
        "seconds_per_iteration_mean": statistics.mean(times),
        "seconds_per_iteration_median": statistics.median(times),
        "seconds_per_iteration_stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2 ** 30
            if device.type == "cuda" else None
        ),
        "checkpointed": checkpointed,
    }
    tensors = {"source": best["source"], "reconstruction": best["reconstruction"]}
    return summary, rows, tensors


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    global ATTENTION_JVP_OVERRIDE, ATTENTION_SCORE_BYTES_OVERRIDE
    ATTENTION_JVP_OVERRIDE = args.attention_jvp_mode
    if args.attention_score_mb is not None:
        ATTENTION_SCORE_BYTES_OVERRIDE = args.attention_score_mb * 1024 * 1024
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    resolutions = [int(value) for value in args.resolutions.split(",")]
    samples = [int(value) for value in args.samples.split(",")]
    algorithms = args.algorithms.split(",")
    jobs = (
        [tuple(map(int, value.split(":"))) for value in args.jobs.split(",")]
        if args.jobs else [(resolution, sample) for resolution in resolutions for sample in samples]
    )
    metadata = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "resolutions": resolutions,
        "samples": samples,
        "algorithms": algorithms,
        "flow_steps": FLOW_STEPS,
        "rank": RANK,
        "trace_probes": TRACE_PROBES,
        "initialization_oversampling": INITIALIZATION_OVERSAMPLING,
        "enrichment_probes": ENRICHMENT_PROBES,
        "factor_stride": 1,
        "angles": N_ANGLES,
        "target_relative_measurement_residual": args.target,
        "max_outer_steps": args.max_outer_steps,
        "optimizer": "LBFGS",
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_max_iter": 1,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "lbfgs_line_search": None,
        "dlr_cuda_graphs": not args.no_cuda_graphs,
        "run_dir_template": args.run_dir_template,
        "checkpoint_epoch": args.epoch,
        "attention_jvp_override": ATTENTION_JVP_OVERRIDE,
        "attention_score_bytes_override": ATTENTION_SCORE_BYTES_OVERRIDE,
    }
    metadata_path = args.out_dir / f"settings_{device.type}{device.index or 0}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    summary_path = args.out_dir / f"samples_{device.type}{device.index or 0}.csv"
    iterations_path = args.out_dir / f"iterations_{device.type}{device.index or 0}.csv"
    summaries = list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
    iteration_rows = list(csv.DictReader(iterations_path.open())) if iterations_path.exists() else []
    completed = {
        (int(row["resolution"]), int(row["sample_index"]), row["algorithm"])
        for row in summaries
    }
    cache: dict[int, tuple] = {}

    for resolution, sample_index in jobs:
        if resolution not in cache:
            configure_kernels(resolution)
            run_dir = (args.run_dir_template.format(res=resolution)
                       if args.run_dir_template else None)
            model, _ = dd.load_flow_model(resolution, 1, args.epoch, str(device),
                                          run_dir=run_dir)
            data = torch.load(
                ROOT / f"data/circles-{resolution}-n1.pt", map_location="cpu",
                weights_only=True, mmap=True,
            )["test"]
            images = data[samples].to(dtype=torch.float32)
            if data.dtype == torch.uint8:
                images.div_(255.0)
            angles = torch.linspace(0, math.pi, N_ANGLES + 1, device=device)[:-1]
            cache[resolution] = model, images, angles
        model, images, angles = cache[resolution]
        configure_kernels(resolution)
        truth = images[samples.index(sample_index):samples.index(sample_index) + 1].to(device)
        operator = lambda image: sparse_ct(image, angles)
        observation = operator(truth).detach()
        source0 = torch.randn(
            truth.shape,
            generator=torch.Generator(device=device).manual_seed(
                resolution * 1000 + sample_index),
            device=device, dtype=torch.float32,
        )

        for algorithm in algorithms:
            key = resolution, sample_index, algorithm
            if key in completed:
                print(f"Skipping completed {key}", flush=True)
                continue
            print(f"Starting {key} on {device}", flush=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            summary, method_rows, tensors = run_method(
                algorithm, model, source0, truth, operator, observation, device,
                args.max_outer_steps, args.target, not args.no_cuda_graphs,
            )
            summary.update(resolution=resolution, sample_index=sample_index)
            summaries.append(summary)
            for row in method_rows:
                iteration_rows.append({
                    "resolution": resolution, "sample_index": sample_index,
                    "algorithm": algorithm, **row,
                })
            write_csv(summary_path, summaries)
            write_csv(iterations_path, iteration_rows)
            torch.save(
                {"truth": truth.cpu(), "observation": observation.cpu(),
                 "summary": summary, "iterations": method_rows, **tensors},
                args.out_dir / f"sample_{resolution}_{sample_index}_{algorithm.lower().replace('-', '_')}.pt",
            )
            completed.add(key)
            print(
                f"Finished {key}: best {100 * summary['best_relative_measurement_residual']:.3f}% "
                f"at step {summary['best_step']}; "
                f"{summary['seconds_per_iteration_mean']:.3f} s/iter",
                flush=True,
            )


if __name__ == "__main__":
    main()

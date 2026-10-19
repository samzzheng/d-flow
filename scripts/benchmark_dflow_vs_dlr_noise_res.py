"""Paired D-Flow / DLR-D-Flow sparse-CT noise benchmark at a chosen resolution.

Generalization of benchmark_dflow_vs_dlr_noise64.py to arbitrary resolution
and noise level(s). For each noise level, N test images are reconstructed
with both algorithms. Each matched pair (D-Flow, DLR-D-Flow) shares the same
ground-truth image, the same noisy sinogram (one noise draw per (sample,
noise level)), and the same random initial source. No source prior is used.
Both arms run fixed-step LBFGS (lr=1.0, max_iter=1, no line search) until the
measurement relative L2 residual reaches 2x the noise level or a step budget
is exhausted. D-Flow uses gradient checkpointing through the flow at
resolution >= 160 to fit memory (matches benchmark_dflow_vs_dlr_sparse_ct.py).

Usage:  python scripts/benchmark_dflow_vs_dlr_noise_res.py <device> \
            --resolution 250 --noise-levels 0.05 --samples 30
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
MAX_OUTER_STEPS = 150
STOP_MULTIPLE_OF_NOISE = 2.0
EPOCH = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device")
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--noise-levels", default="0.05",
                         help="comma-separated relative noise levels")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-dir", default=None,
                         help="model run dir override, e.g. for a noattn variant")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                         help="checkpoint epoch to load (default 50)")
    parser.add_argument("--use-cuda-graphs", action="store_true", default=True)
    parser.add_argument("--no-cuda-graphs", dest="use_cuda_graphs",
                         action="store_false")
    return parser.parse_args()


def configure_kernels(resolution: int) -> None:
    dd.HAND_JVP = True
    dd.HAND_VJP = True
    tangent.ATTENTION_MODE = "moments"
    tangent.ATTENTION_JVP_MODE = "chunked" if resolution <= 128 else "moments"
    tangent.ATTENTION_VJP_MODE = "chunked"
    tangent.GN_STATS_MODE = "native" if resolution <= 96 else "manual"


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


def add_relative_noise(clean: torch.Tensor, level: float, generator: torch.Generator
                        ) -> torch.Tensor:
    """Gaussian noise scaled so that ||eta|| / ||clean|| == level (repo convention)."""
    noise = torch.randn(clean.shape, generator=generator,
                         device=clean.device, dtype=clean.dtype)
    noise = noise / (noise.norm() + 1e-30) * (level * clean.norm())
    return clean + noise


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_lbfgs(source: torch.Tensor) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(
        [source], lr=LBFGS_LR, max_iter=1, history_size=LBFGS_HISTORY_SIZE,
        line_search_fn=None, tolerance_grad=1e-7, tolerance_change=1e-9,
    )


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
    products,
    checkpointed: bool,
) -> tuple[dict, list[dict], dict[str, torch.Tensor]]:
    source = source0.detach().clone().requires_grad_(True)
    optimizer = make_lbfgs(source)
    observation_norm = observation.norm().item()
    rows: list[dict] = []
    best = {"relative_residual": float("inf"), "step": 0, "source": None,
            "reconstruction": None, "image_relative_l2": float("inf"),
            "measurement": None}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, max_outer_steps + 1):
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
                    checkpointed_flow(model, source) if checkpointed
                    else dd.euler_flow(model, source, FLOW_STEPS)
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
            closure_data["measurement"] = operator(terminal).detach()
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
                source=source.detach().cpu(),
                reconstruction=terminal.clamp(0, 1).cpu(),
                image_relative_l2=image_relative_l2,
                measurement=closure_data["measurement"].cpu(),
            )
        rows.append({
            "step": step,
            "seconds": elapsed,
            "relative_measurement_residual": relative_residual,
            "image_relative_l2": image_relative_l2,
            "closure_evaluations": closure_evaluations,
            "is_best_so_far": step == best["step"],
        })
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
    tensors = {
        "source": best["source"], "reconstruction": best["reconstruction"],
        "measurement": best["measurement"],
    }
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
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(8)
    resolution = args.resolution
    noise_levels = [float(v) for v in args.noise_levels.split(",")]
    n_samples = args.samples
    out_dir = args.out_dir or (
        ROOT / f"outputs/dflow_vs_dlr_noise{resolution}")
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_kernels(resolution)
    model, _ = dd.load_flow_model(resolution, 1, args.epoch, str(device), run_dir=args.run_dir)
    data = torch.load(
        ROOT / f"data/circles-{resolution}-n1.pt", map_location="cpu",
        weights_only=True, mmap=True,
    )["test"]
    sample_indices = list(range(n_samples))
    images = data[sample_indices].to(dtype=torch.float32)
    if data.dtype == torch.uint8:
        images.div_(255.0)
    angles = torch.linspace(0, math.pi, N_ANGLES + 1, device=device)[:-1]
    operator = lambda image: sparse_ct(image, angles)
    checkpointed_dflow = resolution >= 160

    metadata = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "resolution": resolution,
        "flow_steps": FLOW_STEPS,
        "rank": RANK,
        "trace_probes": TRACE_PROBES,
        "initialization_oversampling": INITIALIZATION_OVERSAMPLING,
        "enrichment_probes": ENRICHMENT_PROBES,
        "angles": N_ANGLES,
        "noise_levels": noise_levels,
        "n_samples_per_level": n_samples,
        "source_prior": None,
        "stop_multiple_of_noise": STOP_MULTIPLE_OF_NOISE,
        "max_outer_steps": MAX_OUTER_STEPS,
        "optimizer": "LBFGS", "lbfgs_lr": LBFGS_LR, "lbfgs_max_iter": 1,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE, "lbfgs_line_search": None,
        "checkpoint_epoch": args.epoch,
        "run_dir": args.run_dir,
        "dflow_gradient_checkpointing": checkpointed_dflow,
        "dlr_cuda_graphs": args.use_cuda_graphs,
    }
    (out_dir / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")

    summary_path = out_dir / "samples.csv"
    iterations_path = out_dir / "iterations.csv"
    summaries = list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
    iteration_rows = list(csv.DictReader(iterations_path.open())) if iterations_path.exists() else []
    completed = {
        (float(row["noise_level"]), int(row["sample_index"]), row["algorithm"])
        for row in summaries
    }

    products = (
        GraphedAProducts(model, torch.zeros(1, 1, resolution, resolution, device=device))
        if args.use_cuda_graphs else None
    )

    for noise_level in noise_levels:
        target = STOP_MULTIPLE_OF_NOISE * noise_level
        for sample_index in sample_indices:
            truth = images[sample_index:sample_index + 1].to(device)
            clean = operator(truth).detach()
            noise_generator = torch.Generator(device=device).manual_seed(
                int(round(noise_level * 100000)) * 1000 + sample_index)
            observation = add_relative_noise(clean, noise_level, noise_generator)
            source0 = torch.randn(
                truth.shape,
                generator=torch.Generator(device=device).manual_seed(
                    500000 + sample_index),
                device=device, dtype=torch.float32,
            )

            for algorithm in ["D-Flow", "DLR-D-Flow"]:
                key = noise_level, sample_index, algorithm
                if key in completed:
                    print(f"Skipping completed {key}", flush=True)
                    continue
                print(f"Starting res={resolution} noise={noise_level:.0%} "
                      f"sample={sample_index} {algorithm} on {device}", flush=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                summary, method_rows, tensors = run_method(
                    algorithm, model, source0, truth, operator, observation,
                    device, MAX_OUTER_STEPS, target,
                    products if algorithm == "DLR-D-Flow" else None,
                    checkpointed_dflow if algorithm == "D-Flow" else False,
                )
                summary.update(noise_level=noise_level, sample_index=sample_index)
                summaries.append(summary)
                for row in method_rows:
                    iteration_rows.append({
                        "noise_level": noise_level, "sample_index": sample_index,
                        "algorithm": algorithm, **row,
                    })
                write_csv(summary_path, summaries)
                write_csv(iterations_path, iteration_rows)
                torch.save(
                    {"truth": truth.cpu(), "clean_measurement": clean.cpu(),
                     "noisy_measurement": observation.cpu(), "summary": summary,
                     "iterations": method_rows, **tensors},
                    out_dir / (
                        f"sample_noise{int(round(noise_level * 100)):02d}_"
                        f"{sample_index}_{algorithm.lower().replace('-', '_')}.pt"),
                )
                completed.add(key)
                print(
                    f"Finished {key}: best "
                    f"{100 * summary['best_relative_measurement_residual']:.3f}% "
                    f"(target {100 * target:.3f}%) at step {summary['best_step']}; "
                    f"image rel L2 {100 * summary['best_image_relative_l2']:.3f}%; "
                    f"{summary['seconds_per_iteration_mean']:.3f} s/iter; "
                    f"peak {summary['peak_allocated_gib']:.2f} GiB",
                    flush=True,
                )


if __name__ == "__main__":
    main()

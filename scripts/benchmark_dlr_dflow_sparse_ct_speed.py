"""Per-iteration sparse-CT speed and memory for exact D-Flow and optimized DLR.

The benchmark uses one test image and one source per resolution. One measured
iteration consists of a 100-step flow gradient, terminal sparse-CT loss, and
one fixed-step LBFGS update. The warm-up iteration (including CUDA-graph
capture for DLR) is excluded from timing but persistent graph memory is
included in the peak.
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


def checkpointed_flow(model, source: torch.Tensor) -> torch.Tensor:
    step_size = 1.0 / FLOW_STEPS
    value = source
    for index in range(FLOW_STEPS):
        t = index * step_size
        value = checkpoint(
            lambda current, time_value=t: (
                current + step_size * dd.velocity(model, time_value, current)),
            value, use_reentrant=False, preserve_rng_state=False)
    return value


def make_lbfgs(source):
    return torch.optim.LBFGS(
        [source], lr=LBFGS_LR, max_iter=1,
        history_size=LBFGS_HISTORY_SIZE, line_search_fn=None,
        tolerance_grad=1e-7, tolerance_change=1e-9)


def terminal_loss_gradient(x1, operator, observation):
    terminal = x1.detach().requires_grad_(True)
    loss = 0.5 * (operator(terminal) - observation).square().sum()
    gradient, = torch.autograd.grad(loss, terminal)
    return loss.detach(), gradient


def synchronize(device):
    torch.cuda.synchronize(device)


def measure_iterations(step, device, warmup, timed):
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    warmup_start = time.perf_counter()
    warmup_loss = None
    for iteration in range(1, warmup + 1):
        warmup_loss = step(iteration)
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start
    setup_peak_allocated = torch.cuda.max_memory_allocated(device) / 2 ** 30
    setup_peak_reserved = torch.cuda.max_memory_reserved(device) / 2 ** 30

    gc.collect()
    torch.cuda.empty_cache()
    persistent_allocated = torch.cuda.memory_allocated(device) / 2 ** 30
    persistent_reserved = torch.cuda.memory_reserved(device) / 2 ** 30
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    loss = warmup_loss
    for offset in range(timed):
        synchronize(device)
        start = time.perf_counter()
        loss = step(warmup + offset + 1)
        synchronize(device)
        samples.append(time.perf_counter() - start)
    iteration_peak_allocated = torch.cuda.max_memory_allocated(device) / 2 ** 30
    iteration_peak_reserved = torch.cuda.max_memory_reserved(device) / 2 ** 30
    return {
        "warmup_seconds": warmup_seconds,
        "seconds_per_iteration_mean": statistics.mean(samples),
        "seconds_per_iteration_median": statistics.median(samples),
        "iteration_seconds": samples,
        "peak_allocated_gib": max(setup_peak_allocated, iteration_peak_allocated),
        "peak_reserved_gib": max(setup_peak_reserved, iteration_peak_reserved),
        "setup_peak_allocated_gib": setup_peak_allocated,
        "iteration_peak_allocated_gib": iteration_peak_allocated,
        "persistent_allocated_gib": persistent_allocated,
        "persistent_reserved_gib": persistent_reserved,
        "terminal_loss": float(loss),
    }


def benchmark_dflow(model, source, operator, observation, device, warmup, timed,
                    checkpointed):
    value = source.detach().clone().requires_grad_(True)
    optimizer = make_lbfgs(value)
    flow = checkpointed_flow if checkpointed else dd.euler_flow
    closure_evals = []

    def step(iteration):
        count = 0

        def closure():
            nonlocal count
            count += 1
            optimizer.zero_grad(set_to_none=True)
            if checkpointed:
                x1 = flow(model, value)
            else:
                x1 = flow(model, value, FLOW_STEPS)
            loss = 0.5 * (operator(x1) - observation).square().sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        closure_evals.append(count)
        return loss.detach()

    result = measure_iterations(step, device, warmup, timed)
    result.update(
        checkpointed=checkpointed,
        closure_evals=closure_evals,
        final_source_norm=float(value.detach().norm()),
    )
    return result


def benchmark_dlr(model, source, operator, observation, device, warmup, timed,
                  use_cuda_graphs):
    value = source.detach().clone().requires_grad_(True)
    optimizer = make_lbfgs(value)
    products = GraphedAProducts(model, value.detach()) if use_cuda_graphs else None
    closure_evals = []

    def step(iteration):
        count = 0

        def closure():
            nonlocal count
            count += 1
            optimizer.zero_grad(set_to_none=True)
            x1, factors, alpha = dd.sensitivity_factors(
                model, value.detach(), FLOW_STEPS, rank=RANK, method="enriched",
                trace_probes=TRACE_PROBES,
                initialization_oversampling=INITIALIZATION_OVERSAMPLING,
                enrichment_probes=ENRICHMENT_PROBES,
                factor_stride=1, products=products)
            loss, terminal_gradient = terminal_loss_gradient(
                x1, operator, observation)
            value.grad = dd.approx_source_gradient(
                factors, terminal_gradient, alpha).reshape_as(value)
            return loss

        loss = optimizer.step(closure)
        closure_evals.append(count)
        return loss.detach()

    result = measure_iterations(step, device, warmup, timed)
    result.update(
        cuda_graphs=use_cuda_graphs,
        hand_jvp=dd.HAND_JVP,
        hand_vjp=dd.HAND_VJP,
        attention_jvp_mode=tangent.ATTENTION_JVP_MODE or tangent.ATTENTION_MODE,
        attention_vjp_mode=tangent.ATTENTION_VJP_MODE or tangent.ATTENTION_MODE,
        groupnorm_stats_mode=tangent.GN_STATS_MODE,
        closure_evals=closure_evals,
        final_source_norm=float(value.detach().norm()),
    )
    return result


def write_results(path: Path, metadata, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
    csv_path = path.with_suffix(".csv")
    fields = [
        "resolution", "dimension", "algorithm", "seconds_per_iteration_mean",
        "seconds_per_iteration_median", "peak_allocated_gib", "peak_reserved_gib",
        "setup_peak_allocated_gib", "iteration_peak_allocated_gib",
        "persistent_allocated_gib", "warmup_seconds", "checkpointed", "terminal_loss",
        "closure_evals",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolutions", default="64,96,128,160,200,250")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timed", type=int, default=3)
    parser.add_argument("--algorithms", default="D-Flow,DLR-D-Flow")
    parser.add_argument("--attention-mode", choices=("auto", "moments", "chunked"),
                        default="auto")
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/dlr_dflow_sparse_ct_speed_lbfgs/benchmark.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    dd.HAND_JVP = True
    dd.HAND_VJP = True
    tangent.ATTENTION_MODE = "moments"

    resolutions = [int(item) for item in args.resolutions.split(",")]
    metadata = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "resolutions": resolutions,
        "single_sample": True,
        "flow_steps": FLOW_STEPS,
        "rank": RANK,
        "trace_probes": TRACE_PROBES,
        "initialization_oversampling": INITIALIZATION_OVERSAMPLING,
        "enrichment_probes": ENRICHMENT_PROBES,
        "factor_stride": 1,
        "angles": N_ANGLES,
        "optimizer": "LBFGS",
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_max_iter": 1,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "lbfgs_line_search": None,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.timed,
        "dlr_cuda_graphs": not args.no_cuda_graphs,
        "dlr_attention_mode": args.attention_mode,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    rows = []
    for resolution in resolutions:
        if args.attention_mode == "auto":
            tangent.ATTENTION_JVP_MODE = "chunked" if resolution <= 128 else "moments"
            tangent.ATTENTION_VJP_MODE = "chunked"
        else:
            tangent.ATTENTION_JVP_MODE = args.attention_mode
            tangent.ATTENTION_VJP_MODE = args.attention_mode
        tangent.GN_STATS_MODE = "native" if resolution <= 96 else "manual"
        model, _ = dd.load_flow_model(resolution, 1, 50, str(device))
        dataset = torch.load(
            ROOT / f"data/circles-{resolution}-n1.pt", map_location="cpu",
            weights_only=True, mmap=True)["test"]
        truth = dataset[0:1].to(device=device, dtype=torch.float32)
        if dataset.dtype == torch.uint8:
            truth.div_(255.0)
        del dataset
        angles = torch.linspace(0, math.pi, N_ANGLES + 1, device=device)[:-1]
        operator = lambda image: sparse_ct(image, angles)
        observation = operator(truth).detach()
        source = torch.randn(
            1, 1, resolution, resolution,
            generator=torch.Generator().manual_seed(resolution),
            dtype=torch.float32).to(device)

        # Uncheckpointed autograd is faster while its full tape fits. At 160px
        # and above, per-step activation checkpointing is required on 48 GiB.
        use_checkpointing = resolution >= 160
        for algorithm in args.algorithms.split(","):
            gc.collect()
            torch.cuda.empty_cache()
            try:
                if algorithm == "D-Flow":
                    metrics = benchmark_dflow(
                        model, source, operator, observation, device,
                        args.warmup, args.timed, use_checkpointing)
                else:
                    metrics = benchmark_dlr(
                        model, source, operator, observation, device,
                        args.warmup, args.timed, not args.no_cuda_graphs)
            except torch.OutOfMemoryError:
                if algorithm != "D-Flow" or use_checkpointing:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                use_checkpointing = True
                metrics = benchmark_dflow(
                    model, source, operator, observation, device,
                    args.warmup, args.timed, checkpointed=True)
            row = {
                "resolution": resolution,
                "dimension": resolution ** 2,
                "algorithm": algorithm,
                **metrics,
            }
            rows.append(row)
            write_results(args.output, metadata, rows)
            print(
                f"res {resolution:3d} {algorithm:10s} "
                f"{metrics['seconds_per_iteration_mean']:8.3f} s/iter "
                f"(median {metrics['seconds_per_iteration_median']:8.3f}) "
                f"peak {metrics['peak_allocated_gib']:6.2f} GiB",
                flush=True,
            )
        del model, truth, observation, source
        gc.collect()
        torch.cuda.empty_cache()

    write_results(args.output, metadata, rows)
    print(f"Wrote {args.output} and {args.output.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()

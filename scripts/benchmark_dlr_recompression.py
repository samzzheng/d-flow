"""Compare exact recompression implementations with enrichment disabled.

Example: .venv/bin/python scripts/benchmark_dlr_recompression.py --device cpu
The source, scalar path, rank, integrator and probe seeds are held fixed.
CUDA timing requires an otherwise idle GPU; setup is excluded from timings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlr_dflow import dlr_dflow as core
from dlr_dflow import enriched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--trace-probes", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1:
        parser.error("steps must be at least two and repeats must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, _ = core.load_flow_model(args.resolution, device=str(device),
                                    epoch=args.epoch, run_dir=args.run_dir)
    generator = torch.Generator(device=device).manual_seed(2026)
    z = torch.randn(1, 1, args.resolution, args.resolution,
                    device=device, generator=generator)
    directions = torch.randn(z.numel(), 8, device=device, generator=generator)
    terminal_gradient = torch.randn(z.shape, device=device, generator=generator)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(transpose):
        with patch.object(enriched, "RECOMPRESS_FROM_TRANSPOSE", transpose):
            return enriched.evolve_trace_centered_enriched(
                model, z, args.steps, rank=args.rank,
                trace_probes=args.trace_probes, enrichment_probes=0,
                zero_terminal_baseline=False, seed=0)

    def snapshot(result):
        x, (u, s, v), alpha = result
        return dict(state=x, alpha=torch.as_tensor(alpha, device=device),
                    residual_action=u @ (s @ (v.T @ directions)),
                    source_gradient=core.approx_source_gradient(
                        (u, s, v), terminal_gradient, alpha))

    variants = {"explicit_jvp": False, "reuse_vjp": True}
    snapshots = {name: snapshot(run(flag)) for name, flag in variants.items()}
    timings = {name: [] for name in variants}
    for repeat in range(args.repeats):
        order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
        for name in order:
            synchronize()
            start = time.perf_counter()
            result = run(variants[name])
            synchronize()
            timings[name].append(time.perf_counter() - start)
            del result
    errors = {}
    for key, reference in snapshots["explicit_jvp"].items():
        difference = (snapshots["reuse_vjp"][key] - reference).norm().item()
        errors[key] = {"absolute": difference,
                       "relative": difference / max(reference.norm().item(), 1e-30)}
    medians = {key: statistics.median(values) for key, values in timings.items()}
    report = {
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "seconds": timings,
        "median_seconds": medians,
        "speedup": medians["explicit_jvp"] / medians["reuse_vjp"],
        "errors": errors,
        "product_calls": {"explicit_jvp": 3 * args.steps - 1,
                          "reuse_vjp": 2 * args.steps},
        "jvp_columns_removed": (args.steps - 1) * args.rank,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

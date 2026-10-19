"""Benchmark gradient evaluations with fixed numerical settings.

Example: .venv/bin/python scripts/benchmark_dlr_dflow_hotpath.py --cuda-graphs
Capture/setup is reported separately from repeated evaluations. --source-root
can point to a saved copy of dlr_dflow for a before/after comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--trace-probes", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--attention-mode", choices=["moments", "chunked"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.source_root.resolve()))
    import dlr_dflow.dlr_dflow as core
    core.ROOT = ROOT
    if args.attention_mode:
        import dlr_dflow.tangent_unet as tangent
        tangent.ATTENTION_MODE = args.attention_mode
    model, _ = core.load_flow_model(args.resolution, device=str(device))
    z = torch.randn(1, 1, args.resolution, args.resolution, device=device,
                    generator=torch.Generator(device=device).manual_seed(2026))
    operator = core.make_blur(2.0, str(device))
    products = None
    if args.cuda_graphs:
        from dlr_dflow.graphed_products import GraphedAProducts
        products = GraphedAProducts(model, z)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def dlr():
        x, factors, alpha = core.sensitivity_factors(
            model, z, args.steps, rank=args.rank,
            trace_probes=args.trace_probes, products=products)
        gradient = core.approx_source_gradient(
            factors, core.terminal_gradient(x, operator, torch.zeros_like(z)), alpha)
        return x.detach(), gradient.detach(), factors, alpha

    def exact():
        source = z.detach().requires_grad_(True)
        x = core.euler_flow(model, source, args.steps)
        loss = 0.5 * operator(x).square().sum()
        gradient, = torch.autograd.grad(loss, source)
        return x.detach(), gradient.detach()

    report = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    report["torch"] = torch.__version__
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
    artifacts = {}
    for name, fn in [("dlr", dlr)] + ([] if args.skip_exact else [("exact", exact)]):
        synchronize()
        start = time.perf_counter()
        result = fn()
        synchronize()
        setup = time.perf_counter() - start
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(args.repeats):
            synchronize()
            start = time.perf_counter()
            result = fn()
            synchronize()
            elapsed.append(time.perf_counter() - start)
        report[name] = {"first_call_s": setup, "median_s": statistics.median(elapsed),
                        "samples_s": elapsed}
        if device.type == "cuda":
            report[name]["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        artifacts[name] = {"x": result[0].cpu(), "gradient": result[1].cpu()}
        if name == "dlr":
            artifacts[name].update(factors=tuple(t.cpu() for t in result[2]), alpha=result[3])
        print(json.dumps({name: report[name]}), flush=True)
        del result
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save(artifacts, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()

"""Brief, memory-capped GPU timing of larger DLR initialization sketches.

Run under an external 60-second timeout on a shared GPU. No optimization,
exact reverse-mode flow tape, or long sweep is performed. Interleaved warmed
measurements share the identical 99 zero-enrichment step graphs.
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import test_parallel_bug_integrator as helper
from dlr_dflow import dlr_dflow as core, enriched
from dlr_dflow.graphed_products import GraphedAProducts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--memory-fraction", type=float, default=.14)
    parser.add_argument("--chunk-initial-products", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/fast_accuracy_recovery_2026_09_27/gpu_sketch_timing.json")
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= .15:
        parser.error("Memory fraction must be in (0, 0.15] on this shared GPU")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": {"device": str(device), "memory_fraction": args.memory_fraction,
        "resolution": 250, "non_attention": True, "rank": 8, "trace_probes": 4,
        "enrichment_probes": 0, "flow_steps": 100, "terminal_alpha": 0,
        "source": "saved exact CPU sample0 step20", "seed": 0,
        "timing": "complete fresh DLR sensitivity solve and terminal/source gradient, excludes Armijo",
        "gpu_is_shared": True}, "calls": [], "status": "running"}
    report["protocol"]["initial_product_chunks"] = {"A": 20, "AT": 16} if args.chunk_initial_products else None
    started = time.perf_counter()

    def save():
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.out)

    try:
        model, tag = helper.load(250, device)
        assert not model.mid_attn and not any(model.att_channels)
        report["protocol"].update(model=tag, torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(device), cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
        source = torch.load(ROOT / "outputs/accuracy_fast_euler_2026_09_27/exact_from_z0/sources_sample0_exact.pt",
            map_location="cpu", weights_only=True)["sources"][20].to(device)
        angles = torch.linspace(0, torch.pi, 16, device=device)[:-1]
        data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
        _, target = helper.make_problem(250, 0, data, angles, device)
        core.SHARE_LINEARIZATION = True
        enriched.SKIP_UNTRUNCATED_SVD = True
        products = GraphedAProducts(model, source, share_linearization=True,
            full_step_graphs=True,
            max_jvp_columns=20 if args.chunk_initial_products else None,
            max_vjp_columns=16 if args.chunk_initial_products else None)
        save()
        for index, oversampling in enumerate([8, 32, 8, 32, 32, 8]):
            # Leave time for teardown before the external deadline.
            if time.perf_counter() - started > 48:
                report["status"] = "bounded_time_budget"
                break
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            x, factors, _ = core.sensitivity_factors(model, source, 100, rank=8,
                trace_probes=4, enrichment_probes=0, initialization_oversampling=oversampling,
                seed=0, factor_stride=1, zero_terminal_baseline=True, products=products)
            terminal = helper.terminal_gradient(x, target, 250, angles)
            gradient = core.approx_source_gradient(factors, terminal, 0.0)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - begin
            row = {"oversampling": oversampling, "warmup": index < 2, "seconds": elapsed,
                "gradient_norm": float(gradient.norm()),
                "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20}
            report["calls"].append(row)
            save()
            print(json.dumps(row), flush=True)
        else:
            report["status"] = "complete"
        report["medians"] = {str(width): statistics.median([row["seconds"] for row in report["calls"]
            if row["oversampling"] == width and not row["warmup"]]) for width in (8, 32)
            if any(row["oversampling"] == width and not row["warmup"] for row in report["calls"])}
    except torch.cuda.OutOfMemoryError as error:
        report["status"] = "memory_cap_exceeded"
        report["error"] = str(error)
    finally:
        report["total_seconds"] = time.perf_counter() - started
        save()
    print(json.dumps({"status": report["status"], "seconds": report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

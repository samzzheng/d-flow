"""Measure warmed five-step gradient/forward throughput on GPU1."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from .operators import CT, noisy_measurement
from .run import Evaluator, METHODS, write_json


def worker(index: int, resolution: int, repeats: int, barrier: Any, out: Path) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.12, device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = Path("outputs/five_step_noise020_n124_20261013/production")
    settings = json.loads((root / "manifest.json").read_text())["settings"]
    data = torch.load(f"data/circles-{resolution}-n2.pt", map_location="cpu", weights_only=True, mmap=True)
    stored = data["test"][index:index + 1]
    truth = stored.to(device=device, dtype=torch.float32) / (255 if stored.dtype == torch.uint8 else 1)
    seed = settings["seed"] + 10000 * resolution + 200 + index
    model_dir = Path(settings["run_template"].format(resolution=resolution, circles=2))
    model, _ = load_flow_model(resolution, 2, settings["epoch"], str(device), model_dir)
    source = torch.randn(truth.shape, generator=torch.Generator().manual_seed(seed)).to(device).requires_grad_(True)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = "native" if resolution <= 96 else "manual"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    products = GraphedAProducts(model, source.detach(), share_linearization=True,
        full_step_graphs=True, max_step_jvp_columns=6, max_step_vjp_columns=4)
    operator = CT(resolution, 10, device)
    observed = noisy_measurement(operator(truth), settings["noise_relative"], seed + 20000000)
    evaluator = Evaluator(model, operator, observed, settings, seed + 30000000, products)
    results = {}
    for method in METHODS:
        evaluator.objective(source.detach())
        for _ in range(2):
            evaluator.gradient(source, method)
            evaluator.objective(source.detach())
        torch.cuda.synchronize(device)
        barrier.wait(timeout=120)
        started = time.perf_counter()
        for _ in range(repeats):
            gradient = evaluator.gradient(source, method)
            value = evaluator.objective(source.detach())
        torch.cuda.synchronize(device)
        results[method] = {"seconds": time.perf_counter() - started,
                           "gradient_norm": float(gradient.double().norm()), "objective": value}
        assert torch.isfinite(gradient).all()
        barrier.wait(timeout=120)
    write_json(out / f"worker_{index}.json", {"sample": index, "source_seed": seed, "methods": results,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[96, 250])
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 4])
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    context = mp.get_context("spawn")
    records = []
    for resolution in args.resolutions:
        for block, count in enumerate(args.workers):
            out = args.out / f"r{resolution}_block{block}_workers{count}"
            barrier = context.Barrier(count + 1)
            processes = [context.Process(target=worker, args=(i, resolution, args.repeats, barrier, out))
                         for i in range(count)]
            try:
                for process in processes:
                    process.start()
                stages = {}
                for method in METHODS:
                    barrier.wait(timeout=120)
                    started = time.perf_counter()
                    barrier.wait(timeout=120)
                    elapsed = time.perf_counter() - started
                    stages[method] = {"seconds": elapsed, "evaluations_per_second": count * args.repeats / elapsed}
                    print(resolution, count, method, stages[method], flush=True)
                for process in processes:
                    process.join(timeout=30)
                    if process.exitcode != 0:
                        raise RuntimeError(f"Benchmark worker failed: {process.pid}, {process.exitcode}")
                workers = [json.loads((out / f"worker_{i}.json").read_text()) for i in range(count)]
                records.append({"resolution": resolution, "workers": count, "stages": stages,
                                "worker_results": workers})
                write_json(args.out / "results.json", {"repeats": args.repeats, "records": records,
                    "scope": "Distinct test sample/source/noise seeds per worker. One gradient plus one canonical forward per repetition at each fixed source; warmed, no optimizer/line search, not full solve throughput."})
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join()


if __name__ == "__main__":
    main()

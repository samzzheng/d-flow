"""Compare DLR implementation speed and probe budgets on matched sparse CT points.

Examples (run on an otherwise idle GPU):
  .venv/bin/python scripts/benchmark_dlr_speedup.py --resolutions 64 --samples 0
  .venv/bin/python scripts/benchmark_dlr_speedup.py --iterates 0 10 20 \
      --generate-missing-points --points-cache outputs/dlr_speed_points

``shared`` enables the shared primal/pointwise tape and, for zero enrichment,
complete-step graphs. Use --no-full-step-graphs to isolate the tape change.
``baseline`` also disables SKIP_UNTRUNCATED_SVD. Each variant
owns its CUDA graphs. First calls, including capture, are reported separately
from randomized, interleaved warm measurements of a complete source gradient.
Nonzero test points use exact-gradient capped Armijo L-BFGS, never DLR iterates.
The cache stores those sources and exact references for subsequent runs.

--source-root can select a saved repository/package copy in a separate process;
use --implementations separate if that copy predates SHARE_LINEARIZATION.
This is a gradient benchmark, not evidence of end-to-end solve speed or quality.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"trace32_e4": (32, 4), "trace8_e4": (8, 4),
           "trace4_e4": (4, 4), "trace4_e0": (4, 0)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 250])
    parser.add_argument("--samples", type=int, nargs="+", default=[0])
    parser.add_argument("--iterates", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--oversampling", type=int, default=8)
    parser.add_argument("--factor-stride", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--order-seed", type=int, default=20260927)
    parser.add_argument("--implementations", nargs="+", choices=["baseline", "separate", "shared"],
                        default=["baseline", "separate", "shared"])
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-step-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-terminal-baseline", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--points-cache", type=Path)
    parser.add_argument("--generate-missing-points", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/dlr_speedup")
    args = parser.parse_args()
    for name in ("steps", "rank", "repeats", "threads", "factor_stride"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if min(args.iterates + args.samples + [args.warmups, args.oversampling]) < 0:
        parser.error("sample/iterate indices, warmups and oversampling must be nonnegative")
    if args.points_cache is None:
        args.points_cache = args.out / "points"
    return args


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metrics(estimate, target):
    estimate, target = estimate.flatten(), target.flatten()
    target_norm, estimate_norm = target.norm().item(), estimate.norm().item()
    difference = (estimate - target).norm().item()
    if not all(math.isfinite(v) for v in (target_norm, estimate_norm, difference)):
        return {"relative_error": None, "absolute_error": None,
                "cosine": None, "norm_ratio": None, "finite": False}
    return {"relative_error": difference / target_norm if target_norm else None,
            "absolute_error": difference,
            "cosine": (estimate @ target).item() / (target_norm * estimate_norm)
                      if target_norm and estimate_norm else None,
            "norm_ratio": estimate_norm / target_norm if target_norm else None,
            "finite": True}


def load_points(args, helper, model, resolution, sample, z0, y, angles, metadata):
    path = args.points_cache / f"res{resolution}_sample{sample}.pt"
    cached = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
    if cached is not None and cached["metadata"] != metadata:
        raise ValueError(f"Point-cache protocol mismatch: {path}; choose a new --points-cache")
    if cached is None:
        cached = {"metadata": metadata, "points": {}, "references": {}}
    cached["points"].setdefault(0, z0.detach().cpu())
    missing = set(args.iterates) - set(cached["points"])
    if missing:
        if not args.generate_missing_points:
            raise ValueError(f"Missing exact Armijo iterates {sorted(missing)} in {path}. "
                             "Use --generate-missing-points once to create them.")
        print(json.dumps({"event": "generate_exact_points", "resolution": resolution,
                          "sample": sample, "iterates": sorted(missing)}), flush=True)
        points = helper.exact_iterates(model, z0, y, resolution, angles, sorted(missing))
        cached["points"].update({k: value.cpu() for k, value in points.items()})
    args.points_cache.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = path.with_suffix(".tmp")
        torch.save(cached, temporary)
        temporary.replace(path)

    save()
    return cached, save


def main():
    args = parse_args()
    # Package selection must precede helper imports, which themselves import dd.
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.source_root.resolve()))
    import dlr_dflow.dlr_dflow as dd
    dd.ROOT = ROOT
    from dlr_dflow import enriched, tangent_unet
    from dlr_dflow.graphed_products import GraphedAProducts
import test_parallel_bug_integrator as helper
    helper.STEPS = args.steps
    if "shared" in args.implementations and not hasattr(dd, "SHARE_LINEARIZATION"):
        raise ValueError("Selected source has no SHARE_LINEARIZATION; use --implementations separate")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This harness measures CUDA execution; select --device cuda:N")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.order_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "results.json"
    report = {"settings": {k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()},
              "environment": {"torch": str(torch.__version__),
                              "gpu": torch.cuda.get_device_name(device),
                              "source_module": dd.__file__,
                              "cuda": torch.version.cuda,
                              "cudnn_benchmark": torch.backends.cudnn.benchmark,
                              "cudnn_deterministic": torch.backends.cudnn.deterministic,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
              "protocol": {"noise": helper.NOISE, "base_seed": helper.SEED,
                           "angles": 15, "objective": "relative_scaled",
                           "point_optimizer": "exact Armijo L-BFGS, step cap 0.01 sqrt(d)",
                           "armijo_initial_step_policy": helper.bench.ArmijoLBFGS.initial_step_policy,
                           "armijo_progress_policy": helper.bench.ArmijoLBFGS.progress_policy,
                           "timed_operation": "flow, terminal gradient, source gradient",
                           "capture_policy": "independent CUDA graphs for every variant",
                           "checkpoint_exact_at_resolution_ge": 160},
              "resolutions": {}}
    original_share = getattr(dd, "SHARE_LINEARIZATION", False)
    supports_skip_svd = hasattr(enriched, "SKIP_UNTRUNCATED_SVD")
    original_skip_svd = getattr(enriched, "SKIP_UNTRUNCATED_SVD", False)

    def synchronize():
        torch.cuda.synchronize(device)

    def timed(fn):
        synchronize()
        start = time.perf_counter()
        output = fn()
        synchronize()
        return output, time.perf_counter() - start

    def progress(event, **details):
        atomic_json(report_path, report)
        print(json.dumps({"event": event, **details}, allow_nan=False), flush=True)

    try:
        for resolution in args.resolutions:
            model, model_tag = helper.load(resolution, device)
            checkpoint_path = (ROOT / "outputs/fm_unet" /
                ("circles-64/n1/checkpoints/epoch_0050.pt" if resolution == 64 else
                 f"circles-{resolution}-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"))
            with checkpoint_path.open("rb") as handle:
                checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            angles = torch.linspace(0, math.pi, 16, device=device)[:-1]
            data = torch.load(ROOT / f"data/circles-{resolution}-n1.pt", map_location="cpu",
                              weights_only=True, mmap=True)["test"]
            variants, products_by_variant, capture_events = {}, {}, {}
            for implementation in args.implementations:
                for config in args.configs:
                    probes, enrichment = CONFIGS[config]
                    name = f"{implementation}/{config}"
                    dd.SHARE_LINEARIZATION = implementation == "shared"
                    skip_svd = supports_skip_svd and implementation != "baseline"
                    enriched.SKIP_UNTRUNCATED_SVD = skip_svd
                    product_options = {}
                    if implementation == "shared":
                        product_options["full_step_graphs"] = args.full_step_graphs
                    products = GraphedAProducts(model, torch.zeros(
                        1, 1, resolution, resolution, device=device),
                        **product_options) if args.cuda_graphs else None
                    products_by_variant[name] = products
                    capture_events[name] = []
                    if products is not None:
                        for method in ("_capture", "_capture_linearization"):
                            if not hasattr(products, method):
                                continue
                            original_capture = getattr(products, method)

                            def capture(*a, _original=original_capture, _name=name,
                                        _method=method, **kw):
                                _, seconds = timed(lambda: _original(*a, **kw))
                                capture_events[_name].append({"method": _method,
                                    "arguments": [str(v) for v in a], "seconds": seconds})

                            setattr(products, method, capture)
                            del capture, original_capture
                        if getattr(products, "full_step_graphs", False):
                            original_prepare = products.prepare_step_graph

                            def prepare(*a, _original=original_prepare, _name=name, **kw):
                                backend = _original(*a, **kw)
                                if not getattr(backend, "_benchmark_wrapped", False):
                                    original = backend._capture

                                    def capture_step(_original=original, _name=_name):
                                        _, seconds = timed(_original)
                                        capture_events[_name].append({"method": "whole_step._capture",
                                                                     "arguments": [], "seconds": seconds})

                                    backend._capture = capture_step
                                    backend._benchmark_wrapped = True
                                return backend

                            products.prepare_step_graph = prepare
                            del prepare, original_prepare
                    variants[name] = {"shared_linearization": implementation == "shared",
                                      "full_step_graphs": bool(products is not None
                                          and getattr(products, "full_step_graphs", False)
                                          and enrichment == 0 and args.factor_stride == 1),
                                      "skip_untruncated_svd": skip_svd,
                                      "trace_probes": probes, "enrichment_probes": enrichment,
                                      "rank": args.rank, "factor_stride": args.factor_stride}
            res = {"model": model_tag, "checkpoint_sha256": checkpoint_hash,
                   "flags": {key: getattr(enriched, key, None) for key in
                             ("FUSE_K_IMAGE", "RECOMPRESS_FROM_TRANSPOSE",
                              "SHARE_PRIMAL", "PROBES_AS_ENRICHMENT", "V_ENRICH_FROM_Q2")},
                   "variants": variants, "setup": {}, "points": []}
            res["flags"]["FUSE_POINTWISE_DERIVATIVES"] = getattr(
                tangent_unet, "FUSE_POINTWISE_DERIVATIVES", None)
            report["resolutions"][str(resolution)] = res
            for sample in args.samples:
                if sample >= len(data):
                    raise ValueError(f"Sample {sample} outside dataset of size {len(data)}")
                z0, y = helper.make_problem(resolution, sample, data, angles, device)
                metadata = {"checkpoint_sha256": checkpoint_hash, "resolution": resolution,
                            "sample": sample, "steps": args.steps, **report["protocol"],
                            "torch": str(torch.__version__),
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
                cached, save_cache = load_points(args, helper, model, resolution, sample,
                                                z0, y, angles, metadata)
                for iterate in args.iterates:
                    z = cached["points"][iterate].to(device)
                    if iterate not in cached["references"]:
                        (g, residual, x), seconds = timed(lambda: helper.exact_gradient(
                            model, z, y, resolution, angles))
                        cached["references"][iterate] = {"gradient": g.cpu(), "x": x.cpu(),
                                                         "residual": residual,
                                                         "first_reference_s": seconds}
                        save_cache()
                    reference = cached["references"][iterate]
                    g_exact, x_exact = reference["gradient"].to(device), reference["x"].to(device)
                    gx_exact = helper.terminal_gradient(x_exact, y, resolution, angles)
                    point = {"sample": sample, "iterate": iterate,
                             "residual": reference["residual"],
                             "source_norm_over_sqrt_d": z.norm().item() / math.sqrt(z.numel()),
                             "accuracy": {}, "timing": {}, "orders": []}
                    res["points"].append(point)

                    def evaluate(name):
                        if name == "exact":
                            gradient, _, x = helper.exact_gradient(model, z, y, resolution, angles)
                            return x, gradient, None, 0.0
                        config = variants[name]
                        dd.SHARE_LINEARIZATION = config["shared_linearization"]
                        enriched.SKIP_UNTRUNCATED_SVD = config["skip_untruncated_svd"]
                        x, factors, alpha = enriched.evolve_trace_centered_enriched(
                            model, z, args.steps, rank=args.rank,
                            trace_probes=config["trace_probes"],
                            initialization_oversampling=args.oversampling,
                            enrichment_probes=config["enrichment_probes"],
                            factor_stride=args.factor_stride,
                            products=products_by_variant[name], zero_terminal_baseline=False)
                        gx = helper.terminal_gradient(x, y, resolution, angles)
                        gradient = dd.approx_source_gradient(
                            factors, gx, 0.0 if args.zero_terminal_baseline else alpha).flatten()
                        return x, gradient, factors, alpha

                    names = ["exact", *variants]
                    snapshots = {}
                    for name in names:
                        allocated_before = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        result, first_seconds = timed(lambda name=name: evaluate(name))
                        x, gradient, factors, alpha = result
                        if name not in res["setup"]:
                            res["setup"][name] = {"first_call_s": first_seconds,
                                                  "peak_above_existing_mib": (
                                                      torch.cuda.max_memory_allocated(device) - allocated_before) / 2**20,
                                                  "persistent_added_mib": (
                                                      torch.cuda.memory_allocated(device) - allocated_before) / 2**20,
                                                  "capture_events": capture_events.get(name, [])}
                        accuracy = {"gradient": metrics(gradient, g_exact),
                                    "terminal_state": metrics(x, x_exact),
                                    "evolved_alpha": float(alpha)}
                        if factors is not None:
                            gx = helper.terminal_gradient(x, y, resolution, angles)
                            accuracy["zero_terminal_baseline"] = metrics(
                                dd.approx_source_gradient(factors, gx, 0.0), g_exact)
                            accuracy["evolved_terminal_baseline"] = metrics(
                                dd.approx_source_gradient(factors, gx, alpha), g_exact)
                            accuracy["common_terminal_gradient"] = metrics(
                                dd.approx_source_gradient(factors, gx_exact,
                                    0.0 if args.zero_terminal_baseline else alpha), g_exact)
                        snapshots[name] = (x.detach().clone(), gradient.detach().clone())
                        point["accuracy"][name] = accuracy
                        for _ in range(args.warmups):
                            evaluate(name)
                        synchronize()
                        progress("accuracy", resolution=resolution, sample=sample,
                                 iterate=iterate, variant=name, **accuracy)
                        del result, factors
                    for config in args.configs:
                        old, new = f"separate/{config}", f"shared/{config}"
                        if old in snapshots and new in snapshots:
                            point["accuracy"][new]["versus_separate"] = {
                                "terminal_state": metrics(snapshots[new][0], snapshots[old][0]),
                                "gradient": metrics(snapshots[new][1], snapshots[old][1])}
                        baseline = f"baseline/{config}"
                        if baseline in snapshots:
                            for implementation in ("separate", "shared"):
                                candidate = f"{implementation}/{config}"
                                if candidate in snapshots:
                                    point["accuracy"][candidate]["versus_baseline"] = {
                                        "terminal_state": metrics(snapshots[candidate][0], snapshots[baseline][0]),
                                        "gradient": metrics(snapshots[candidate][1], snapshots[baseline][1])}
                    del snapshots
                    times = {name: [] for name in names}
                    rng = random.Random(args.order_seed + resolution * 10000 + sample * 100 + iterate)
                    for repeat in range(args.repeats):
                        order = list(names)
                        rng.shuffle(order)
                        point["orders"].append(order)
                        for name in order:
                            result, seconds = timed(lambda name=name: evaluate(name))
                            del result
                            times[name].append(seconds)
                            point["timing"][name] = {
                                "median_s": statistics.median(times[name]),
                                "min_s": min(times[name]), "samples_s": times[name]}
                            progress("timing", resolution=resolution, sample=sample,
                                     iterate=iterate, repeat=repeat, variant=name, seconds=seconds)
                    exact_time = point["timing"]["exact"]["median_s"]
                    for name in variants:
                        point["timing"][name]["speedup_vs_exact"] = (
                            exact_time / point["timing"][name]["median_s"])
                    progress("point_complete", resolution=resolution, sample=sample, iterate=iterate)
            products_by_variant.clear()
            del products, model, data
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        dd.SHARE_LINEARIZATION = original_share
        enriched.SKIP_UNTRUNCATED_SVD = original_skip_svd
    progress("complete", output=str(report_path))


if __name__ == "__main__":
    main()

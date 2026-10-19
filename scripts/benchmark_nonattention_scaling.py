"""Spatial scaling of exact D-Flow and fast DLR with non-attention UNets.

Each timing is one fresh source gradient, with no optimizer or line search.
Checkpoint policies and derivative kernels stay fixed across image sizes.
The trained suite uses each resolution's checkpoint; fixed250 uses the same
250px checkpoint at every input size as a pure architecture control.
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
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
from dlr_dflow import dlr_dflow as core, enriched, tangent_unet as tangent
from dlr_dflow.graphed_products import GraphedAProducts
from models.unet.unet import AttentionBlock


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resolutions", type=int, nargs="+", default=[64, 96, 128, 160, 200, 250])
    p.add_argument("--suites", nargs="+", choices=["trained", "fixed250"], default=["trained", "fixed250"])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--trace-probes", type=int, default=4)
    p.add_argument("--oversampling", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--components", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gn-stats", choices=["native", "manual"], default="manual")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/nonattention_scaling_2026_09_27")
    args = p.parse_args()
    if min(args.steps, args.rank, args.trace_probes, args.repeats) < 1:
        p.error("Step, rank, probe, and repeat counts must be positive")
    return args


class Timeline:
    def __init__(self):
        self.marks = []

    def mark(self, label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.marks.append((label, event))

    def durations(self):
        # CUDA events include GPU idle gaps caused by CPU dispatch between
        # recorded events. These are elapsed stage times, not kernel busy time.
        return {label: left.elapsed_time(right) / 1000
                for (_, left), (label, right) in zip(self.marks, self.marks[1:])}


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    core.HAND_JVP = core.HAND_VJP = core.SHARE_LINEARIZATION = True
    tangent.GN_STATS_MODE = args.gn_stats
    enriched.SKIP_UNTRUNCATED_SVD = True
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "threads": torch.get_num_threads(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "protocol": {"seed": 20260807, "angles": 15, "noise": .01,
            "timed_operation": "one fresh gradient; no optimizer, line search, or diagnostics",
            "dlr": "rank8/trace4/enrichment0 defaults; full-step CUDA graphs; terminal alpha0",
            "exact_checkpoint_policies": ["never", "every UNet step"],
            "stage_timing": "CUDA event elapsed time, including possible host-dispatch gaps",
            "repetitions": "one initial warm/capture call; randomized interleaving of measured calls"},
        "rows": []}

    def save(event, **fields):
        (args.out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)

    def terminal(x, target, resolution, angles):
        leaf = x.detach().requires_grad_(True)
        value, _ = bench.data_objective(bench.sparse_radon(leaf, angles), target,
                                        resolution, "relative_scaled")
        gradient, = torch.autograd.grad(value, leaf)
        return gradient

    for suite in args.suites:
        for resolution in args.resolutions:
            model_resolution = resolution if suite == "trained" else 250
            run = ROOT / f"outputs/fm_unet/circles-{model_resolution}-noattn-ch16-b8/n1"
            config = json.loads((run / "config.json").read_text())
            checkpoint_path = run / "checkpoints/epoch_0100.pt"
            with checkpoint_path.open("rb") as handle:
                checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            model = bench.load_model(model_resolution, 100, device,
                "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
            assert not any(isinstance(m, AttentionBlock) for m in model.modules())
            angles = torch.linspace(0, math.pi, 16, device=device)[:-1]
            data = torch.load(ROOT / f"data/circles-{resolution}-n1.pt", mmap=True,
                              weights_only=True, map_location="cpu")["test"]
            truth = bench.normalize_images(data[:1]).to(device)
            del data
            with torch.no_grad():
                clean = bench.sparse_radon(truth, angles)
            target, _, _ = bench.add_relative_noise(clean.cpu(), .01,
                torch.Generator().manual_seed(20260807 + resolution * 1000))
            target = target.to(device)
            z = torch.randn(truth.shape, generator=torch.Generator().manual_seed(
                20260807 + 10_000_000 + resolution * 1000)).to(device)
            products = GraphedAProducts(model, z)
            row = {"suite": suite, "resolution": resolution, "pixels": resolution**2,
                "architecture": config["arch"], "parameters": sum(p.numel() for p in model.parameters()),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash,
                "methods": {}, "accuracy": {}}
            report["rows"].append(row)
            save("start", suite=suite, resolution=resolution)

            def exact(checkpointed, timeline):
                source = z.detach().requires_grad_(True)
                x = bench.solve_flow(model, source, args.steps, "forward_euler", checkpointed)
                timeline.mark("forward")
                gx = terminal(x, target, resolution, angles)
                timeline.mark("terminal_gradient")
                grad, = torch.autograd.grad(x, source, gx)
                timeline.mark("backward")
                return x.detach(), grad.detach()

            def fast(timeline):
                original_prepare = products.prepare_step_graph
                original_canonical = enriched._canonicalize_factors

                def prepare(*values, **kw):
                    out = original_prepare(*values, **kw)
                    timeline.mark("initialization")
                    return out

                def canonical(factors):
                    timeline.mark("recurring_evolution")
                    result = original_canonical(factors)
                    timeline.mark("final_canonicalization")
                    return result

                with patch.object(products, "prepare_step_graph", prepare), \
                        patch.object(enriched, "_canonicalize_factors", canonical):
                    x, factors, alpha = core.sensitivity_factors(model, z, args.steps,
                        rank=args.rank, trace_probes=args.trace_probes,
                        initialization_oversampling=args.oversampling, enrichment_probes=0,
                        products=products, zero_terminal_baseline=True)
                timeline.mark("return_bookkeeping")
                gx = terminal(x, target, resolution, angles)
                timeline.mark("terminal_gradient")
                grad = core.approx_source_gradient(factors, gx, alpha).reshape_as(z)
                timeline.mark("factor_transpose_action")
                return x.detach(), grad.detach()

            functions = {"dflow": lambda t: exact(False, t),
                         "dflow_checkpointed": lambda t: exact(True, t), "fast_dlr": fast}
            saved = {}

            def measure(name):
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                timeline = Timeline()
                begin = time.perf_counter()
                timeline.mark("start")
                result = functions[name](timeline)
                torch.cuda.synchronize(device)
                wall = time.perf_counter() - begin
                peak = (torch.cuda.max_memory_allocated(device) - before) / 2**20
                return result, {"wall_s": wall, "stages_s": timeline.durations(),
                                "peak_above_existing_mib": peak}

            active = []
            for name in functions:
                try:
                    result, timing = measure(name)
                except torch.OutOfMemoryError:
                    row["methods"][name] = {"status": "out_of_memory"}
                    gc.collect()
                    torch.cuda.empty_cache()
                    save("oom", suite=suite, resolution=resolution, method=name)
                    continue
                saved[name] = tuple(value.cpu() for value in result)
                del result
                row["methods"][name] = {"status": "ok", "first_call": timing, "samples": []}
                active.append(name)
                save("warm", suite=suite, resolution=resolution, method=name, seconds=timing["wall_s"])
            order = random.Random(20260927 + resolution)
            for repeat in range(args.repeats):
                sequence = active.copy()
                order.shuffle(sequence)
                for name in sequence:
                    result, timing = measure(name)
                    del result
                    row["methods"][name]["samples"].append(timing)
                    save("timing", suite=suite, resolution=resolution, repeat=repeat,
                         method=name, seconds=timing["wall_s"])
            for name in active:
                data = row["methods"][name]
                data["median_s"] = statistics.median(p["wall_s"] for p in data["samples"])
                data["stages_median_s"] = {stage: statistics.median(p["stages_s"][stage]
                    for p in data["samples"]) for stage in data["samples"][0]["stages_s"]}
                data["peak_increment_mib"] = max(p["peak_above_existing_mib"] for p in data["samples"])
            reference_name = "dflow" if "dflow" in saved else "dflow_checkpointed"
            xr, gr = saved[reference_name]
            for name, (x, g) in saved.items():
                row["accuracy"][name] = {"gradient_relative_error": float((g-gr).norm()/gr.norm()),
                    "terminal_state_relative_error": float((x-xr).norm()/xr.norm()),
                    "gradient_cosine": float(torch.nn.functional.cosine_similarity(g.flatten(),gr.flatten(),dim=0))}
            if args.components:
                from profile_dlr_scaling_components import profile_components
                del products
                gc.collect()
                torch.cuda.empty_cache()
                row["components"] = profile_components(model, z, rank=args.rank,
                    trace_probes=args.trace_probes, oversampling=args.oversampling,
                    repeats=args.repeats, steps=args.steps)
            else:
                del products
            save("resolution_complete", suite=suite, resolution=resolution,
                timings={n: row["methods"][n]["median_s"] for n in active})
            del model, z, truth, target, saved
            gc.collect()
            torch.cuda.empty_cache()
    save("complete", output=str(args.out / "results.json"))


if __name__ == "__main__":
    main()

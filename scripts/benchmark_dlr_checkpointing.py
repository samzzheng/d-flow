"""Bounded checkpointed/uncheckpointed DLR comparison, one policy per process.

Use --mode tape on CPU to measure retained local activations and derivative
actions, or --mode flow for a complete fresh fast-DLR gradient without Armijo.
CUDA runs are memory capped; use an external timeout on shared devices.
Compare the saved .pt outputs across policies to check numerical agreement.
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
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import test_parallel_bug_integrator as helper
from dlr_dflow import dlr_dflow as core, enriched
from dlr_dflow.graphed_products import GraphedAProducts
from dlr_dflow.tangent_unet import unet_linearize


def retained_activation_bytes(tape, model):
    """Count unique retained tensor storage, excluding model weights/buffers."""
    def key(tensor):
        storage = tensor.untyped_storage()
        return (str(tensor.device), storage.data_ptr(), storage.nbytes())

    excluded = {key(tensor) for tensor in (*model.parameters(), *model.buffers())}
    seen_objects, storages = set(), set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            entry = key(value)
            if entry not in excluded:
                storages.add(entry)
            return
        if id(value) in seen_objects or isinstance(value, torch.nn.Module):
            return
        seen_objects.add(id(value))
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(vars(tape))
    return sum(entry[2] for entry in storages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("tape", "flow"), default="tape")
    parser.add_argument("--checkpoint-blocks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resolution", type=int, default=250)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=.14)
    parser.add_argument("--max-seconds", type=float, default=45)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.resolution, args.steps, args.repeats, args.threads) < 1:
        parser.error("Resolution, steps, repeats and threads must be positive")
    if not 0 < args.memory_fraction <= .15:
        parser.error("Memory fraction must be in (0, 0.15] on shared GPUs")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": {**vars(args), "out": str(args.out), "torch": str(torch.__version__),
        "non_attention": True, "epoch": 100, "seed": 0,
        "timing": "one warmup excluded; no Armijo or optimization",
        "peak_memory": "absolute allocator peak including model and graph capture; fresh process per policy"},
        "calls": [], "status": "running"}
    started = time.perf_counter()

    def save():
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.out)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def memory():
        if device.type != "cuda":
            return {}
        return {"allocated_MiB": torch.cuda.memory_allocated(device) / 2**20,
                "reserved_MiB": torch.cuda.memory_reserved(device) / 2**20,
                "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20}

    outputs = None
    try:
        model_template = "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1"
        model = bench.load_model(args.resolution, 100, device, model_template)
        assert not model.mid_attn and not any(model.att_channels)
        bench.configure_dlr_kernels(args.resolution)
        core.SHARE_LINEARIZATION = True
        enriched.SKIP_UNTRUNCATED_SVD = True
        report["protocol"].update(model=model_template.format(res=args.resolution),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
        if device.type == "cuda":
            report["protocol"]["gpu"] = torch.cuda.get_device_name(device)
        if args.resolution == 250:
            source_path = ROOT / "outputs/accuracy_fast_euler_2026_09_27/exact_from_z0/sources_sample0_exact.pt"
            source = torch.load(source_path, map_location="cpu", weights_only=True)["sources"][20].to(device)
            report["protocol"]["source"] = "saved exact CPU sample0 step20"
        else:
            source = torch.randn((1, 1, args.resolution, args.resolution),
                generator=torch.Generator().manual_seed(0)).to(device)
            report["protocol"]["source"] = "CPU torch.randn seed0"
        if args.mode == "tape":
            tangents = torch.randn((12, 1, args.resolution, args.resolution),
                generator=torch.Generator().manual_seed(1)).to(device)
            cotangents = torch.randn((8, 1, args.resolution, args.resolution),
                generator=torch.Generator().manual_seed(2)).to(device)
            report["protocol"].update(time=0., jvp_columns=12, vjp_columns=8)

            def run():
                tape = unet_linearize(model, 0., source, checkpoint_blocks=args.checkpoint_blocks)
                report["retained_activation_MiB"] = retained_activation_bytes(tape, model) / 2**20
                return {"primal": tape.primal, "jvp": tape.jvp(tangents), "vjp": tape.vjp(cotangents)}
        else:
            angles = torch.linspace(0, torch.pi, 16, device=device)[:-1]
            data = torch.load(ROOT / f"data/circles-{args.resolution}-n1.pt",
                map_location="cpu", weights_only=True, mmap=True)["test"]
            _, target = helper.make_problem(args.resolution, 0, data, angles, device)
            products = (GraphedAProducts(model, source, share_linearization=True,
                full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
                checkpoint_blocks=args.checkpoint_blocks) if device.type == "cuda" else None)
            report["protocol"].update(rank=8, trace_probes=4, enrichment_probes=0,
                initialization_oversampling=8, terminal_alpha=0., factor_stride=1,
                full_step_cuda_graphs=device.type == "cuda", initial_product_chunks=[20, 16])

            def run():
                x, factors, _ = core.sensitivity_factors(model, source, args.steps,
                    rank=8, trace_probes=4, enrichment_probes=0, initialization_oversampling=8,
                    seed=0, factor_stride=1, zero_terminal_baseline=True, products=products,
                    checkpoint_blocks=args.checkpoint_blocks)
                terminal = helper.terminal_gradient(x, target, args.resolution, angles)
                gradient = core.approx_source_gradient(factors, terminal, 0.)
                return {"image": x, "source_gradient": gradient}
        sync()
        save()
        for index in range(args.repeats + 1):
            if time.perf_counter() - started > args.max_seconds:
                report["status"] = "bounded_time_budget"
                break
            outputs = None
            sync()
            begin = time.perf_counter()
            outputs = run()
            sync()
            row = {"warmup": index == 0, "seconds": time.perf_counter() - begin, **memory()}
            report["calls"].append(row)
            save()
            print(json.dumps(row), flush=True)
            if index == 0 and device.type == "cuda":
                # Warm peaks still include live persistent graph allocations.
                report["capture_memory"] = memory()
                torch.cuda.reset_peak_memory_stats(device)
        else:
            report["status"] = "complete"
        measured = [row["seconds"] for row in report["calls"] if not row["warmup"]]
        if measured:
            report["median_seconds"] = statistics.median(measured)
        if outputs is not None:
            torch.save({key: value.detach().cpu() for key, value in outputs.items()}, args.out.with_suffix(".pt"))
    except torch.cuda.OutOfMemoryError as error:
        report["status"] = "memory_cap_exceeded"
        report["error"] = str(error)
    except Exception as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        save()
    print(json.dumps({"status": report["status"], "seconds": report["total_seconds"]}), flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

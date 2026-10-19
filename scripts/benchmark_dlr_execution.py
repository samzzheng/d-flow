"""Bounded DLR execution comparison against D-Flow, one policy per process.

Use --mode tape on CPU to measure retained local activations and derivative
actions, or --mode flow for a complete fresh fast-DLR gradient without Armijo.
CUDA runs are memory capped; use an external timeout on shared devices.
Compare the saved .pt outputs across policies to check numerical agreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import test_parallel_bug_integrator as helper
from dlr_dflow import dlr_dflow as core, enriched
from dlr_dflow import tangent_unet as tangent
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
    parser.add_argument("--mode", choices=("tape", "flow"), default="flow")
    parser.add_argument("--method", choices=("dlr", "dflow-checkpointed", "dflow-uncheckpointed", "compare", "compare-all"), default="dlr")
    parser.add_argument("--memory-format", choices=("contiguous", "channels-last"), default="contiguous")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--disable-conv-tf32", action="store_true")
    parser.add_argument("--fused-norm", action="store_true")
    parser.add_argument("--specialized-algebra", action="store_true")
    parser.add_argument("--source-iterate", type=int, default=20)
    parser.add_argument("--step-jvp-columns", type=int)
    parser.add_argument("--step-vjp-columns", type=int)
    parser.add_argument("--selective-block-chunks", action="store_true",
                        help="Exploratory four-column chunks only within large residual blocks")
    parser.add_argument("--fused-primal", choices=("none", "affine", "full"), default="none")
    parser.add_argument("--checkpoint-blocks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resolution", type=int, default=250)
    parser.add_argument("--trace-probes", type=int, default=4,
                        help="0 = analytic alpha_t = 1 - t (no probe columns)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=.14)
    parser.add_argument("--max-seconds", type=float, default=45)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.resolution, args.steps, args.repeats, args.threads) < 1:
        parser.error("Resolution, steps, repeats and threads must be positive")
    comparison = args.method in {"compare", "compare-all"}
    includes_dlr = comparison or args.method == "dlr"
    if comparison and args.mode != "flow":
        parser.error("Matched comparison requires --mode flow")
    if args.selective_block_chunks and args.checkpoint_blocks:
        parser.error("Selective-block prototype requires cached block tapes")
    if not 0 < args.memory_fraction <= 1:
        parser.error("Memory fraction must be in (0, 1]")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.benchmark_limit = 4
    if args.disable_conv_tf32:
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if args.memory_fraction * total_bytes > free_bytes - 512 * 2**20:
            parser.error("Requested allocator cap exceeds currently free VRAM minus a 512 MiB margin; lower --memory-fraction")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": {**vars(args), "out": str(args.out), "torch": str(torch.__version__),
        "non_attention": True, "epoch": 100, "seed": 0,
        "timing": "one warmup excluded; no Armijo or optimization",
        "peak_memory": "absolute allocator peak including model and graph capture; fresh process per policy"},
        "calls": [], "status": "running"}
    if comparison:
        report["protocol"]["timing"] = "one warmup per method, then randomized interleaving; no Armijo or optimization"
        report["protocol"]["peak_memory"] = "shared process pools: absolute peaks cannot be attributed to individual methods"
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
        if args.memory_format == "channels-last":
            model = model.to(memory_format=torch.channels_last)
        bench.configure_dlr_kernels(args.resolution)
        tangent.GN_ACTION_BACKEND = "triton" if args.fused_norm else "native"
        enriched.SPECIALIZE_ZERO_ENRICHMENT = args.specialized_algebra
        tangent.GN_PRIMAL_BACKEND = "native" if args.fused_primal == "none" else args.fused_primal
        if args.selective_block_chunks:
            # A bounded experiment only: reduce wide fine-level intermediates
            # while leaving coarse-level derivative batches intact.
            def chunk_large_blocks(original):
                def apply(module, record, lanes, *positional, **keywords):
                    if lanes.shape[0] > 4 and lanes.shape[-2] * lanes.shape[-1] >= 15625:
                        return torch.cat([original(module, record, part, *positional, **keywords)
                                          for part in lanes.split(4, dim=0)], dim=0)
                    return original(module, record, lanes, *positional, **keywords)
                return apply
            for name in ("_resblock_J", "_resblock_T", "_block_J", "_block_T"):
                setattr(tangent, name, chunk_large_blocks(getattr(tangent, name)))
        core.SHARE_LINEARIZATION = True
        enriched.SKIP_UNTRUNCATED_SVD = True
        report["protocol"].update(model=model_template.format(res=args.resolution),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
        if device.type == "cuda":
            report["protocol"]["gpu"] = torch.cuda.get_device_name(device)
        if args.resolution == 250:
            source_path = ROOT / "outputs/accuracy_fast_euler_2026_09_27/exact_from_z0/sources_sample0_exact.pt"
            source = torch.load(source_path, map_location="cpu", weights_only=True)["sources"][args.source_iterate].to(device)
            report["protocol"]["source"] = f"saved exact CPU sample0 step{args.source_iterate}"
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

            def run(method):
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
                max_step_jvp_columns=args.step_jvp_columns,
                max_step_vjp_columns=args.step_vjp_columns,
                checkpoint_blocks=args.checkpoint_blocks) if device.type == "cuda" and includes_dlr else None)
            report["protocol"].update(dlr_settings_apply_to="DLR only", rank=8, trace_probes=args.trace_probes, enrichment_probes=0,
                initialization_oversampling=8, terminal_alpha=0., factor_stride=1,
                full_step_cuda_graphs=device.type == "cuda" and includes_dlr, initial_product_chunks=[20, 16],
                exact_dflow_execution="eager autograd through the native model")

            def run(method):
                if method.startswith("dflow-"):
                    variable = source.detach().requires_grad_(True)
                    x = bench.solve_flow(model, variable, args.steps, "forward_euler", method == "dflow-checkpointed")
                    terminal = helper.terminal_gradient(x, target, args.resolution, angles)
                    gradient, = torch.autograd.grad(x, variable, terminal.reshape_as(x))
                    return {"image": x.detach(), "source_gradient": gradient.detach().flatten()}
                x, factors, _ = core.sensitivity_factors(model, source, args.steps,
                    rank=8, trace_probes=args.trace_probes, enrichment_probes=0, initialization_oversampling=8,
                    seed=0, factor_stride=1, zero_terminal_baseline=True, products=products,
                    checkpoint_blocks=args.checkpoint_blocks)
                terminal = helper.terminal_gradient(x, target, args.resolution, angles)
                gradient = core.approx_source_gradient(factors, terminal, 0.)
                return {"image": x, "source_gradient": gradient}
        sync()
        save()
        methods = ["dlr", "dflow-checkpointed"] if comparison else [args.method]
        if args.method == "compare-all":
            methods.append("dflow-uncheckpointed")
        random_order = random.Random(20260928)
        schedule = []
        for index in range(args.repeats + 1):
            order = methods.copy()
            if index:
                random_order.shuffle(order)
            schedule.extend((index, method) for method in order)
        saved_outputs = {}
        for index, method in schedule:
            if time.perf_counter() - started > args.max_seconds:
                report["status"] = "bounded_time_budget"
                break
            outputs = None
            sync()
            begin = time.perf_counter()
            outputs = run(method)
            sync()
            row = {"method": method, "warmup": index == 0, "seconds": time.perf_counter() - begin, **memory()}
            report["calls"].append(row)
            save()
            print(json.dumps(row), flush=True)
            # Defer device-to-host copies until every timed call is complete.
            saved_outputs[method] = {key: value.detach().clone() for key, value in outputs.items()}
            if index == 0 and device.type == "cuda":
                # Warm peaks still include live persistent graph allocations.
                report["capture_memory"] = memory()
                torch.cuda.reset_peak_memory_stats(device)
        else:
            report["status"] = "complete"
        report["medians"] = {}
        for method in methods:
            measured = [row["seconds"] for row in report["calls"] if not row["warmup"] and row["method"] == method]
            if measured:
                report["medians"][method] = statistics.median(measured)
        if len(methods) == 1 and report["medians"]:
            report["median_seconds"] = report["medians"][methods[0]]
        if outputs is not None:
            cpu_outputs = {method: {key: value.cpu() for key, value in values.items()}
                           for method, values in saved_outputs.items()}
            torch.save(cpu_outputs if comparison else cpu_outputs[methods[0]], args.out.with_suffix(".pt"))
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

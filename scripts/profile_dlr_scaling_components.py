"""Measure exact D-Flow/DLR kernels on a fixed, attention-free UNet state.

This is a component microbenchmark, not an end-to-end gradient benchmark.
Stages run independently on representative fixed tensors. Their sum is useful
for attribution but need not equal a single combined graph's latency. All
timings exclude input preparation, warmup, and graph capture. CUDA event times
measure elapsed stream intervals (including launch gaps), not active GPU time.

Example:
  .venv/bin/python scripts/profile_dlr_scaling_components.py \
      --run-dir outputs/fm_unet/circles-64-noattn/n1 --resolution 64 \
      --output outputs/nonattention_scaling/components64.json
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dlr_dflow import dlr_dflow as core
from dlr_dflow import enriched, tangent_unet
from models.unet.unet import AttentionBlock


def _samples(fn, device, repeats, iterations):
    wall, event = [], []
    retained_output = None
    is_cuda = device.type == "cuda"
    for _ in range(repeats):
        if is_cuda:
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        for _ in range(iterations):
            retained_output = fn()
        if is_cuda:
            stop_event.record()
            stop_event.synchronize()
        wall.append(1000 * (time.perf_counter() - start) / iterations)
        if is_cuda:
            event.append(start_event.elapsed_time(stop_event) / iterations)
    # Keep eager outputs alive until their queued operations have completed.
    del retained_output
    return {
        "wall_ms_median": statistics.median(wall), "wall_ms_samples": wall,
        "cuda_event_ms_median": statistics.median(event) if event else None,
        "cuda_event_ms_samples": event,
    }


def _measure(fn, device, repeats, iterations, *, allow_graph=True, capture_stream=None):
    context = torch.cuda.stream(capture_stream) if capture_stream is not None else nullcontext()
    with context:
        return _measure_on_stream(fn, device, repeats, iterations,
                                  allow_graph=allow_graph, capture_stream=capture_stream)


def _measure_on_stream(fn, device, repeats, iterations, *, allow_graph=True, capture_stream=None):
    """Time the same operations eagerly and, if supported, as one CUDA graph."""
    for _ in range(2):
        warm_output = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del warm_output
    result = {"eager": _samples(fn, device, repeats, iterations)}
    if not allow_graph:
        result["graph"] = None
        result["graph_unavailable_reason"] = "SVD stage intentionally timed without capture"
        return result
    if device.type != "cuda":
        result["graph"] = None
        result["graph_unavailable_reason"] = "CUDA tensors required"
        return result

    graph = captured_output = None
    setup_start = time.perf_counter()
    try:
        gc.collect()
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            side = capture_stream if capture_stream is not None else torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    warm_output = fn()
            torch.cuda.current_stream().wait_stream(side)
            del warm_output
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured_output = fn()
        torch.cuda.synchronize(device)
        result["graph_setup_wall_s"] = time.perf_counter() - setup_start
        result["graph"] = _samples(graph.replay, device, repeats, iterations)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        # Keep the successful eager measurements if a backend cannot capture
        # a different stage. SVD stages never enter this capture path.
        result["graph"] = None
        result["graph_setup_wall_s"] = time.perf_counter() - setup_start
        result["graph_unavailable_reason"] = f"{type(error).__name__}: {error}"
    finally:
        # Do not accumulate a private graph pool for every component.
        del captured_output, graph
        gc.collect()
    return result


def profile_components(model, x, rank=8, trace_probes=4, oversampling=8,
                       repeats=3, steps=100, *, iterations=5):
    """Return JSON-serializable stage timings for the current fast p=0 DLRA.

    ``x`` is a fixed (1, 1, H, W) image, and the model must be frozen and in
    evaluation mode. All normalization statistics use ``manual`` throughout
    this study. Steady stages use t=0.5; initialization stages use t=0.0.
    Derivative widths are m+r / r and m+(r+oversampling) / (r+oversampling).
    The native input VJP differentiates one retained model forward tape; it
    excludes that forward and does not include checkpoint recomputation.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("pass a frozen model in evaluation mode")
    if any(isinstance(module, AttentionBlock) for module in model.modules()):
        raise ValueError("this study requires an attention-free model")
    if x.ndim != 4 or tuple(x.shape[:2]) != (1, 1):
        raise ValueError("x must have shape (1, 1, H, W)")
    if min(rank, trace_probes, repeats, steps, iterations) < 1 or oversampling < 0:
        raise ValueError("rank, probes, repeats, steps and iterations must be positive")
    dimension = x.numel()
    if rank > dimension:
        raise ValueError("rank exceeds the image dimension")
    flag_names = ("FUSE_K_IMAGE", "M_FROM_TRANSPOSE",
                  "RECOMPRESS_FROM_TRANSPOSE", "SKIP_UNTRUNCATED_SVD")
    if not all(getattr(enriched, name) for name in flag_names):
        raise ValueError("component counts require the current fast algebra flags enabled")
    x = x.detach()
    device, dtype = x.device, x.dtype
    device_context = torch.cuda.device(device) if device.type == "cuda" else None
    with (device_context if device_context is not None else nullcontext()), \
            patch.object(tangent_unet, "GN_STATS_MODE", "manual"):
        return _profile(model, x, rank, trace_probes, oversampling,
                        repeats, steps, iterations, flag_names)


def _profile(model, x, rank, m, oversampling, repeats, steps, iterations, flag_names):
    device, dtype = x.device, x.dtype
    height, width = x.shape[-2:]
    dimension, sketch_width, h = x.numel(), min(x.numel(), rank + oversampling), 1.0 / steps
    generator = torch.Generator(device=device).manual_seed(731)
    original_qr = torch.linalg.qr
    u = original_qr(torch.randn(dimension, rank, generator=generator,
                                device=device, dtype=dtype), mode="reduced")[0]
    v = original_qr(torch.randn(dimension, rank, generator=generator,
                                device=device, dtype=dtype), mode="reduced")[0]
    s = torch.diag(torch.linspace(.2, .02, rank, device=device, dtype=dtype))
    factors = (u, s, v)
    alpha = x.new_tensor(.5)
    probes = enriched._rademacher(dimension, m, generator, device, dtype)
    sensitivities = alpha * probes + u @ (s @ (v.T @ probes))
    cotangent = torch.randn(x.shape, generator=generator, device=device, dtype=dtype)
    t, t0 = x.new_tensor([.5]), x.new_zeros(1)
    tape = tangent_unet.unet_linearize(model, t, x)
    tape0 = tangent_unet.unet_linearize(model, t0, x)

    def action(tape_here, kind, q):
        return getattr(tape_here, kind)(q.T.reshape(-1, 1, height, width)).reshape(
            q.shape[1], dimension).T

    # Record real inputs/outputs from the production factor step. Its algebra
    # stage below calls that same function, substituting only these expensive
    # operators, so no approximate or separately maintained formula is timed.
    recorded = {"qr": []}

    def record_a(q):
        result = action(tape, "jvp", q)
        recorded["jvp_input"], recorded["jvp_output"] = q, result
        return result

    def record_at(q):
        result = action(tape, "vjp", q)
        recorded["vjp_input"], recorded["vjp_output"] = q, result
        return result

    def record_qr(value, *args, **kwargs):
        result = original_qr(value, *args, **kwargs)
        recorded["qr"].append((value, args, kwargs, result))
        return result

    with patch.object(torch.linalg, "qr", record_qr):
        next_factors, derivative, _ = enriched._fused_enriched_step(
            record_a, record_at, factors, sensitivities, probes, h, alpha, 0, None)
    if len(recorded["qr"]) != 2:
        raise RuntimeError("production p=0 step changed: expected K and L QR calls")
    expected_widths = (m + rank, rank)
    observed_widths = (recorded["jvp_input"].shape[1], recorded["vjp_input"].shape[1])
    if observed_widths != expected_widths:
        raise RuntimeError(f"production Jacobian widths changed: {observed_widths}")

    sketch = enriched._rademacher(dimension, sketch_width, generator, device, dtype)
    initial_q = torch.cat([probes, sketch], dim=1)
    initial_image = action(tape0, "jvp", initial_q)
    initial_sensitivities = probes + h * initial_image[:, :m]
    initial_alpha = (probes * initial_sensitivities).sum() / (m * dimension)
    range_input = h * initial_image[:, m:] + (1.0 - initial_alpha) * sketch
    range_basis = original_qr(range_input, mode="reduced")[0]
    initial_transpose = action(tape0, "vjp", range_basis)
    transpose_input = h * initial_transpose + (1.0 - initial_alpha) * range_basis
    thin_basis, triangular = original_qr(transpose_input, mode="reduced")
    small_u, singular_values, small_vt = torch.linalg.svd(triangular, full_matrices=False)

    def trace_state_update():
        updated = sensitivities + h * derivative
        scalar = (probes * updated).sum() / (m * dimension)
        state = x + h * tape.primal
        return updated, scalar, state

    updated_sensitivities, updated_alpha, updated_x = trace_state_update()
    copy_sources = (*next_factors, updated_sensitivities, updated_alpha, updated_x)
    copy_destinations = tuple(torch.empty_like(value) for value in copy_sources)

    def state_buffer_copies():
        for destination, source in zip(copy_destinations, copy_sources):
            destination.copy_(source)
        return copy_destinations

    def complete_fixed_state_step():
        shared = tangent_unet.unet_linearize(model, t, x)
        new_factors, dq, _ = enriched._fused_enriched_step(
            lambda q: action(shared, "jvp", q), lambda q: action(shared, "vjp", q),
            factors, sensitivities, probes, h, alpha, 0, None)
        updated = sensitivities + h * dq
        scalar = (probes * updated).sum() / (m * dimension)
        state = x + h * shared.primal
        for destination, source in zip(copy_destinations,
                                       (*new_factors, updated, scalar, state)):
            destination.copy_(source)
        return copy_destinations

    def initialization_algebra():
        q = torch.cat([probes, sketch], dim=1)
        updated = probes + h * initial_image[:, :m]
        scalar = (probes * updated).sum() / (m * dimension)
        range_value = h * initial_image[:, m:] + (1.0 - scalar) * sketch
        transpose_value = h * initial_transpose + (1.0 - scalar) * range_basis
        state = x + h * tape0.primal
        return q, updated, scalar, range_value, transpose_value, state

    def initialization_rotations():
        right_vectors = thin_basis @ small_u
        return (range_basis @ small_vt.T[:, :rank],
                torch.diag(singular_values[:rank]), right_vectors[:, :rank])

    @torch.no_grad()
    def native_forward():
        return model(x, t)

    # Deliberately outside no_grad: this is the actual native reverse-mode
    # input derivative through one retained forward tape, not the hand VJP.
    # Autograd schedules CUDA backward operations on their forward streams.
    # Build this retained tape on the exact stream that will capture its VJP;
    # otherwise backward escapes the capture and invalidates the graph.
    native_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
    if native_stream is not None:
        native_stream.wait_stream(torch.cuda.current_stream(device))
    native_context = torch.cuda.stream(native_stream) if native_stream is not None else nullcontext()
    with native_context, torch.enable_grad():
        variable = x.detach().clone().requires_grad_(True)
        native_primal = model(variable, t)
    if native_stream is not None:
        torch.cuda.current_stream(device).wait_stream(native_stream)

    def native_input_vjp():
        return torch.autograd.grad(native_primal, variable, cotangent,
                                   retain_graph=True, create_graph=False)[0]

    components = {}

    def measure(name, function, *, allow_graph=True, capture_stream=None, **metadata):
        components[name] = dict(_measure(function, device, repeats, iterations,
                                         allow_graph=allow_graph, capture_stream=capture_stream),
                                **metadata)

    measure("native_forward", native_forward, group="dflow", calls_per_gradient=steps)
    measure("native_input_vjp", native_input_vjp, group="dflow", calls_per_gradient=steps,
            retained_forward_tape=True, includes_forward=False, capture_stream=native_stream)
    measure("steady_primal_linearization", lambda: tangent_unet.unet_linearize(model, t, x),
            group="steady", calls_per_gradient=steps - 1)
    measure("steady_jvp", lambda: action(tape, "jvp", recorded["jvp_input"]),
            group="steady", columns=m + rank, calls_per_gradient=steps - 1)
    measure("steady_vjp", lambda: action(tape, "vjp", recorded["vjp_input"]),
            group="steady", columns=rank, calls_per_gradient=steps - 1)
    for name, (value, args, kwargs, _) in zip(("steady_k_qr", "steady_l_qr"), recorded["qr"]):
        measure(name, lambda value=value, args=args, kwargs=kwargs: original_qr(value, *args, **kwargs),
                group="steady", matrix_shape=list(value.shape), calls_per_gradient=steps - 1)

    qr_index = 0

    def replay_qr(value, *args, **kwargs):
        nonlocal qr_index
        result = recorded["qr"][qr_index % 2][3]
        qr_index += 1
        return result

    def factor_algebra():
        return enriched._fused_enriched_step(
            lambda q: recorded["jvp_output"], lambda q: recorded["vjp_output"],
            factors, sensitivities, probes, h, alpha, 0, None)

    with patch.object(torch.linalg, "qr", replay_qr):
        measure("steady_factor_algebra", factor_algebra, group="steady",
                calls_per_gradient=steps - 1, includes_alpha_rate_reduction=True)
    measure("steady_trace_state_update", trace_state_update, group="steady", calls_per_gradient=steps - 1)
    measure("steady_state_buffer_copies", state_buffer_copies, group="steady", calls_per_gradient=steps - 1)
    measure("complete_fixed_state_step", complete_fixed_state_step, group="combined",
            calls_per_gradient=steps - 1)

    measure("initial_primal_linearization", lambda: tangent_unet.unet_linearize(model, t0, x),
            group="initialization", calls_per_gradient=1)
    measure("initial_jvp", lambda: action(tape0, "jvp", initial_q), group="initialization",
            columns=m + sketch_width, calls_per_gradient=1)
    measure("initial_vjp", lambda: action(tape0, "vjp", range_basis), group="initialization",
            columns=sketch_width, calls_per_gradient=1)
    measure("initial_range_qr", lambda: original_qr(range_input, mode="reduced"),
            group="initialization", matrix_shape=list(range_input.shape), calls_per_gradient=1)
    measure("initial_transpose_qr", lambda: original_qr(transpose_input, mode="reduced"),
            group="initialization", matrix_shape=list(transpose_input.shape), calls_per_gradient=1)
    measure("initial_small_svd", lambda: torch.linalg.svd(triangular, full_matrices=False),
            group="initialization", matrix_shape=list(triangular.shape), calls_per_gradient=1,
            allow_graph=False)
    measure("initial_algebra_and_state_update", initialization_algebra, group="initialization",
            calls_per_gradient=1)
    measure("initial_factor_rotations", initialization_rotations, group="initialization",
            calls_per_gradient=1)
    measure("final_svd_and_rotations", lambda: enriched._canonicalize_factors(next_factors),
            group="terminal", matrix_shape=[rank, rank], calls_per_gradient=int(steps > 1),
            allow_graph=False)
    measure("terminal_transpose_action", lambda: core.approx_source_gradient(next_factors, cotangent, alpha),
            group="terminal", calls_per_gradient=1)

    sums = {}
    for execution in ("eager", "graph"):
        steady = [value[execution] for value in components.values() if value["group"] == "steady"]
        if all(value is not None for value in steady):
            sums[execution] = {
                metric: sum(value[metric] for value in steady)
                for metric in ("wall_ms_median", "cuda_event_ms_median")
                if all(value[metric] is not None for value in steady)
            }
        else:
            sums[execution] = None
    result = {
        "schema_version": 1, "device": str(device), "dtype": str(dtype),
        "torch": torch.__version__, "image_shape": list(x.shape),
        "dimension": dimension, "rank": rank, "trace_probes": m,
        "oversampling": oversampling, "initial_sketch_width": sketch_width,
        "steps": steps, "repeats": repeats, "iterations_per_sample": iterations,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": {name: getattr(model, name, None) for name in
                         ("ch", "ch_mul", "groups", "att_channels", "mid_attn")},
        "flags": {**{name: getattr(enriched, name) for name in flag_names},
                  "GN_STATS_MODE": tangent_unet.GN_STATS_MODE,
                  "FUSED_GN": tangent_unet.FUSED_GN,
                  "FUSE_POINTWISE_DERIVATIVES": tangent_unet.FUSE_POINTWISE_DERIVATIVES},
        "total_jvp_columns_per_gradient": m + sketch_width + (steps - 1) * (m + rank),
        "total_vjp_columns_per_gradient": sketch_width + (steps - 1) * rank,
        "components": components, "sum_of_independent_steady_components_ms": sums,
        "notes": [
            "Fixed representative state: steady t=0.5, initialization t=0.0; no trajectory timing.",
            "Synthetic orthonormal factors and trace sensitivities have production shapes.",
            "Independent stage graphs incur separate replay overhead; their sum is not an end-to-end latency.",
            "Complete fixed-state step executes production algebra and output buffer copies together.",
            "Native input VJP excludes its retained primal forward and any checkpoint recomputation.",
            "Probe/sketch random generation, terminal objective gradient, prior gradient, and optimizer are excluded.",
            "CUDA event intervals can include CPU launch gaps; they are not GPU active-kernel sums.",
        ],
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--trace-probes", type=int, default=4)
    parser.add_argument("--oversampling", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, _ = core.load_flow_model(args.resolution, epoch=args.epoch,
                                    device=args.device, run_dir=args.run_dir)
    device = next(model.parameters()).device
    x = torch.randn(1, 1, args.resolution, args.resolution, device=device,
                    generator=torch.Generator(device=device).manual_seed(2026))
    report = profile_components(model, x, rank=args.rank, trace_probes=args.trace_probes,
                                oversampling=args.oversampling, repeats=args.repeats,
                                steps=args.steps, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "components": len(report["components"])}))


if __name__ == "__main__":
    main()

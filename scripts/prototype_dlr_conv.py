"""Bounded standalone convolution experiment; never enabled in DLR production.

Example: timeout 50s .venv/bin/python scripts/prototype_dlr_conv.py \
    --device cuda:1 --channels 16 --lanes 12 --resolution 250 \
    --out outputs/conv_prototype.json
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gather_conv3x3_rows(x, rows):
    """CPU-verifiable indexing for implicit-GEMM rows, including zero padding."""
    batch, channels, height, width = x.shape
    rows = rows.to(device=x.device, dtype=torch.long)
    if rows.ndim != 1 or bool(((rows < 0) | (rows >= batch * height * width)).any()):
        raise ValueError("rows must contain valid flattened output-pixel indices")
    k = torch.arange(channels * 9, device=x.device)
    batch_index = rows // (height * width)
    spatial = rows % (height * width)
    input_y = spatial[:, None] // width + (k[None, :] % 9) // 3 - 1
    input_x = spatial[:, None] % width + k[None, :] % 3 - 1
    offset = ((batch_index[:, None] * channels + k[None, :] // 9) * height + input_y) * width + input_x
    valid = (input_y >= 0) & (input_y < height) & (input_x >= 0) & (input_x < width)
    values = x.reshape(-1)[offset.clamp(0, x.numel() - 1)]
    return torch.where(valid, values, torch.zeros_like(values))


@lru_cache(maxsize=1)
def _kernel():
    from scripts.prototype_dlr_conv_triton import conv3x3_nchw
    return conv3x3_nchw


@torch.no_grad()
def conv3x3_nchw(x, weight):
    """Bias-free3x3, stride1, padding1; only CUDAFP32 Cin16/48→Cout16."""
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("expected NCHW input and OIHW weight tensors")
    if x.shape[1] not in (16, 48) or tuple(weight.shape) != (16, x.shape[1], 3, 3):
        raise ValueError("prototype only supports Cin16/48→Cout16,3x3 weights")
    if (not x.is_cuda or x.dtype != torch.float32 or weight.device != x.device
            or weight.dtype != torch.float32):
        raise ValueError("prototype requires matching CUDA float32 tensors")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("prototype requires contiguous NCHW/OIHW without layout conversion")
    if any(size < 1 for size in x.shape):
        raise ValueError("input axes must be nonempty")
    batch, channels, height, width = x.shape
    with torch.cuda.device(x.device):
        output = torch.empty((batch, 16, height, width), device=x.device, dtype=x.dtype)
        _kernel()[((batch * height * width + 63) // 64,)](
            x, weight, output, BATCH=batch, CHANNELS=channels, HEIGHT=height, WIDTH=width,
            BLOCK_M=64, BLOCK_K=32, num_warps=4, num_stages=3)
    return output


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--channels", type=int, choices=(16, 48), default=16)
    parser.add_argument("--lanes", type=int, choices=(6, 12), default=12)
    parser.add_argument("--resolution", type=int, default=250)
    parser.add_argument("--memory-fraction", type=float, default=.08)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.resolution, args.repeats, args.iterations) < 1 or not 0 < args.memory_fraction <= 1:
        parser.error("sizes/counts must be positive; memory fraction must be in(0,1]")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.manual_seed(929)
    x = torch.randn(args.lanes, args.channels, args.resolution, args.resolution, device=device)
    weight = torch.randn(16, args.channels, 3, 3, device=device) / (9 * args.channels) ** .5
    report = {"protocol": {**vars(args), "out": str(args.out), "input_dtype": "float32",
                            "custom_input_precision": "tf32x3",
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                            "cudnn_benchmark": torch.backends.cudnn.benchmark,
                            "gpu": torch.cuda.get_device_name(device),
                            "torch": torch.__version__, "includes_backward": False},
              "measurements": {}}
    print(json.dumps(report["protocol"]), flush=True)
    graphs = {}
    setup_start = time.perf_counter()
    for name, fn in (("cudnn", lambda: F.conv2d(x, weight, padding=1)),
                     ("implicit_gemm_tf32x3", lambda: conv3x3_nchw(x, weight))):
        graphs[name] = _capture(fn)
    torch.cuda.synchronize()
    report["setup_wall_seconds"] = time.perf_counter() - setup_start
    timings = {name: [] for name in graphs}
    for repeat in range(args.repeats):
        order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for name in order:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graphs[name][0].replay()
            stop.record()
            stop.synchronize()
            timings[name].append(start.elapsed_time(stop) / args.iterations)
    # One strict-FP32cuDNN evaluation supplies a same-device numerical reference;
    # timings above use the original recorded backend settings throughout.
    with torch.backends.cudnn.flags(enabled=True, benchmark=torch.backends.cudnn.benchmark,
                                    deterministic=torch.backends.cudnn.deterministic,
                                    allow_tf32=False):
        reference = F.conv2d(x, weight, padding=1)
    for name, (_, output) in graphs.items():
        error = output - reference
        report["measurements"][name] = {
            "graph_ms_samples": timings[name], "graph_ms_median": statistics.median(timings[name]),
            "relative_l2_vs_strict_fp32": float(error.norm() / reference.norm()),
            "max_abs_vs_strict_fp32": float(error.abs().max())}
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["measured_scope"] = "one synthetic bias-free forward convolution; no full-network speed claim"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

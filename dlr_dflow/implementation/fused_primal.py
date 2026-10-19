"""Experimental fusion of a frozen GN/conditioning/SiLU primal record.

The caller supplies the already-computed normalized activation, retaining the
existing mean/variance arithmetic. Shared UNet tapes opt in through
``tangent_unet.GN_PRIMAL_BACKEND``; native execution remains the default.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _cuda_kernel():
    from ._fused_primal_triton import pointwise_record
    return pointwise_record


@torch.no_grad()
def normalization_activation_record(normalized, weight, bias, scale=None, shift=None,
                                    *, backend="auto", fuse_silu=False):
    """Return SiLU output and its coefficient for the GN input derivative.

    ``normalized`` is the existing batch-one normalized activation, e.g.
    ``_gn_stats(gn, p)[2].reshape_as(p)``. Weights/biases and optional conditioning
    each contain one scalar per channel. Supplying the existing normalized
    tensor, rather than recomputing it, preserves the statistics calculation.

    The conservative CUDA mode fuses only affine normalization and conditioning,
    retaining native ATen SiLU and backward. ``fuse_silu=True`` additionally
    evaluates both SiLU outputs in the same kernel; libdevice exp and rounded
    division implement the same formulas, but bitwise agreement is not promised.
    GPU support is float32 contiguous NCHW (other strides explicitly convert).
    CPU/other dtype auto falls back to torch. This is not an autograd operation.
    """
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    if normalized.ndim != 4 or normalized.shape[0] != 1 or normalized.numel() == 0:
        raise ValueError("normalized must have shape (1,C,H,W) with nonempty axes")
    if (scale is None) != (shift is None):
        raise ValueError("scale and shift must both be provided or omitted")
    channels = normalized.shape[1]
    for name, value in (("weight", weight), ("bias", bias), ("scale", scale), ("shift", shift)):
        if value is not None and (value.numel() != channels or value.device != normalized.device
                                  or value.dtype != normalized.dtype):
            raise ValueError(f"{name} must have one value per channel and share the input device/dtype")
    weight, bias = weight.reshape(1, channels, 1, 1), bias.reshape(1, channels, 1, 1)
    if scale is not None:
        scale, shift = scale.reshape(1, channels, 1, 1), shift.reshape(1, channels, 1, 1)
    supported = normalized.is_cuda and normalized.dtype == torch.float32
    if backend == "triton" and not supported:
        raise ValueError("Triton primal fusion requires CUDA float32 tensors")
    if backend == "torch" or not supported:
        preactivation = normalized * weight + bias
        affine = weight
        if scale is not None:
            preactivation = preactivation * (1 + scale) + shift
            affine = affine * (1 + scale)
        if fuse_silu:
            denominator = 1 + torch.exp(-preactivation)
            sigmoid = 1 / denominator
            output = preactivation / denominator
            coefficient = affine * sigmoid * (1 + preactivation * (1 - sigmoid))
            return output, coefficient
        return F.silu(preactivation), torch.ops.aten.silu_backward(affine.expand_as(preactivation), preactivation)
    with torch.cuda.device(normalized.device):
        return _cuda_record(normalized, weight, bias, scale, shift, fuse_silu)


def _cuda_record(normalized, weight, bias, scale, shift, fuse_silu):
    normalized = normalized.contiguous()
    weight, bias = weight.contiguous(), bias.contiguous()
    if scale is not None:
        scale, shift = scale.contiguous(), shift.contiguous()
    output = torch.empty_like(normalized)
    coefficient = torch.empty_like(normalized) if fuse_silu else output
    count = normalized.numel()
    _cuda_kernel()[((count + 255) // 256,)](
        normalized, weight, bias, weight if scale is None else scale,
        bias if shift is None else shift, output, coefficient,
        COUNT=count, SPATIAL=normalized.shape[2] * normalized.shape[3],
        CONDITIONED=scale is not None, FUSE_SILU=bool(fuse_silu), BLOCK=256,
        num_warps=4, enable_fp_fusion=False)
    if fuse_silu:
        return output, coefficient
    affine = weight if scale is None else weight * (1 + scale)
    return F.silu(output), torch.ops.aten.silu_backward(affine.expand_as(output), output)

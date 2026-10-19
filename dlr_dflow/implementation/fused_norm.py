"""Optional exact GroupNorm derivative with a shared batch-one primal.

The CUDA float32 backend avoids materializing one copy of the primal per
derivative lane. Triton is imported lazily and is unnecessary for CPU use.
This module implements a local linear map; it is not an autograd operation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn


__all__ = ["FastGroupNorm", "group_norm_action", "group_norm_action_torch",
           "swap_groupnorm_fast"]


class _FastGroupNormFn(torch.autograd.Function):
    """GroupNorm forward with a closed-form forward-mode derivative."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                num_groups: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.aten.native_group_norm(
            x, weight, bias, x.shape[0], x.shape[1], x.shape[2] * x.shape[3],
            num_groups, eps)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
        x, weight, _, num_groups, _ = inputs
        _, mean, rstd = output
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.save_for_forward(x, weight, mean, rstd)
        ctx.num_groups = num_groups
        ctx.mark_non_differentiable(mean, rstd)

    @staticmethod
    def jvp(ctx: Any, x_t: torch.Tensor, weight_t: torch.Tensor | None,
            bias_t: torch.Tensor | None, *_: Any) -> tuple[torch.Tensor, None, None]:
        x, weight, mean, rstd = ctx.saved_tensors
        batch, channels, height, width = x.shape
        groups = ctx.num_groups
        x_grouped = x.reshape(batch, groups, -1)
        normalized = (x_grouped - mean.reshape(batch, groups, 1)) * rstd.reshape(batch, groups, 1)
        tangent = x_t.reshape(batch, groups, -1)
        normalized_tangent = rstd.reshape(batch, groups, 1) * (
            tangent - tangent.mean(dim=2, keepdim=True)
            - normalized * (normalized * tangent).mean(dim=2, keepdim=True))
        output_tangent = normalized_tangent.reshape(batch, channels, height, width)
        if weight is not None:
            output_tangent = weight.view(1, channels, 1, 1) * output_tangent
        if weight_t is not None:
            output_tangent = output_tangent + weight_t.view(1, channels, 1, 1) * normalized.reshape(
                batch, channels, height, width)
        if bias_t is not None:
            output_tangent = output_tangent + bias_t.view(1, channels, 1, 1)
        return output_tangent, None, None

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor, grad_mean: torch.Tensor | None,
                 grad_rstd: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        x, weight, mean, rstd = ctx.saved_tensors
        batch, channels, height, width = x.shape
        needs_grad = [ctx.needs_input_grad[0], ctx.needs_input_grad[1],
                      ctx.needs_input_grad[2]]
        dx, dw, db = torch.ops.aten.native_group_norm_backward(
            grad_out.contiguous(), x, mean, rstd, weight,
            batch, channels, height * width, ctx.num_groups, needs_grad)
        return dx, dw, db, None, None


class FastGroupNorm(nn.GroupNorm):
    """GroupNorm module with the hand-written forward-mode rule."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_dtype = x.dtype if self.weight is None else self.weight.dtype
        output, _, _ = _FastGroupNormFn.apply(
            x.contiguous().to(weight_dtype), self.weight, self.bias,
            self.num_groups, self.eps)
        return output


def swap_groupnorm_fast(model: nn.Module) -> int:
    """Replace ``nn.GroupNorm`` children in-place and return the count."""
    swapped = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if type(child) is not nn.GroupNorm:
                continue
            fast = FastGroupNorm(child.num_groups, child.num_channels,
                                 eps=child.eps, affine=child.affine)
            fast.load_state_dict(child.state_dict())
            if child.affine:
                fast.to(device=child.weight.device, dtype=child.weight.dtype)
            for parameter in fast.parameters():
                parameter.requires_grad_(False)
            setattr(module, name, fast)
            swapped += 1
    return swapped


def group_norm_action_torch(p, y, mean, rstd, groups, coefficient=None, transpose=False):
    """Reference formula using saved normalization statistics, without k copies."""
    lanes = y.shape[0]
    centered = p.reshape(1, groups, -1) - mean.reshape(1, groups, 1)
    scale = rstd.reshape(1, groups, 1)
    weighted = y * coefficient if coefficient is not None and transpose else y
    grouped = weighted.reshape(lanes, groups, -1)
    mean_y = grouped.mean(-1, keepdim=True)
    mean_centered_y = (centered * grouped).mean(-1, keepdim=True)
    result = (scale * (grouped - mean_y - centered * scale.square() * mean_centered_y)).reshape_as(y)
    return result * coefficient if coefficient is not None and not transpose else result


@lru_cache(maxsize=1)
def _cuda_kernels():
    from ._fused_norm_triton import reduce_partials, apply_partials, apply_single_row
    return reduce_partials, apply_partials, apply_single_row


@torch.no_grad()
def group_norm_action(p, y, mean, rstd, groups, coefficient=None, transpose=False,
                      *, backend="auto", block_size=1024):
    """Apply L(y), coefficient*L(y), or L(coefficient*y) using saved GN stats.

    ``p`` is (1,C,H,W), ``y`` is (lanes,C,H,W), and the statistics each have
    ``groups`` elements. A coefficient has the primal shape, or is broadcastable
    to it. ``transpose=False`` places it after L (JVP); True places it before L
    (VJP). ``L`` is the symmetric input derivative of unaffined GroupNorm.

    ``auto`` uses Triton only for CUDA float32 and otherwise the torch formula.
    ``triton`` requests that backend explicitly and rejects unsupported tensors.
    Each primal/lane/coefficient input independently supports contiguous NCHW
    or channels-last storage. Other strides are explicitly made contiguous;
    the primal is never expanded over derivative lanes. Padding does not enter
    the normalization counts. CUDA output preserves the supported lane layout.
    """
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    if p.ndim != 4 or p.shape[0] != 1 or y.ndim != 4 or y.shape[1:] != p.shape[1:]:
        raise ValueError("expected p=(1,C,H,W), y=(lanes,C,H,W)")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0 or p.shape[1] % groups:
        raise ValueError("groups must be a positive divisor of the channel count")
    if y.shape[0] < 1 or p.numel() == 0:
        raise ValueError("primal and derivative lanes must be nonempty")
    if (p.device != y.device or p.dtype != y.dtype or mean.device != p.device
            or rstd.device != p.device or mean.dtype != p.dtype or rstd.dtype != p.dtype):
        raise ValueError("all tensors must share a device and dtype")
    if mean.numel() != groups or rstd.numel() != groups:
        raise ValueError("mean and rstd must contain one statistic per group")
    if coefficient is not None:
        if coefficient.device != p.device or coefficient.dtype != p.dtype:
            raise ValueError("coefficient must share the primal device and dtype")
        try:
            coefficient = coefficient.expand_as(p)
        except RuntimeError as error:
            raise ValueError("coefficient must broadcast to the primal shape") from error
    if (not isinstance(block_size, int) or isinstance(block_size, bool)
            or block_size < 32 or block_size & (block_size - 1)):
        raise ValueError("block_size must be a power of two at least 32")
    supported = p.is_cuda and p.dtype == torch.float32
    if backend == "triton" and not supported:
        raise ValueError("Triton GN derivatives require CUDA float32 tensors")
    if backend == "torch" or not supported:
        return group_norm_action_torch(p, y, mean, rstd, groups, coefficient, transpose)
    with torch.cuda.device(y.device):
        return _group_norm_action_triton(p, y, mean, rstd, groups, coefficient,
                                         transpose, block_size)


def _group_norm_action_triton(p, y, mean, rstd, groups, coefficient, transpose,
                              block_size):
    reduce_partials, apply_partials, apply_single_row = _cuda_kernels()

    def supported_layout(value):
        if value.is_contiguous():
            return value, False
        if value.is_contiguous(memory_format=torch.channels_last):
            return value, True
        return value.contiguous(), False

    p, p_channels_last = supported_layout(p)
    y, y_channels_last = supported_layout(y)
    mean, rstd = mean.contiguous().reshape(groups), rstd.contiguous().reshape(groups)
    coefficient_channels_last = False
    if coefficient is not None:
        coefficient, coefficient_channels_last = supported_layout(coefficient)
    group_size = p.numel() // groups
    tiles = (group_size + block_size - 1) // block_size
    rows = y.shape[0] * groups
    result = torch.empty_like(y, memory_format=torch.preserve_format)
    coefficient_ptr = p if coefficient is None else coefficient
    layouts = dict(CHANNELS=p.shape[1], SPATIAL=p.shape[2] * p.shape[3],
                   P_CHANNELS_LAST=p_channels_last, Y_CHANNELS_LAST=y_channels_last,
                   COEFFICIENT_CHANNELS_LAST=coefficient_channels_last)
    if group_size <= 4096:
        apply_single_row[(rows,)](
            p, y, mean, rstd, coefficient_ptr, result,
            GROUPS=groups, GROUP_SIZE=group_size,
            HAS_COEFFICIENT=coefficient is not None, TRANSPOSE=bool(transpose),
            BLOCK=1 << (group_size - 1).bit_length(),
            num_warps=4 if group_size <= 1024 else 8, enable_fp_fusion=False, **layouts)
        return result
    partials = torch.empty((2, rows, tiles), device=y.device, dtype=y.dtype)
    # A group/lane/tile first computes two scalar partial reductions. Each
    # output tile then reduces those small arrays and applies the exact map.
    grid = (rows, tiles)
    launch = dict(num_warps=4, enable_fp_fusion=False)
    reduce_partials[grid](p, y, mean, coefficient_ptr, partials,
                         GROUPS=groups, GROUP_SIZE=group_size, ROWS=rows, TILES=tiles,
                         HAS_COEFFICIENT=coefficient is not None, TRANSPOSE=bool(transpose),
                         BLOCK=block_size, **layouts, **launch)
    apply_partials[grid](p, y, mean, rstd, coefficient_ptr, partials, result,
                        GROUPS=groups, GROUP_SIZE=group_size, ROWS=rows, TILES=tiles,
                        HAS_COEFFICIENT=coefficient is not None, TRANSPOSE=bool(transpose),
                        BLOCK=block_size, REDUCTION_BLOCK=1 << (tiles - 1).bit_length(),
                        **layouts, **launch)
    return result

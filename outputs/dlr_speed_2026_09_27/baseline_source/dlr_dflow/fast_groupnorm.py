"""GroupNorm with a hand-written forward-mode (JVP) rule.

torch.func's decomposition of native_group_norm's JVP materializes an unfused
elementwise storm (measured 10-33x the forward cost, ~27% of every
Jacobian-product call on this UNet).  With frozen affine parameters the
tangent has the closed form

    v     = (x - mean) * rstd
    y0_t  = rstd * (x_t - mean_g(x_t) - v * mean_g(v * x_t))
    y_t   = weight * y0_t

- two group reductions and a few fused multiplies.  Backward delegates to the
native kernel, so reverse mode (VJPs, exact D-Flow gradients) is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _FastGroupNormFn(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, weight, bias, num_groups, eps):
        out, mean, rstd = torch.ops.aten.native_group_norm(
            x, weight, bias, x.shape[0], x.shape[1], x.shape[2] * x.shape[3],
            num_groups, eps)
        return out, mean, rstd

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, num_groups, eps = inputs
        out, mean, rstd = output
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.save_for_forward(x, weight, mean, rstd)
        ctx.num_groups = num_groups
        ctx.mark_non_differentiable(mean, rstd)

    @staticmethod
    def jvp(ctx, x_t, weight_t, bias_t, *_):
        x, weight, mean, rstd = ctx.saved_tensors
        N, C, H, W = x.shape
        G = ctx.num_groups
        xg = x.reshape(N, G, -1)
        m = mean.reshape(N, G, 1)
        r = rstd.reshape(N, G, 1)
        v = (xg - m) * r
        tg = x_t.reshape(N, G, -1)
        y0 = r * (tg - tg.mean(dim=2, keepdim=True)
                  - v * (v * tg).mean(dim=2, keepdim=True))
        y_t = weight.view(1, C, 1, 1) * y0.reshape(N, C, H, W)
        if weight_t is not None:
            y_t = y_t + weight_t.view(1, C, 1, 1) * v.reshape(N, C, H, W)
        if bias_t is not None:
            y_t = y_t + bias_t.view(1, C, 1, 1)
        return y_t, None, None

    @staticmethod
    def backward(ctx, grad_out, grad_mean, grad_rstd):
        x, weight, mean, rstd = ctx.saved_tensors
        N, C, H, W = x.shape
        mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[1],
                ctx.needs_input_grad[2]]
        dx, dw, db = torch.ops.aten.native_group_norm_backward(
            grad_out.contiguous(), x, mean, rstd, weight,
            N, C, H * W, ctx.num_groups, mask)
        return dx, dw, db, None, None


class FastGroupNorm(nn.GroupNorm):
    def forward(self, x):
        # Custom autograd Functions bypass autocast's casting, so a BF16
        # activation would meet FP32 affine parameters inside the kernel.
        # Match vanilla GroupNorm-under-autocast: compute in the weight dtype.
        out, _, _ = _FastGroupNormFn.apply(
            x.contiguous().to(self.weight.dtype), self.weight, self.bias,
            self.num_groups, self.eps)
        return out


def swap_groupnorm_fast(model: nn.Module) -> int:
    """Replace every nn.GroupNorm in-place with FastGroupNorm; returns count."""
    swapped = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if type(child) is nn.GroupNorm:
                fast = FastGroupNorm(child.num_groups, child.num_channels,
                                     eps=child.eps, affine=child.affine)
                fast.load_state_dict(child.state_dict())
                fast.to(device=child.weight.device, dtype=child.weight.dtype)
                for p in fast.parameters():
                    p.requires_grad_(False)
                setattr(module, name, fast)
                swapped += 1
    return swapped

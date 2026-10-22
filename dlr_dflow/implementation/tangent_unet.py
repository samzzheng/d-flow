"""Hand-written tangent-mode (forward-mode) pass of the flow UNet.

`unet_jvp(model, t, x, T)` returns the primal v(t, x) for ONE image x and the
Jacobian action d_x v(t, x) T for a batch T of k tangent images in a single
sweep: the primal activations are computed once (batch 1) and every layer's
exact directional derivative is pushed through the k tangent lanes with the
layer's own linearisation (conv: same conv without bias; GroupNorm/SiLU/
softmax: analytic rules). Mathematically identical to torch.func.jvp column
by column, but with ~1 forward-equivalent of FLOPs per column and none of the
unfused elementwise kernels functorch's decomposition emits (2.5x fewer at
128px; see prof_jvp.py). Frozen weights are assumed (their tangent is zero).
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from models.unet.unet import AttentionBlock, get_timestep_embedding


def _conv(m: nn.Conv2d, p, d):
    return (F.conv2d(p, m.weight, m.bias, m.stride, m.padding, m.dilation, m.groups),
            F.conv2d(d, m.weight, None, m.stride, m.padding, m.dilation, m.groups))


# GroupNorm's linearisation at the primal x is the symmetric map (per group)
#   L(y) = rstd * (y - mean(y) - xhat * mean(xhat * y)),   xhat = (x - mu) * rstd,
# so both the tangent  d -> L(d) * w  and the adjoint  g -> L(g * w)  are one
# application of L.  The fused aten group-norm backward kernel computes exactly
# L(g * w) for grad_out = g; omitting its weight gives L(g), including when some
# learned weights are zero. Native forward statistics are shared with the JVP.
FUSED_GN = True
FUSED_GN_CONTIG = True
# Match the model's forward map. Small reduction differences from manual
# statistics can amplify along sensitive trajectories and reverse gradients.
GN_STATS_MODE = "native"
# Shared tapes can fold the normalization affine weight, conditioning scale,
# and SiLU derivative into one primal-sized coefficient. Each derivative lane
# then needs one multiply instead of recomputing sigmoid and affine products.
# False retains the separate tape operations for numerical/performance A/B.
FUSE_POINTWISE_DERIVATIVES = True
# Optional CUDA float32 kernel: broadcast the single primal instead of copying
# it to every derivative lane, and fuse the cached pointwise coefficient.
# CPU/other dtypes retain the native path. Keep fixed for a graph's lifetime.
GN_ACTION_BACKEND = "native"  # "native" or "triton"
# Optional shared-tape primal fusion, retaining the existing manual GN stats.
# "affine" combines affine/conditioning operations; "full" also combines
# SiLU and its derivative coefficient. CUDA float32 only; other cases retain
# the native path. Choose before constructing any reusable graph provider.
GN_PRIMAL_BACKEND = "native"  # "native", "affine", or "full"

# The moments implementation avoids per-lane attention score matrices, at the
# cost of head_dim times more matrix-multiply work. The chunked implementation
# uses the ordinary softmax derivative with a bounded query dimension.
ATTENTION_MODE = "moments"
# Optional per-sweep overrides let the benchmark choose the faster exact
# contraction without coupling the JVP and VJP implementations.
ATTENTION_JVP_MODE = None
ATTENTION_VJP_MODE = None
ATTENTION_SCORE_BYTES = 32 * 1024 * 1024


def _gn_stats(m: nn.GroupNorm, p):
    G = m.num_groups
    pg = p.reshape(1, G, -1)
    mu = pg.mean(-1, keepdim=True)
    xc = pg - mu
    rstd = torch.rsqrt((xc * xc).mean(-1, keepdim=True) + m.eps)
    return mu, rstd, xc * rstd


def _L(m: nn.GroupNorm, p, y, weight=None, stats=None):
    """Apply GroupNorm's (symmetric) input-linearisation at primal p to k lanes y.

    With `weight` given, returns L(y * weight) (the adjoint) - the fused kernel
    folds that channel scaling in for free; otherwise returns L(y).
    """
    k, C, H, W = y.shape
    G = m.num_groups
    if FUSED_GN:
        if stats is None:
            if GN_STATS_MODE == "native":
                _, mu, rstd = torch.ops.aten.native_group_norm(
                    p, None, None, 1, C, H * W, G, m.eps)
            else:
                mu, rstd, _ = _gn_stats(m, p)
        else:
            mu, rstd = stats[:2]
        pk = p.expand(k, C, H, W)
        if FUSED_GN_CONTIG:
            pk = pk.contiguous()
        out, _, _ = torch.ops.aten.native_group_norm_backward(
            y.contiguous(), pk, mu.reshape(1, G).expand(k, G).contiguous(),
            rstd.reshape(1, G).expand(k, G).contiguous(),
            weight, k, C, H * W, G, [True, False, False])
        return out
    mu, rstd, xhat = _gn_stats(m, p) if stats is None else stats
    if weight is not None:
        y = y * weight.view(1, C, 1, 1)
    yg = y.reshape(k, G, -1)
    out = (yg - yg.mean(-1, keepdim=True)
           - xhat * (xhat * yg).mean(-1, keepdim=True)) * rstd
    return out.reshape(y.shape)


def _gn(m: nn.GroupNorm, p, d):
    C = p.shape[1]
    w = m.weight.view(1, C, 1, 1)
    b = m.bias.view(1, C, 1, 1)
    if FUSED_GN and GN_STATS_MODE == "native":
        primal, mu, rstd = torch.ops.aten.native_group_norm(
            p, m.weight, m.bias, 1, C, p.shape[-2] * p.shape[-1],
            m.num_groups, m.eps)
        return primal, _L(m, p, d, stats=(mu, rstd)) * w
    stats = _gn_stats(m, p)
    return stats[2].reshape(p.shape) * w + b, _L(m, p, d, stats=stats) * w


def _gn_p(m: nn.GroupNorm, p):
    """Primal-only GroupNorm (used by the reverse pass's forward sweep)."""
    return F.group_norm(p, m.num_groups, m.weight, m.bias, m.eps)


def _gn_record(m: nn.GroupNorm, p):
    """Primal normalization and the statistics shared by both derivatives."""
    if FUSED_GN and GN_STATS_MODE == "native":
        _, C, H, W = p.shape
        primal, mu, rstd = torch.ops.aten.native_group_norm(
            p, m.weight, m.bias, 1, C, H * W, m.num_groups, m.eps)
        return primal, (mu, rstd)
    stats = _gn_stats(m, p)
    primal = (stats[2].reshape_as(p) * m.weight.view(1, -1, 1, 1)
              + m.bias.view(1, -1, 1, 1))
    # The fused derivative needs only mean/rstd; do not retain another full
    # normalized activation for every block in the shared tape.
    return primal, stats[:2] if FUSED_GN else stats


def _silu(p, d):
    # tangent = d * silu'(p): the fused silu_backward kernel is that product
    # (one pass, broadcasting the batch-1 primal over the k lanes)
    return F.silu(p), torch.ops.aten.silu_backward(d, p.expand_as(d))


def _pmm(P, Z):
    """P @ Z per lane for P (1,nh,N,N) and Z (k,nh,N,c) -> (k,nh,N,c).

    torch.matmul would broadcast P over the k lanes and MATERIALISE the
    (k,nh,N,N) expansion (4-7 GB per call at 250px); folding k into the column
    dimension makes it one bmm over heads with nothing expanded.
    """
    k, nh, N, c = Z.shape
    Zf = Z.permute(1, 2, 0, 3).reshape(nh, N, k * c)
    return torch.bmm(P[0], Zf).reshape(nh, N, k, c).permute(2, 0, 1, 3)


def _attention_query_chunk(lanes):
    k, nh, N, _ = lanes.shape
    # This bounds each score-shaped temporary, not the sum of all temporaries.
    return max(1, min(N, ATTENTION_SCORE_BYTES // (k * nh * N * lanes.element_size())))


def _attention_jvp_chunked(P, qp, kp, vp, qd, kd, vd, scale):
    """Exact attention JVP using O(k * heads * chunk * N) score workspace."""
    chunk = _attention_query_chunk(qd)
    kt, dkt = kp.transpose(-1, -2), kd.transpose(-1, -2)
    parts = []
    for start in range(0, qp.shape[-2], chunk):
        stop = start + chunk
        Pc = P[..., start:stop, :]
        ds = (qd[..., start:stop, :] @ kt + qp[..., start:stop, :] @ dkt) * scale
        dp = Pc * (ds - (Pc * ds).sum(-1, keepdim=True))
        parts.append(dp @ vp)
    return torch.cat(parts, dim=-2) + _pmm(P, vd)


def _attention_vjp_chunked(P, qp, kp, vp, op, go, scale):
    """Transpose of the chunked JVP; key gradients sum over query chunks."""
    chunk = _attention_query_chunk(go)
    vt = vp.transpose(-1, -2)
    q_parts = []
    k_bar = torch.zeros_like(go)
    for start in range(0, qp.shape[-2], chunk):
        stop = start + chunk
        gc = go[..., start:stop, :]
        rho = (gc * op[..., start:stop, :]).sum(-1, keepdim=True)
        ds = P[..., start:stop, :] * (gc @ vt - rho)
        q_parts.append((ds @ kp) * scale)
        k_bar.add_((ds.transpose(-1, -2) @ qp[..., start:stop, :]) * scale)
    return torch.cat(q_parts, dim=-2), k_bar, _pmm(P.transpose(-1, -2), go)


def _attn(m: AttentionBlock, p, d):
    hp, hd = _gn(m.norm, p, d)
    _, _, H, W = p.shape
    nh, hd_ = m.num_heads, m.head_dim

    def heads(z):                                   # (B,C,H,W) -> (B,nh,N,hd)
        return z.reshape(z.shape[0], nh, hd_, H * W).permute(0, 1, 3, 2)

    def unheads(z):
        return z.permute(0, 1, 3, 2).reshape(z.shape[0], nh * hd_, H, W)

    qp, qd = _conv(m.Q, hp, hd)
    kp, kd = _conv(m.K, hp, hd)
    vp, vd = _conv(m.V, hp, hd)
    qp, qd, kp, kd, vp, vd = map(heads, (qp, qd, kp, kd, vp, vd))
    scale = 1.0 / math.sqrt(hd_)                    # SDPA default scale
    P = torch.softmax((qp @ kp.transpose(-1, -2)) * scale, dim=-1)   # (1,nh,N,N)
    op = P @ vp
    # The moments path avoids any per-lane N x N tensor. With dS = scale*(dq k^T +
    # q dk^T) and dP = P o (dS - r 1^T), r_i = sum_j P_ij dS_ij:
    #   (dP v)_i = sum_j P_ij dS_ij v_j - r_i (P v)_i
    #   sum_j P_ij dS_ij v_j = scale * ( dq_i . M_i + q_i . (P X)_i ),
    #   M_i = sum_j P_ij k_j v_j^T (primal only), X_j = dk_j v_j^T (per lane).
    # Its extra workspace is O(k N hd^2), in addition to the shared primal P.
    attention_mode = ATTENTION_JVP_MODE or ATTENTION_MODE
    if attention_mode == "chunked":
        od = _attention_jvp_chunked(P, qp, kp, vp, qd, kd, vd, scale)
    elif attention_mode == "moments":
        N = H * W
        Pk = P @ kp                                                     # (1,nh,N,hd)
        Pdk = _pmm(P, kd)                                               # (k,nh,N,hd)
        r = ((qd * Pk).sum(-1, keepdim=True)
             + (qp * Pdk).sum(-1, keepdim=True)) * scale                # (k,nh,N,1)
        M = (P @ (kp.unsqueeze(-1) * vp.unsqueeze(-2)).reshape(1, nh, N, hd_ * hd_)
             ).reshape(1, nh, N, hd_, hd_)
        PX = _pmm(P, (kd.unsqueeze(-1) * vp.unsqueeze(-2)).reshape(-1, nh, N, hd_ * hd_)
                  ).reshape(-1, nh, N, hd_, hd_)
        term = (torch.einsum("khic,bhicd->khid", qd, M)
                + torch.einsum("bhic,khicd->khid", qp, PX)) * scale
        od = term - r * op + _pmm(P, vd)
    else:
        raise ValueError(f"Unknown JVP attention mode: {attention_mode!r}")
    pp, pd = _conv(m.proj, unheads(op), unheads(od))
    return (p + pp) / math.sqrt(2.0), (d + pd) / math.sqrt(2.0)


def _block(seq: nn.Sequential, p, d, scale_shift=None):
    """GroupNorm -> [scale/shift] -> SiLU -> (Dropout: identity in eval) -> Conv."""
    for layer in seq:
        if isinstance(layer, nn.GroupNorm):
            p, d = _gn(layer, p, d)
            if scale_shift is not None:
                scale, shift = scale_shift
                p, d = p * (1 + scale) + shift, d * (1 + scale)
        elif isinstance(layer, nn.SiLU):
            p, d = _silu(p, d)
        elif isinstance(layer, nn.Conv2d):
            p, d = _conv(layer, p, d)
        elif isinstance(layer, (nn.Identity, nn.Dropout)):
            pass
        else:
            raise NotImplementedError(type(layer))
    return p, d


def _resblock(m, p, d, emb):
    e = m.cond_block(emb)
    hp, hd = _block(m.block1, p, d)
    if m.scale_shift:
        hp, hd = _block(m.block2, hp, hd, scale_shift=e.chunk(2, dim=1))
    else:
        hp, hd = _block(m.block2, hp + e, hd)
    if isinstance(m.skip_connection, nn.Identity):
        sp, sd = p, d
    else:
        sp, sd = _conv(m.skip_connection, p, d)
    hp, hd = (sp + hp) / math.sqrt(2.0), (sd + hd) / math.sqrt(2.0)
    if isinstance(m.attn, AttentionBlock):
        return _attn(m.attn, hp, hd)
    return hp, hd


@torch.no_grad()
def unet_jvp(model, t: float, x: torch.Tensor, T: torch.Tensor):
    """x: (1,1,H,W) primal image; T: (k,1,H,W) tangents. Returns (v, d_x v T)."""
    # t may be a python float or a 0-dim tensor (tensor keeps torch.compile
    # from baking the time into the graph and recompiling every step)
    tvec = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(1)
    emb = model.time_proj(get_timestep_embedding(tvec * 1000.0, model.time_embed_dim))
    xp, xd = _conv(model.input_proj, x, T)
    hp, hd = xp, xd
    down = []
    n_levels = len(model.down)
    for i, blocks in enumerate(model.down):
        hp, hd = _resblock(blocks[0], hp, hd, emb)
        down.append((hp, hd))
        hp, hd = _resblock(blocks[1], hp, hd, emb)
        down.append((hp, hd))
        if i < n_levels - 1:
            hp, hd = _conv(blocks[2].conv, hp, hd)
    hp, hd = _resblock(model.mid[0], hp, hd, emb)
    hp, hd = _resblock(model.mid[1], hp, hd, emb)
    for i, blocks in enumerate(model.up):
        sp, sd = down.pop()
        hp, hd = _resblock(blocks[0], torch.cat((hp, sp), 1), torch.cat((hd, sd), 1), emb)
        sp, sd = down.pop()
        hp, hd = _resblock(blocks[1], torch.cat((hp, sp), 1), torch.cat((hd, sd), 1), emb)
        if i < n_levels - 1:
            size = tuple(down[-1][0].shape[-2:])
            hp = F.interpolate(hp, size=size, mode="nearest")
            hd = F.interpolate(hd, size=size, mode="nearest")
            hp, hd = _conv(blocks[2].conv, hp, hd)
    hp, hd = _block(model.final, torch.cat((hp, xp), 1), torch.cat((hd, xd), 1))
    return hp, hd


# ---------------------------------------------------------------------------
# Hand-written reverse-mode pass: A_t^T G for a batch G of k cotangent images.
# A primal forward (batch 1) records what each layer's adjoint needs; the
# adjoints are the transposes of the tangent rules above (GroupNorm's and
# softmax's linearisations are symmetric, conv's adjoint is conv2d_input).
# ---------------------------------------------------------------------------
# Adjoint of the (bias-free) conv on k lanes: the aten convolution_backward op
# autograd itself uses (only the input's shape/strides are read for grad_input)
# versus the conv_transpose route of torch.nn.grad.conv2d_input.
CONV_T_ATEN = True


def _conv_T(m: nn.Conv2d, g, in_hw):
    shape = (g.shape[0], m.in_channels, *in_hw)
    if CONV_T_ATEN:
        channels_last = (
            (g.is_contiguous(memory_format=torch.channels_last) and not g.is_contiguous())
            or (m.weight.is_contiguous(memory_format=torch.channels_last)
                and not m.weight.is_contiguous()))
        input_layout = torch.channels_last if channels_last else torch.contiguous_format
        gi, _, _ = torch.ops.aten.convolution_backward(
            g, torch.empty(shape, device=g.device, dtype=g.dtype, memory_format=input_layout), m.weight, None,
            m.stride, m.padding, m.dilation, False, [0, 0], m.groups,
            [True, False, False])
        return gi
    return torch.nn.grad.conv2d_input(shape, m.weight, g, m.stride, m.padding,
                                      m.dilation, m.groups)


def _gn_T(m: nn.GroupNorm, p_in, g, stats=None):
    # adjoint of d -> L(d) * w  is  g -> L(g * w)  (L symmetric per group)
    return _L(m, p_in, g, weight=m.weight, stats=stats)


def _silu_T(p_in, g):
    return torch.ops.aten.silu_backward(g, p_in.expand_as(g))


def _interp_T(g, in_hw):
    """Adjoint of nearest-neighbour F.interpolate(size=g.shape[-2:])."""
    k, C, out_h, out_w = g.shape
    in_h, in_w = in_hw
    return torch.ops.aten.upsample_nearest2d_backward(
        g, [out_h, out_w], [k, C, in_h, in_w], None, None)


def _block_fwd(seq, p, scale_shift=None, cache_linearization=False):
    if (GN_PRIMAL_BACKEND != "native" and cache_linearization
            and GN_STATS_MODE == "manual" and FUSED_GN and FUSE_POINTWISE_DERIVATIVES
            and p.is_cuda and p.dtype == torch.float32
            and isinstance(seq[0], nn.GroupNorm) and isinstance(seq[-1], nn.Conv2d)
            and sum(isinstance(layer, nn.SiLU) for layer in seq[1:-1]) == 1
            and all(isinstance(layer, (nn.SiLU, nn.Identity, nn.Dropout)) for layer in seq[1:-1])):
        from .fused_primal import normalization_activation_record
        norm, convolution = seq[0], seq[-1]
        stats = _gn_stats(norm, p)
        scale, shift = (None, None) if scale_shift is None else scale_shift
        activated, coefficient = normalization_activation_record(
            stats[2].reshape_as(p), norm.weight, norm.bias, scale, shift,
            backend="triton", fuse_silu=GN_PRIMAL_BACKEND == "full")
        output = F.conv2d(activated, convolution.weight, convolution.bias,
                          convolution.stride, convolution.padding,
                          convolution.dilation, convolution.groups)
        return output, {"p0": p, "gn_stats": stats[:2],
                        "pointwise_derivative": coefficient, "hw2": activated.shape[-2:]}
    rec = {"p0": p}
    for layer in seq:
        if isinstance(layer, nn.GroupNorm):
            if cache_linearization:
                p, rec["gn_stats"] = _gn_record(layer, p)
            else:
                p = _gn_p(layer, p)
            if scale_shift is not None:
                p = p * (1 + scale_shift[0]) + scale_shift[1]
            rec["p1"] = p
        elif isinstance(layer, nn.SiLU):
            if cache_linearization and FUSE_POINTWISE_DERIVATIVES:
                affine = seq[0].weight.view(1, -1, 1, 1)
                if scale_shift is not None:
                    affine = affine * (1 + scale_shift[0])
                rec["pointwise_derivative"] = torch.ops.aten.silu_backward(
                    affine.expand_as(p), p)
                # The coefficient replaces the equally sized preactivation;
                # neither derivative action needs that preactivation now.
                rec.pop("p1")
            p = F.silu(p)
            rec["hw2"] = p.shape[-2:]
        elif isinstance(layer, nn.Conv2d):
            p = F.conv2d(p, layer.weight, layer.bias, layer.stride, layer.padding,
                         layer.dilation, layer.groups)
        elif isinstance(layer, (nn.Identity, nn.Dropout)):
            pass
        else:
            raise NotImplementedError(type(layer))
    return p, rec


def _block_T(seq, rec, g, scale_shift=None):
    if "checkpoint_input" in rec:
        # This record lives only for the current action; never grow the stored
        # tape when a different derivative width is evaluated later.
        _, rec = _block_fwd(seq, rec["checkpoint_input"],
                            scale_shift=scale_shift, cache_linearization=True)
    gn, conv = seq[0], seq[-1]
    g = _conv_T(conv, g, rec["hw2"])
    if "pointwise_derivative" in rec:
        if GN_ACTION_BACKEND == "triton" and g.is_cuda and g.dtype == torch.float32:
            from .fused_norm import group_norm_action
            return group_norm_action(
                rec["p0"], g, *rec["gn_stats"][:2], gn.num_groups,
                coefficient=rec["pointwise_derivative"], transpose=True,
                backend="triton")
        # The coefficient already contains GN's channel weight, so the
        # symmetric normalization derivative must not multiply by it again.
        return _L(gn, rec["p0"], g * rec["pointwise_derivative"],
                  stats=rec["gn_stats"])
    g = _silu_T(rec["p1"], g)
    if scale_shift is not None:
        g = g * (1 + scale_shift[0])
    return _gn_T(gn, rec["p0"], g, stats=rec.get("gn_stats"))


def _attn_fwd(m: AttentionBlock, p, cache_linearization=False):
    if cache_linearization:
        hp, stats = _gn_record(m.norm, p)
    else:
        hp = _gn_p(m.norm, p)
    _, _, H, W = p.shape
    nh, hd_ = m.num_heads, m.head_dim

    def heads(z):
        return z.reshape(z.shape[0], nh, hd_, H * W).permute(0, 1, 3, 2)

    qp, kp, vp = (heads(F.conv2d(hp, c.weight, c.bias)) for c in (m.Q, m.K, m.V))
    scale = 1.0 / math.sqrt(hd_)
    P = torch.softmax((qp @ kp.transpose(-1, -2)) * scale, dim=-1)
    op = P @ vp
    out = op.permute(0, 1, 3, 2).reshape(1, nh * hd_, H, W)
    res = (p + F.conv2d(out, m.proj.weight, m.proj.bias)) / math.sqrt(2.0)
    rec = dict(p=p, q=qp, k=kp, v=vp, P=P, op=op, hw=(H, W))
    if cache_linearization:
        rec["gn_stats"] = stats
        if "moments" in (ATTENTION_JVP_MODE or ATTENTION_MODE,
                         ATTENTION_VJP_MODE or ATTENTION_MODE):
            rec["Pk"] = P @ kp
            # M_ij = E_P[k_i v_j]; the reverse pass uses M transposed.
            rec["M"] = (P @ (kp.unsqueeze(-1) * vp.unsqueeze(-2)).reshape(
                1, nh, H * W, hd_ * hd_)).reshape(1, nh, H * W, hd_, hd_)
    return res, rec


def _attn_T(m: AttentionBlock, rec, g):
    H, W = rec["hw"]
    nh, hd_ = m.num_heads, m.head_dim
    N = H * W
    k = g.shape[0]
    qp, kp, vp, P, op = rec["q"], rec["k"], rec["v"], rec["P"], rec["op"]
    scale = 1.0 / math.sqrt(hd_)
    g2 = g / math.sqrt(2.0)
    go = _conv_T(m.proj, g2, (H, W)).reshape(k, nh, hd_, N).permute(0, 1, 3, 2)  # (k,nh,N,hd)
    attention_mode = ATTENTION_VJP_MODE or ATTENTION_MODE
    if attention_mode == "chunked":
        q_bar, k_bar, v_bar = _attention_vjp_chunked(P, qp, kp, vp, op, go, scale)
    elif attention_mode == "moments":
        Pt = P.transpose(-1, -2)
        v_bar = _pmm(Pt, go)
        rho = (go * op).sum(-1, keepdim=True)                                      # (k,nh,N,1)
        if "M" in rec:
            Pk, Mp = rec["Pk"], rec["M"].transpose(-1, -2)
        else:
            Pk = P @ kp
            Mp = (P @ (vp.unsqueeze(-1) * kp.unsqueeze(-2)).reshape(1, nh, N, hd_ * hd_)
                  ).reshape(1, nh, N, hd_, hd_)                                      # sum_j P_ij v_jc k_jd
        q_bar = (torch.einsum("khic,bhicd->khid", go, Mp) - rho * Pk) * scale
        Y = (go.unsqueeze(-1) * qp.unsqueeze(-2)).reshape(k, nh, N, hd_ * hd_)      # go_ic q_id
        PtY = _pmm(Pt, Y).reshape(k, nh, N, hd_, hd_)
        k_bar = (torch.einsum("bhjc,khjcd->khjd", vp, PtY) - _pmm(Pt, rho * qp)) * scale
    else:
        raise ValueError(f"Unknown VJP attention mode: {attention_mode!r}")

    def unheads(z):
        return z.permute(0, 1, 3, 2).reshape(k, nh * hd_, H, W)

    gh = (_conv_T(m.Q, unheads(q_bar), (H, W)) + _conv_T(m.K, unheads(k_bar), (H, W))
          + _conv_T(m.V, unheads(v_bar), (H, W)))
    return g2 + _gn_T(m.norm, rec["p"], gh, stats=rec.get("gn_stats"))


def _resblock_fwd(m, p, emb, cache_linearization=False):
    e = m.cond_block(emb)
    rec = {"x": p, "e": e}
    h, rec["b1"] = _block_fwd(m.block1, p, cache_linearization=cache_linearization)
    if m.scale_shift:
        h, rec["b2"] = _block_fwd(m.block2, h, scale_shift=e.chunk(2, dim=1),
                                 cache_linearization=cache_linearization)
    else:
        h, rec["b2"] = _block_fwd(m.block2, h + e,
                                 cache_linearization=cache_linearization)
    s = p if isinstance(m.skip_connection, nn.Identity) else F.conv2d(
        p, m.skip_connection.weight, m.skip_connection.bias)
    h = (s + h) / math.sqrt(2.0)
    if isinstance(m.attn, AttentionBlock):
        h, rec["attn"] = _attn_fwd(m.attn, h, cache_linearization=cache_linearization)
    return h, rec


def _resblock_T(m, rec, g):
    if "checkpoint_input" in rec:
        _, rec = _resblock_fwd(m, rec["checkpoint_input"],
                               rec["checkpoint_emb"], cache_linearization=True)
    if "attn" in rec:
        g = _attn_T(m.attn, rec["attn"], g)
    g = g / math.sqrt(2.0)
    ss = rec["e"].chunk(2, dim=1) if m.scale_shift else None
    gh = _block_T(m.block2, rec["b2"], g, scale_shift=ss)
    gx = _block_T(m.block1, rec["b1"], gh)
    if isinstance(m.skip_connection, nn.Identity):
        return gx + g
    return gx + _conv_T(m.skip_connection, g, rec["x"].shape[-2:])


@torch.no_grad()
def unet_vjp(model, t, x: torch.Tensor, G: torch.Tensor):
    """x: (1,1,H,W) primal image; G: (k,1,H,W) cotangents. Returns (v, (d_x v)^T G)."""
    tvec = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(1)
    emb = model.time_proj(get_timestep_embedding(tvec * 1000.0, model.time_embed_dim))
    n_levels = len(model.down)
    # ---- primal forward with tape
    xp = F.conv2d(x, model.input_proj.weight, model.input_proj.bias, padding=model.input_proj.padding)
    hp = xp
    down, tape_down, tape_up = [], [], []
    for i, blocks in enumerate(model.down):
        hp, r0 = _resblock_fwd(blocks[0], hp, emb); down.append(hp)
        hp, r1 = _resblock_fwd(blocks[1], hp, emb); down.append(hp)
        rec = {"r0": r0, "r1": r1}
        if i < n_levels - 1:
            rec["hw"] = hp.shape[-2:]
            c = blocks[2].conv
            hp = F.conv2d(hp, c.weight, c.bias, c.stride, c.padding)
        tape_down.append(rec)
    hp, m0 = _resblock_fwd(model.mid[0], hp, emb)
    hp, m1 = _resblock_fwd(model.mid[1], hp, emb)
    for i, blocks in enumerate(model.up):
        rec = {}
        s = down.pop(); rec["c0"] = hp.shape[1]
        hp, rec["r0"] = _resblock_fwd(blocks[0], torch.cat((hp, s), 1), emb)
        s = down.pop(); rec["c1"] = hp.shape[1]
        hp, rec["r1"] = _resblock_fwd(blocks[1], torch.cat((hp, s), 1), emb)
        if i < n_levels - 1:
            rec["hw_in"] = hp.shape[-2:]
            size = tuple(down[-1].shape[-2:])
            hp = F.interpolate(hp, size=size, mode="nearest")
            rec["hw_up"] = hp.shape[-2:]
            c = blocks[2].conv
            hp = F.conv2d(hp, c.weight, c.bias, c.stride, c.padding)
        tape_up.append(rec)
    cfin = hp.shape[1]
    v, fin = _block_fwd(model.final, torch.cat((hp, xp), 1))
    # ---- reverse sweep over k cotangent lanes
    g = _block_T(model.final, fin, G)
    gh, gx = g[:, :cfin], g[:, cfin:]
    skip_g = []                                   # cotangents flowing into the skips
    for i in reversed(range(len(model.up))):
        blocks, rec = model.up[i], tape_up[i]
        if i < n_levels - 1:
            gh = _conv_T(blocks[2].conv, gh, rec["hw_up"])
            gh = _interp_T(gh, rec["hw_in"])
        gh = _resblock_T(blocks[1], rec["r1"], gh)
        gh, gs = gh[:, :rec["c1"]], gh[:, rec["c1"]:]; skip_g.append(gs)
        gh = _resblock_T(blocks[0], rec["r0"], gh)
        gh, gs = gh[:, :rec["c0"]], gh[:, rec["c0"]:]; skip_g.append(gs)
    gh = _resblock_T(model.mid[1], m1, gh)
    gh = _resblock_T(model.mid[0], m0, gh)
    for i in reversed(range(n_levels)):
        blocks, rec = model.down[i], tape_down[i]
        if i < n_levels - 1:
            gh = _conv_T(blocks[2].conv, gh, rec["hw"])
        gh = gh + skip_g.pop()
        gh = _resblock_T(blocks[1], rec["r1"], gh)
        gh = gh + skip_g.pop()
        gh = _resblock_T(blocks[0], rec["r0"], gh)
    gx = gx + gh
    return v, _conv_T(model.input_proj, gx, x.shape[-2:])


def _conv_J(m: nn.Conv2d, d):
    return F.conv2d(d, m.weight, None, m.stride, m.padding, m.dilation, m.groups)


def _block_J(seq, rec, d, scale_shift=None):
    if "checkpoint_input" in rec:
        _, rec = _block_fwd(seq, rec["checkpoint_input"],
                            scale_shift=scale_shift, cache_linearization=True)
    gn, conv = seq[0], seq[-1]
    if "pointwise_derivative" in rec:
        if GN_ACTION_BACKEND == "triton" and d.is_cuda and d.dtype == torch.float32:
            from .fused_norm import group_norm_action
            action = group_norm_action(
                rec["p0"], d, *rec["gn_stats"][:2], gn.num_groups,
                coefficient=rec["pointwise_derivative"], transpose=False,
                backend="triton")
            return _conv_J(conv, action)
        d = _L(gn, rec["p0"], d, stats=rec["gn_stats"])
        return _conv_J(conv, d * rec["pointwise_derivative"])
    d = _L(gn, rec["p0"], d, stats=rec["gn_stats"]) * gn.weight.view(1, -1, 1, 1)
    if scale_shift is not None:
        d = d * (1 + scale_shift[0])
    return _conv_J(conv, _silu_T(rec["p1"], d))


def _attn_J(m: AttentionBlock, rec, d):
    H, W = rec["hw"]
    nh, hd_ = m.num_heads, m.head_dim
    N = H * W
    qp, kp, vp, P, op = rec["q"], rec["k"], rec["v"], rec["P"], rec["op"]
    hd = (_L(m.norm, rec["p"], d, stats=rec["gn_stats"])
          * m.norm.weight.view(1, -1, 1, 1))

    def heads(z):
        return z.reshape(z.shape[0], nh, hd_, N).permute(0, 1, 3, 2)

    qd, kd, vd = (heads(_conv_J(layer, hd)) for layer in (m.Q, m.K, m.V))
    scale = 1.0 / math.sqrt(hd_)
    mode = ATTENTION_JVP_MODE or ATTENTION_MODE
    if mode == "chunked":
        od = _attention_jvp_chunked(P, qp, kp, vp, qd, kd, vd, scale)
    elif mode == "moments":
        if "M" in rec:
            Pk, M = rec["Pk"], rec["M"]
        else:
            Pk = P @ kp
            M = (P @ (kp.unsqueeze(-1) * vp.unsqueeze(-2)).reshape(
                1, nh, N, hd_ * hd_)).reshape(1, nh, N, hd_, hd_)
        Pdk = _pmm(P, kd)
        r = ((qd * Pk).sum(-1, keepdim=True)
             + (qp * Pdk).sum(-1, keepdim=True)) * scale
        PX = _pmm(P, (kd.unsqueeze(-1) * vp.unsqueeze(-2)).reshape(
            -1, nh, N, hd_ * hd_)).reshape(-1, nh, N, hd_, hd_)
        term = (torch.einsum("khic,bhicd->khid", qd, M)
                + torch.einsum("bhic,khicd->khid", qp, PX)) * scale
        od = term - r * op + _pmm(P, vd)
    else:
        raise ValueError(f"Unknown JVP attention mode: {mode!r}")
    od = od.permute(0, 1, 3, 2).reshape(d.shape[0], nh * hd_, H, W)
    return (d + _conv_J(m.proj, od)) / math.sqrt(2.0)


def _resblock_J(m, rec, d):
    if "checkpoint_input" in rec:
        _, rec = _resblock_fwd(m, rec["checkpoint_input"],
                               rec["checkpoint_emb"], cache_linearization=True)
    h = _block_J(m.block1, rec["b1"], d)
    scale_shift = rec["e"].chunk(2, dim=1) if m.scale_shift else None
    h = _block_J(m.block2, rec["b2"], h, scale_shift=scale_shift)
    skip = d if isinstance(m.skip_connection, nn.Identity) else _conv_J(m.skip_connection, d)
    h = (skip + h) / math.sqrt(2.0)
    return _attn_J(m.attn, rec["attn"], h) if "attn" in rec else h


class UnetLinearization:
    """One frozen-state primal tape supporting arbitrary derivative widths.

    The model must stay in evaluation mode with unchanged weights for the
    tape's lifetime. Actions do not modify the tape, their inputs, or earlier
    outputs. Storage is one velocity evaluation's primal activations, with
    shared normalization statistics and attention moments; tangent/adjoint
    activations are temporary and independent of the number of flow steps.

    With ``checkpoint_blocks=True``, retain only residual-block inputs and a
    shared time embedding (plus the final head input), then rebuild each
    block's internal tape when a JVP or VJP reaches it. This trades additional
    primal work on every action for lower retained activation storage. It
    does not checkpoint across flow steps or change either derivative map.
    """

    @torch.no_grad()
    def __init__(self, model, t, x, *, checkpoint_blocks=False):
        if model.training:
            raise ValueError("UNet linearization requires model.eval()")
        if GN_ACTION_BACKEND not in {"native", "triton"}:
            raise ValueError("GN_ACTION_BACKEND must be 'native' or 'triton'")
        if GN_PRIMAL_BACKEND not in {"native", "affine", "full"}:
            raise ValueError("GN_PRIMAL_BACKEND must be 'native', 'affine', or 'full'")
        self.model = model
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.input_hw = x.shape[-2:]
        tvec = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(1)
        emb = model.time_proj(get_timestep_embedding(tvec * 1000.0, model.time_embed_dim))
        n_levels = len(model.down)
        c = model.input_proj
        xp = F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)
        hp = xp

        def record_resblock(block, value):
            output, record = _resblock_fwd(block, value, emb, cache_linearization=True)
            if self.checkpoint_blocks:
                # Every value originates from an out-of-place operation in
                # this sweep. No caller-owned x/t storage is retained, and
                # the shared embedding likewise owns its computed storage.
                record = {"checkpoint_input": value, "checkpoint_emb": emb}
            return output, record

        down, self.tape_down, self.tape_up = [], [], []
        for i, blocks in enumerate(model.down):
            hp, r0 = record_resblock(blocks[0], hp)
            down.append(hp)
            hp, r1 = record_resblock(blocks[1], hp)
            down.append(hp)
            rec = {"r0": r0, "r1": r1}
            if i < n_levels - 1:
                rec["hw"] = hp.shape[-2:]
                c = blocks[2].conv
                hp = F.conv2d(hp, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)
            self.tape_down.append(rec)
        hp, self.mid0 = record_resblock(model.mid[0], hp)
        hp, self.mid1 = record_resblock(model.mid[1], hp)
        for i, blocks in enumerate(model.up):
            rec = {"c0": hp.shape[1]}
            hp, rec["r0"] = record_resblock(blocks[0], torch.cat((hp, down.pop()), 1))
            rec["c1"] = hp.shape[1]
            hp, rec["r1"] = record_resblock(blocks[1], torch.cat((hp, down.pop()), 1))
            if i < n_levels - 1:
                rec["hw_in"] = hp.shape[-2:]
                hp = F.interpolate(hp, size=tuple(down[-1].shape[-2:]), mode="nearest")
                rec["hw_up"] = hp.shape[-2:]
                c = blocks[2].conv
                hp = F.conv2d(hp, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)
            self.tape_up.append(rec)
        self.final_channels = hp.shape[1]
        final_input = torch.cat((hp, xp), 1)
        self.primal, self.final = _block_fwd(
            model.final, final_input, cache_linearization=True)
        if self.checkpoint_blocks:
            self.final = {"checkpoint_input": final_input}

    @torch.no_grad()
    def jvp(self, tangents):
        """Apply d_x v; checkpointed blocks recompute their local primal tape."""
        model = self.model
        n_levels = len(model.down)
        dx = _conv_J(model.input_proj, tangents)
        dh, down = dx, []
        for i, blocks in enumerate(model.down):
            rec = self.tape_down[i]
            dh = _resblock_J(blocks[0], rec["r0"], dh)
            down.append(dh)
            dh = _resblock_J(blocks[1], rec["r1"], dh)
            down.append(dh)
            if i < n_levels - 1:
                dh = _conv_J(blocks[2].conv, dh)
        dh = _resblock_J(model.mid[0], self.mid0, dh)
        dh = _resblock_J(model.mid[1], self.mid1, dh)
        for i, blocks in enumerate(model.up):
            rec = self.tape_up[i]
            dh = _resblock_J(blocks[0], rec["r0"], torch.cat((dh, down.pop()), 1))
            dh = _resblock_J(blocks[1], rec["r1"], torch.cat((dh, down.pop()), 1))
            if i < n_levels - 1:
                dh = F.interpolate(dh, size=tuple(rec["hw_up"]), mode="nearest")
                dh = _conv_J(blocks[2].conv, dh)
        return _block_J(model.final, self.final, torch.cat((dh, dx), 1))

    @torch.no_grad()
    def vjp(self, cotangents):
        """Apply (d_x v)^T to (k, 1, H, W) cotangents using the same tape."""
        model = self.model
        n_levels = len(model.down)
        g = _block_T(model.final, self.final, cotangents)
        gh, gx = g[:, :self.final_channels], g[:, self.final_channels:]
        skip_g = []
        for i in reversed(range(len(model.up))):
            blocks, rec = model.up[i], self.tape_up[i]
            if i < n_levels - 1:
                gh = _conv_T(blocks[2].conv, gh, rec["hw_up"])
                gh = _interp_T(gh, rec["hw_in"])
            gh = _resblock_T(blocks[1], rec["r1"], gh)
            gh, gs = gh[:, :rec["c1"]], gh[:, rec["c1"]:]
            skip_g.append(gs)
            gh = _resblock_T(blocks[0], rec["r0"], gh)
            gh, gs = gh[:, :rec["c0"]], gh[:, rec["c0"]:]
            skip_g.append(gs)
        gh = _resblock_T(model.mid[1], self.mid1, gh)
        gh = _resblock_T(model.mid[0], self.mid0, gh)
        for i in reversed(range(n_levels)):
            blocks, rec = model.down[i], self.tape_down[i]
            if i < n_levels - 1:
                gh = _conv_T(blocks[2].conv, gh, rec["hw"])
            gh = _resblock_T(blocks[1], rec["r1"], gh + skip_g.pop())
            gh = _resblock_T(blocks[0], rec["r0"], gh + skip_g.pop())
        return _conv_T(model.input_proj, gx + gh, self.input_hw)


def unet_linearize(model, t, x: torch.Tensor, *, checkpoint_blocks=False):
    """Build a shared hand linearization, optionally recomputing block internals.

    ``checkpoint_blocks`` keeps block inputs instead of their full local
    activation tapes. Repeated A/A^T actions reconstruct and discard each
    local tape independently, preserving the primal and derivative rules.
    """
    return UnetLinearization(model, t, x, checkpoint_blocks=checkpoint_blocks)

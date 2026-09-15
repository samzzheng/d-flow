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
# L(g * w) for grad_out = g, so L(y) = backward(grad_out = y / w) in ~3 memory
# passes instead of the ~7 an eager formula needs (fallback kept).
FUSED_GN = True
FUSED_GN_CONTIG = True


def _gn_stats(m: nn.GroupNorm, p):
    G = m.num_groups
    pg = p.reshape(1, G, -1)
    mu = pg.mean(-1, keepdim=True)
    xc = pg - mu
    rstd = torch.rsqrt((xc * xc).mean(-1, keepdim=True) + m.eps)
    return mu, rstd, xc * rstd


def _L(m: nn.GroupNorm, p, y):
    """Apply GroupNorm's (symmetric) input-linearisation at primal p to k lanes y."""
    k, C, H, W = y.shape
    G = m.num_groups
    mu, rstd, xhat = _gn_stats(m, p)
    if FUSED_GN:
        w = m.weight.view(1, C, 1, 1)
        pk = p.expand(k, C, H, W)
        if FUSED_GN_CONTIG:
            pk = pk.contiguous()
        out, _, _ = torch.ops.aten.native_group_norm_backward(
            y / w, pk, mu.reshape(1, G).expand(k, G).contiguous(),
            rstd.reshape(1, G).expand(k, G).contiguous(),
            m.weight, k, C, H * W, G, [True, False, False])
        return out
    yg = y.reshape(k, G, -1)
    out = (yg - yg.mean(-1, keepdim=True)
           - xhat * (xhat * yg).mean(-1, keepdim=True)) * rstd
    return out.reshape(y.shape)


def _gn(m: nn.GroupNorm, p, d):
    C = p.shape[1]
    w = m.weight.view(1, C, 1, 1)
    b = m.bias.view(1, C, 1, 1)
    _, _, xhat = _gn_stats(m, p)
    return xhat.reshape(p.shape) * w + b, _L(m, p, d) * w


def _gn_p(m: nn.GroupNorm, p):
    """Primal-only GroupNorm (used by the reverse pass's forward sweep)."""
    return F.group_norm(p, m.num_groups, m.weight, m.bias, m.eps)


def _silu(p, d):
    # tangent = d * silu'(p): the fused silu_backward kernel is that product
    # (one pass, broadcasting the batch-1 primal over the k lanes)
    return F.silu(p), torch.ops.aten.silu_backward(d, p.expand_as(d))


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
    # Softmax JVP without any per-lane N x N tensor. With dS = scale*(dq k^T +
    # q dk^T) and dP = P o (dS - r 1^T), r_i = sum_j P_ij dS_ij:
    #   (dP v)_i = sum_j P_ij dS_ij v_j - r_i (P v)_i
    #   sum_j P_ij dS_ij v_j = scale * ( dq_i . M_i + q_i . (P X)_i ),
    #   M_i = sum_j P_ij k_j v_j^T (primal only), X_j = dk_j v_j^T (per lane).
    # Everything is a matmul against the primal P; peak memory is O(k N hd^2).
    N = H * W
    Pk = P @ kp                                                     # (1,nh,N,hd)
    Pdk = P @ kd                                                    # (k,nh,N,hd)
    r = ((qd * Pk).sum(-1, keepdim=True)
         + (qp * Pdk).sum(-1, keepdim=True)) * scale                # (k,nh,N,1)
    M = (P @ (kp.unsqueeze(-1) * vp.unsqueeze(-2)).reshape(1, nh, N, hd_ * hd_)
         ).reshape(1, nh, N, hd_, hd_)
    PX = (P @ (kd.unsqueeze(-1) * vp.unsqueeze(-2)).reshape(-1, nh, N, hd_ * hd_)
          ).reshape(-1, nh, N, hd_, hd_)
    term = (torch.einsum("khic,bhicd->khid", qd, M)
            + torch.einsum("bhic,khicd->khid", qp, PX)) * scale
    od = term - r * op + P @ vd
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
    emb = model.time_proj(get_timestep_embedding(tvec * 1000.0, model.ch))
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
        gi, _, _ = torch.ops.aten.convolution_backward(
            g, torch.empty(shape, device=g.device, dtype=g.dtype), m.weight, None,
            m.stride, m.padding, m.dilation, False, [0, 0], m.groups,
            [True, False, False])
        return gi
    return torch.nn.grad.conv2d_input(shape, m.weight, g, m.stride, m.padding,
                                      m.dilation, m.groups)


def _gn_T(m: nn.GroupNorm, p_in, g):
    # adjoint of d -> L(d) * w  is  g -> L(g * w)  (L symmetric per group)
    return _L(m, p_in, g * m.weight.view(1, -1, 1, 1))


def _silu_T(p_in, g):
    return torch.ops.aten.silu_backward(g, p_in.expand_as(g))


def _interp_T(g, in_hw):
    """Adjoint of nearest-neighbour F.interpolate(size=g.shape[-2:])."""
    k, C, out_h, out_w = g.shape
    in_h, in_w = in_hw
    dev = g.device
    rows = F.interpolate(torch.arange(in_h, device=dev, dtype=torch.float32
                                      ).view(1, 1, in_h, 1), size=(out_h, 1),
                         mode="nearest").view(-1).long()
    cols = F.interpolate(torch.arange(in_w, device=dev, dtype=torch.float32
                                      ).view(1, 1, 1, in_w), size=(1, out_w),
                         mode="nearest").view(-1).long()
    tmp = torch.zeros(k, C, in_h, out_w, device=dev, dtype=g.dtype).index_add_(2, rows, g)
    return torch.zeros(k, C, in_h, in_w, device=dev, dtype=g.dtype).index_add_(3, cols, tmp)


def _block_fwd(seq, p, scale_shift=None):
    rec = {"p0": p}
    for layer in seq:
        if isinstance(layer, nn.GroupNorm):
            p = _gn_p(layer, p)
            if scale_shift is not None:
                p = p * (1 + scale_shift[0]) + scale_shift[1]
            rec["p1"] = p
        elif isinstance(layer, nn.SiLU):
            p = p * torch.sigmoid(p)
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
    gn, conv = seq[0], seq[-1]
    g = _conv_T(conv, g, rec["hw2"])
    g = _silu_T(rec["p1"], g)
    if scale_shift is not None:
        g = g * (1 + scale_shift[0])
    return _gn_T(gn, rec["p0"], g)


def _attn_fwd(m: AttentionBlock, p):
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
    return res, dict(p=p, q=qp, k=kp, v=vp, P=P, op=op, hw=(H, W))


def _attn_T(m: AttentionBlock, rec, g):
    H, W = rec["hw"]
    nh, hd_ = m.num_heads, m.head_dim
    N = H * W
    k = g.shape[0]
    qp, kp, vp, P, op = rec["q"], rec["k"], rec["v"], rec["P"], rec["op"]
    scale = 1.0 / math.sqrt(hd_)
    g2 = g / math.sqrt(2.0)
    go = _conv_T(m.proj, g2, (H, W)).reshape(k, nh, hd_, N).permute(0, 1, 3, 2)  # (k,nh,N,hd)
    Pt = P.transpose(-1, -2)
    v_bar = Pt @ go
    rho = (go * op).sum(-1, keepdim=True)                                      # (k,nh,N,1)
    Pk = P @ kp
    Mp = (P @ (vp.unsqueeze(-1) * kp.unsqueeze(-2)).reshape(1, nh, N, hd_ * hd_)
          ).reshape(1, nh, N, hd_, hd_)                                          # sum_j P_ij v_jc k_jd
    q_bar = (torch.einsum("khic,bhicd->khid", go, Mp) - rho * Pk) * scale
    Y = (go.unsqueeze(-1) * qp.unsqueeze(-2)).reshape(k, nh, N, hd_ * hd_)      # go_ic q_id
    PtY = (Pt @ Y).reshape(k, nh, N, hd_, hd_)
    k_bar = (torch.einsum("bhjc,khjcd->khjd", vp, PtY) - Pt @ (rho * qp)) * scale

    def unheads(z):
        return z.permute(0, 1, 3, 2).reshape(k, nh * hd_, H, W)

    gh = (_conv_T(m.Q, unheads(q_bar), (H, W)) + _conv_T(m.K, unheads(k_bar), (H, W))
          + _conv_T(m.V, unheads(v_bar), (H, W)))
    return g2 + _gn_T(m.norm, rec["p"], gh)


def _resblock_fwd(m, p, emb):
    e = m.cond_block(emb)
    rec = {"x": p, "e": e}
    h, rec["b1"] = _block_fwd(m.block1, p)
    if m.scale_shift:
        h, rec["b2"] = _block_fwd(m.block2, h, scale_shift=e.chunk(2, dim=1))
    else:
        h, rec["b2"] = _block_fwd(m.block2, h + e)
    s = p if isinstance(m.skip_connection, nn.Identity) else F.conv2d(
        p, m.skip_connection.weight, m.skip_connection.bias)
    h = (s + h) / math.sqrt(2.0)
    if isinstance(m.attn, AttentionBlock):
        h, rec["attn"] = _attn_fwd(m.attn, h)
    return h, rec


def _resblock_T(m, rec, g):
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
    emb = model.time_proj(get_timestep_embedding(tvec * 1000.0, model.ch))
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

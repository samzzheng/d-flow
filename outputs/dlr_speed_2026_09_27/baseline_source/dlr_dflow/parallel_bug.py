"""Parallel rank-adaptive BUG integrator for the flow sensitivity, fixed rank.

Ceruti, Kusch, Lubich, "A parallel rank-adaptive integrator for dynamical
low-rank approximation" (arXiv:2304.05660), applied to the residual ODE of the
notes,

    J_t ~= alpha_t I + Y_t,     dY/dt = F(t, Y) = alpha_t A_t + A_t Y - alpha_dot_t I,

with one explicit Euler substep per flow step and the rank held at r (the
2r x 2r augmented core is always truncated back to r).  This is a test-bed
module: enriched.evolve_trace_centered_enriched (projector splitting, KSL) is
untouched and stays the algorithm of record.

Per flow step, from Y_0 = U_0 S_0 V_0^T (all from the OLD bases, so the two
Jacobian-product calls are independent and can run concurrently):

    JVP call   A_t [ J_t P | alpha V_0 + U_0 S_0 | alpha w + Y_0 w ]      (m + r + p columns)
    VJP call   A_t^T [ U_0 | w ]                                           (r + p columns)

    K_1  = U_0 S_0 + h ( A_t(alpha V_0 + U_0 S_0) - alpha_dot V_0 )      K-step
    L_1  = V_0 S_0^T + h ( alpha A_t^T U_0 + V_0 S_0^T M_0^T - alpha_dot U_0 ),  M_0^T = U_0^T A_t^T U_0
    Sbar = U_0^T K_1                                                      S-step (exact identity
                                                                          with the fused K image)
    U~   = orth. complement part of [K_1 | F_0 w] against U_0             (r + p new left dirs)
    V~   = orth. complement part of [L_1 | F_0^T w] against V_0           (r + p new right dirs)
    S^   = [[Sbar, L_1^T V~], [U~^T K_1, S11]]                            ((2r+p) x (2r+p))
    SVD(S^) -> keep r:  U_1 = [U_0 U~] P_r,  S_1 = diag(sigma_r),  V_1 = [V_0 V~] Q_r

w are p Rademacher enrichment probes (p = 0 is the paper's integrator).  S11 is
0 in the paper's parallel integrator; with s11="euler" it is the Remark 3.1
value h U~^T F(t_0, Y_0) V~ = h (alpha U~^T A_t V~ - alpha_dot U~^T V~), which
costs one more, serial, JVP on V~ (r + p columns) -- and with one Euler substep
that variant is exactly the augmented-basis BUG (Galerkin) integrator.

Column count per step (m trace probes, rank r, p enrichment):
    parallel, s11=0      m + 2r + 2p    one stage
    parallel, s11=euler  m + 3r + 3p    two stages
    KSL (enriched.py)    m + 2r + 3p    three dependent stages (JVP -> QR -> VJP)

The first flow step (Y_0 = 0 has no bases) uses the same randomized sketch of
Y_h as enriched.py, so the comparison isolates the per-step integrator.
"""
from __future__ import annotations

import torch

from . import dlr_dflow as core
from .enriched import _rademacher, _thin_svd, _trace_probes

Factors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

_SIDE_STREAMS: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    key = device.index
    if key not in _SIDE_STREAMS:
        _SIDE_STREAMS[key] = torch.cuda.Stream(device)
    return _SIDE_STREAMS[key]


ORTH_REL_TOL = 1e-4


def _orth_against(basis: torch.Tensor, block: torch.Tensor,
                  rel_tol: float = ORTH_REL_TOL) -> torch.Tensor:
    """Orthonormal basis of the numerically independent part of `block`
    outside span(basis).

    Two passes of block Gram-Schmidt against the (orthonormal) basis, a thin
    QR of the remainder, and a rank-revealing cut: remainder directions whose
    singular value is below rel_tol of the largest are dropped (the paper's
    "filled with zero columns").  The cut matters: when the new directions all
    come from the range of a numerically low-rank A_t, the remainder is
    rank-deficient and QR of it returns directions with O(1) overlap with the
    basis -- the (1,1) block then couples those garbage directions with
    non-zero entries and the factors lose orthonormality (defect ~10 observed).
    The kept directions are well conditioned, so one more pass restores their
    orthogonality to the basis to rounding.
    """
    for _ in range(2):
        block = block - basis @ (basis.T @ block)
    q, r = torch.linalg.qr(block, mode="reduced")
    left, singular_values, _ = torch.linalg.svd(r, full_matrices=False)
    keep = singular_values > rel_tol * singular_values[0]
    if not bool(keep.any()):
        return q[:, :0]
    q = q @ left[:, keep]
    q = q - basis @ (basis.T @ q)
    q, _ = torch.linalg.qr(q, mode="reduced")
    return q


def parallel_bug_step(
    A, AT, factors: Factors, probe_sensitivities: torch.Tensor,
    probes: torch.Tensor, step_size: float, alpha: float,
    enrichment_probes: int, generator: torch.Generator,
    s11: str = "zero", overlap: bool = False,
):
    """One fixed-rank parallel BUG step.  Returns (factors, A J_t P, alpha_dot, diag).

    `diag` holds the relative size of the (1,1) block when s11="euler"
    (h * eta / ||S^|| in the paper's notation), else None.
    """
    U0, S0, V0 = factors
    dimension, rank = U0.shape
    m = probe_sensitivities.shape[1]
    p = enrichment_probes
    U0S0 = U0 @ S0
    jvp_inputs = [probe_sensitivities, alpha * V0 + U0S0]
    vjp_inputs = [U0]
    if p:
        omega = _rademacher(dimension, p, generator, U0.device, U0.dtype)
        jvp_inputs.append(alpha * omega + U0 @ (S0 @ (V0.T @ omega)))
        vjp_inputs.append(omega)
    vjp_in = torch.cat(vjp_inputs, dim=1)

    # ---- the two independent Jacobian-product calls ----------------------
    if overlap and U0.is_cuda:
        main = torch.cuda.current_stream(U0.device)
        side = _side_stream(U0.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            transpose_block = AT(vjp_in)
        first = A(torch.cat(jvp_inputs, dim=1))
        main.wait_stream(side)
        transpose_block.record_stream(main)
    else:
        transpose_block = AT(vjp_in)
        first = A(torch.cat(jvp_inputs, dim=1))

    probe_derivative = first[:, :m]
    k_image = first[:, m:m + rank]
    ATU0 = transpose_block[:, :rank]
    alpha_dot = (probes * probe_derivative).sum() / (m * dimension)   # 0-dim, no sync

    # ---- K, L and S-bar steps (explicit Euler, all from the old bases) -----
    K1 = U0S0 + step_size * (k_image - alpha_dot * V0)
    M0_T = U0.T @ ATU0                                     # U_0^T A_t^T U_0
    L1 = V0 @ S0.T + step_size * (
        alpha * ATU0 + V0 @ (S0.T @ M0_T) - alpha_dot * U0)
    Sbar1 = U0.T @ K1

    new_left, new_right = [K1], [L1]
    if p:
        forcing_omega = first[:, m + rank:] - alpha_dot * omega          # F_0 w
        AT_omega = transpose_block[:, rank:]
        transpose_forcing_omega = (                                       # F_0^T w
            alpha * AT_omega + V0 @ (S0.T @ (U0.T @ AT_omega)) - alpha_dot * omega)
        new_left.append(forcing_omega)
        new_right.append(transpose_forcing_omega)
    U_new = _orth_against(U0, torch.cat(new_left, dim=1))                # (d, r+p)
    V_new = _orth_against(V0, torch.cat(new_right, dim=1))               # (d, r+p)

    # ---- augmented core -----------------------------------------------------
    S_K = U_new.T @ K1                                                    # (r+p, r)
    S_L = L1.T @ V_new                                                    # (r, r+p)
    diag = None
    if s11 == "euler":
        # Remark 3.1 / eq. (3.9): the block the parallel integrator drops.
        # Y_0 V~ = 0 since V~ is orthogonal to V_0, so F_0 V~ = alpha A V~ - alpha_dot V~.
        A_Vnew = A(V_new)                                                 # serial, r+p cols
        S11 = step_size * (alpha * (U_new.T @ A_Vnew) - alpha_dot * (U_new.T @ V_new))
    else:
        S11 = U_new.new_zeros(U_new.shape[1], V_new.shape[1])
    core_matrix = torch.cat([
        torch.cat([Sbar1, S_L], dim=1),
        torch.cat([S_K, S11], dim=1),
    ], dim=0)
    if s11 == "euler":
        diag = (S11.norm() / core_matrix.norm()).item()
    P, singular_values, Q_T = torch.linalg.svd(core_matrix, full_matrices=False)
    U_hat = torch.cat([U0, U_new], dim=1)
    V_hat = torch.cat([V0, V_new], dim=1)
    new_factors = (
        U_hat @ P[:, :rank],
        torch.diag(singular_values[:rank]),
        V_hat @ Q_T.T[:, :rank],
    )
    return new_factors, probe_derivative, alpha_dot, diag


def evolve_parallel_bug(
    model: torch.nn.Module,
    z: torch.Tensor,
    n_steps: int,
    rank: int = 8,
    trace_probes: int = 4,
    initialization_oversampling: int = 8,
    enrichment_probes: int = 0,
    seed: int = 0,
    zero_terminal_baseline: bool = False,
    products=None,
    s11: str = "zero",
    overlap: bool = False,
    return_diagnostics: bool = False,
):
    """Evolve J ~= alpha I + U S V^T along the Euler flow with the parallel BUG step.

    Same interface and returns as enriched.evolve_trace_centered_enriched:
    (x_1, (U, S, V), alpha_1), plus a diagnostics dict when asked for.
    """
    if s11 not in ("zero", "euler"):
        raise ValueError("s11 must be 'zero' or 'euler'")
    height, width = z.shape[-2:]
    dimension = height * width
    h = 1.0 / n_steps
    probes, generator = _trace_probes(
        dimension, trace_probes, seed + 1, z.device, z.dtype)
    sensitivities = probes.clone()
    x = z.detach()
    alpha = 1.0
    s11_sizes: list[float] = []

    # First step: identical to enriched.py -- Y_0 = 0 has no bases, so sketch
    # the first nonzero residual Y_h ~= h A_0 + (1 - alpha_h) I.
    if products is not None:
        products.set_state(0.0, x)
        A, AT = products.A, products.AT
    else:
        A, AT = core.make_A_products(model, 0.0, x)
    sketch_size = min(dimension, rank + initialization_oversampling)
    sketch = _rademacher(dimension, sketch_size, generator, z.device, z.dtype)
    first = A(torch.cat([sensitivities, sketch], dim=1))
    sensitivity_derivative = first[:, :trace_probes]
    sketch_image = first[:, trace_probes:]
    sensitivities = (sensitivities + h * sensitivity_derivative).detach()
    alpha = (probes * sensitivities).sum() / (trace_probes * dimension)
    range_basis, _ = torch.linalg.qr(
        h * sketch_image + (1.0 - alpha) * sketch, mode="reduced")
    transpose_image = h * AT(range_basis) + (1.0 - alpha) * range_basis
    right_vectors, singular_values, left_coordinates_t = _thin_svd(transpose_image)
    retained = min(rank, singular_values.numel())
    factors = (
        range_basis @ left_coordinates_t.T[:, :retained],
        torch.diag(singular_values[:retained]),
        right_vectors[:, :retained],
    )
    with torch.no_grad():
        velocity = getattr(A, "last_primal", None)
        if velocity is None:
            velocity = core.velocity(model, 0.0, x)
        x = x + h * velocity.reshape_as(x)

    for index in range(1, n_steps):
        t = index * h
        if products is not None:
            products.set_state(t, x)
        else:
            A, AT = core.make_A_products(model, t, x)
        factors, sensitivity_derivative, _, diag = parallel_bug_step(
            A, AT, factors, sensitivities, probes, h, alpha,
            enrichment_probes, generator, s11=s11, overlap=overlap)
        if diag is not None:
            s11_sizes.append(diag)
        sensitivities = (sensitivities + h * sensitivity_derivative).detach()
        alpha = (probes * sensitivities).sum() / (trace_probes * dimension)
        with torch.no_grad():
            velocity = getattr(A, "last_primal", None)
            if velocity is None:
                velocity = core.velocity(model, t, x)
            x = x + h * velocity.reshape_as(x)

    alpha = float(alpha)
    if zero_terminal_baseline:
        alpha = 0.0
    if return_diagnostics:
        U, _, V = factors
        eye = torch.eye(rank, device=z.device, dtype=z.dtype)
        diagnostics = {
            "s11_relative_size_mean": (sum(s11_sizes) / len(s11_sizes)) if s11_sizes else None,
            "s11_relative_size_max": max(s11_sizes) if s11_sizes else None,
            "U_orthogonality_defect": (U.T @ U - eye).norm().item(),
            "V_orthogonality_defect": (V.T @ V - eye).norm().item(),
        }
        return x, factors, alpha, diagnostics
    return x, factors, alpha


def columns_per_step(rank: int, trace_probes: int, enrichment_probes: int,
                     integrator: str, s11: str = "zero") -> dict:
    """Jacobian-product columns per flow step and dependent stages, by design."""
    r, m, p = rank, trace_probes, enrichment_probes
    if integrator == "ksl":
        # enriched._fused_enriched_step with FUSE_K_IMAGE + RECOMPRESS_FROM_TRANSPOSE
        return {"jvp": m + r + p, "vjp": r + 2 * p, "serial_jvp": 0, "stages": 3}
    if s11 == "euler":
        return {"jvp": m + r + p, "vjp": r + p, "serial_jvp": r + p, "stages": 2}
    return {"jvp": m + r + p, "vjp": r + p, "serial_jvp": 0, "stages": 1}

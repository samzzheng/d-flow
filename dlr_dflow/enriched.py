"""Trace-centered and basis-enriched rank-r flow-sensitivity approximations."""

from __future__ import annotations

from typing import Callable

import torch

from . import dlr_dflow as core

# Derive the S-step core M = U1^T A_t U1 from the L-step's A_t^T U1 instead of a
# separate JVP call. Exact in real arithmetic; flag kept for A/B measurement.
M_FROM_TRANSPOSE = True
# Advance the flow state with the primal v(t, x) already computed inside the
# step's JVP call instead of a separate forward pass. Exact; flag for A/B.
SHARE_PRIMAL = True
# Compute the recompression core U_c^T A_t X as (A_t^T U_c)^T X: fix the
# candidate U basis [U1 | Q2] before the L-step's VJP call and ride the extra
# p columns in it, deleting the separate recompression JVP call (r+p forward-
# mode columns plus a primal forward per step). Exact in real arithmetic.
RECOMPRESS_FROM_TRANSPOSE = True


Factors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _rademacher(rows: int, columns: int, generator: torch.Generator,
                 device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (2 * torch.randint(
        0, 2, (rows, columns), generator=generator, device=device,
        dtype=torch.int64) - 1).to(dtype=dtype)


def _randomized_svd(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    transpose_matvec: Callable[[torch.Tensor], torch.Tensor],
    dimension: int,
    rank: int,
    oversampling: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    warm_start: torch.Tensor = None,
) -> Factors:
    """Randomized SVD of an implicit operator.

    warm_start (d, k) supplies leading sketch directions instead of random
    ones.  For a slowly-moving source the previous right singular vectors
    already span most of the dominant subspace, so seeding them lets the same
    sketch width resolve it more accurately (columns are rescaled to the
    Rademacher column norm to keep the QR well conditioned)."""
    sketch_size = min(dimension, rank + oversampling)
    omega = _rademacher(
        dimension, sketch_size, generator, device=device, dtype=dtype)
    if warm_start is not None and warm_start.numel():
        seeded = min(warm_start.shape[1], sketch_size)
        scale = dimension ** 0.5
        omega[:, :seeded] = scale * warm_start[:, :seeded].to(
            device=device, dtype=dtype)
    image = matvec(omega)
    range_basis, _ = torch.linalg.qr(image, mode="reduced")
    transpose_image = transpose_matvec(range_basis)
    # Factor the tall (d, k) matrix as QR first and take the SVD of the small
    # k x k triangular factor.  Equivalent, cheaper, and far better
    # conditioned: J_1 is numerically rank ~2 here, so the tall matrix has
    # repeated near-zero singular values that make a direct GPU SVD fail to
    # converge (worse with a warm-started sketch, which by design overlaps the
    # dominant subspace).
    basis_t, triangular = torch.linalg.qr(transpose_image, mode="reduced")
    try:
        small_u, singular_values, left_coordinates_t = torch.linalg.svd(
            triangular, full_matrices=False)
    except Exception:                      # last-resort: LAPACK on the host
        small_u, singular_values, left_coordinates_t = (
            t.to(triangular.device) for t in torch.linalg.svd(
                triangular.detach().cpu(), full_matrices=False))
    right_vectors = basis_t @ small_u
    retained = min(rank, singular_values.numel())
    left_vectors = range_basis @ left_coordinates_t.T[:, :retained]
    return (
        left_vectors,
        torch.diag(singular_values[:retained]),
        right_vectors[:, :retained],
    )


def _trace_probes(dimension: int, count: int, seed: int,
                  device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    probes = _rademacher(
        dimension, count, generator, device=device, dtype=dtype)
    return probes, generator


def trace_centered_flow(model: torch.nn.Module, z: torch.Tensor, n_steps: int,
                        trace_probes: int = 8, seed: int = 0,
                        return_alpha_path: bool = False):
    """Evolve x and probe sensitivities Q=J P, estimating a=tr(J)/d."""
    dimension = z.numel()
    probes, _ = _trace_probes(
        dimension, trace_probes, seed + 1, z.device, z.dtype)
    sensitivities = probes.clone()
    x = z.detach()
    step_size = 1.0 / n_steps
    alpha_path = [1.0]

    for index in range(n_steps):
        t = index * step_size
        A, _ = core.make_A_products(model, t, x)
        sensitivity_derivative = A(sensitivities)
        sensitivities = (sensitivities + step_size * sensitivity_derivative).detach()
        alpha = (probes * sensitivities).sum().item() / (trace_probes * dimension)
        with torch.no_grad():
            x = x + step_size * core.velocity(model, t, x)
        alpha_path.append(alpha)

    result = (x, alpha_path[-1])
    return (*result, alpha_path) if return_alpha_path else result


def _recompress_enriched_euler(
    A,
    AT,
    factors: Factors,
    projector_factors: Factors,
    step_size: float,
    alpha: float,
    alpha_dot: float,
    enrichment_probes: int,
    generator: torch.Generator,
) -> Factors:
    """Augment the projector bases with residual forcing directions and truncate."""
    U, S, V = factors
    projector_U, _, projector_V = projector_factors
    dimension, rank = U.shape
    omega = _rademacher(
        dimension, enrichment_probes, generator, U.device, U.dtype)

    residual_omega = U @ (S @ (V.T @ omega))
    forcing_omega = A(alpha * omega + residual_omega) - alpha_dot * omega

    transpose_A_omega = AT(omega)
    transpose_forcing_omega = (
        alpha * transpose_A_omega
        + V @ (S.T @ (U.T @ transpose_A_omega))
        - alpha_dot * omega
    )

    candidate_U, _ = torch.linalg.qr(
        torch.cat([projector_U, forcing_omega], dim=1), mode="reduced")
    candidate_V, _ = torch.linalg.qr(
        torch.cat([projector_V, transpose_forcing_omega], dim=1), mode="reduced")

    residual_candidate_V = U @ (S @ (V.T @ candidate_V))
    forcing_candidate_V = (
        A(alpha * candidate_V + residual_candidate_V)
        - alpha_dot * candidate_V
    )
    euler_target_candidate_V = (
        residual_candidate_V + step_size * forcing_candidate_V)
    small_core = candidate_U.T @ euler_target_candidate_V
    core_U, singular_values, core_V_t = torch.linalg.svd(
        small_core, full_matrices=False)
    return (
        candidate_U @ core_U[:, :rank],
        torch.diag(singular_values[:rank]),
        candidate_V @ core_V_t.T[:, :rank],
    )


def _fused_enriched_step(
    A,
    AT,
    factors: Factors,
    probe_sensitivities: torch.Tensor,
    probes: torch.Tensor,
    step_size: float,
    alpha: float,
    enrichment_probes: int,
    generator: torch.Generator,
    adaptive_rank: bool = False,
    rank_min: int = 4,
    rank_max: int = 16,
    rank_tol: float = 1e-2,
):
    """One enriched projector-splitting step in two Jacobian-product calls.

    Mathematically identical to projector_split_step followed by
    _recompress_enriched_euler, but every product whose input is available up
    front shares one batched call: the trace-probe derivative, A V0, A K and
    the enrichment forcing all ride in the first JVP, and AT(U1) shares its
    VJP with AT(omega).  Returns the new factors, the probe derivative A P
    (for the trace update outside), and alpha_dot.
    """
    U, S, V = factors
    dimension, rank = U.shape
    n_probes = probe_sensitivities.shape[1]
    omega = _rademacher(
        dimension, enrichment_probes, generator, U.device, U.dtype)

    US = U @ S
    residual_omega = U @ (S @ (V.T @ omega))
    first = A(torch.cat(
        [probe_sensitivities, V, US, alpha * omega + residual_omega], dim=1))
    probe_derivative = first[:, :n_probes]
    AV0 = first[:, n_probes:n_probes + rank]
    AK = first[:, n_probes + rank:n_probes + 2 * rank]
    forcing_image = first[:, n_probes + 2 * rank:]

    alpha_dot = (probes * probe_derivative).sum() / (
        n_probes * dimension)          # 0-dim GPU tensor: no sync
    forcing_omega = forcing_image - alpha_dot * omega

    # K-step
    K1 = US + step_size * (alpha * AV0 + AK - alpha_dot * V)
    U1, Shat = torch.linalg.qr(K1)
    # The L-step's VJP block A^T [U1 | omega] is needed anyway; computing it
    # first lets the S-step take M = U1^T A U1 = (U1^T A^T U1)^T from it and
    # drop the separate A(U1) JVP call (8 forward-mode columns per step).
    via_transpose = RECOMPRESS_FROM_TRANSPOSE and enrichment_probes > 0
    if via_transpose:
        # candidate U = [U1 | Q2] with Q2 the orthonormalised part of the
        # enrichment forcing outside span(U1): same span as qr([U1 | forcing]),
        # known before the VJP call so A^T of it can share that call.
        Q2, _ = torch.linalg.qr(
            forcing_omega - U1 @ (U1.T @ forcing_omega), mode="reduced")
        transpose_block = AT(torch.cat([U1, Q2, omega], dim=1))
        ATU1 = transpose_block[:, :rank]
        ATQ2 = transpose_block[:, rank:rank + enrichment_probes]
        transpose_A_omega = transpose_block[:, rank + enrichment_probes:]
    else:
        transpose_block = AT(torch.cat([U1, omega], dim=1))
        ATU1 = transpose_block[:, :rank]
        transpose_A_omega = transpose_block[:, rank:]
    if M_FROM_TRANSPOSE:
        M = (U1.T @ ATU1).T
    else:
        M = U1.T @ A(U1)
    # S-step
    Stil = Shat + step_size * (
        -alpha * (U1.T @ AV0) - M @ Shat + alpha_dot * (U1.T @ V))
    # L-step
    L0 = V @ Stil.T
    L1 = L0 + step_size * (alpha * ATU1 + L0 @ M.T - alpha_dot * U1)
    V1, _ = torch.linalg.qr(L1)

    # Enrichment and recompression (the math of _recompress_enriched_euler).
    transpose_forcing_omega = (
        alpha * transpose_A_omega
        + V @ (S.T @ (U.T @ transpose_A_omega))
        - alpha_dot * omega
    )
    candidate_V, _ = torch.linalg.qr(
        torch.cat([V1, transpose_forcing_omega], dim=1), mode="reduced")
    residual_candidate_V = U @ (S @ (V.T @ candidate_V))
    if via_transpose:
        candidate_U = torch.cat([U1, Q2], dim=1)
        transpose_A_candidate_U = torch.cat([ATU1, ATQ2], dim=1)
        # U_c^T [Y V_c + h (alpha A V_c + A Y V_c - alpha_dot V_c)] with every
        # U_c^T A X written as (A^T U_c)^T X
        small_core = (
            candidate_U.T @ residual_candidate_V
            + step_size * (
                alpha * (transpose_A_candidate_U.T @ candidate_V)
                + transpose_A_candidate_U.T @ residual_candidate_V
                - alpha_dot * (candidate_U.T @ candidate_V)))
    else:
        candidate_U, _ = torch.linalg.qr(
            torch.cat([U1, forcing_omega], dim=1), mode="reduced")
        forcing_candidate_V = (
            A(alpha * candidate_V + residual_candidate_V)
            - alpha_dot * candidate_V
        )
        euler_target_candidate_V = (
            residual_candidate_V + step_size * forcing_candidate_V)
        small_core = candidate_U.T @ euler_target_candidate_V
    core_U, singular_values, core_V_t = torch.linalg.svd(
        small_core, full_matrices=False)
    if adaptive_rank:
        # notes Sec. 13: truncate the augmented core by singular-value
        # threshold - rank floats in [rank_min, rank_max]
        keep = int((singular_values >= rank_tol * singular_values[0]).sum().item())
        r_keep = max(rank_min, min(keep, rank_max, singular_values.numel()))
    else:
        r_keep = rank
    new_factors = (
        candidate_U @ core_U[:, :r_keep],
        torch.diag(singular_values[:r_keep]),
        candidate_V @ core_V_t.T[:, :r_keep],
    )
    return new_factors, probe_derivative, alpha_dot


def evolve_trace_centered_enriched(
    model: torch.nn.Module,
    z: torch.Tensor,
    n_steps: int,
    rank: int = 8,
    trace_probes: int = 8,
    initialization_oversampling: int = 8,
    enrichment_probes: int = 4,
    seed: int = 0,
    return_alpha_path: bool = False,
    return_diagnostics: bool = False,
    zero_terminal_baseline: bool = False,
    seed_basis: torch.Tensor = None,
    products=None,
    factor_stride: int = 1,
    adaptive_rank: bool = False,
    rank_min: int = 4,
    rank_max: int = 16,
    rank_tol: float = 1e-2,
):
    """Evolve J ~= a I + U S V^T with trace centering and basis enrichment.

    The zero residual at t=0 is handled by delaying factor initialization until
    after one Euler step and sketching that first nonzero residual. Subsequent
    steps use projector splitting, augment its bases with fresh forcing
    directions, and truncate the enriched approximation back to ``rank``.

    seed_basis (d x k, optional) implements notes Sec. 9 options A/C: its
    columns replace the first k random columns of the initialization sketch,
    so directions the optimizer already uses (previous source gradient, or
    the previous iteration's V subspace) get range-finding priority at equal
    cost. None (default) is option B: purely random probes.
    """
    height, width = z.shape[-2:]
    dimension = height * width
    if rank + enrichment_probes > dimension:
        raise ValueError("rank plus enrichment probes exceeds the state dimension")
    step_size = 1.0 / n_steps
    probes, generator = _trace_probes(
        dimension, trace_probes, seed + 1, z.device, z.dtype)
    sensitivities = probes.clone()
    x = z.detach()
    alpha = 1.0
    alpha_path = [alpha]
    diagnostics = [{
        "t": 0.0,
        "alpha": alpha,
        "residual_singular_values": [0.0] * rank,
    }]

    # First step: Y_0=0 has no singular vectors. Sketch the first nonzero
    # residual Y_h ~= h A_0 + (1-a_h) I instead of inserting epsilon noise.
    # The probe derivative and the sketch image share one batched JVP.
    if products is not None:
        products.set_state(0.0, x)
        A, AT = products.A, products.AT
    else:
        A, AT = core.make_A_products(model, 0.0, x)
    sketch_rank = rank_max if adaptive_rank else rank
    sketch_size = min(dimension, sketch_rank + initialization_oversampling)
    sketch = _rademacher(dimension, sketch_size, generator, z.device, z.dtype)
    if seed_basis is not None and seed_basis.numel() > 0:
        # Notes Sec. 9 options A/C: give supplied directions range-finding
        # priority. Columns are scaled to the Rademacher norm sqrt(d) so the
        # (1 - alpha) identity correction treats them like the random probes.
        seeds = seed_basis.detach().reshape(dimension, -1).to(z.dtype)
        norms = seeds.norm(dim=0, keepdim=True).clamp_min(1e-12)
        seeds = seeds / norms * dimension ** 0.5
        k = min(seeds.shape[1], sketch_size)
        sketch = torch.cat([seeds[:, :k], sketch[:, k:]], dim=1)
    first = A(torch.cat([sensitivities, sketch], dim=1))
    sensitivity_derivative = first[:, :trace_probes]
    sketch_image = first[:, trace_probes:]
    sensitivities = (sensitivities + step_size * sensitivity_derivative).detach()
    next_alpha = (probes * sensitivities).sum() / (
        trace_probes * dimension)
    range_basis, _ = torch.linalg.qr(
        step_size * sketch_image + (1.0 - next_alpha) * sketch, mode="reduced")
    transpose_image = (
        step_size * AT(range_basis) + (1.0 - next_alpha) * range_basis)
    right_vectors, singular_values, left_coordinates_t = torch.linalg.svd(
        transpose_image, full_matrices=False)
    if adaptive_rank:
        keep = int((singular_values >= rank_tol * singular_values[0]).sum().item())
        retained = max(rank_min, min(keep, rank_max, singular_values.numel()))
    else:
        retained = min(rank, singular_values.numel())
    factors = (
        range_basis @ left_coordinates_t.T[:, :retained],
        torch.diag(singular_values[:retained]),
        right_vectors[:, :retained],
    )
    with torch.no_grad():
        x = x + step_size * core.velocity(model, 0.0, x)
    alpha = next_alpha
    alpha_path.append(alpha)
    diagnostics.append({
        "t": step_size,
        "alpha": alpha,
        "residual_singular_values": torch.diag(factors[1]).detach().cpu().tolist(),
    })

    # factor_stride > 1: multirate integration - the state x advances on the
    # fine grid every step, while the factor/trace ODEs take Euler steps of
    # h = stride*dt evaluated at the fine trajectory's current (t, x). A
    # remainder step below lands the factor time exactly on t = 1.
    factor_time = step_size
    for index in range(1, n_steps):
        t = index * step_size
        if index % factor_stride == 0:
            h = step_size * factor_stride
            if products is not None:
                products.set_state(t, x)
            else:
                A, AT = core.make_A_products(model, t, x)
            factors, sensitivity_derivative, _ = _fused_enriched_step(
                A, AT, factors, sensitivities, probes, h, alpha,
                enrichment_probes, generator, adaptive_rank=adaptive_rank,
                rank_min=rank_min, rank_max=rank_max, rank_tol=rank_tol)
            sensitivities = (sensitivities + h * sensitivity_derivative).detach()
            alpha = (probes * sensitivities).sum() / (
                trace_probes * dimension)
            factor_time += h
            velocity_here = getattr(A, "last_primal", None) if SHARE_PRIMAL else None
        else:
            velocity_here = None
        with torch.no_grad():
            if velocity_here is not None:
                x = x + step_size * velocity_here.reshape_as(x)
            else:
                x = x + step_size * core.velocity(model, t, x)
        alpha_path.append(alpha)
        if return_diagnostics:
            diagnostics.append({
                "t": (index + 1) * step_size,
                "alpha": float(alpha),
                "residual_singular_values": torch.diag(factors[1]).detach().cpu().tolist(),
            })

    if factor_time < 1.0 - 1e-9:
        h = 1.0 - factor_time
        if products is not None:
            products.set_state(1.0 - h, x)
        else:
            A, AT = core.make_A_products(model, 1.0 - h, x)
        factors, sensitivity_derivative, _ = _fused_enriched_step(
            A, AT, factors, sensitivities, probes, h, alpha,
            enrichment_probes, generator, adaptive_rank=adaptive_rank,
            rank_min=rank_min, rank_max=rank_max, rank_tol=rank_tol)
        sensitivities = (sensitivities + h * sensitivity_derivative).detach()
        alpha = (probes * sensitivities).sum() / (
            trace_probes * dimension)
        alpha_path.append(alpha)

    alpha = float(alpha)                 # one sync for the whole evolution
    if zero_terminal_baseline:
        # J_1 is numerically low rank for contractive flows, so the additive
        # alpha_1 I term is pure error at t=1 (notes Sec. 4 revision).
        alpha = 0.0
        alpha_path[-1] = 0.0
    if return_alpha_path or return_diagnostics:
        alpha_path = [float(a) for a in alpha_path]
    result = (x, factors, alpha)
    if return_diagnostics:
        return (*result, diagnostics)
    return (*result, alpha_path) if return_alpha_path else result


def make_terminal_flow_products(model: torch.nn.Module, z: torch.Tensor,
                                n_steps: int):
    """Return products J_1 Q and J_1^T Q for the discrete flow.

    The tangent/cotangent columns are vmapped over a single unbatched primal
    trajectory, so the flow itself is solved once per call rather than once
    per column."""
    height, width = z.shape[-2:]
    dimension = height * width
    z1 = z.reshape(1, 1, height, width)

    def flow(value):
        return core.euler_flow(model, value, n_steps)

    def _lanes(Q):
        return Q.T.reshape(-1, 1, 1, height, width)

    def J(Q):
        out = torch.func.vmap(
            lambda tangent: torch.func.jvp(flow, (z1,), (tangent,))[1])(_lanes(Q))
        return out.reshape(Q.shape[1], dimension).T

    def JT(Q):
        _, vjp_function = torch.func.vjp(flow, z1)
        out = torch.func.vmap(lambda cot: vjp_function(cot)[0])(_lanes(Q))
        return out.reshape(Q.shape[1], dimension).T

    return J, JT


def direct_terminal_rank_factors(
    model: torch.nn.Module,
    z: torch.Tensor,
    n_steps: int,
    rank: int,
    alpha: float,
    oversampling: int = 8,
    seed: int = 0,
    sketch_steps: int = None,
    warm_start: torch.Tensor = None,
) -> Factors:
    """Randomized terminal approximation of J_1-alpha I.

    sketch_steps overrides the number of Euler steps used to differentiate the
    flow.  The objective keeps its own (finer) step count; the factors only
    have to supply a descent direction, so a coarser flow here trades a little
    direction accuracy for a proportional cut in refresh cost."""
    dimension = z.numel()
    generator = torch.Generator(device=z.device).manual_seed(seed)
    steps = n_steps if sketch_steps is None else sketch_steps
    J, JT = make_terminal_flow_products(model, z, steps)
    return _randomized_svd(
        lambda Q: J(Q) - alpha * Q,
        lambda Q: JT(Q) - alpha * Q,
        dimension, rank, oversampling, generator, z.device, z.dtype,
        warm_start=warm_start)

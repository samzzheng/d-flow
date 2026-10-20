"""Trace-centered and basis-enriched rank-r flow-sensitivity approximations."""

from __future__ import annotations

from typing import Callable

import torch

from sampler.ode_samplers import euler_flow

from .. import dlr_dflow as core
from .types import Factors

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
# Form the K-step image A(alpha V + US) directly instead of applying A to V
# and US separately. The S-step recovers U1^T A V from (A^T U1)^T V, using
# the existing transpose product. This saves rank JVP columns per step.
FUSE_K_IMAGE = True
# With no enrichment and a fixed rank, the square recompression core is not
# truncated. Keep that general core during evolution instead of diagonalizing
# it after every step. Public factors are still returned in SVD form.
SKIP_UNTRUNCATED_SVD = True
# Experimental algebra specialization for the fixed-rank p=0 fast path. It
# reuses the K/L QR triangular factors to evaluate the same projected Euler
# core with fewer tall matrix products. No approximation budget changes.
SPECIALIZE_ZERO_ENRICHMENT = False


def _canonicalize_factors(factors: Factors) -> Factors:
    """Put an orthonormal low-rank factorization into diagonal-core form."""
    U, S, V = factors
    left, values, right_t = torch.linalg.svd(S, full_matrices=False)
    return U @ left, torch.diag(values), V @ right_t.T


def _rademacher(rows: int, columns: int, generator: torch.Generator,
                 device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (2 * torch.randint(
        0, 2, (rows, columns), generator=generator, device=device,
        dtype=torch.int64) - 1).to(dtype=dtype)


def _thin_svd(matrix: torch.Tensor) -> Factors:
    """SVD of a tall sketch through its small QR factor.

    The near-zero singular values of these sketches can make a direct tall
    GPU SVD fail to converge. Only the small triangular factor needs an SVD;
    the tall left singular vectors follow from multiplying its QR basis.
    """
    basis, triangular = torch.linalg.qr(matrix, mode="reduced")
    try:
        small_u, singular_values, right_vectors_t = torch.linalg.svd(
            triangular, full_matrices=False)
    except torch.linalg.LinAlgError:       # last-resort: LAPACK on the host
        small_u, singular_values, right_vectors_t = (
            t.to(triangular.device) for t in torch.linalg.svd(
                triangular.detach().cpu(), full_matrices=False))
    return basis @ small_u, singular_values, right_vectors_t


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
    right_vectors, singular_values, left_coordinates_t = _thin_svd(transpose_image)
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


def _zero_enrichment_step(
    A, AT, factors: Factors, probe_sensitivities: torch.Tensor,
    probes: torch.Tensor, step_size: float, alpha: float,
):
    """Equivalent p=0 KSL/recompression algebra using the two QR factors.

    Write W=alpha*A^T U1-alpha_dot*U1, M=(A^T U1)^T U1 and P=W^T V.
    The K-step QR gives R_K=U1^T K1. With C=P+M R_K, the original S/L
    steps are S_tilde=R_K-h C and L1=V(S_tilde+h M S_tilde)^T+h W.
    If L1=V1 R_L, the projected Euler recompression is exactly

        S1 = R_L^T + h^2 M C (V^T V1).

    This identity retains both ordinary QR factorizations. It removes tall
    recompression products; reassociation can change floating-point rounding.
    Only call with zero enrichment, fixed rank and the default alpha rule.
    """
    U, S, V = factors
    dimension, rank = U.shape
    n_probes = probe_sensitivities.shape[1]
    US = U @ S
    first = A(torch.cat([probe_sensitivities, alpha * V + US], dim=1))
    probe_derivative = first[:, :n_probes]
    # n_probes == 0: analytic plain-OT baseline alpha_t = 1 - t (alpha_dot = -1)
    alpha_dot = ((probes * probe_derivative).sum() / (n_probes * dimension)
                 if n_probes else -1.0)
    U1, R_K = torch.linalg.qr(
        US + step_size * (first[:, n_probes:n_probes + rank] - alpha_dot * V))
    ATU1 = AT(U1)
    W = alpha * ATU1 - alpha_dot * U1
    M = ATU1.T @ U1
    C = W.T @ V + M @ R_K
    S_tilde = R_K - step_size * C
    L1 = V @ (S_tilde + step_size * (M @ S_tilde)).T + step_size * W
    V1, R_L = torch.linalg.qr(L1)
    small_core = R_L.T + step_size * step_size * (M @ C) @ (V.T @ V1)
    return (U1, small_core, V1), probe_derivative, alpha_dot


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
    alpha_rate=None,
):
    """One enriched projector-splitting step in two Jacobian-product calls.

    Mathematically identical to projector_split_step followed by
    _recompress_enriched_euler, but every product whose input is available up
    front shares one batched call: the trace-probe derivative, A(alpha V0+K)
    and the enrichment forcing all ride in the first JVP, and AT(U1) shares
    its VJP with the enrichment directions. Projected products in the S-step
    reuse AT(U1). Returns the new factors, the probe derivative A P (for the
    trace update outside), and alpha_dot.
    """
    U, S, V = factors
    dimension, rank = U.shape
    n_probes = probe_sensitivities.shape[1]
    # A custom rate receives U and might depend on its particular basis,
    # rather than only its span. Preserve the old gauge for those hooks.
    untruncated_core = (SKIP_UNTRUNCATED_SVD and enrichment_probes == 0
                        and not adaptive_rank and alpha_rate is None)
    if (SPECIALIZE_ZERO_ENRICHMENT and untruncated_core
            and FUSE_K_IMAGE and M_FROM_TRANSPOSE and RECOMPRESS_FROM_TRANSPOSE):
        return _zero_enrichment_step(
            A, AT, factors, probe_sensitivities, probes, step_size, alpha)
    # A zero-width view avoids an unnecessary RNG operation and lets the
    # fixed-rank, zero-enrichment step be captured without a generator.
    omega = (U[:, :0] if enrichment_probes == 0 else _rademacher(
        dimension, enrichment_probes, generator, U.device, U.dtype))

    US = U @ S
    k_inputs = [alpha * V + US] if FUSE_K_IMAGE else [V, US]
    residual_omega = U @ (S @ (V.T @ omega))
    first = A(torch.cat(
        [probe_sensitivities, *k_inputs, alpha * omega + residual_omega], dim=1))
    probe_derivative = first[:, :n_probes]
    if FUSE_K_IMAGE:
        k_image = first[:, n_probes:n_probes + rank]
        forcing_image = first[:, n_probes + rank:]
    else:
        AV0 = first[:, n_probes:n_probes + rank]
        AK = first[:, n_probes + rank:n_probes + 2 * rank]
        k_image = alpha * AV0 + AK
        forcing_image = first[:, n_probes + 2 * rank:]
    if alpha_rate is None and n_probes == 0:
        alpha_dot = -1.0               # analytic alpha_t = 1 - t
    elif alpha_rate is None:
        alpha_dot = (probes * probe_derivative).sum() / (
            n_probes * dimension)      # 0-dim GPU tensor: no sync
    else:
        alpha_dot = alpha_rate(probe_derivative, U)
    forcing_omega = forcing_image - alpha_dot * omega

    # K-step
    K1 = US + step_size * (k_image - alpha_dot * V)
    U1, Shat = torch.linalg.qr(K1)
    # The L-step's VJP block A^T [U1 | omega] is needed anyway; computing it
    # first lets the S-step take M = U1^T A U1 = (U1^T A^T U1)^T from it and
    # drop the separate A(U1) JVP call (8 forward-mode columns per step).
    via_transpose = RECOMPRESS_FROM_TRANSPOSE
    if via_transpose:
        # candidate U = [U1 | Q2] with Q2 the orthonormalised part of the
        # enrichment forcing outside span(U1): same span as qr([U1 | forcing]),
        # known before the VJP call so A^T of it can share that call.
        if enrichment_probes > 0:
            # A QR of the projected forcing alone can complete a deficient
            # range with vectors inside span(U1), invalidating the small-core
            # SVD. A joint QR supplies an orthogonal completion even when the
            # forcing is zero, without changing the fixed product widths.
            augmented_U, _ = torch.linalg.qr(
                torch.cat([U1, forcing_omega], dim=1), mode="reduced")
            Q2 = augmented_U[:, rank:]
        else:
            # U_c = U1 without enrichment, so AT(U1) already supplies the
            # entire recompression core. Avoid a redundant rank-column JVP.
            Q2 = U1[:, :0]
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
    projected_AV0 = ATU1.T @ V if FUSE_K_IMAGE else U1.T @ AV0
    Stil = Shat + step_size * (
        -alpha * projected_AV0 - M @ Shat + alpha_dot * (U1.T @ V))
    # L-step
    L0 = V @ Stil.T
    L1 = L0 + step_size * (alpha * ATU1 + L0 @ M.T - alpha_dot * U1)
    V1, _ = torch.linalg.qr(L1)

    # Enrichment and recompression (the math of _recompress_enriched_euler).
    if untruncated_core:
        # V1 is already orthonormal, and no columns are being appended.
        candidate_V = V1
    else:
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
        if untruncated_core:
            candidate_U = U1
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
    if untruncated_core:
        return (candidate_U, small_core, candidate_V), probe_derivative, alpha_dot
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
    factor_callback=None,
    alpha_rate=None,
    checkpoint_blocks: bool = False,
    _canonicalize_terminal: bool = True,
    residual_sketch_steps: int | None = None,
):
    """Evolve J ~= a I + U S V^T with trace centering and basis enrichment.

    trace_probes=0 selects the analytic plain-OT baseline alpha_t = 1 - t
    (alpha_dot = -1; the scalar Euler update telescopes exactly and reaches 0
    at t = 1), removing the probe columns from every JVP.

    alpha_rate(step, probe_derivative, U) -> alpha_dot, if given, replaces the
    Hutchinson rate tr(P^T A_t J_t P)/(m d) and alpha is integrated from it,
    alpha_{k+1} = alpha_k + h alpha_dot_k, instead of being re-read from the
    probe sensitivities (identical in exact arithmetic for the default rate).
    probe_derivative is A_t J_t P for the trace probes P; U is the current left
    factor (zero columns at step 0). Off by default.

    factor_callback(step, alpha, factors), if given, is called with the
    current model J_t ~= alpha I + U S V^T after every completed Euler step
    (step = 0 is J_0 = I with empty factors). It is off by default and does
    not change the evolution. Callback and returned factors have a diagonal
    core of descending nonnegative singular values, even when the evolution
    uses a general square core internally.
    Internal gradient-only callers may set _canonicalize_terminal=False to
    retain that core at return, avoiding an SVD and two tall basis rotations.
    Callback factors remain canonical; graph-backed outputs remain owned.

    residual_sketch_steps=2 opts into a Gaussian sketch of the exact two-step
    Euler residual; 1 provides a matched Gaussian control. Requires analytical
    alpha, fixed rank, zero enrichment and factor_stride=1. Numerically zero
    sketches extend the exact prefix on the original grid; unresolved rank is
    not filled with arbitrary QR directions. None preserves legacy startup.

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
    if products is not None and bool(getattr(products, "checkpoint_blocks", False)) != bool(checkpoint_blocks):
        raise ValueError("checkpoint_blocks must match the supplied product provider")
    height, width = z.shape[-2:]
    dimension = height * width
    if rank + enrichment_probes > dimension:
        raise ValueError("rank plus enrichment probes exceeds the state dimension")
    if residual_sketch_steps is not None:
        if not 1 <= residual_sketch_steps <= n_steps:
            raise ValueError("residual_sketch_steps must lie between 1 and n_steps")
        if trace_probes or alpha_rate is not None or adaptive_rank or enrichment_probes or factor_stride != 1:
            raise ValueError("residual sketch startup requires analytical alpha, fixed rank, zero enrichment and stride 1")
    step_size = 1.0 / n_steps
    untruncated_core = (SKIP_UNTRUNCATED_SVD and enrichment_probes == 0
                        and not adaptive_rank and alpha_rate is None)
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
    }] if return_diagnostics else None
    if factor_callback is not None:
        empty = z.new_zeros(dimension, 0)
        factor_callback(0, 1.0, (empty, z.new_zeros(0, 0), empty))

    startup_steps = 1
    if residual_sketch_steps is not None:
        from .residual_startup import residual_sketch_startup
        x, factors, startup_steps = residual_sketch_startup(
            model, z, n_steps, rank, initialization_oversampling,
            residual_sketch_steps, generator, checkpoint_blocks=checkpoint_blocks,
            products=products, seed_basis=seed_basis,
            factor_callback=factor_callback, diagnostics=diagnostics)
        alpha = 1.0 - startup_steps * step_size
        alpha_path = [1.0 - index * step_size for index in range(startup_steps + 1)]
        if products is not None:
            A, AT = products.A, products.AT
    else:
        # First step: Y_0=0 has no singular vectors. Sketch the first nonzero
        # residual Y_h ~= h A_0 + (1-a_h) I instead of inserting epsilon noise.
        # The probe derivative and the sketch image share one batched JVP.
        if products is not None:
            products.set_state(0.0, x)
            A, AT = products.A, products.AT
        else:
            A, AT = core.make_A_products(model, 0.0, x, checkpoint_blocks=checkpoint_blocks)
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
        if alpha_rate is None and trace_probes == 0:
            next_alpha = 1.0 - step_size
        elif alpha_rate is None:
            next_alpha = (probes * sensitivities).sum() / (
                trace_probes * dimension)
        else:
            next_alpha = alpha + step_size * alpha_rate(
                0, sensitivity_derivative, z.new_zeros(dimension, 0))
        range_basis, _ = torch.linalg.qr(
            step_size * sketch_image + (1.0 - next_alpha) * sketch, mode="reduced")
        transpose_image = (
            step_size * AT(range_basis) + (1.0 - next_alpha) * range_basis)
        right_vectors, singular_values, left_coordinates_t = _thin_svd(transpose_image)
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
            velocity_here = getattr(A, "last_primal", None) if SHARE_PRIMAL else None
            if velocity_here is None:
                velocity_here = core.velocity(model, 0.0, x)
            x = x + step_size * velocity_here.reshape_as(x)
        alpha = next_alpha
        alpha_path.append(alpha)
        if factor_callback is not None:
            factor_callback(1, float(alpha), factors)
        if return_diagnostics:
            diagnostics.append({
                "t": step_size,
                "alpha": float(alpha),
                "residual_singular_values": torch.diag(factors[1]).detach().cpu().tolist(),
            })

    # An explicitly enabled provider can capture the complete fixed-rank,
    # zero-enrichment step, including QR/core algebra and the state update.
    # The initial randomized sketch remains outside that graph.
    whole_step = None
    if (getattr(products, "full_step_graphs", False) and SHARE_PRIMAL and untruncated_core
            and factor_stride == 1 and n_steps > startup_steps):
        whole_step = products.prepare_step_graph(
            x, factors, sensitivities, probes, alpha, step_size)

    # factor_stride > 1: multirate integration - the state x advances on the
    # fine grid every step, while the factor/trace ODEs take Euler steps of
    # h = stride*dt evaluated at the fine trajectory's current (t, x). A
    # remainder step below lands the factor time exactly on t = 1.
    factor_time = startup_steps * step_size
    for index in range(startup_steps, n_steps):
        t = index * step_size
        if whole_step is not None:
            whole_step.run(t)
            x, factors = whole_step.x, whole_step.factors
            sensitivities, alpha = whole_step.sensitivities, whole_step.alpha
            factor_time += step_size
        elif index % factor_stride == 0:
            h = step_size * factor_stride
            if products is not None:
                products.set_state(t, x)
            else:
                A, AT = core.make_A_products(model, t, x, checkpoint_blocks=checkpoint_blocks)
            step_rate = (None if alpha_rate is None else
                         (lambda pd, U_now, k=index: alpha_rate(k, pd, U_now)))
            factors, sensitivity_derivative, alpha_dot = _fused_enriched_step(
                A, AT, factors, sensitivities, probes, h, alpha,
                enrichment_probes, generator, adaptive_rank=adaptive_rank,
                rank_min=rank_min, rank_max=rank_max, rank_tol=rank_tol,
                alpha_rate=step_rate)
            sensitivities = (sensitivities + h * sensitivity_derivative).detach()
            if alpha_rate is None and trace_probes == 0:
                alpha = alpha - h          # telescopes to 1 - t exactly
            elif alpha_rate is None:
                alpha = (probes * sensitivities).sum() / (
                    trace_probes * dimension)
            else:
                alpha = alpha + h * alpha_dot
            factor_time += h
            velocity_here = getattr(A, "last_primal", None) if SHARE_PRIMAL else None
        else:
            velocity_here = None
        if whole_step is None:
            with torch.no_grad():
                if velocity_here is not None:
                    x = x + step_size * velocity_here.reshape_as(x)
                else:
                    x = x + step_size * core.velocity(model, t, x)
        # The whole-step backend updates its scalar in place. Requested paths
        # need a snapshot; otherwise every entry would alias the final alpha.
        alpha_path.append(alpha.clone() if whole_step is not None
                          and (return_alpha_path or return_diagnostics) else alpha)
        if factor_callback is not None:
            # Canonicalizing for an observer must not change the internal
            # gauge, or enabling diagnostics could change subsequent steps.
            observed_factors = (_canonicalize_factors(factors)
                                if untruncated_core else factors)
            factor_callback(index + 1, float(alpha), observed_factors)
        if return_diagnostics:
            if factor_callback is not None:
                values = torch.diag(observed_factors[1])
            elif untruncated_core:
                values = torch.linalg.svdvals(factors[1])
            else:
                values = torch.diag(factors[1])
            diagnostics.append({
                "t": (index + 1) * step_size,
                "alpha": float(alpha),
                "residual_singular_values": values.detach().cpu().tolist(),
            })

    if factor_time < 1.0 - 1e-9:
        h = 1.0 - factor_time
        if products is not None:
            products.set_state(1.0 - h, x)
        else:
            A, AT = core.make_A_products(model, 1.0 - h, x, checkpoint_blocks=checkpoint_blocks)
        factors, sensitivity_derivative, _ = _fused_enriched_step(
            A, AT, factors, sensitivities, probes, h, alpha,
            enrichment_probes, generator, adaptive_rank=adaptive_rank,
            rank_min=rank_min, rank_max=rank_max, rank_tol=rank_tol)
        # The legacy multirate remainder uses the default scalar rule even
        # when the main loop has a custom alpha hook. Its core can therefore
        # be general although the main-loop fast path was disabled.
        if (not untruncated_core and SKIP_UNTRUNCATED_SVD
                and enrichment_probes == 0 and not adaptive_rank):
            factors = _canonicalize_factors(factors)
        sensitivities = (sensitivities + h * sensitivity_derivative).detach()
        alpha = (alpha - h if trace_probes == 0 else
                 (probes * sensitivities).sum() / (trace_probes * dimension))
        alpha_path.append(alpha)

    alpha = float(alpha)                 # one sync for the whole evolution
    if zero_terminal_baseline:
        # J_1 is numerically low rank for contractive flows, so the additive
        # alpha_1 I term is pure error at t=1 (notes Sec. 4 revision).
        alpha = 0.0
        alpha_path[-1] = 0.0
    if return_alpha_path or return_diagnostics:
        alpha_path = [float(a) for a in alpha_path]
    if whole_step is not None:
        # The provider can reset/replay for the next source. Return owned
        # tensors; canonicalization below also creates independent factors.
        x = x.clone()
    if untruncated_core and n_steps > startup_steps and _canonicalize_terminal:
        factors = _canonicalize_factors(factors)
    elif whole_step is not None:
        factors = tuple(value.clone() for value in factors)
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
        return euler_flow(model, value, n_steps)

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

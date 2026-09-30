"""Independent exact-map checks, spectral references, and noise diagnostics."""
from __future__ import annotations

import math
import time
from typing import Callable

import torch

from dlr_dflow.api import approx_source_gradient, make_terminal_flow_products, sensitivity_factors, terminal_gradient
from sampler.ode_samplers import euler_flow
from .study import BudgetExpired, Problem, Record, Tensor, candidate, tangent

Product = Callable[[Tensor], Tensor]


def comparison(estimate: Tensor, reference: Tensor) -> Record:
    a, b = estimate.flatten().double(), reference.flatten().double()
    na, nb = float(a.norm()), float(b.norm())
    difference = float((a - b).norm())
    return {"absolute_error": difference, "relative_error": difference / nb if nb else None,
            "cosine": float(a @ b) / (na * nb) if na and nb else None,
            "norm_ratio": na / nb if nb else None}


def reference_svd(J: Product, JT: Product, gx: Tensor, exact: Tensor, rank: int,
                  seed: int, deadline: float, output: Record, max_refinements: int = 6,
                  refine_cutoff: bool = False) -> tuple[Tensor, Tensor, Tensor] | None:
    """Independent block iteration; certification uses both singular residuals.

    Keep one extra singular value to expose ambiguity at the rank cutoff.
    A small residual alone cannot prove that a randomized range found the
    globally largest singular values; record the reference as numerical.
    """
    dimension = gx.numel()
    k = min(dimension, rank + 9)
    generator = torch.Generator().manual_seed(seed)
    omega = torch.randn(dimension, k, generator=generator, dtype=gx.dtype)
    output.update(status="uncertified", refinements=[])
    previous = None
    factors = None
    q, _ = torch.linalg.qr(J(omega), mode="reduced")
    for iteration in range(max_refinements):
        if time.monotonic() >= deadline:
            output["status"] = "uncertified_deadline"
            break
        b = JT(q)
        v, singular, small_ut = torch.linalg.svd(b, full_matrices=False)
        u = q @ small_ut.T
        estimate = v[:, :rank] @ (singular[:rank, None] * (u[:, :rank].T @ gx.reshape(-1, 1)))
        left = J(v[:, :rank]) - u[:, :rank] * singular[:rank]
        right = JT(u[:, :rank]) - v[:, :rank] * singular[:rank]
        norm = float(singular[0])
        residual = max(float(left.norm(dim=0).max()), float(right.norm(dim=0).max())) / max(norm, 1e-30)
        stability = float((estimate - previous).norm()) / max(float(exact.norm()), 1e-30) if previous is not None else None
        gap = float((singular[rank - 1] - singular[rank]) / singular[0]) if rank < len(singular) else None
        row = {"iteration": iteration, "relative_triplet_residual": residual,
               "gradient_change_over_exact_norm": stability, "relative_cutoff_gap": gap,
               "singular_values": singular.tolist()}
        output["refinements"].append(row)
        factors = (u[:, :rank], torch.diag(singular[:rank]), v[:, :rank])
        output["gradient"] = comparison(estimate, exact)
        if residual < 1e-3 and stability is not None and stability < .01:
            output["status"] = "certified_numerical" if gap is None or gap > max(2 * residual, 1e-6) else "ambiguous_cutoff"
            if output["status"] == "certified_numerical" or not refine_cutoff:
                break
        previous = estimate
        if time.monotonic() >= deadline:
            output["status"] = "uncertified_deadline"
            break
        q, _ = torch.linalg.qr(J(b), mode="reduced")
    return factors


def preflight(problem: Problem) -> Record:
    torch.manual_seed(problem.seed)
    z = problem.source
    start = time.perf_counter()
    x, _, exact = problem.evaluate(z, "exact")
    exact_seconds = time.perf_counter() - start
    start = time.perf_counter()
    native, _, alpha, path = sensitivity_factors(
        problem.model, z, problem.steps, rank=8, speed_profile="fast", trace_probes=0,
        enrichment_probes=0, initialization_oversampling=8, seed=problem.seed,
        zero_terminal_baseline=True, return_alpha_path=True)
    dlr_seconds = time.perf_counter() - start
    expected = torch.linspace(1, 0, problem.steps + 1, dtype=torch.float64)
    alpha_error = float((torch.tensor(path, dtype=torch.float64) - expected).abs().max())
    endpoint_error = float((native - x).norm() / x.norm())
    if alpha != 0 or alpha_error > 1e-12 or endpoint_error > 1e-4:
        raise AssertionError(f"Analytical alpha or matched endpoint check failed: {alpha}, {alpha_error}, {endpoint_error}")
    direction = torch.randn_like(z)
    direction /= direction.norm()
    _, plus = problem.trial(z + .02 * direction)
    _, minus = problem.trial(z - .02 * direction)
    finite_difference = (plus - minus) / .04
    derivative = float((exact * direction).sum())
    derivative_error = abs(finite_difference - derivative) / max(abs(derivative), 1e-8)
    J, JT = make_terminal_flow_products(problem.model, z, problem.steps)
    q, p = torch.randn(z.numel(), 1), torch.randn(z.numel(), 1)
    tick = time.perf_counter()
    jq, jtp = J(q), JT(p)
    adjoint_error = abs(float((p * jq).sum() - (q * jtp).sum())) / max(
        float(p.norm() * jq.norm()), float(q.norm() * jtp.norm()), 1e-30)
    if derivative_error > .08 or adjoint_error > 1e-4:
        raise AssertionError(f"Exact AD checks failed: finite difference {derivative_error}, adjoint {adjoint_error}")
    return {"exact_gradient_seconds": exact_seconds, "fresh_dlr_gradient_seconds": dlr_seconds,
            "jvp_vjp_pair_seconds": time.perf_counter() - tick,
            "endpoint_relative_difference": endpoint_error, "alpha_max_error": alpha_error,
            "terminal_alpha": float(alpha), "finite_difference_relative_error": derivative_error,
            "adjoint_scaled_error": adjoint_error}


def diagnose(problem: Problem, z: Tensor, output: Record, deadline: float, projected: bool) -> None:
    start = time.perf_counter()
    x, loss, exact = problem.evaluate(z, "exact")
    native, factors, alpha = sensitivity_factors(
        problem.model, z, problem.steps, rank=8, speed_profile="fast", trace_probes=0,
        enrichment_probes=0, initialization_oversampling=8,
        zero_terminal_baseline=True, seed=problem.seed)
    gx = terminal_gradient(x, problem.operator, problem.noisy)
    dlr = approx_source_gradient(factors, gx, alpha)
    v = factors[2]
    subspace = (v @ (v.T @ exact.reshape(-1, 1))).reshape_as(exact)
    lhs = float((dlr.double() - exact.double()).square().sum())
    rhs = float((dlr.double() - subspace.double()).square().sum() + (subspace.double() - exact.double()).square().sum())
    output.update(endpoint_relative_difference=float((native - x).norm() / x.norm()),
                  dlr=comparison(dlr, exact), subspace=comparison(subspace, exact),
                  coefficient=comparison(dlr, subspace), decomposition_relative_error=abs(lhs - rhs) / max(lhs, 1e-30),
                  tangent_dlr=comparison(tangent(dlr, z, projected), tangent(exact, z, projected)))
    raw_J, raw_JT = make_terminal_flow_products(problem.model, z, problem.steps)
    # Bound memory by processing two tangent/cotangent columns at a time.
    def J(q: Tensor) -> Tensor:
        return torch.cat([raw_J(chunk) for chunk in q.split(2, dim=1)], dim=1)

    def JT(q: Tensor) -> Tensor:
        return torch.cat([raw_JT(chunk) for chunk in q.split(2, dim=1)], dim=1)

    clean_gx = terminal_gradient(x, problem.operator, problem.clean)
    clean_gradient = JT(clean_gx.reshape(-1, 1)).reshape_as(z)
    output["clean_gradient"] = []
    for noise_index in range(11):
        measurement = problem.noisy
        if noise_index:
            noise = torch.randn(problem.clean.shape, generator=torch.Generator().manual_seed(problem.seed + 30000 + noise_index))
            measurement = problem.clean + noise * (.01 * problem.clean.norm() / noise.norm())
        noisy_gx = terminal_gradient(x, problem.operator, measurement)
        noisy_gradient = exact if noise_index == 0 else JT(noisy_gx.reshape(-1, 1)).reshape_as(z)
        approximation = approx_source_gradient(factors, noisy_gx, alpha)
        exact_error = float((noisy_gradient - clean_gradient).norm())
        approximate_error = float((approximation - clean_gradient).norm())
        ratio = approximate_error / exact_error if exact_error else None
        output["clean_gradient"].append({"noise_index": noise_index, "trajectory_noise": noise_index == 0,
            "exact_to_clean": comparison(noisy_gradient, clean_gradient),
            "dlr_to_clean": comparison(approximation, clean_gradient),
            "dlr_to_noisy": comparison(approximation, noisy_gradient), "R": ratio,
            "closer_to_clean": ratio < 1 if ratio is not None else None})
    # Selected true Strong Wolfe tests on the SAME initial capped search curve.
    # Diagnose the conditions; this is not a surrogate-gradient Wolfe optimizer.
    output["strong_wolfe"] = []
    for label, gradient in (("exact", exact), ("dlr", dlr)):
        direction = -tangent(gradient, z, projected)
        slope0 = float((exact * direction).sum())
        rate = min(min(1., 1 / max(float(gradient.abs().sum()), 1e-30)), .64 / max(float(direction.norm()), 1e-30))
        for backtrack in range(3):
            trial = candidate(z, direction, rate, projected)
            _, trial_loss, trial_gradient = problem.evaluate(trial, "exact")
            curve_derivative = direction
            if projected:
                raw = z + rate * direction
                curve_derivative = (math.sqrt(z.numel()) / raw.norm()) * tangent(direction, raw, True)
            slope = float((trial_gradient * curve_derivative).sum())
            output["strong_wolfe"].append({"estimator_direction": label, "backtrack": backtrack, "rate": rate,
                "true_initial_slope": slope0, "true_trial_slope": slope,
                "true_armijo": slope0 < 0 and trial_loss <= loss + 1e-4 * rate * slope0,
                "strong_curvature": slope0 < 0 and abs(slope) <= .9 * abs(slope0)})
            rate *= .5
    output["spectral"] = {}
    # Reserve capacity for other trajectory points instead of spending the phase at one source.
    local_deadline = min(deadline, time.monotonic() + 360)
    reference_svd(J, JT, gx, exact, 8, problem.seed + 40000, local_deadline, output["spectral"])
    output["seconds"] = time.perf_counter() - start

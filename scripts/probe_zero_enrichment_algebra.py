"""CPU-only prototype of equivalent fixed-rank zero-enrichment DLRA algebra.

This does not change production code. It reuses the K/L QR triangular factors
to avoid recompressing the projected Euler target through tall matrices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlr_dflow import enriched


def zero_step(a, at, factors, sensitivities, probes, h, alpha, *, qr_core=True):
    """Same p=0 KSL/recompression step, with reassociated exact algebra."""
    u, s, v = factors
    d, rank = u.shape
    m = probes.shape[1]
    us = u @ s
    first = a(torch.cat((sensitivities, alpha * v + us), dim=1))
    derivative = first[:, :m]
    alpha_dot = (probes * derivative).sum() / (m * d)
    u1, r_k = torch.linalg.qr(us + h * (first[:, m:] - alpha_dot * v))
    atu1 = at(u1)
    w = alpha * atu1 - alpha_dot * u1
    matrix = atu1.T @ u1
    projected_w = w.T @ v
    correction = projected_w + matrix @ r_k
    s_tilde = r_k - h * correction
    # (V S_tilde^T) (I+h M^T) = V [S_tilde+h M S_tilde]^T.
    l1 = v @ (s_tilde + h * (matrix @ s_tilde)).T + h * w
    v1, r_l = torch.linalg.qr(l1)
    overlap = v.T @ v1
    if qr_core:
        # R_L^T = L1^T V1. Recompression differs from L1^T V1 by
        # h^2 M(P+M R_K) V^T V1, with P=W^T V.
        core = r_l.T + h * h * (matrix @ correction) @ overlap
    else:
        # Intermediate form, without using the L-step QR identity.
        core = (r_k - h * projected_w) @ overlap + h * (w.T @ v1)
    return (u1, core, v1), derivative, alpha_dot


def reference_step(a, at, factors, sensitivities, probes, h, alpha):
    return enriched._fused_enriched_step(
        a, at, factors, sensitivities, probes, h, alpha, 0, None)


def problem(d, rank, dtype, condition=100., seed=3091):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, dtype=dtype, generator=generator)
    u = torch.linalg.qr(randn(d, rank))[0]
    v = torch.linalg.qr(randn(d, rank))[0]
    s = torch.diag(torch.logspace(0., -torch.log10(torch.tensor(condition)).item(), rank, dtype=dtype))
    probes = (2 * torch.randint(0, 2, (d, 4), generator=generator) - 1).to(dtype)
    sensitivities = .7 * probes + u @ (s @ (v.T @ probes))
    diagonal = torch.linspace(-.8, .3, d, dtype=dtype)
    left = randn(d, 3) / d ** .5
    right = randn(d, 3) / d ** .5
    a = lambda q: diagonal[:, None] * q + left @ (right.T @ q)
    at = lambda q: diagonal[:, None] * q + right @ (left.T @ q)
    return a, at, (u, s, v), sensitivities, probes, randn(d, 3)


def relative(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-20))


def action(factors, cotangents):
    u, s, v = factors
    return v @ (s.T @ (u.T @ cotangents))


@torch.no_grad()
def validate():
    rows = []
    for dtype in (torch.float32, torch.float64):
        for condition in (1., 1e3, 1e6):
            a, at, initial, q0, probes, cotangents = problem(128, 8, dtype, condition)
            results = []
            one_step_results = []
            for implementation in (reference_step, zero_step):
                factors, sensitivities, alpha = initial, q0, .7
                for step in range(100):
                    factors, derivative, _ = implementation(a, at, factors, sensitivities, probes, .01, alpha)
                    sensitivities = sensitivities + .01 * derivative
                    alpha = (probes * sensitivities).sum() / probes.numel()
                    if step == 0:
                        one_step_results.append(action(factors, cotangents))
                u, s, v = factors
                results.append({"action": action(factors, cotangents), "q": sensitivities,
                                "alpha": alpha, "u_orth": (u.T @ u - torch.eye(8, dtype=dtype)).norm(),
                                "v_orth": (v.T @ v - torch.eye(8, dtype=dtype)).norm()})
            reference, candidate = results
            row = {"dtype": str(dtype), "initial_core_condition": condition,
                   "one_step_action_relative_error": relative(one_step_results[1], one_step_results[0]),
                   "100_step_action_relative_error": relative(candidate["action"], reference["action"]),
                   "100_step_trace_sensitivity_relative_error": relative(candidate["q"], reference["q"]),
                   "alpha_abs_difference": float((candidate["alpha"] - reference["alpha"]).abs()),
                   "candidate_u_orthogonality": float(candidate["u_orth"]),
                   "candidate_v_orthogonality": float(candidate["v_orth"])}
            tolerance = 2e-3 if dtype == torch.float32 else 1e-10
            if row["100_step_action_relative_error"] > tolerance:
                raise RuntimeError(f"prototype parity failed: {row}")
            rows.append(row)
    return rows


@torch.no_grad()
def benchmark(d, repeats):
    a, at, factors, q, probes, _ = problem(d, 8, torch.float32)
    functions = {"production": lambda: reference_step(a, at, factors, q, probes, .01, .7),
                 "specialized_qr_core": lambda: zero_step(a, at, factors, q, probes, .01, .7),
                 "specialized_direct_core": lambda: zero_step(a, at, factors, q, probes, .01, .7, qr_core=False)}
    timings = {name: [] for name in functions}
    for function in functions.values():
        function()
    for iteration in range(repeats):
        names = list(functions)
        if iteration % 2:
            names.reverse()
        for name in names:
            start = time.perf_counter()
            functions[name]()
            timings[name].append((time.perf_counter() - start) * 1000)
    return {name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in timings.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--dimension", type=int, default=62500)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/dlr_zero_algebra_probe_2026_09_28.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    payload = {"device": "cpu", "threads": args.threads, "dimension": args.dimension,
               "rank": 8, "trace_probes": 4, "enrichment": 0, "steps": 100,
               "caveat": "Synthetic linear velocity and CPU timings; not model or GPU speed estimates.",
               "validation": validate(), "timing": benchmark(args.dimension, args.repeats)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

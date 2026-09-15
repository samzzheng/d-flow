"""Timing comparison for the DLRA toy matrix ODE.

This benchmark compares two ways to integrate

    A'(t) = L A(t) + A(t) R,    t in [0, 1],

with rank-r initial data.  The dense solver advances the full n x n matrix.
The DLRA solver advances only U, S, V and uses the linear structure of the
right-hand side to compute the required skinny products without forming A(t).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def random_generator(n: int, rng: np.random.Generator) -> np.ndarray:
    m_skew = rng.standard_normal((n, n))
    skew = 1.5 * (m_skew - m_skew.T) / np.sqrt(n)
    m_sym = rng.standard_normal((n, n))
    sym = 0.3 * (m_sym + m_sym.T) / np.sqrt(n)
    return skew + sym


def make_low_rank_a0(n: int, r: int, rng: np.random.Generator):
    q1, _ = np.linalg.qr(rng.standard_normal((n, r)))
    q2, _ = np.linalg.qr(rng.standard_normal((n, r)))
    s = np.diag(0.5 ** np.arange(r))
    return q1, s, q2, (q1 @ s) @ q2.T


def dense_rhs(a: np.ndarray, l_mat: np.ndarray, r_mat: np.ndarray) -> np.ndarray:
    return l_mat @ a + a @ r_mat


def dense_rk4(a0: np.ndarray, l_mat: np.ndarray, r_mat: np.ndarray, n_steps: int) -> np.ndarray:
    h = 1.0 / n_steps
    a = a0.copy()
    for _ in range(n_steps):
        k1 = dense_rhs(a, l_mat, r_mat)
        k2 = dense_rhs(a + 0.5 * h * k1, l_mat, r_mat)
        k3 = dense_rhs(a + 0.5 * h * k2, l_mat, r_mat)
        k4 = dense_rhs(a + h * k3, l_mat, r_mat)
        a = a + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return a


def reorthonormalize(u: np.ndarray, s: np.ndarray, v: np.ndarray):
    qu, ru = np.linalg.qr(u)
    qv, rv = np.linalg.qr(v)
    return qu, ru @ s @ rv.T, qv


def factor_derivative(
    u: np.ndarray, s: np.ndarray, v: np.ndarray, l_mat: np.ndarray, r_mat: np.ndarray
):
    """Projected factor RHS using skinny products for Z = LUSV^T + USV^TR."""
    lu = l_mat @ u
    rv = r_mat.T @ v
    ul = u.T @ lu
    vr = v.T @ r_mat @ v

    # Z V and Z^T U, using U^T U = V^T V = I after each accepted step.
    z_v = lu @ s + u @ (s @ vr)
    z_t_u = v @ (s.T @ ul.T) + rv @ s.T

    s_dot = ul @ s + s @ vr
    s_inv = np.linalg.inv(s)
    u_dot = (z_v - u @ (u.T @ z_v)) @ s_inv
    v_dot = (z_t_u - v @ (v.T @ z_t_u)) @ s_inv.T
    return u_dot, s_dot, v_dot


def axpy(a: float, x, y):
    return y[0] + a * x[0], y[1] + a * x[1], y[2] + a * x[2]


def dlra_rk4(
    u0: np.ndarray, s0: np.ndarray, v0: np.ndarray, l_mat: np.ndarray, r_mat: np.ndarray, n_steps: int
):
    h = 1.0 / n_steps
    factors = (u0.copy(), s0.copy(), v0.copy())
    for _ in range(n_steps):
        k1 = factor_derivative(*factors, l_mat=l_mat, r_mat=r_mat)
        k2 = factor_derivative(*axpy(0.5 * h, k1, factors), l_mat=l_mat, r_mat=r_mat)
        k3 = factor_derivative(*axpy(0.5 * h, k2, factors), l_mat=l_mat, r_mat=r_mat)
        k4 = factor_derivative(*axpy(h, k3, factors), l_mat=l_mat, r_mat=r_mat)
        u = factors[0] + (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        s = factors[1] + (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        v = factors[2] + (h / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        factors = reorthonormalize(u, s, v)
    return factors


def time_call(fn, repeats: int):
    best = float("inf")
    value = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - t0)
    return best, value


def run_benchmark(sizes: list[int], rank: int, n_steps: int, repeats: int, seed: int):
    rows = []
    for n in sizes:
        rng = np.random.default_rng(seed + n)
        l_mat = random_generator(n, rng)
        r_mat = random_generator(n, rng)
        u0, s0, v0, a0 = make_low_rank_a0(n, rank, rng)

        dense_time, dense_final = time_call(
            lambda: dense_rk4(a0, l_mat, r_mat, n_steps), repeats
        )
        dlra_time, factors = time_call(
            lambda: dlra_rk4(u0, s0, v0, l_mat, r_mat, n_steps), repeats
        )
        y_final = factors[0] @ factors[1] @ factors[2].T
        rel_error = np.linalg.norm(y_final - dense_final) / np.linalg.norm(dense_final)
        rows.append(
            {
                "n": n,
                "rank": rank,
                "n_steps": n_steps,
                "dense_time_s": dense_time,
                "dlra_time_s": dlra_time,
                "speedup": dense_time / dlra_time,
                "endpoint_rel_error": rel_error,
            }
        )
        print(
            f"n={n:4d} dense={dense_time:.4f}s dlra={dlra_time:.4f}s "
            f"speedup={dense_time / dlra_time:.1f}x rel_err={rel_error:.2e}",
            flush=True,
        )
    return rows


def save_outputs(rows: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dlra_timing.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = np.array([row["n"] for row in rows])
    dense = np.array([row["dense_time_s"] for row in rows])
    dlra = np.array([row["dlra_time_s"] for row in rows])
    speedup = np.array([row["speedup"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].loglog(n, dense, "o-", label="dense RK4 on A(t)")
    axes[0].loglog(n, dlra, "o-", label="DLRA RK4 on U,S,V")
    axes[0].set_xlabel("matrix size n for A(t) in R^{n x n}")
    axes[0].set_ylabel("time to integrate to t=1 (s)")
    axes[0].set_title("Toy ODE runtime")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].semilogx(n, speedup, "o-", color="C2")
    axes[1].set_xlabel("matrix size n")
    axes[1].set_ylabel("dense time / DLRA time")
    axes[1].set_title("DLRA speedup")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig_path = out_dir / "dlra_timing.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"saved {csv_path}")
    print(f"saved {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dlra/outputs")
    args = parser.parse_args()

    rows = run_benchmark(args.sizes, args.rank, args.n_steps, args.repeats, args.seed)
    save_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()

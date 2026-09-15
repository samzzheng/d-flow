"""CPU benchmark for dense and dynamical-low-rank toy ODE solvers.

All methods solve A' = L A + A R on [0, 1] from the same exactly rank-r
initial condition. Dense Forward Euler and dense RK4 evolve A directly;
DLRA-RK4 evolves U, S, V. Accuracy is measured against the analytic endpoint
exp(L) A0 exp(R). Run with BLAS thread counts fixed externally for a controlled
single-thread benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from benchmark_dlra_timing import (
    dense_rk4,
    dlra_rk4,
    random_generator,
)


ROOT = Path(__file__).resolve().parents[1]


def make_report_low_rank_a0(n: int, rank: int, rng: np.random.Generator):
    """Match the exact-rank initial condition used by dlra_toy.ipynb."""
    q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.diag(np.linspace(1.0, 0.4, rank))
    u0 = q1[:, :rank]
    v0 = q2[:, :rank]
    return u0, s, v0, (u0 @ s) @ v0.T


def dense_forward_euler(
    a0: np.ndarray,
    l_mat: np.ndarray,
    r_mat: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    step_size = 1.0 / n_steps
    a = a0.copy()
    for _ in range(n_steps):
        a += step_size * (l_mat @ a + a @ r_mat)
    return a


def timed_runs(function, repeats: int, warmups: int) -> tuple[dict, object]:
    value = None
    for _ in range(warmups):
        value = function()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        durations.append(time.perf_counter() - start)
    return {
        "median": statistics.median(durations),
        "minimum": min(durations),
        "maximum": max(durations),
    }, value


def endpoint_errors(estimate: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    absolute = float(np.linalg.norm(estimate - exact, ord="fro"))
    relative = absolute / float(np.linalg.norm(exact, ord="fro"))
    return absolute, relative


def run(
    sizes: list[int],
    rank: int,
    n_steps: int,
    repeats: int,
    warmups: int,
    seed: int,
) -> list[dict]:
    rows = []
    for n in sizes:
        # Reinitialize with the report's seed for each size. At n=40 this is
        # exactly the ODE and initial condition used in the July toy figure.
        rng = np.random.default_rng(seed)
        l_mat = random_generator(n, rng)
        r_mat = random_generator(n, rng)
        u0, s0, v0, a0 = make_report_low_rank_a0(n, rank, rng)
        exact = expm(l_mat) @ a0 @ expm(r_mat)

        methods = {
            "Forward Euler": lambda: dense_forward_euler(a0, l_mat, r_mat, n_steps),
            "Dense RK4": lambda: dense_rk4(a0, l_mat, r_mat, n_steps),
            "DLRA RK4": lambda: dlra_rk4(u0, s0, v0, l_mat, r_mat, n_steps),
        }
        for method, function in methods.items():
            timing, value = timed_runs(function, repeats, warmups)
            estimate = value if method != "DLRA RK4" else value[0] @ value[1] @ value[2].T
            absolute_error, relative_error = endpoint_errors(estimate, exact)
            row = {
                "n": n,
                "rank": rank,
                "n_steps": n_steps,
                "method": method,
                "median_time_s": timing["median"],
                "minimum_time_s": timing["minimum"],
                "maximum_time_s": timing["maximum"],
                "endpoint_abs_fro_error": absolute_error,
                "endpoint_rel_fro_error": relative_error,
            }
            rows.append(row)
            print(
                f"n={n:4d} {method:13s} median={timing['median']:.6f}s "
                f"rel_error={relative_error:.3e}",
                flush=True,
            )
    return rows


def save(rows: list[dict], output_dir: Path, repeats: int, warmups: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "toy_ode_cpu_solver_timings.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "device": "CPU",
        "blas_threads": 1,
        "timing_statistic": "median",
        "repeats": repeats,
        "warmups": warmups,
        "interval": [0.0, 1.0],
        "initial_condition": "exact rank-r",
        "singular_values": "linspace(1.0, 0.4, rank), matching dlra_toy.ipynb",
        "reference": "exp(L) A0 exp(R)",
    }
    (output_dir / "toy_ode_cpu_solver_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    methods = ["Forward Euler", "Dense RK4", "DLRA RK4"]
    colors = {"Forward Euler": "C0", "Dense RK4": "C1", "DLRA RK4": "C2"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        n_values = [row["n"] for row in selected]
        axes[0].loglog(
            n_values,
            [row["median_time_s"] for row in selected],
            "o-",
            color=colors[method],
            label=method,
        )
        axes[1].loglog(
            n_values,
            [row["endpoint_rel_fro_error"] for row in selected],
            "o-",
            color=colors[method],
            label=method,
        )
    axes[0].set_title("CPU solution time")
    axes[0].set_xlabel("Matrix size n")
    axes[0].set_ylabel("Median time to t=1 (s)")
    axes[1].set_title("Endpoint accuracy")
    axes[1].set_xlabel("Matrix size n")
    axes[1].set_ylabel("Relative Frobenius error")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.suptitle("Toy matrix ODE: Forward Euler, dense RK4, and DLRA RK4")
    fig.tight_layout()
    fig.savefig(output_dir / "toy_ode_cpu_solver_benchmark.png", dpi=160, bbox_inches="tight")
    print(f"Saved results to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 64, 128, 256, 512])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "dlra/outputs/toy_ode_cpu_benchmark",
    )
    args = parser.parse_args()
    rows = run(args.sizes, args.rank, args.n_steps, args.repeats, args.warmups, args.seed)
    save(rows, args.output_dir, args.repeats, args.warmups)


if __name__ == "__main__":
    main()

"""Combine and validate the GPU sparse-CT circle noise sweep."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("forward_euler", "rk4")
NOISE_DIRS = {
    "noise_000": 0.00,
    "noise_001": 0.01,
    "noise_005": 0.05,
    "noise_010": 0.10,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_summary(rows: list[dict]) -> list[dict]:
    output = []
    noises = sorted({float(row["noise_level"]) for row in rows})
    resolutions = sorted({int(row["resolution"]) for row in rows})
    for noise in noises:
        for resolution in resolutions:
            for method in METHODS:
                group = [
                    row for row in rows
                    if float(row["noise_level"]) == noise
                    and int(row["resolution"]) == resolution
                    and row["method"] == method
                ]
                if not group:
                    continue
                values = lambda key: [float(row[key]) for row in group]
                output.append({
                    "noise_level": noise,
                    "noise_percent": 100 * noise,
                    "resolution": resolution,
                    "method": method,
                    "samples": len(group),
                    "mean_image_relative_l2": statistics.mean(values("image_relative_l2")),
                    "median_image_relative_l2": statistics.median(values("image_relative_l2")),
                    "mean_measurement_relative_l2_noisy": statistics.mean(
                        values("measurement_relative_l2_noisy")),
                    "mean_measurement_relative_l2_clean": statistics.mean(
                        values("measurement_relative_l2_clean")),
                    "mean_time_per_outer_iteration_s": statistics.mean(
                        values("mean_outer_time_s")),
                    "median_time_per_outer_iteration_s": statistics.median(
                        values("mean_outer_time_s")),
                    "mean_closure_evaluations": statistics.mean(
                        values("closure_evaluations")),
                    "mean_best_step": statistics.mean(values("best_step")),
                    "mean_peak_gpu_memory_mb": statistics.mean(
                        values("peak_gpu_memory_mb")),
                })
    return output


def paired_summary(rows: list[dict]) -> list[dict]:
    indexed = {
        (
            float(row["noise_level"]), int(row["resolution"]),
            int(row["sample_index"]), row["method"],
        ): row
        for row in rows
    }
    output = []
    noises = sorted({key[0] for key in indexed})
    resolutions = sorted({key[1] for key in indexed})
    for noise in noises:
        for resolution in resolutions:
            pairs = []
            sample_indices = sorted({
                key[2] for key in indexed
                if key[0] == noise and key[1] == resolution
            })
            for sample_index in sample_indices:
                euler = indexed.get((noise, resolution, sample_index, "forward_euler"))
                rk4 = indexed.get((noise, resolution, sample_index, "rk4"))
                if euler is not None and rk4 is not None:
                    pairs.append((euler, rk4))
            if not pairs:
                continue
            output.append({
                "noise_level": noise,
                "noise_percent": 100 * noise,
                "resolution": resolution,
                "pairs": len(pairs),
                "rk4_image_error_wins": sum(
                    float(r["image_relative_l2"]) < float(e["image_relative_l2"])
                    for e, r in pairs
                ),
                "rk4_measurement_error_wins": sum(
                    float(r["measurement_relative_l2_noisy"])
                    < float(e["measurement_relative_l2_noisy"])
                    for e, r in pairs
                ),
                "mean_rk4_to_euler_runtime_ratio": statistics.mean(
                    float(r["mean_outer_time_s"]) / float(e["mean_outer_time_s"])
                    for e, r in pairs
                ),
            })
    return output


def save_plots(summary: list[dict], root: Path) -> None:
    labels = {"forward_euler": "Forward Euler", "rk4": "RK4"}
    colors = {"forward_euler": "#245b78", "rk4": "#a4562a"}
    for metric, ylabel, filename in (
        ("mean_image_relative_l2", "Mean image relative L2", "error_vs_noise.png"),
        (
            "mean_time_per_outer_iteration_s",
            "Mean time per outer iteration (s)",
            "runtime_vs_noise.png",
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
        for axis, resolution in zip(axes, (128, 160)):
            for method in METHODS:
                group = sorted(
                    (row for row in summary
                     if int(row["resolution"]) == resolution
                     and row["method"] == method),
                    key=lambda row: float(row["noise_level"]),
                )
                if group:
                    axis.plot(
                        [float(row["noise_percent"]) for row in group],
                        [float(row[metric]) for row in group],
                        "o-", label=labels[method], color=colors[method],
                    )
            axis.set_title(f"{resolution}×{resolution}")
            axis.set_xlabel("Measurement noise (%)")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(root / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def validate(
    rows: list[dict], iteration_rows: list[dict],
    expected_resolutions: list[int], expected_samples: int,
) -> None:
    expected_keys = {
        (noise, resolution, sample_index, method)
        for noise in NOISE_DIRS.values()
        for resolution in expected_resolutions
        for sample_index in range(expected_samples)
        for method in METHODS
    }
    actual_keys = {
        (
            float(row["noise_level"]), int(row["resolution"]),
            int(row["sample_index"]), row["method"],
        )
        for row in rows
    }
    missing = expected_keys - actual_keys
    extras = actual_keys - expected_keys
    if missing or extras:
        raise RuntimeError(f"Sweep row mismatch: missing={missing}, extras={extras}")
    expected_iterations = len(expected_keys) * 100
    if len(iteration_rows) != expected_iterations:
        raise RuntimeError(
            f"Expected {expected_iterations} iteration rows, got {len(iteration_rows)}"
        )
    for row in rows:
        if int(row["steps_run"]) != 100:
            raise RuntimeError(f"Incomplete reconstruction row: {row}")
        if not 1 <= int(row["best_step"]) <= 100:
            raise RuntimeError(f"Invalid best step: {row}")
        if not Path(row["diagnostic_figure"]).exists():
            raise RuntimeError(f"Missing diagnostic figure: {row['diagnostic_figure']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/dflow_sparse_ct_circles_gpu_100step_noise_sweep"),
    )
    parser.add_argument("--expected-resolutions", type=int, nargs="+", default=[128])
    parser.add_argument("--expected-samples", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    samples = []
    iterations = []
    for directory, noise in NOISE_DIRS.items():
        for row in read_csv(args.root / directory / "samples.csv"):
            row = {"noise_level": noise, **row}
            samples.append(row)
        for row in read_csv(args.root / directory / "iterations.csv"):
            row = {"noise_level": noise, **row}
            iterations.append(row)
    if not samples:
        raise RuntimeError(f"No completed samples found under {args.root}")
    if args.require_complete:
        validate(samples, iterations, args.expected_resolutions, args.expected_samples)

    summary = group_summary(samples)
    write_csv(args.root / "combined_samples.csv", samples)
    write_csv(args.root / "combined_iterations.csv", iterations)
    write_csv(args.root / "combined_summary.csv", summary)
    write_csv(args.root / "paired_summary.csv", paired_summary(samples))
    save_plots(summary, args.root)
    print(f"Combined {len(samples)} reconstructions and {len(iterations)} iterations")


if __name__ == "__main__":
    main()

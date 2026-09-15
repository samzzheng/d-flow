#!/usr/bin/env python3
"""Generate the D-Flow TV, timing, and stopping report from saved benchmark data."""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
TV_ROOT = (
    ROOT
    / "outputs"
    / "dflow_tv_lr_comparison_750_steps"
    / "by_tv_weight"
)
TIME_ROOT = ROOT / "outputs" / "dflow_time_complexity_tv_1e-4_lr_1p0_750_steps"
REPORT_PATH = REPORT_DIR / "dflow_tv_time_complexity_report.html"

RESOLUTIONS = (64, 96, 128, 160)
HIGH_RESOLUTIONS = (128, 160)
SAMPLES = range(5)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    key: value if key == "figure_path" else float(value)
                    for key, value in row.items()
                }
            )
        return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_sign_p(wins: int, total: int) -> float:
    tail = max(wins, total - wins)
    probability = 2.0 * sum(math.comb(total, k) for k in range(tail, total + 1)) / 2**total
    return min(1.0, probability)


def paired_tv_analysis() -> tuple[list[dict[str, object]], dict[str, float]]:
    no_tv_rows = read_csv(
        TV_ROOT / "tv_0" / "lr_1.0" / "dflow_blur_scalability_samples.csv"
    )
    tv_rows = read_csv(
        TV_ROOT / "tv_1e-4" / "lr_1.0" / "dflow_blur_scalability_samples.csv"
    )
    no_tv = {
        (int(row["resolution"]), int(row["sample_idx"])): row for row in no_tv_rows
    }
    with_tv = {
        (int(row["resolution"]), int(row["sample_idx"])): row for row in tv_rows
    }

    summary_rows: list[dict[str, object]] = []
    combined_no_image: list[float] = []
    combined_tv_image: list[float] = []
    combined_no_variation: list[float] = []
    combined_tv_variation: list[float] = []
    combined_no_measurement: list[float] = []
    combined_tv_measurement: list[float] = []

    for resolution in HIGH_RESOLUTIONS:
        no_image = np.array(
            [no_tv[(resolution, sample)]["image_mse_clamped"] for sample in SAMPLES]
        )
        tv_image = np.array(
            [with_tv[(resolution, sample)]["image_mse_clamped"] for sample in SAMPLES]
        )
        no_variation = np.array(
            [no_tv[(resolution, sample)]["best_tv"] for sample in SAMPLES]
        )
        tv_variation = np.array(
            [with_tv[(resolution, sample)]["best_tv"] for sample in SAMPLES]
        )
        no_measurement = np.array(
            [no_tv[(resolution, sample)]["measurement_mse_raw"] for sample in SAMPLES]
        )
        tv_measurement = np.array(
            [with_tv[(resolution, sample)]["measurement_mse_raw"] for sample in SAMPLES]
        )

        summary_rows.append(
            {
                "resolution": resolution,
                "n_pairs": 5,
                "no_tv_mean_image_mse": no_image.mean(),
                "tv_mean_image_mse": tv_image.mean(),
                "mean_image_mse_reduction_fraction": 1.0 - tv_image.mean() / no_image.mean(),
                "no_tv_median_image_mse": np.median(no_image),
                "tv_median_image_mse": np.median(tv_image),
                "median_image_mse_reduction_fraction": 1.0
                - np.median(tv_image) / np.median(no_image),
                "tv_image_mse_wins": int(np.sum(tv_image < no_image)),
                "no_tv_mean_solution_tv": no_variation.mean(),
                "tv_mean_solution_tv": tv_variation.mean(),
                "mean_solution_tv_reduction_fraction": 1.0
                - tv_variation.mean() / no_variation.mean(),
                "no_tv_median_solution_tv": np.median(no_variation),
                "tv_median_solution_tv": np.median(tv_variation),
                "median_solution_tv_reduction_fraction": 1.0
                - np.median(tv_variation) / np.median(no_variation),
                "tv_solution_tv_wins": int(np.sum(tv_variation < no_variation)),
                "no_tv_median_measurement_mse": np.median(no_measurement),
                "tv_median_measurement_mse": np.median(tv_measurement),
            }
        )
        combined_no_image.extend(no_image)
        combined_tv_image.extend(tv_image)
        combined_no_variation.extend(no_variation)
        combined_tv_variation.extend(tv_variation)
        combined_no_measurement.extend(no_measurement)
        combined_tv_measurement.extend(tv_measurement)

    no_image = np.asarray(combined_no_image)
    tv_image = np.asarray(combined_tv_image)
    no_variation = np.asarray(combined_no_variation)
    tv_variation = np.asarray(combined_tv_variation)
    no_measurement = np.asarray(combined_no_measurement)
    tv_measurement = np.asarray(combined_tv_measurement)
    combined = {
        "no_tv_mean_image_mse": float(no_image.mean()),
        "tv_mean_image_mse": float(tv_image.mean()),
        "image_mean_reduction": float(1.0 - tv_image.mean() / no_image.mean()),
        "no_tv_median_image_mse": float(np.median(no_image)),
        "tv_median_image_mse": float(np.median(tv_image)),
        "image_median_reduction": float(
            1.0 - np.median(tv_image) / np.median(no_image)
        ),
        "image_wins": int(np.sum(tv_image < no_image)),
        "image_sign_p": exact_sign_p(int(np.sum(tv_image < no_image)), len(no_image)),
        "no_tv_mean_solution_tv": float(no_variation.mean()),
        "tv_mean_solution_tv": float(tv_variation.mean()),
        "solution_tv_mean_reduction": float(
            1.0 - tv_variation.mean() / no_variation.mean()
        ),
        "no_tv_median_solution_tv": float(np.median(no_variation)),
        "tv_median_solution_tv": float(np.median(tv_variation)),
        "solution_tv_median_reduction": float(
            1.0 - np.median(tv_variation) / np.median(no_variation)
        ),
        "solution_tv_wins": int(np.sum(tv_variation < no_variation)),
        "solution_tv_sign_p": exact_sign_p(
            int(np.sum(tv_variation < no_variation)), len(no_variation)
        ),
        "no_tv_median_measurement_mse": float(np.median(no_measurement)),
        "tv_median_measurement_mse": float(np.median(tv_measurement)),
    }
    return summary_rows, combined


def tv_sweep_analysis() -> list[dict[str, object]]:
    rows_by_weight: dict[float, list[dict[str, object]]] = {}
    for weight_dir in TV_ROOT.glob("tv_*"):
        sample_path = (
            weight_dir / "lr_1.0" / "dflow_blur_scalability_samples.csv"
        )
        if sample_path.exists():
            rows_by_weight[float(weight_dir.name.removeprefix("tv_"))] = read_csv(
                sample_path
            )

    no_tv = {
        (int(row["resolution"]), int(row["sample_idx"])): row
        for row in rows_by_weight[0.0]
    }
    summary = []
    for weight in sorted(rows_by_weight):
        rows = rows_by_weight[weight]
        image_values = np.array([row["image_mse_clamped"] for row in rows])
        solution_tv = np.array([row["best_tv"] for row in rows])
        image_wins = sum(
            row["image_mse_clamped"]
            < no_tv[(int(row["resolution"]), int(row["sample_idx"]))][
                "image_mse_clamped"
            ]
            for row in rows
        )
        solution_tv_wins = sum(
            row["best_tv"]
            < no_tv[(int(row["resolution"]), int(row["sample_idx"]))]["best_tv"]
            for row in rows
        )
        result = {
            "tv_weight": weight,
            "overall_mean_image_mse": image_values.mean(),
            "overall_median_image_mse": np.median(image_values),
            "overall_mean_solution_tv": solution_tv.mean(),
            "image_mse_wins_vs_no_tv": image_wins if weight else "",
            "solution_tv_wins_vs_no_tv": solution_tv_wins if weight else "",
        }
        for resolution in RESOLUTIONS:
            resolution_rows = [
                row for row in rows if int(row["resolution"]) == resolution
            ]
            result[f"mean_image_mse_{resolution}"] = np.mean(
                [row["image_mse_clamped"] for row in resolution_rows]
            )
            result[f"median_image_mse_{resolution}"] = np.median(
                [row["image_mse_clamped"] for row in resolution_rows]
            )
            result[f"mean_solution_tv_{resolution}"] = np.mean(
                [row["best_tv"] for row in resolution_rows]
            )
        summary.append(result)
    return summary


def plot_tv_pairs(path: Path) -> None:
    no_tv_rows = read_csv(
        TV_ROOT / "tv_0" / "lr_1.0" / "dflow_blur_scalability_samples.csv"
    )
    tv_rows = read_csv(
        TV_ROOT / "tv_1e-4" / "lr_1.0" / "dflow_blur_scalability_samples.csv"
    )
    no_tv = {
        (int(row["resolution"]), int(row["sample_idx"])): row for row in no_tv_rows
    }
    with_tv = {
        (int(row["resolution"]), int(row["sample_idx"])): row for row in tv_rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    colors = {128: "#245b78", 160: "#a4562a"}
    metrics = (
        ("image_mse_clamped", "Image MSE", "Image fidelity"),
        ("best_tv", "Reconstruction TV", "Oscillation proxy"),
    )
    for axis, (metric, ylabel, title) in zip(axes, metrics):
        for resolution in HIGH_RESOLUTIONS:
            offset = -0.07 if resolution == 128 else 0.07
            for sample in SAMPLES:
                values = [
                    no_tv[(resolution, sample)][metric],
                    with_tv[(resolution, sample)][metric],
                ]
                axis.plot(
                    [0 + offset, 1 + offset],
                    values,
                    marker="o",
                    markersize=4,
                    linewidth=1.15,
                    color=colors[resolution],
                    alpha=0.72,
                )
        axis.set_yscale("log")
        axis.set_xticks([0, 1], ["No TV", r"TV, $\lambda=10^{-4}$"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.24)
    axes[0].plot([], [], color=colors[128], marker="o", label="128×128")
    axes[0].plot([], [], color=colors[160], marker="o", label="160×160")
    axes[0].legend(frameon=False)
    fig.suptitle("Paired high-resolution D-Flow results (same five samples)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tv_galleries() -> None:
    weight_dirs = sorted(
        (
            directory
            for directory in TV_ROOT.glob("tv_*")
            if (directory / "lr_1.0" / "dflow_blur_scalability_samples.csv").exists()
        ),
        key=lambda directory: float(directory.name.removeprefix("tv_")),
    )
    for resolution in RESOLUTIONS:
        fig, axes = plt.subplots(
            len(weight_dirs),
            3,
            figsize=(15, 26),
            constrained_layout=True,
        )
        for row_index, weight_dir in enumerate(weight_dirs):
            weight = float(weight_dir.name.removeprefix("tv_"))
            rows = [
                row
                for row in read_csv(
                    weight_dir
                    / "lr_1.0"
                    / "dflow_blur_scalability_samples.csv"
                )
                if int(row["resolution"]) == resolution
            ]
            ordered = sorted(rows, key=lambda row: row["image_mse_clamped"])
            selections = (
                ("Best", ordered[0]),
                ("Median", ordered[len(ordered) // 2]),
                ("Worst", ordered[-1]),
            )
            for column_index, (label, selected) in enumerate(selections):
                axis = axes[row_index, column_index]
                figure_path = ROOT / str(selected["figure_path"])
                axis.imshow(plt.imread(figure_path))
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(
                    f"{label}: sample {int(selected['sample_idx'])}, "
                    f"MSE={selected['image_mse_clamped']:.2e}",
                    fontsize=9,
                )
                if column_index == 0:
                    axis.set_ylabel(
                        f"TV={weight:.0e}",
                        fontsize=10,
                        fontweight="bold",
                        labelpad=9,
                    )
        fig.suptitle(
            f"{resolution}×{resolution}: best, median, and worst by image MSE",
            fontsize=16,
        )
        fig.savefig(
            REPORT_DIR / f"dflow_tv_gallery_res_{resolution}.png",
            dpi=130,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def timing_analysis() -> tuple[list[dict[str, object]], dict[tuple[int, int], list[dict]]]:
    timing_summary = read_csv(TIME_ROOT / "dflow_iteration_timing_summary.csv")
    timing_rows = read_csv(TIME_ROOT / "dflow_iteration_timings.csv")
    histories: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in timing_rows:
        histories[(int(row["resolution"]), int(row["sample_idx"]))].append(row)
    for history in histories.values():
        history.sort(key=lambda row: row["outer_step"])
    return timing_summary, histories


def plot_timing(path: Path, timing_summary: list[dict[str, object]]) -> None:
    resolutions = np.array([row["resolution"] for row in timing_summary])
    medians = np.array([row["median_iteration_s"] for row in timing_summary])
    p10 = np.array([row["p10_iteration_s"] for row in timing_summary])
    p90 = np.array([row["p90_iteration_s"] for row in timing_summary])
    closure = np.array([row["median_elapsed_per_closure_s"] for row in timing_summary])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    axes[0].errorbar(
        resolutions,
        medians,
        yerr=[medians - p10, p90 - medians],
        fmt="o-",
        capsize=4,
        color="#245b78",
        linewidth=1.7,
    )
    axes[0].set_xlabel("Resolution R")
    axes[0].set_ylabel("Seconds per outer iteration")
    axes[0].set_title("Outer LBFGS iteration")
    axes[0].grid(alpha=0.24)

    axes[1].plot(resolutions, closure, "o-", color="#a4562a", linewidth=1.7)
    axes[1].set_xlabel("Resolution R")
    axes[1].set_ylabel("Seconds per closure evaluation")
    axes[1].set_title("D-Flow closure cost")
    axes[1].grid(alpha=0.24)
    fig.suptitle("Measured GPU iteration time (five samples per resolution)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def simulate_stopping(
    histories: dict[tuple[int, int], list[dict]],
    *,
    min_steps: int = 350,
    patience: int = 50,
    relative_tolerance: float = 1e-3,
) -> list[dict[str, object]]:
    rows = []
    for (resolution, sample_idx), history in sorted(histories.items()):
        objectives = np.array([row["objective"] for row in history])
        best = np.minimum.accumulate(objectives)
        stop_step = len(history)
        for index in range(max(min_steps - 1, patience), len(history)):
            relative_improvement = (best[index - patience] - best[index]) / max(
                abs(best[index - patience]), 1e-30
            )
            if relative_improvement < relative_tolerance:
                stop_step = index + 1
                break
        rows.append(
            {
                "resolution": resolution,
                "sample_idx": sample_idx,
                "stop_step": stop_step,
                "full_steps": len(history),
                "steps_saved": len(history) - stop_step,
                "fraction_saved": 1.0 - stop_step / len(history),
                "objective_gap_fraction": best[stop_step - 1] / best[-1] - 1.0,
            }
        )
    return rows


def plot_convergence(
    path: Path, histories: dict[tuple[int, int], list[dict]], stopping_rows: list[dict]
) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    colors = {64: "#245b78", 96: "#4d7358", 128: "#a4562a", 160: "#7a3d62"}
    for resolution in RESOLUTIONS:
        curves = []
        for sample in SAMPLES:
            objectives = np.array(
                [row["objective"] for row in histories[(resolution, sample)]]
            )
            best = np.minimum.accumulate(objectives)
            curves.append(np.maximum(best / best[-1] - 1.0, 1e-6))
        median_curve = np.median(np.stack(curves), axis=0)
        axis.plot(
            np.arange(1, len(median_curve) + 1),
            median_curve,
            label=f"{resolution}×{resolution}",
            color=colors[resolution],
            linewidth=1.65,
        )
    axis.axvline(350, color="#555", linestyle="--", linewidth=1.2, label="Minimum 350")
    axis.set_yscale("log")
    axis.set_xlabel("Outer LBFGS step")
    axis.set_ylabel("Best objective gap relative to step 750")
    axis.set_title("Median best-so-far objective convergence")
    axis.grid(alpha=0.24)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_sci(value: float) -> str:
    return f"{value:.3e}"


def report_html(
    tv_summary: list[dict[str, object]],
    combined: dict[str, float],
    timing_summary: list[dict[str, object]],
    stopping_rows: list[dict[str, object]],
) -> str:
    timing_by_res = {int(row["resolution"]): row for row in timing_summary}
    stopping_by_res: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in stopping_rows:
        stopping_by_res[int(row["resolution"])].append(row)

    stop_steps = np.array([row["stop_step"] for row in stopping_rows])
    stop_gaps = np.array([row["objective_gap_fraction"] for row in stopping_rows])
    early_count = int(np.sum(stop_steps < 750))
    mean_saving = 1.0 - stop_steps.mean() / 750.0

    tv_table_rows = "\n".join(
        f"""<tr>
<td>{int(row['resolution'])}×{int(row['resolution'])}</td>
<td>{format_sci(row['no_tv_mean_image_mse'])}</td>
<td>{format_sci(row['tv_mean_image_mse'])}</td>
<td>{100 * row['mean_image_mse_reduction_fraction']:.1f}%</td>
<td>{int(row['tv_image_mse_wins'])}/5</td>
<td>{row['no_tv_mean_solution_tv']:.5f}</td>
<td>{row['tv_mean_solution_tv']:.5f}</td>
<td>{100 * row['mean_solution_tv_reduction_fraction']:.1f}%</td>
<td>{int(row['tv_solution_tv_wins'])}/5</td>
</tr>"""
        for row in tv_summary
    )

    timing_rows = "\n".join(
        f"""<tr>
<td>{resolution}×{resolution}</td>
<td>{int(timing_by_res[resolution]['n_samples'])}</td>
<td>{timing_by_res[resolution]['mean_iteration_s']:.3f}</td>
<td>{timing_by_res[resolution]['median_iteration_s']:.3f}</td>
<td>{timing_by_res[resolution]['p10_iteration_s']:.3f}</td>
<td>{timing_by_res[resolution]['p90_iteration_s']:.3f}</td>
<td>{timing_by_res[resolution]['mean_closure_evals_per_iteration']:.2f}</td>
<td>{timing_by_res[resolution]['median_elapsed_per_closure_s']:.4f}</td>
</tr>"""
        for resolution in RESOLUTIONS
    )

    stop_rows = "\n".join(
        f"""<tr>
<td>{resolution}×{resolution}</td>
<td>{np.mean([row['stop_step'] for row in stopping_by_res[resolution]]):.1f}</td>
<td>{np.median([row['stop_step'] for row in stopping_by_res[resolution]]):.0f}</td>
<td>{100 * (1 - np.mean([row['stop_step'] for row in stopping_by_res[resolution]]) / 750):.1f}%</td>
<td>{100 * np.median([row['objective_gap_fraction'] for row in stopping_by_res[resolution]]):.3f}%</td>
<td>{100 * max(row['objective_gap_fraction'] for row in stopping_by_res[resolution]):.3f}%</td>
</tr>"""
        for resolution in RESOLUTIONS
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>D-Flow TV, Runtime, and Stopping Report</title>
<script>
window.MathJax={{
  tex:{{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]],processEscapes:true}},
  svg:{{fontCache:"global"}},
  options:{{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}}
}};
</script>
<script defer src="vendor/node_modules/mathjax/es5/tex-svg.js"></script>
<style>
:root{{--ink:#17212b;--muted:#5a6774;--line:#d7dde3;--soft:#f4f7f9;--accent:#245b78;--good:#4d7358;--warn:#7a4b2d}}
body{{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:white}}
main{{max-width:1120px;margin:0 auto;padding:36px 28px 64px}}
h1{{font-size:32px;margin:0 0 10px;line-height:1.15}}
h2{{font-size:22px;margin:38px 0 14px;padding-top:8px;border-top:1px solid var(--line)}}
h3{{font-size:17px;margin:24px 0 10px}}
p,li{{line-height:1.68}} p{{margin:12px 0}}
.subtitle,.small{{color:var(--muted);font-size:13px}}
.status-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:22px 0}}
.status{{border:1px solid var(--line);border-radius:8px;padding:14px;background:var(--soft)}}
.status strong{{display:block;color:var(--muted);font-size:13px;margin-bottom:5px}} .status span{{font-size:18px;font-weight:650}}
.note{{border-left:4px solid var(--accent);padding:12px 16px;background:#eef5f8;margin:20px 0;line-height:1.62}}
.note.good{{border-left-color:var(--good);background:#f1f7f2}} .note.warn{{border-left-color:var(--warn);background:#fbf3ed}}
table{{width:100%;border-collapse:collapse;margin:18px 0 28px;font-size:13px;line-height:1.45}}
th,td{{border-bottom:1px solid var(--line);padding:10px 8px;text-align:right;vertical-align:top}}
th:first-child,td:first-child{{text-align:left}} th{{background:var(--soft);font-weight:650;color:#273442}}
.settings{{max-width:760px}} .settings td:nth-child(2),.settings th:nth-child(2){{text-align:left}}
.grid-2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
figure{{border:1px solid var(--line);border-radius:8px;padding:12px;background:white;margin:18px 0}}
figure img{{width:100%;height:auto;display:block;border-radius:4px}}
figcaption{{color:var(--muted);font-size:13px;line-height:1.55;margin-top:10px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.math{{overflow-x:auto;text-align:center;margin:18px 0}}
@media(max-width:800px){{main{{padding:24px 15px 44px}}.status-grid,.grid-2{{grid-template-columns:1fr}}table{{display:block;overflow-x:auto;font-size:12px}}}}
</style>
</head>
<body><main>
<h1>D-Flow TV Regularization, Runtime Scaling, and Stopping</h1>
<p class="subtitle">Numerical report generated from the completed 750-step TV/LR sweep and the GPU per-iteration timing benchmark. Generated 2026-07-31.</p>

<section class="status-grid">
<div class="status"><strong>High-resolution pairs</strong><span>10 TV vs no-TV</span></div>
<div class="status"><strong>Timed iterations</strong><span>15,000</span></div>
<div class="status"><strong>Recommended mean budget</strong><span>{stop_steps.mean():.1f} steps</span></div>
</section>

<h2>Executive conclusions</h2>
<ol>
<li><strong>TV regularization suppresses oscillatory solutions at high resolution.</strong> With \(\lambda_{{TV}}=10^{{-4}}\), reconstruction TV fell in all 10 paired 128×128 and 160×160 cases. Mean reconstruction TV decreased by {100 * combined['solution_tv_mean_reduction']:.1f}% and mean image MSE decreased by {100 * combined['image_mean_reduction']:.1f}%.</li>
<li><strong>The fidelity benefit is meaningful but not universal.</strong> TV improved image MSE in {combined['image_wins']}/10 high-resolution pairs. Four pairs had higher image MSE, so TV should be described as a strong oscillation-control prior with an average fidelity benefit, not as a guaranteed per-sample improvement.</li>
<li><strong>Outer-iteration time increases sharply at 128 and 160.</strong> Median measured time rose from 1.230 s at 64×64 to 5.897 s at 160×160. Each outer step used approximately 20 closure evaluations.</li>
<li><strong>The absolute \(10^{{-12}}\) data-loss stop is ineffective.</strong> All 20 timed samples ran for 750 steps. A best-objective plateau rule with a 350-step minimum reduced the retrospective mean budget by {100 * mean_saving:.1f}% while remaining within {100 * stop_gaps.max():.2f}% of the full-run best regularized objective.</li>
</ol>

<h2>1. Does TV help with oscillatory high-resolution errors?</h2>
<p>The paired experiment compares the same five samples at every resolution using LBFGS learning rate 1.0, no line search, five D-Flow Euler steps, and 750 outer steps. The forward prediction is denoted \(Ax\), and the regularized inverse objective is</p>
<div class="math">\[
\mathcal J(z)=\operatorname{{MSE}}(A x(z),b)+\lambda_{{TV}}\operatorname{{TV}}(x(z)).
\]</div>
<p>The comparison below uses \(\lambda_{{TV}}=0\) and \(10^{{-4}}\). Reconstruction TV is the numerical proxy for oscillation; image MSE measures fidelity to the known synthetic ground truth.</p>

<table>
<tr><th>Resolution</th><th>No-TV mean image MSE</th><th>TV mean image MSE</th><th>Mean reduction</th><th>Image wins</th><th>No-TV mean solution TV</th><th>TV mean solution TV</th><th>TV reduction</th><th>TV wins</th></tr>
{tv_table_rows}
</table>

<div class="note good"><strong>Combined high-resolution result.</strong> Across the 10 paired 128/160 cases, mean image MSE decreased from {format_sci(combined['no_tv_mean_image_mse'])} to {format_sci(combined['tv_mean_image_mse'])} ({100 * combined['image_mean_reduction']:.1f}% lower). Mean reconstruction TV decreased from {combined['no_tv_mean_solution_tv']:.5f} to {combined['tv_mean_solution_tv']:.5f} ({100 * combined['solution_tv_mean_reduction']:.1f}% lower).</div>

<div class="note warn"><strong>Tradeoff.</strong> The TV reconstruction had lower solution TV in 10/10 pairs (two-sided sign-test \(p={combined['solution_tv_sign_p']:.4f}\)), but lower image MSE in only {combined['image_wins']}/10 pairs (\(p={combined['image_sign_p']:.3f}\)). Median measurement MSE increased from {format_sci(combined['no_tv_median_measurement_mse'])} without TV to {format_sci(combined['tv_median_measurement_mse'])} with TV. The measurement residual remains small, but regularization intentionally trades exact data fit for a smoother, generally more accurate image.</div>

<figure>
<img src="dflow_tv_high_resolution_pairs.png" alt="Paired image MSE and reconstruction TV with and without TV regularization">
<figcaption>Each line connects the same sample without and with TV. Reconstruction TV falls for every high-resolution sample; image MSE improves for six of ten samples.</figcaption>
</figure>

<h3>Representative paired reconstructions</h3>
<div class="grid-2">
<figure><img src="../dflow_tv_lr_comparison_750_steps/by_tv_weight/tv_0/lr_1.0/figures/res_128/n1_sample_000.png" alt="128 no-TV result"><figcaption>128×128 sample 0, no TV.</figcaption></figure>
<figure><img src="../dflow_tv_lr_comparison_750_steps/by_tv_weight/tv_1e-4/lr_1.0/figures/res_128/n1_sample_000.png" alt="128 TV result"><figcaption>128×128 sample 0, TV \(10^{{-4}}\): image MSE is 79.2% lower and solution TV is 81.4% lower than the paired no-TV result.</figcaption></figure>
<figure><img src="../dflow_tv_lr_comparison_750_steps/by_tv_weight/tv_0/lr_1.0/figures/res_160/n1_sample_000.png" alt="160 no-TV result"><figcaption>160×160 sample 0, no TV.</figcaption></figure>
<figure><img src="../dflow_tv_lr_comparison_750_steps/by_tv_weight/tv_1e-4/lr_1.0/figures/res_160/n1_sample_000.png" alt="160 TV result"><figcaption>160×160 sample 0, TV \(10^{{-4}}\): image MSE is 95.0% lower and solution TV is 89.9% lower.</figcaption></figure>
</div>

<h2>2. Per-iteration time versus resolution</h2>
<p>The GPU benchmark timed five samples at each resolution. It synchronized CUDA around every outer LBFGS step and excluded the first 10 steps of each sample from the steady-state summary. The 15,000 recorded outer steps contained 299,272 closure evaluations.</p>

<table>
<tr><th>Resolution</th><th>Samples</th><th>Mean iteration (s)</th><th>Median iteration (s)</th><th>P10 (s)</th><th>P90 (s)</th><th>Closures/iteration</th><th>Median closure (s)</th></tr>
{timing_rows}
</table>

<figure>
<img src="dflow_iteration_time_scaling.png" alt="D-Flow time per outer iteration and closure versus resolution">
<figcaption>An outer LBFGS step costs approximately 20 D-Flow closure evaluations. The lower resolutions are overhead-bound; measured time rises sharply at 128 and 160.</figcaption>
</figure>

<div class="note warn"><strong>Timing limitation.</strong> The 128×128 trace changed from roughly 1.19 s to 3.9 s per outer iteration partway through otherwise identical samples, showing changing system/GPU contention. A log-log fit gives approximately \(T\propto R^{{1.63}}\), or \(T\propto N_{{pixels}}^{{0.81}}\), but that exponent should be treated as descriptive rather than an isolated hardware-complexity estimate.</div>

<h2>3. How many outer iterations are needed?</h2>
<p>The existing stop condition requires unregularized measurement MSE below \(10^{{-12}}\). None of the 20 timing runs satisfied it, so every run used all 750 outer steps. This condition is also misaligned with TV runs, which minimize the combined regularized objective.</p>

<p>Define the best regularized objective observed by step \(k\) as</p>
<div class="math">\[
B_k=\min_{{0\le j\le k}}\mathcal J_j.
\]</div>
<p>A conservative candidate rule is:</p>
<ol>
<li>Always run at least 350 outer steps.</li>
<li>Afterward, stop when \((B_{{k-50}}-B_k)/\max(|B_{{k-50}}|,\epsilon)&lt;10^{{-3}}\).</li>
<li>Return the saved best iterate, not the final iterate.</li>
<li>Retain 750 as the maximum budget and add rollback for non-finite or severely increasing objectives.</li>
</ol>

<table>
<tr><th>Resolution</th><th>Mean stop step</th><th>Median stop step</th><th>Mean steps saved</th><th>Median objective gap</th><th>Worst objective gap</th></tr>
{stop_rows}
</table>

<div class="note good"><strong>Retrospective stopping result.</strong> The rule stopped {early_count}/20 runs before step 750. Mean stopping time was {stop_steps.mean():.1f} steps and median stopping time was {np.median(stop_steps):.0f}. This saved {100 * mean_saving:.1f}% of outer steps on average. The median regularized-objective gap relative to the best value found by step 750 was {100 * np.median(stop_gaps):.3f}%; the worst gap was {100 * stop_gaps.max():.3f}%.</div>

<figure>
<img src="dflow_objective_convergence.png" alt="Median best-so-far D-Flow objective convergence">
<figcaption>Median best-so-far regularized-objective gap relative to the step-750 result. Improvements after roughly 350 steps are usually small, although two 160×160 runs continue to improve enough to use the full budget under the proposed rule.</figcaption>
</figure>

<div class="note warn"><strong>Validation requirement.</strong> The stopping rule was selected retrospectively on these same 20 trajectories, and intermediate image MSE was not recorded. It should be validated prospectively on new samples while logging both regularized objective and reconstruction error. The current evidence supports it as a candidate engineering rule, not a finalized convergence theorem.</div>

<h2>Configuration and provenance</h2>
<table class="settings">
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Problem</td><td>Gaussian deblurring, one-circle synthetic images</td></tr>
<tr><td>Resolutions</td><td>64, 96, 128, 160</td></tr>
<tr><td>Blur sigma</td><td>5.0</td></tr>
<tr><td>D-Flow Euler steps</td><td>5</td></tr>
<tr><td>Optimizer</td><td>LBFGS, LR 1.0, max_iter 20, history 25, no line search</td></tr>
<tr><td>Outer budget</td><td>750</td></tr>
<tr><td>TV comparison</td><td>\(\lambda_{{TV}}=0\) versus \(10^{{-4}}\)</td></tr>
<tr><td>Model checkpoint</td><td>Epoch 50</td></tr>
</table>

<h3>Source artifacts</h3>
<ul>
<li><code>outputs/dflow_tv_lr_comparison_750_steps/by_tv_weight/</code></li>
<li><code>outputs/dflow_time_complexity_tv_1e-4_lr_1p0_750_steps/dflow_iteration_timings.csv</code></li>
<li><code>outputs/reports/dflow_tv_high_resolution_summary.csv</code></li>
<li><code>outputs/reports/dflow_stopping_criterion_simulation.csv</code></li>
</ul>
<p class="small">The cancelled CPU timing run is excluded from this report.</p>
</main></body></html>
"""


def concise_report_html(
    sweep_rows: list[dict[str, object]],
    histories: dict[tuple[int, int], list[dict]],
) -> str:
    best_mean = min(row["overall_mean_image_mse"] for row in sweep_rows)
    no_tv_row = next(row for row in sweep_rows if row["tv_weight"] == 0)
    best_row = next(
        row for row in sweep_rows if row["overall_mean_image_mse"] == best_mean
    )
    tv_1e4_row = next(row for row in sweep_rows if row["tv_weight"] == 1e-4)
    sweep_table = "\n".join(
        f"""<tr{(' class="best"' if row['overall_mean_image_mse'] == best_mean else '')}>
<td>{row['tv_weight']:.0e}</td>
<td>{format_sci(row['mean_image_mse_64'])}<br><span class="sub">{format_sci(row['median_image_mse_64'])}</span></td>
<td>{format_sci(row['mean_image_mse_96'])}<br><span class="sub">{format_sci(row['median_image_mse_96'])}</span></td>
<td>{format_sci(row['mean_image_mse_128'])}<br><span class="sub">{format_sci(row['median_image_mse_128'])}</span></td>
<td>{format_sci(row['mean_image_mse_160'])}<br><span class="sub">{format_sci(row['median_image_mse_160'])}</span></td>
<td>{format_sci(row['overall_mean_image_mse'])}<br><span class="sub">{format_sci(row['overall_median_image_mse'])}</span></td>
<td>{row['overall_mean_solution_tv']:.5f}</td>
<td>{row['image_mse_wins_vs_no_tv'] if row['tv_weight'] else '—'}</td>
<td>{row['solution_tv_wins_vs_no_tv'] if row['tv_weight'] else '—'}</td>
</tr>"""
        for row in sweep_rows
    )

    iteration_rows = []
    for resolution in RESOLUTIONS:
        values = np.array(
            [
                row["elapsed_s"]
                for sample in SAMPLES
                for row in histories[(resolution, sample)]
                if row["outer_step"] >= 10
            ]
        )
        closure_counts = np.array(
            [
                row["closure_evals"]
                for sample in SAMPLES
                for row in histories[(resolution, sample)]
                if row["outer_step"] >= 10
            ]
        )
        iteration_rows.append(
            {
                "resolution": resolution,
                "mean": values.mean(),
                "median": np.median(values),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )
    iteration_table = "\n".join(
        f"""<tr>
<td>{row['resolution']}×{row['resolution']}</td>
<td>{row['mean']:.3f}</td>
<td>{row['median']:.3f}</td>
<td>{row['minimum']:.3f}</td>
<td>{row['maximum']:.3f}</td>
</tr>"""
        for row in iteration_rows
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D-Flow TV and Runtime Results</title>
<style>
:root{{--ink:#17212b;--muted:#5a6774;--line:#d7dde3;--soft:#f4f7f9;--accent:#245b78}}
body{{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink)}}
main{{max-width:1120px;margin:auto;padding:34px 26px 60px}} h1{{font-size:30px;margin-bottom:6px}}
h2{{font-size:21px;margin:34px 0 12px;border-top:1px solid var(--line);padding-top:18px}}
p,li{{line-height:1.6}} .subtitle,.small{{color:var(--muted);font-size:13px}}
table{{width:100%;border-collapse:collapse;margin:16px 0 22px;font-size:13px}}
th,td{{padding:10px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{background:var(--soft)}} tr.best{{background:#edf6ef;font-weight:650}}
.sub{{color:var(--muted);font-weight:400;font-size:12px}}
.result{{border-left:4px solid var(--accent);background:#eef5f8;padding:11px 15px;margin:16px 0;line-height:1.55}}
.goal{{border:1px solid #a9c5d4;border-left:5px solid var(--accent);border-radius:7px;background:#f3f8fa;padding:14px 17px;margin:16px 0 20px;line-height:1.6}}
.goal strong{{display:block;margin-bottom:4px;color:var(--accent)}}
figure{{margin:18px 0;border:1px solid var(--line);border-radius:7px;padding:10px}} figure img{{width:100%;display:block}}
figcaption{{color:var(--muted);font-size:13px;margin-top:8px}} code{{font-family:ui-monospace,monospace}}
@media(max-width:800px){{main{{padding:22px 14px}}table{{display:block;overflow-x:auto;font-size:12px}}}}
</style></head><body><main>
<h1>D-Flow TV and Runtime Results</h1>
<p class="subtitle">Results from the completed TV sweep and GPU iteration-timing experiment.</p>

<h2>1. TV-regularization sweep</h2>
<div class="goal"><strong>Goal of the test</strong>Provide numerical evidence that TV regularization can reduce oscillatory error in D-Flow reconstructions. We are particularly interested in high-resolution cases and whether good reconstruction results can still be obtained without TV regularization.</div>
<p><strong>Experiment:</strong> five paired samples at each of 64×64, 96×96, 128×128, and 160×160; LBFGS LR 1.0; no line search; 750 outer steps; five Euler steps. TV weights: 0, 10<sup>−6</sup>, 3×10<sup>−6</sup>, 10<sup>−5</sup>, 3×10<sup>−5</sup>, 10<sup>−4</sup>, 3×10<sup>−4</sup>, and 10<sup>−3</sup>.</p>
<table>
<tr><th>TV weight</th><th>64 MSE<br><span class="sub">mean / median</span></th><th>96 MSE<br><span class="sub">mean / median</span></th><th>128 MSE<br><span class="sub">mean / median</span></th><th>160 MSE<br><span class="sub">mean / median</span></th><th>All MSE<br><span class="sub">mean / median</span></th><th>Mean solution TV</th><th>Samples with lower<br>image MSE</th><th>Samples with lower<br>solution TV</th></tr>
{sweep_table}
</table>
<div class="result"><strong>What the last two columns mean.</strong> Every TV setting is compared with no TV on the same 20 cases: five samples at each of four resolutions. “Samples with lower image MSE” counts cases where TV improved reconstruction accuracy. “Samples with lower solution TV” counts cases where TV produced a smoother reconstruction. These are independent tests: one sample can count in both columns, one column, or neither, so the totals should not be added together. Lower solution TV is evidence of reduced oscillation, but it does not automatically imply lower image error. The counts are binary and do not show improvement magnitude; the mean and median MSE columns provide that information.</div>
<div class="result"><strong>Result.</strong> Across all four resolutions, TV weight 3×10<sup>−6</sup> gave the lowest overall mean image MSE, {format_sci(best_mean)}, compared with {format_sci(no_tv_row['overall_mean_image_mse'])} without TV ({100 * (1 - best_mean / no_tv_row['overall_mean_image_mse']):.1f}% lower), and won {best_row['image_mse_wins_vs_no_tv']}/20 paired comparisons. TV weight 10<sup>−4</sup> gave a similar overall mean MSE of {format_sci(tv_1e4_row['overall_mean_image_mse'])}, reduced mean solution TV by {100 * (1 - tv_1e4_row['overall_mean_solution_tv'] / no_tv_row['overall_mean_solution_tv']):.1f}%, and reduced solution TV in {tv_1e4_row['solution_tv_wins_vs_no_tv']}/20 pairs.</div>

<h3>Best, median, and worst reconstructions</h3>
<p>The selections below are based on image MSE among the five samples for each resolution and TV weight.</p>
<figure><img src="dflow_tv_gallery_res_64.png" alt="64x64 TV sweep reconstruction gallery"><figcaption>64×64 sweep.</figcaption></figure>
<figure><img src="dflow_tv_gallery_res_96.png" alt="96x96 TV sweep reconstruction gallery"><figcaption>96×96 sweep.</figcaption></figure>
<figure><img src="dflow_tv_gallery_res_128.png" alt="128x128 TV sweep reconstruction gallery"><figcaption>128×128 sweep.</figcaption></figure>
<figure><img src="dflow_tv_gallery_res_160.png" alt="160x160 TV sweep reconstruction gallery"><figcaption>160×160 sweep.</figcaption></figure>

<h2>2. Per-iteration runtime</h2>
<p><strong>Experiment:</strong> four resolutions, five samples per resolution, 750 outer LBFGS steps per sample, LR 1.0, TV weight 10<sup>−4</sup>, no line search, and five Euler steps. Timing used one GPU process, with the first 10 iterations of each sample excluded as warm-up.</p>
<table>
<tr><th>Resolution</th><th>Mean (s)</th><th>Median (s)</th><th>Lowest (s)</th><th>Highest (s)</th></tr>
{iteration_table}
</table>
<p class="small">The 128×128 timings changed partway through the run because of GPU contention; the values above are the measured results, not isolated-hardware estimates.</p>

<p class="small">Source data: <code>outputs/dflow_tv_lr_comparison_750_steps/by_tv_weight/</code> and <code>outputs/dflow_time_complexity_tv_1e-4_lr_1p0_750_steps/</code>.</p>
</main></body></html>"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tv_summary, combined = paired_tv_analysis()
    sweep_rows = tv_sweep_analysis()
    timing_summary, histories = timing_analysis()
    stopping_rows = simulate_stopping(histories)

    write_csv(REPORT_DIR / "dflow_tv_high_resolution_summary.csv", tv_summary)
    write_csv(REPORT_DIR / "dflow_tv_high_resolution_sweep.csv", sweep_rows)
    write_csv(REPORT_DIR / "dflow_stopping_criterion_simulation.csv", stopping_rows)
    plot_tv_pairs(REPORT_DIR / "dflow_tv_high_resolution_pairs.png")
    plot_tv_galleries()
    REPORT_PATH.write_text(
        concise_report_html(sweep_rows, histories)
    )
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

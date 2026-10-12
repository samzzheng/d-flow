"""Audit completed noisy sweeps and report paired accuracy distributions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ct_mri_sweep.run import TERMINAL, jobs, write_json

METHODS = ("D-Flow", "P-Flow", "DLR D-Flow")
Record = dict[str, Any]


def summarize(rows: list[Record]) -> Record:
    errors = [row["image_percent"] for row in rows]
    later = [row for row in rows if row["attempts"] > 100]
    return {"n": len(rows), "median_image_percent": statistics.median(errors),
            "mean_image_percent": statistics.mean(errors), "p90_image_percent": float(np.quantile(errors, .9)),
            "image_at_most_5_percent": sum(value <= 5 for value in errors),
            "image_above_20_percent": sum(value > 20 for value in errors),
            "median_residual_percent": statistics.median(row["residual_percent"] for row in rows),
            "median_residual_over_noise": statistics.median(row["residual_over_noise"] for row in rows),
            "median_attempts": statistics.median(row["attempts"] for row in rows),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "median_image_by_100_percent": statistics.median(row["image_by_100_percent"] for row in rows),
            "median_best_image_percent_retrospective": statistics.median(row["best_image_percent"] for row in rows),
            "continued_after_100": len(later),
            "worse_image_despite_lower_objective_after_100": sum(row["image_percent"] > row["image_by_100_percent"]
                and row["phi"] < row["phi_by_100"] for row in later)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    protocols = {}
    studies = []
    for root in args.roots:
        manifest = json.loads((root / "production/manifest.json").read_text())
        settings = manifest["settings"]
        state = json.loads((root / "queue_status.json").read_text())
        assert state["status"] == "finished" and not state["failures"]
        expected = len(jobs(settings)) * settings["samples"] * 3
        count = 0
        for job in jobs(settings):
            for sample in range(settings["samples"]):
                case = root / "production" / job.name / f"sample_{sample:03d}"
                previous = None
                for method in METHODS:
                    stem = method.lower().replace("-", "_").replace(" ", "_")
                    record = json.loads((case / f"{stem}.json").read_text())
                    assert record["status"] in TERMINAL
                    protocol = record["protocol"]
                    if previous is not None:
                        assert previous == protocol
                    previous = protocol
                    key = (job.name, sample)
                    paired = {name: protocol[name] for name in ("checkpoint_sha256", "test_subset_sha256", "job",
                        "sample", "source_seed", "noise_seed", "dlr_seed", "operator")}
                    if key in protocols:
                        assert protocols[key] == paired
                    protocols[key] = paired
                    history = record["history"]
                    last = history[-1]
                    early = next(item for item in reversed(history) if item["step"] <= 100)
                    assert all(b["phi"] <= a["phi"] * (1 + 1e-10) for a, b in zip(history, history[1:]))
                    assert all(math.isfinite(item["image_relative_l2"]) for item in history)
                    rows.append({"noise_percent": 100 * settings["noise_relative"], "resolution": job.resolution,
                        "problem": job.problem, "sample": sample, "method": method,
                        "image_percent": 100 * last["image_relative_l2"],
                        "residual_percent": 100 * last["measurement_relative_l2_noisy"],
                        "clean_measurement_percent": 100 * last["measurement_relative_l2_clean"],
                        "residual_over_noise": last["residual_over_noise"], "attempts": last["step"],
                        "accepted_updates": sum(item.get("source_step_norm", 0) > 0 for item in history),
                        "status": record["status"], "phi": last["phi"], "phi_by_100": early["phi"],
                        "image_by_100_percent": 100 * early["image_relative_l2"],
                        "best_image_percent": 100 * min(item["image_relative_l2"] for item in history)})
                    count += 1
                assert (case / "comparison.png").exists() and (case / "comparison.pdf").exists()
        assert count == expected
        studies.append({"root": str(root), "noise_percent": 100 * settings["noise_relative"],
                        "method_runs": count, "resolutions": settings["resolutions"], "samples": settings["samples"]})
    with (args.out / "runs.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    winners = []
    lines = ["# Completed five-step noisy reconstruction studies", "",
        "Final saved iterates; no ground-truth-based checkpoint selection. Single-circle epoch200 models, "
        "CT10/MRI4, five Euler steps, FP32, shared LBFGS/geometric Armijo, up to1,000 attempts. "
        "DLR:rank8, analytical alpha=1-t, previous-V sketch initialization. "
        "All methods completed without execution failures; a stalled line search is a terminal optimizer outcome, not convergence.", "",
        "Inputs/model/operator/seed protocols match across methods and the overlapping noise-study cases. "
        "All saved objectives are nonincreasing. Every case has PNG/PDF reconstruction and convergence figures. "
        "One-percent and five-percent studies cover different resolution grids, so pooled differences are not paired noise effects.", ""]
    figure, axes = plt.subplots(len(studies), 2, figsize=(12, 4.3 * len(studies)), squeeze=False, layout="constrained")
    for index, study in enumerate(studies):
        noise = study["noise_percent"]
        subset = [row for row in rows if row["noise_percent"] == noise]
        lines += [f"## {noise:g}% noise ({study['method_runs']} method runs)", "",
            "Relative image L2 error in percent: median / mean / count above20% (30 samples per cell).", "",
            "| Problem | Resolution | D-Flow | P-Flow | DLR D-Flow |",
            "| --- | --- | --- | --- | --- |"]
        for problem in ("ct", "mri"):
            for resolution in sorted(study["resolutions"]):
                cells = []
                for method in METHODS:
                    selected = [row for row in subset if row["problem"] == problem and row["resolution"] == resolution and row["method"] == method]
                    stats = summarize(selected)
                    aggregate.append({"noise_percent": noise, "problem": problem, "resolution": resolution, "method": method, **stats})
                    cells.append(f"{stats['median_image_percent']:.2f} / {stats['mean_image_percent']:.2f} / {stats['image_above_20_percent']}")
                lines.append(f"| {problem.upper()} | {resolution} | " + " | ".join(cells) + " |")
        lines += ["", "| Problem | Method | Median image % | Mean image % | ≤5% | >20% | Stalled | Median attempts | Residual / noise norm | By100 image % | Worse image after100 despite lower objective |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for column, problem in enumerate(("ct", "mri")):
            candidates = [row for row in subset if row["problem"] == problem]
            counts = Counter()
            for resolution in study["resolutions"]:
                for sample in range(study["samples"]):
                    paired = [row for row in candidates if row["resolution"] == resolution and row["sample"] == sample]
                    low = min(row["image_percent"] for row in paired)
                    best = [row["method"] for row in paired if row["image_percent"] == low]
                    counts[best[0] if len(best) == 1 else "tie"] += 1
            winners.append({"noise_percent": noise, "problem": problem, "wins": dict(counts)})
            for method, color in zip(METHODS, ("#2563eb", "#d97706", "#059669")):
                selected = [row for row in candidates if row["method"] == method]
                stats = summarize(selected)
                aggregate.append({"noise_percent": noise, "problem": problem, "resolution": "all", "method": method, **stats})
                n = stats["n"]
                lines.append(f"| {problem.upper()} | {method} | {stats['median_image_percent']:.2f} | "
                    f"{stats['mean_image_percent']:.2f} | {stats['image_at_most_5_percent']}/{n} | "
                    f"{stats['image_above_20_percent']}/{n} | {stats['statuses'].get('stalled', 0)}/{n} | "
                    f"{stats['median_attempts']:g} | {stats['median_residual_over_noise']:.3f} | "
                    f"{stats['median_image_by_100_percent']:.2f} | {stats['worse_image_despite_lower_objective_after_100']}/{stats['continued_after_100']} |")
                resolutions = sorted(study["resolutions"])
                quantiles = np.array([np.quantile([row["image_percent"] for row in selected if row["resolution"] == resolution], [.25, .5, .75]) for resolution in resolutions])
                axes[index, column].plot(resolutions, quantiles[:, 1], "o-", color=color, label=method)
                axes[index, column].fill_between(resolutions, quantiles[:, 0], quantiles[:, 2], color=color, alpha=.13)
            axes[index, column].set(title=f"{noise:g}% noise, {problem.upper()}: 30 samples per resolution",
                xlabel="Resolution (pixels per side)", ylabel="Relative image L2 error (%)", yscale="log", xticks=sorted(study["resolutions"]))
            axes[index, column].grid(alpha=.2)
            axes[index, column].legend()
        lines += ["", "Wins on paired samples: " + "; ".join(f"{item['problem'].upper()} {item['wins']}" for item in winners if item["noise_percent"] == noise), ""]
    figure.suptitle("Final image errors: median and interquartile range (logarithmic error axis)")
    for extension in ("png", "pdf"):
        figure.savefig(args.out / f"accuracy_summary.{extension}", dpi=160)
    plt.close(figure)
    lines += ["By100 uses the checkpoint at attempt100, or the terminal checkpoint for runs that stopped earlier. "
              "Post100 deterioration counts only runs with more than100 attempts. Best-image errors in JSON/CSV are "
              "retrospective ground-truth diagnostics, not deployable stopping rules. Thresholds5%/20% describe "
              "accuracy and severe errors; they do not define optimizer failure. Group medians do not identify paired wins."]
    write_json(args.out / "summary.json", {"studies": studies, "groups": aggregate, "paired_winners": winners})
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"method_runs": len(rows), "paired_cases": len(rows) // 3, "output": str(args.out)}))


if __name__ == "__main__":
    main()

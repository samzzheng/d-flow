"""CPU-only paired reconstruction trajectory audit from saved histories."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METHODS = {"D-Flow": "d_flow", "P-Flow": "p_flow", "DLR D-Flow": "dlr_d_flow"}
RESOLUTIONS = (64, 96, 128, 160, 200, 250)
Record = dict[str, Any]


def summarize(rows: list[Record]) -> Record:
    return {
        "cases": len(rows),
        "median_final_image_percent": median(row["final_image_percent"] for row in rows),
        "median_common_all_image_percent": median(row["common_all_image_percent"] for row in rows),
        "median_common_pair_image_percent": median(row["common_pair_image_percent"] for row in rows if row["common_pair_image_percent"] is not None) if rows[0]["method"] != "P-Flow" else None,
        "median_common_all_attempts": median(row["common_all_attempts"] for row in rows),
        "median_common_pair_attempts": median(row["common_pair_attempts"] for row in rows),
        "median_attempts": median(row["attempts"] for row in rows),
        "median_best_image_percent_retrospective": median(row["best_image_percent"] for row in rows),
        "median_best_image_step_retrospective": median(row["best_image_step"] for row in rows),
        "median_final_residual_over_noise": median(row["final_residual_over_noise"] for row in rows),
        "median_best_image_residual_over_noise_retrospective": median(row["best_image_residual_over_noise"] for row in rows),
        "median_image_by100_percent": median(row["image_by100_percent"] for row in rows),
        "over20_final": sum(row["final_image_percent"] > 20 for row in rows),
        "over20_final_previously_atmost10": sum(row["final_image_percent"] > 20 and row["best_image_percent"] <= 10 for row in rows),
        "over20_final_never_atmost10": sum(row["final_image_percent"] > 20 and row["best_image_percent"] > 10 for row in rows),
        "over20_final_residual_atmost1_1noise": sum(row["final_image_percent"] > 20 and row["final_residual_over_noise"] <= 1.1 for row in rows),
        "over20_common_all": sum(row["common_all_image_percent"] > 20 for row in rows),
        "continued_after100": sum(row["attempts"] > 100 for row in rows),
        "worsened_after100_despite_lower_loss": sum(row["attempts"] > 100 and row["final_image_percent"] > row["image_by100_percent"] and row["final_phi"] < row["phi_by100"] for row in rows),
        "worsened_final_vs_best_by_over1point": sum(row["final_image_percent"] - row["best_image_percent"] > 1 for row in rows),
        **{
            f"noise_floor_{threshold}_{key}": value
            for threshold in (1.0, 1.05, 1.1)
            for key, value in {
                "reached": sum(row[f"noise_floor_{threshold}_image_percent"] is not None for row in rows),
                "median_image_percent_reached_only": median([row[f"noise_floor_{threshold}_image_percent"] for row in rows if row[f"noise_floor_{threshold}_image_percent"] is not None]) if any(row[f"noise_floor_{threshold}_image_percent"] is not None for row in rows) else None,
                "median_firsthit_or_final_image_percent": median(row[f"noise_floor_{threshold}_or_final_image_percent"] for row in rows),
            }.items()
        },
    }


def main() -> None:
    rows: list[Record] = []
    cases: list[Record] = []
    protocols: dict[tuple[int, str, int], Record] = {}
    for noise in (1, 5, 10, 20):
        root = ROOT / f"five_step_noise{noise:03d}_n124_20261013" / "production"
        for resolution in RESOLUTIONS:
            for problem, suffix in (("ct", "ct10"), ("mri", "mri4")):
                for sample in range(30):
                    directory = root / f"r{resolution}_n1_{suffix}" / f"sample_{sample:03d}"
                    records = {name: json.loads((directory / f"{stem}.json").read_text()) for name, stem in METHODS.items()}
                    previous: Record | None = None
                    histories = {name: {int(item["step"]): item for item in record["history"]} for name, record in records.items()}
                    common = min(max(history) for history in histories.values())
                    pair_common = min(max(histories[name]) for name in ("D-Flow", "DLR D-Flow"))
                    metadata: Record = {"noise_percent": noise, "resolution": resolution, "problem": problem, "sample": sample, "directory": str(directory), "common_all_attempts": common, "common_pair_attempts": pair_common}
                    case = dict(metadata)
                    for method, record in records.items():
                        assert record["status"] in ("stalled", "step_limit")
                        assert previous is None or previous == record["protocol"]
                        previous = record["protocol"]
                        protocol = {key: previous[key] for key in ("checkpoint_sha256", "test_subset_sha256", "job", "sample", "source_seed", "noise_seed", "dlr_seed", "operator")}
                        key = (resolution, problem, sample)
                        assert key not in protocols or protocols[key] == protocol
                        protocols[key] = protocol
                        history = record["history"]
                        assert all(b["phi"] <= a["phi"] * (1 + 1e-10) for a, b in zip(history, history[1:]))
                        assert all(math.isfinite(item["image_relative_l2"]) for item in history)
                        last = history[-1]
                        early = histories[method][min(100, last["step"])]
                        best = min(history, key=lambda item: item["image_relative_l2"])
                        at_common = histories[method][common]
                        at_pair = histories[method][pair_common] if method != "P-Flow" else None
                        row: Record = {**metadata, "method": method, "status": record["status"], "attempts": last["step"],
                            "final_image_percent": 100 * last["image_relative_l2"], "final_residual_over_noise": last["residual_over_noise"], "final_phi": last["phi"],
                            "common_all_image_percent": 100 * at_common["image_relative_l2"], "common_all_residual_over_noise": at_common["residual_over_noise"],
                            "common_pair_image_percent": 100 * at_pair["image_relative_l2"] if at_pair is not None else None,
                            "image_by100_percent": 100 * early["image_relative_l2"], "phi_by100": early["phi"],
                            "best_image_percent": 100 * best["image_relative_l2"], "best_image_step": best["step"], "best_image_residual_over_noise": best["residual_over_noise"]}
                        for threshold in (1.0, 1.05, 1.1):
                            hit = next((item for item in history if item["residual_over_noise"] <= threshold), None)
                            row[f"noise_floor_{threshold}_image_percent"] = None if hit is None else 100 * hit["image_relative_l2"]
                            row[f"noise_floor_{threshold}_step"] = None if hit is None else hit["step"]
                            row[f"noise_floor_{threshold}_or_final_image_percent"] = 100 * (last if hit is None else hit)["image_relative_l2"]
                        rows.append(row)
                        case[method] = {key: value for key, value in row.items() if key not in metadata}
                    cases.append(case)

    flat_rows = rows
    with (OUT / "runs.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    groups: list[Record] = []
    comparisons: list[Record] = []
    for noise in (1, 5, 10, 20):
        for problem in ("ct", "mri"):
            for resolution in ("all", *RESOLUTIONS):
                selected = [case for case in cases if case["noise_percent"] == noise and case["problem"] == problem and (resolution == "all" or case["resolution"] == resolution)]
                prefix = {"noise_percent": noise, "problem": problem, "resolution": resolution}
                for method in METHODS:
                    selected_rows = [row for row in flat_rows if row["noise_percent"] == noise and row["problem"] == problem and row["method"] == method and (resolution == "all" or row["resolution"] == resolution)]
                    groups.append({**prefix, "method": method, **summarize(selected_rows)})
                comparisons.append({**prefix, "cases": len(selected),
                    **{f"dlr_beats_d_{metric}": sum(case["DLR D-Flow"][metric] < case["D-Flow"][metric] for case in selected) for metric in ("final_image_percent", "common_all_image_percent", "common_pair_image_percent", "best_image_percent")},
                    "dlr_final_beats_d_retrospective_best": sum(case["DLR D-Flow"]["final_image_percent"] < case["D-Flow"]["best_image_percent"] for case in selected),
                    "dlr_final_beats_p_retrospective_best": sum(case["DLR D-Flow"]["final_image_percent"] < case["P-Flow"]["best_image_percent"] for case in selected),
                    "dlr_final_atmost10_d_final_above20": sum(case["DLR D-Flow"]["final_image_percent"] <= 10 and case["D-Flow"]["final_image_percent"] > 20 for case in selected),
                })
    candidates = []
    for noise in (5, 20):
        for resolution in (64, 128):
            for problem in ("ct", "mri"):
                selected = [case for case in cases if case["noise_percent"] == noise and case["resolution"] == resolution and case["problem"] == problem]
                good = [case for case in selected if case["DLR D-Flow"]["final_image_percent"] <= 10]
                good.sort(key=lambda case: case["D-Flow"]["final_image_percent"] - case["DLR D-Flow"]["final_image_percent"], reverse=True)
                bad = max(selected, key=lambda case: case["DLR D-Flow"]["final_image_percent"] - case["D-Flow"]["final_image_percent"])
                candidates.extend([{**case, "selection": "DLR good / D-Flow poor"} for case in good[:2]])
                candidates.append({**bad, "selection": "largest DLR disadvantage"})
    summary = {"method_runs": len(rows), "paired_cases": len(cases), "groups": groups, "paired_comparisons": comparisons,
        "interpretation": "Common prefixes are the last attempt completed by all three methods in each paired case, or by D-Flow and DLR only. Prefix length varies by case. Best checkpoints use ground truth retrospectively. Noise-floor firsthit-or-final includes all cases and explicitly falls back to final when no crossing occurs. No reruns or new GPU work."}
    summary["persistent_dlr_large_errors"] = []
    for problem in ("ct", "mri"):
        large_error_sets = [{(row["resolution"], row["sample"]) for row in rows if row["problem"] == problem and row["noise_percent"] == noise and row["method"] == "DLR D-Flow" and row["final_image_percent"] > 20} for noise in (1, 5, 10, 20)]
        summary["persistent_dlr_large_errors"].append({"problem": problem, "noise_levels": [1, 5, 10, 20], "large_error_counts": list(map(len, large_error_sets)), "intersection_count": len(set.intersection(*large_error_sets)), "union_count": len(set.union(*large_error_sets))})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "diagnostic_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    pooled = {(group["noise_percent"], group["problem"], group["method"]): group for group in groups if group["resolution"] == "all"}
    lines = ["# Noise regularization: paired trajectory audit", "", "Read-only CPU analysis of 4,320 completed method histories: 1,440 paired cases, single-circle models, six resolutions, CT10/MRI4, 30 samples per configuration, five Euler steps, shared LBFGS/Armijo settings, rank-8 DLR. Each table entry pools the same 180 cases per problem/noise. Protocol hashes, seeds, operators, and monotone saved objectives were checked. All numbers below are relative image L2 percentages unless labeled otherwise.", "", "## Final reconstruction versus retrospective best", "", "Best checkpoints are selected using ground truth and are an oracle diagnostic, not an available stopping criterion. A low retrospective error shows that the trajectory found an accurate reconstruction before its final iterate.", "", "| Problem | Noise | D-Flow final / best | P-Flow final / best | DLR final / best |", "| --- | --- | --- | --- | --- |"]
    for problem in ("ct", "mri"):
        for noise in (1, 5, 10, 20):
            cells = [f"{pooled[noise, problem, method]['median_final_image_percent']:.2f} / {pooled[noise, problem, method]['median_best_image_percent_retrospective']:.2f}" for method in METHODS]
            lines.append(f"| {problem.upper()} | {noise}% | " + " | ".join(cells) + " |")
    lines += ["", "## Matched attempts and residuals", "", "D-Flow/DLR are compared at the last attempt completed by both methods in each case. This avoids granting D-Flow its later additional attempts, although the common attempt number varies by sample and the methods still use different directions and accepted steps. Residual ratios use final observed residual divided by the known noise norm. Near one is consistent with leaving measurement noise unfitted; it is not proof that image content is correct.", "", "| Problem | Noise | Median common attempt | D-Flow common error | DLR common error | DLR wins / 180 at common attempt | Final residual/noise: D / P / DLR |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for problem in ("ct", "mri"):
        for noise in (1, 5, 10, 20):
            d = pooled[noise, problem, "D-Flow"]
            dlr = pooled[noise, problem, "DLR D-Flow"]
            comparison = next(item for item in comparisons if item["noise_percent"] == noise and item["problem"] == problem and item["resolution"] == "all")
            residuals = " / ".join(f"{pooled[noise, problem, method]['median_final_residual_over_noise']:.3f}" for method in METHODS)
            lines.append(f"| {problem.upper()} | {noise}% | {d['median_common_pair_attempts']:g} | {d['median_common_pair_image_percent']:.2f} | {dlr['median_common_pair_image_percent']:.2f} | {comparison['dlr_beats_d_common_pair_image_percent']} | {residuals} |")
    lines += ["", "## Distinguishing late degradation from failure to reach an accurate reconstruction", "", "Here a large final error means >20%; a previously accurate checkpoint means <=10%. These are descriptive thresholds, not objective convergence or certified optimizer failure.", "", "| Problem | Noise | Method | Final error >20% | Of these, previously <=10% | Of these, never <=10% |", "| --- | --- | --- | --- | --- | --- |"]
    for problem in ("ct", "mri"):
        for noise in (5, 20):
            for method in METHODS:
                item = pooled[noise, problem, method]
                lines.append(f"| {problem.upper()} | {noise}% | {method} | {item['over20_final']} | {item['over20_final_previously_atmost10']} | {item['over20_final_never_atmost10']} |")
    lines += ["", "DLR errors above20% persist at every noise level for22 CT cases and35 MRI cases. This suggests a separate basin/subspace issue in addition to late noise fitting; error alone does not identify optimizer failure.", "", "## Interpretation and next checks", "", "- The strongest observed regularization signature is resistance to late noise fitting: D-Flow frequently visits an accurate image and then decreases the observed loss while worsening the image. At20% noise DLR final beats D-Flow's retrospective best in only9/180 CT and30/180 MRI cases.", "- The DLR advantage survives matched D-Flow/DLR attempt prefixes at high noise, so different final stopping counts alone are insufficient to explain it. With a prefix common to all three methods, P-Flow often limits the comparison to an earlier optimization stage; those results are preserved separately in summary.json.", "- A direct first crossing of residual <=1.05 times the known noise norm is not uniformly effective: at20% CT it gives D-Flow median16.15% versus3.14% retrospective best. At20% MRI it gives7.47% versus1.78%. These numbers include final checkpoints for cases that never cross. A stopping-rule study should not assume that one crossing certifies best image quality.", "- Histories cannot establish that the retained Jacobian subspace removes noise. Compare exact and approximate gradients at the same source, separating the clean signal and added noise terms, and measure the restricted forward sensitivity A J V as well as J. Candidate paired cases are in diagnostic_candidates.json.", "", "Artifacts: runs.csv contains per-run matched-prefix and first-crossing metrics; summary.json contains each resolution as well as pooled results; trajectory_summary.png/pdf show medians. No new reconstruction jobs were started.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    colors = {"D-Flow": "#1f77b4", "P-Flow": "#d97706", "DLR D-Flow": "#16803d"}
    for row_index, problem in enumerate(("ct", "mri")):
        for method in METHODS:
            values = [pooled[noise, problem, method] for noise in (1, 5, 10, 20)]
            axes[row_index, 0].plot([1, 5, 10, 20], [item["median_final_image_percent"] for item in values], "o-", color=colors[method], label=method)
            axes[row_index, 0].plot([1, 5, 10, 20], [item["median_best_image_percent_retrospective"] for item in values], "--", color=colors[method], alpha=.8)
            if method != "P-Flow":
                axes[row_index, 1].plot([1, 5, 10, 20], [item["median_common_pair_image_percent"] for item in values], "o-", color=colors[method], label=method)
            axes[row_index, 2].plot([1, 5, 10, 20], [item["median_final_residual_over_noise"] for item in values], "o-", color=colors[method], label=method)
        axes[row_index, 0].set_title(f"{problem.upper()}: final (solid), oracle best (dashed)")
        axes[row_index, 1].set_title(f"{problem.upper()}: matched D-Flow/DLR attempts")
        axes[row_index, 2].set_title(f"{problem.upper()}: final residual / noise norm")
        axes[row_index, 2].axhline(1, color="black", linestyle=":", linewidth=1)
        for column_index, axis in enumerate(axes[row_index]):
            axis.set_xlabel("Measurement noise (%)")
            axis.set_xticks([1, 5, 10, 20])
            axis.set_yscale("log")
            axis.set_ylabel("Median relative image error (%)" if column_index < 2 else "Median norm ratio")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    figure.savefig(OUT / "trajectory_summary.png", dpi=150)
    figure.savefig(OUT / "trajectory_summary.pdf")
    plt.close(figure)
    print(json.dumps({"counts": [len(rows), len(cases)], "groups": [group for group in groups if group["resolution"] == "all"], "comparisons": [comparison for comparison in comparisons if comparison["resolution"] == "all"]}, indent=2))


if __name__ == "__main__":
    main()

"""Freeze paired terminal histories and summarize the unfinished CT/MRI sweep."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, spearmanr

METHODS = ("D-Flow", "P-Flow", "DLR D-Flow")
STEMS = ("d_flow", "p_flow", "dlr_d_flow")
TERMINAL = {"stalled", "step_limit", "objective_plateau"}
Record = dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (args.root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    records: list[Record] = []
    for case in sorted(args.root.glob("r*/sample_*")):
        paths = [case / f"{stem}.json" for stem in STEMS]
        if not all(path.exists() for path in paths):
            continue
        items = [json.loads(path.read_text()) for path in paths]
        if not all(item["status"] in TERMINAL for item in items):
            continue
        assert all(item["protocol"] == items[0]["protocol"] for item in items)
        records.append({"case": str(case), "methods": dict(zip(METHODS, items))})
    snapshot = {"created": datetime.now().astimezone().isoformat(), "root": str(args.root),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "settings": manifest["settings"], "records": records}
    (args.output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    rows: list[Record] = []
    for record in records:
        for method, item in record["methods"].items():
            history = item["history"]
            last, initial = history[-1], history[0]
            prior = history[max(0, len(history) - 21)]
            best = min(history, key=lambda entry: entry["image_relative_l2"])
            cap = .01 * item["protocol"]["job"]["resolution"]
            rows.append({"case": record["case"], "method": method, **item["protocol"]["job"],
                         "sample": item["protocol"]["sample"], "status": item["status"],
                         "attempts": last["step"],
                         "accepted_updates": sum(entry.get("source_step_norm", 0) > 0 for entry in history),
                         "image_pct": 100 * last["image_relative_l2"],
                         "residual_pct": 100 * last["measurement_relative_l2_clean"],
                         "initial_image_pct": 100 * initial["image_relative_l2"],
                         "initial_residual_pct": 100 * initial["measurement_relative_l2_clean"],
                         "best_image_pct_retrospective": 100 * best["image_relative_l2"],
                         "best_image_step_retrospective": best["step"],
                         "last20_objective_reduction_pct": 100 * (prior["phi"] - last["phi"]) / prior["phi"] if prior["phi"] else None,
                         "steps_at_cap": sum(entry.get("source_step_norm", 0) >= .999 * cap for entry in history),
                         "failed_searches": sum(entry.get("source_step_norm") == 0 for entry in history),
                         "max_increase_phi": max((b["phi"] - a["phi"] for a, b in zip(history, history[1:])), default=0)})
    with (args.output / "cases.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rng = np.random.default_rng(20261007)
    summaries: list[Record] = []
    for job, grouped in itertools.groupby(rows, key=lambda row: Path(row["case"]).parent.name):
        group = list(grouped)
        errors = np.array([[row["image_pct"] for row in group if row["method"] == method] for method in METHODS]).T
        comparisons: list[Record] = []
        for threshold in (2, 5, 10, 20):
            success = errors <= threshold
            for a, b in ((2, 0), (2, 1), (1, 0)):
                differences = success[:, a].astype(float) - success[:, b]
                bootstrap = differences[rng.integers(len(errors), size=(10000, len(errors)))].mean(1)
                wins = int((differences > 0).sum())
                losses = int((differences < 0).sum())
                comparisons.append({"threshold_pct": threshold, "a": METHODS[a], "b": METHODS[b],
                                    "a_only_success": wins, "b_only_success": losses,
                                    "success_difference_percentage_points": 100 * differences.mean(),
                                    "paired_bootstrap_95ci_pp": (100 * np.quantile(bootstrap, [.025, .975])).tolist(),
                                    "discordance_binomial_p_unadjusted": binomtest(wins, wins + losses).pvalue if wins + losses else 1.0})
        summaries.append({"job": job, "n": len(errors), "complete": len(errors) == manifest["settings"]["samples"],
                          "methods": {method: {
                              "median_image_pct": float(np.median(errors[:, i])), "mean_image_pct": float(errors[:, i].mean()),
                              "success5": int((errors[:, i] <= 5).sum()), "failure20": int((errors[:, i] > 20).sum()),
                              "stalled": sum(row["status"] == "stalled" for row in group if row["method"] == method),
                              "median_residual_pct": float(np.median([row["residual_pct"] for row in group if row["method"] == method])),
                              "initial_final_spearman": float(spearmanr([row["initial_image_pct"] for row in group if row["method"] == method], errors[:, i]).statistic)
                          } for i, method in enumerate(METHODS)}, "comparisons": comparisons})
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    complete_jobs = {entry["job"] for entry in summaries if entry["complete"]}
    complete_rows = [row for row in rows if Path(row["case"]).parent.name in complete_jobs]
    lines = [f"# Completed noiseless sweep audit: {snapshot['created']}", "",
             f"Frozen {len(records)} paired terminal cases; {len(complete_jobs)} acquisitions have all 30 cases.",
             "Only matched terminal triples are included. Partial acquisitions may be completion-biased.",
             "Success below is an exploratory <=5% final relative image L2 threshold; severe error is >20%.",
             "A stalled run is not necessarily inaccurate, and a 100-attempt run is not necessarily converged.", "",
             "| Acquisition | n | Median image error %: D / P / DLR | Success <=5%: D / P / DLR | Severe >20%: D / P / DLR |",
             "| --- | --- | --- | --- | --- |"]
    for summary in summaries:
        values = list(summary["methods"].values())
        lines.append(f"| {summary['job']} | {summary['n']}{'' if summary['complete'] else ' (partial)'} | " +
                     " / ".join(f"{v['median_image_pct']:.2f}" for v in values) + " | " +
                     " / ".join(str(v['success5']) for v in values) + " | " +
                     " / ".join(str(v['failure20']) for v in values) + " |")
    lines += ["", "## Stopping and initialization (complete acquisitions only)", ""]
    for method in METHODS:
        subset = [row for row in complete_rows if row["method"] == method]
        bad = [row for row in subset if row["image_pct"] > 20]
        limits = [row for row in bad if row["status"] == "step_limit"]
        lines.append(f"- {method}: {len(bad)}/{len(subset)} severe; {sum(r['status']=='stalled' for r in bad)} severe stalls; "
                     f"{len(limits)} severe step limits, of which {sum(r['last20_objective_reduction_pct'] > 1 for r in limits)} "
                     "still reduced the objective by >1% over the last 20 attempts. "
                     f"Among severe stalls, median attempts {np.median([r['attempts'] for r in bad if r['status']=='stalled']):.0f}.")
    lines += ["", "## Paired <=5% success comparisons", "",
              "A-only / B-only are discordant paired cases. Intervals are paired percentile bootstrap (10,000 resamples).",
              "P-values are exploratory exact discordance tests, unadjusted for inspecting multiple regimes/thresholds.", "",
              "| Acquisition | A vs B | A-only / B-only | Success difference pp (95% interval) | p |",
              "| --- | --- | --- | --- | --- |"]
    for summary in summaries:
        if not summary["complete"]:
            continue
        for comparison in summary["comparisons"]:
            if comparison["threshold_pct"] != 5:
                continue
            lo, hi = comparison["paired_bootstrap_95ci_pp"]
            lines.append(f"| {summary['job']} | {comparison['a']} vs {comparison['b']} | "
                         f"{comparison['a_only_success']} / {comparison['b_only_success']} | "
                         f"{comparison['success_difference_percentage_points']:.1f} ({lo:.1f}, {hi:.1f}) | "
                         f"{comparison['discordance_binomial_p_unadjusted']:.4g} |")
    lines += ["", "Source manifests, full frozen histories, threshold sensitivity (2/5/10/20%), and per-case data are saved alongside this report.",
              "Best-image values in cases.csv use retrospective ground truth selection, not an available stopping rule.",
              "No additional reconstructions were run by this script."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for problem, marker in (("ct", "o"), ("mri", "^")):
        for ax, other in zip(axes, METHODS[:2]):
            selected = [record for record in records if Path(record["case"]).parent.name in complete_jobs and record["methods"][METHODS[0]]["protocol"]["job"]["problem"] == problem]
            x = [100 * record["methods"][other]["history"][-1]["image_relative_l2"] for record in selected]
            y = [100 * record["methods"][METHODS[2]]["history"][-1]["image_relative_l2"] for record in selected]
            ax.scatter(x, y, label=problem.upper(), marker=marker, alpha=.6, s=22)
            ax.set(xscale="log", yscale="log", xlabel=f"{other}: relative image error (%)", ylabel="DLR D-Flow: relative image error (%)")
    for ax in axes:
        ax.plot([.3, 200], [.3, 200], "k--", lw=1)
        ax.axhline(5, color="gray", lw=.6)
        ax.axvline(5, color="gray", lw=.6)
        ax.legend()
        ax.grid(alpha=.15)
    fig.suptitle("Paired final errors; complete 30-case acquisitions only")
    fig.savefig(args.output / "paired_errors.png", dpi=160)
    plt.close(fig)
    assert max(row["max_increase_phi"] for row in rows) <= 1e-9
    print("\n".join(lines))


if __name__ == "__main__":
    main()

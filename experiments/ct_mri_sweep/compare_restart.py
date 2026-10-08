"""Compare frozen completed triples against the earlier sweep on paired cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
import torch

from .analyze_completed import METHODS, STEMS, TERMINAL

Record = dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("outputs/ct_mri_sweep_100steps_noiseless_20261007"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    snapshot = json.loads((args.audit / "snapshot.json").read_text())
    summaries = json.loads((args.audit / "summary.json").read_text())
    complete = {row["job"] for row in summaries if row["complete"]}
    snapshot_path = args.audit / "restart_snapshot.json"
    if snapshot_path.exists():
        saved = json.loads(snapshot_path.read_text())
        assert saved["previous_root"] == str(args.previous)
        records = saved["records"]
    else:
        records = []
        for record in snapshot["records"]:
            case = Path(record["case"]).relative_to(snapshot["root"])
            paths = [args.previous / case / f"{stem}.json" for stem in STEMS]
            if not all(path.exists() for path in paths):
                continue
            previous = dict(zip(METHODS, (json.loads(path.read_text()) for path in paths)))
            if not all(row["status"] in TERMINAL for row in previous.values()):
                continue
            records.append({"case": str(case), "complete_acquisition": case.parent.name in complete,
                            "old": previous, "new": record["methods"]})
        snapshot_path.write_text(json.dumps({"previous_root": str(args.previous), "records": records}, indent=2) + "\n")
    for record in records:
        for method in METHODS:
            old = dict(record["old"][method]["protocol"])
            new = dict(record["new"][method]["protocol"])
            old.pop("manifest_sha256")
            new.pop("manifest_sha256")
            assert old == new, (record["case"], method)
        old_inputs = torch.load(args.previous / record["case"] / "inputs.pt", map_location="cpu", weights_only=True)
        new_inputs = torch.load(Path(snapshot["root"]) / record["case"] / "inputs.pt", map_location="cpu", weights_only=True)
        for key in ("truth", "initial", "clean", "observed", "mask"):
            a, b = old_inputs[key], new_inputs[key]
            assert (a is None and b is None) or torch.equal(a, b), (record["case"], key)
    rng = np.random.default_rng(20261008)
    groups = {name: [record for record in records if Path(record["case"]).parent.name == name]
              for name in sorted({Path(record["case"]).parent.name for record in records})}
    groups["all_complete_acquisitions"] = [record for record in records if record["complete_acquisition"]]
    changes: list[Record] = []
    for group, rows in groups.items():
        for method in METHODS:
            old_errors = np.array([100 * row["old"][method]["history"][-1]["image_relative_l2"] for row in rows])
            new_errors = np.array([100 * row["new"][method]["history"][-1]["image_relative_l2"] for row in rows])
            difference = (new_errors <= 5).astype(float) - (old_errors <= 5)
            rescues, regressions = int((difference > 0).sum()), int((difference < 0).sum())
            bootstrap = difference[rng.integers(len(rows), size=(10000, len(rows)))].mean(1)
            changes.append({"group": group, "method": method, "n": len(rows),
                            "complete": group in complete or group == "all_complete_acquisitions",
                            "old_success5": int((old_errors <= 5).sum()), "new_success5": int((new_errors <= 5).sum()),
                            "rescues5": rescues, "regressions5": regressions,
                            "lower_image_error": int((new_errors < old_errors).sum()),
                            "old_mean_image_pct": float(old_errors.mean()), "new_mean_image_pct": float(new_errors.mean()),
                            "old_median_image_pct": float(np.median(old_errors)), "new_median_image_pct": float(np.median(new_errors)),
                            "success_delta_pp": float(100 * difference.mean()),
                            "paired_bootstrap_95ci_pp": (100 * np.quantile(bootstrap, [.025, .975])).tolist(),
                            "discordance_p_unadjusted": binomtest(rescues, rescues + regressions).pvalue if rescues + regressions else 1.})
    complete_records = [record for record in snapshot["records"] if Path(record["case"]).parent.name in complete]
    severe = [record for record in complete_records if record["methods"]["DLR D-Flow"]["history"][-1]["image_relative_l2"] > .2]
    stalls = [record["methods"]["DLR D-Flow"] for record in severe if record["methods"]["DLR D-Flow"]["status"] == "stalled"]
    failures = {"dlr_severe": len(severe), "also_exact_severe": sum(record["methods"]["D-Flow"]["history"][-1]["image_relative_l2"] > .2 for record in severe),
                "proxy_success_among_dlr_severe": sum(record["methods"]["P-Flow"]["history"][-1]["image_relative_l2"] <= .05 for record in severe),
                "severe_stalls": len(stalls), "median_stall_attempt": float(np.median([row["history"][-1]["step"] for row in stalls])),
                "severe_stalls_last_search_at_trial_budget": sum(row["history"][-1]["line_search_trials"] == snapshot["settings"]["optimizer"]["max_ls"] for row in stalls),
                "minimum_severe_measurement_residual_pct": min(100 * record["methods"]["DLR D-Flow"]["history"][-1]["measurement_relative_l2_clean"] for record in severe),
                "severe_ever_image_success5": sum(min(point["image_relative_l2"] for point in record["methods"]["DLR D-Flow"]["history"]) <= .05 for record in severe)}
    comparisons = [dict(comparison, job=row["job"]) for row in summaries if row["complete"]
                   for comparison in row["comparisons"] if comparison["threshold_pct"] == 5]
    previous_adjusted = 0.
    for index, comparison in enumerate(sorted(comparisons, key=lambda row: row["discordance_binomial_p_unadjusted"])):
        previous_adjusted = max(previous_adjusted, min(1., (len(comparisons) - index) * comparison["discordance_binomial_p_unadjusted"]))
        comparison["holm_p_within_complete_5pct_comparisons"] = previous_adjusted
    result = {"snapshot_created": snapshot["created"], "paired_cases": len(records), "validated_input_pairs": len(records),
              "changes": changes, "failure_patterns": failures, "method_comparisons_at_5pct": comparisons}
    (args.audit / "restart_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# Restart comparison: {snapshot['created']}", "",
             f"All {len(records)} old/new pairs have matching protocols except manifest hash and bitwise-equal truths, sources, clean/observed measurements and masks.",
             "Current sweep uses geometric halving for every method and previous-V sketch reuse for DLR. These changes are not isolated in the DLR comparison.", "",
             "| Acquisition | n | Method | <=5% old / new | Rescues / regressions | Median image % old / new |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in changes:
        lines.append(f"| {row['group']}{'' if row['complete'] else ' (partial)'} | {row['n']} | {row['method']} | "
                     f"{row['old_success5']} / {row['new_success5']} | {row['rescues5']} / {row['regressions5']} | "
                     f"{row['old_median_image_pct']:.2f} / {row['new_median_image_pct']:.2f} |")
    lines += ["", "## Changes across the same complete acquisitions", ""]
    for row in changes:
        if row["group"] == "all_complete_acquisitions":
            lo, hi = row["paired_bootstrap_95ci_pp"]
            lines.append(f"- {row['method']}: success change {row['success_delta_pp']:.1f} percentage points, "
                         f"paired bootstrap 95% interval [{lo:.1f}, {hi:.1f}], exploratory discordance p={row['discordance_p_unadjusted']:.4g}.")
    lines += ["", "## DLR failure pattern", "", json.dumps(failures, indent=2), "",
              "Intervals and tests are exploratory. The Holm adjustment in JSON covers the 15 pairwise method comparisons at the 5% threshold within complete acquisitions, not every analysis inspected.",
              "No extra reconstructions or GPU work were performed. A 30-trial failure establishes budget exhaustion, not that further trials would succeed."]
    (args.audit / "restart_comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"paired_cases": len(records), "overall_changes": [row for row in changes if row["group"] == "all_complete_acquisitions"], "failures": failures}, indent=2))


if __name__ == "__main__":
    main()

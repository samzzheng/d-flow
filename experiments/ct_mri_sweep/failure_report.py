"""Regenerate fixed-source precision and rank diagnostic plots and tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Record = dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/dlr_failure_tests_20261008"))
    args = parser.parse_args()
    root = args.root
    factors = json.loads((root / "factors/results.json").read_text())
    variants = ("rank8_random", "rank8_saved_v", "rank16_random", "rank32_random", "rank8_enrichment2")
    labels = ("r8 random", "r8 saved V", "r16 random", "r32 random", "r8 enrichment 2")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    rows = []
    for case in sorted(factors, key=lambda item: item["case"]):
        values = [100 * case["variants"][variant]["relative_error"] for variant in variants]
        axis = axes[1] if case["case"].endswith("sample_019") else axes[0]
        axis.plot(labels, values, marker="o", label=case["case"])
        rows.extend({"case": case["case"], "variant": variant,
                     **{key: case["variants"][variant][key] for key in
                        ("relative_error", "cosine", "subspace_relative_error", "coefficient_relative_error")}}
                    for variant in variants)
    for axis, title in zip(axes, ("Three failed reconstruction sources", "Accurate reconstruction control")):
        axis.set(title=title, ylabel="Relative source-gradient error (%)")
        axis.tick_params(axis="x", labelrotation=20)
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    fig.suptitle("Native CPU32 reference: same source, endpoint and terminal gradient\nThese are gradient errors, not reconstruction errors")
    fig.savefig(root / "factor_accuracy.png", dpi=160)
    plt.close(fig)
    with (root / "factor_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    summary: list[Record] = []
    for axis, case, label in zip(axes, ("r128_n3_ct10/sample_008", "r128_n2_mri4/sample_003"),
                                 ("CT, three circles, sample 008", "MRI, two circles, sample 003")):
        for flag, color in (("on", "tab:red"), ("off", "tab:blue")):
            result = json.loads((root / "gpu" / case / f"tf32_{flag}" / "result.json").read_text())
            for direction, style, name in (("refreshed_v8", "-", "DLR r8"), ("exact", "--", "exact")):
                record = result["directions"][direction]
                trials = record["trials"]
                axis.plot([trial["rate"] * record["norm"] for trial in trials],
                          [trial["phi"] - result["phi"] for trial in trials],
                          linestyle=style, marker=".", color=color, label=f"{name}, TF32 {flag}")
                summary.append({"case": case, "tf32": flag, "direction": direction,
                                "phi_at_same_source": result["phi"],
                                "endpoint_drift_from_saved": result["endpoint_drift_from_saved"],
                                "relative_gradient_error": record["relative_error_exact"],
                                "gradient_cosine": record["cosine_exact"],
                                "accepted_trial": record["first_armijo_trial"],
                                "unchanged_source_trial": record.get("unchanged_at_trial"),
                                "accepted_phi": trials[-1]["phi"] if record["first_armijo_trial"] else None})
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(title=label, xscale="log", xlabel="Proposed source displacement norm",
                 ylabel="Objective change (negative is better)")
        if "mri" in case:
            axis.set_yscale("symlog")
            axis.set_ylabel("Objective change (symmetric log scale)")
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    fig.suptitle("Halving search at saved failed sources\nEach curve uses its own precision-matched baseline objective")
    fig.savefig(root / "precision_line_search.png", dpi=160)
    plt.close(fig)
    (root / "gpu_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

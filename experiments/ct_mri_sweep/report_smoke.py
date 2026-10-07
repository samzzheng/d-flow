"""Describe late progress in the 100-update smoke histories without assuming convergence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.reconstruction_figures import METHODS
from .run import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    cases = sorted(args.out.glob("r*/sample_000"))
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 3 * len(cases)), squeeze=False)
    rows = []
    for case, panels in zip(cases, axes):
        for method in METHODS:
            path = case / (method.lower().replace("-", "_").replace(" ", "_") + ".json")
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            history = record["history"]
            final = history[-1]
            drops = None
            image_change = None
            if final["step"] >= 40:
                values = [history[index]["phi"] for index in (-41, -21, -1)]
                drops = [100 * (left - right) / max(abs(left), 1e-30) for left, right in zip(values, values[1:])]
                image_change = 100 * (final["image_relative_l2"] - history[-21]["image_relative_l2"])
            row = {"case": case.parent.name, "method": method, "status": record["status"],
                   "updates": final["step"], "image_error_percent": 100 * final["image_relative_l2"],
                   "residual_over_noise": final["residual_over_noise"],
                   "objective_drop_last_two_20_step_windows_percent": drops,
                   "image_error_change_last_20_steps_percentage_points": image_change,
                   "objective_plateau_at_100": bool(final["step"] == 100 and drops is not None and max(drops) < .1),
                   "mean_line_search_trials": sum(point.get("line_search_trials", 0) for point in history) / max(final["step"], 1)}
            rows.append(row)
            label = f"{method} ({record['status']}, {final['step']})"
            panels[0].plot([point["step"] for point in history], [100 * point["image_relative_l2"] for point in history], label=label)
            panels[1].semilogy([point["step"] for point in history], [point["residual_over_noise"] for point in history], label=label)
        panels[0].set(title=case.parent.name, ylabel="Relative image error (%)", xlabel="Optimizer attempt")
        panels[1].set(title="Observed residual / noise norm", ylabel="Residual / noise", xlabel="Optimizer attempt")
        panels[1].axhline(1, color="black", linewidth=.7, linestyle="--")
        for panel in panels:
            panel.grid(alpha=.2)
            panel.legend(fontsize=7)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.out / f"convergence.{extension}", dpi=140)
    plt.close(fig)
    write_json(args.out / "convergence_summary.json", {
        "interpretation": "One image from the three-circle model per resolution/operator; not population accuracy evidence. A step limit or stall is not convergence. Plateau means less than 0.1% objective decrease in each of the final two 20-update windows at update 100.",
        "results": rows})
    lines = ["# 100-update smoke checks", "",
             "One test image from the three-circle model at each resolution/operator. These are smoke checks, not 30-sample results.", "",
             "A numerical stall and a 100-update limit do not establish convergence. Objective plateau requires less than 0.1% decrease in each of the final two 20-update windows.", "",
             "| Case | Method | Updates | Status | Image error | Residual / noise | Objective decrease, last 20 | Plateau at 100 |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        drops = row["objective_drop_last_two_20_step_windows_percent"]
        decrease = f"{drops[-1]:.2f}%" if drops is not None else "n/a"
        lines.append(f"| {row['case']} | {row['method']} | {row['updates']} | {row['status']} | {row['image_error_percent']:.2f}% | {row['residual_over_noise']:.2f} | {decrease} | {row['objective_plateau_at_100']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

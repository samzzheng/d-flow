"""Plot measured LBFGS + Armijo iteration costs for the paired CPU pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    methods = ("pflow", "dflow", "fast_dlr")
    positions = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(10, 5.8))
    figure.patch.set_facecolor("#fafbfc")
    axis.set_facecolor("#fafbfc")
    width = .34
    for offset, key, label, color in (
        (-width / 2, "mean_seconds_excluding_extra_trial_flows", "Excluding extra Armijo trial flows", "#348a9b"),
        (width / 2, "mean_iteration_seconds", "Including Armijo", "#d88443"),
    ):
        values = [report["summary"][method][key] for method in methods]
        bars = axis.bar(positions + offset, values, width, label=label, color=color, zorder=3)
        axis.bar_label(bars, labels=[f"{value:.2f} s" for value in values], padding=6, fontsize=12, weight="bold")
    axis.set_xticks(positions, ["P-Flow", "D-Flow", "Fast DLR D-Flow"], fontsize=12)
    axis.set_ylabel("Seconds per iteration · lower is faster", fontsize=11)
    axis.set_ylim(0, max(report["summary"][method]["mean_iteration_seconds"] for method in methods) * 1.23)
    axis.grid(axis="y", alpha=.2, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left", fontsize=10)
    settings = report["settings"]
    figure.suptitle("Per-iteration speed: shared projected LBFGS + Armijo", fontsize=15, weight="bold", x=.1, ha="left", y=.98)
    figure.text(.1, .91, f"CPU · {settings['threads']} threads · non-attention {settings['resolution']}×{settings['resolution']} model · {settings['steps']} Euler steps · 3 test images", fontsize=11, color="#414853")
    figure.text(.1, .06,
                "Bars average the per-image iteration times. DLR: rank 8, factor refresh every 5 steps.\n"
                "15-step budget; DLR stopped after 8 steps on two images. Accepted forward solves stay counted.\n"
                "Small CPU pilot; not a GPU benchmark or a comparison of converged reconstruction quality.",
                fontsize=9, color="#414853", linespacing=1.5)
    figure.subplots_adjust(left=.1, right=.98, bottom=.25, top=.86)
    output = args.results.parent / "iteration_speed.png"
    figure.savefig(output, dpi=190, facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".svg"), facecolor=figure.get_facecolor())
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()

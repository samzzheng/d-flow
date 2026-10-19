"""Plot the bounded sample-0 comparison with explicit CPU/GPU provenance."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_parallel_bug_integrator as helper


def main():
    torch.set_num_threads(1)
    out = ROOT / "outputs/accuracy_fast_euler_2026_09_27"
    cpu = json.loads((out / "exact_from_z0/results.json").read_text())["runs"][0]
    if "final" not in cpu:
        raise ValueError("Wait for the CPU reconstruction run to finish")
    history = json.loads((ROOT / "outputs/dlr_speed_2026_09_27/short_solves/results.json").read_text())
    fast = next(run for run in history["runs"] if run["resolution"] == 250
                and run["sample"] == 0 and run["method"] == "fast")
    cache = torch.load(ROOT / "outputs/dlr_speed_2026_09_27/initial/points/res250_sample0.pt",
                       map_location="cpu", weights_only=True)
    data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu",
                      weights_only=True, mmap=True)["test"]
    truth = helper.bench.normalize_images(data[:1])
    markers = [{"step": step, "residual": cache["references"][step]["residual"],
                "image_relative_l2": float(helper.bench.relative_l2(
                    cache["references"][step]["x"].clamp(0, 1), truth))}
               for step in (0, 10)]
    comparison = {"scope": "sample 0, 250px no-attention, 1% noise, 100 Euler flow steps",
                  "limitation": "CPU exact versus historical GPU fast; shared image/noise seeds, with forward/Radon roundoff differences; exploratory matched-problem comparison",
                  "cpu_exact": {"final": cpu["final"], "stop_reason": cpu["stop_reason"]},
                  "gpu_fast": {"final": fast["final"], "stop_reason": fast["stop_reason"]},
                  "gpu_exact_markers": markers}
    for name, run in (("cpu_exact", cpu), ("gpu_fast", fast)):
        comparison[name]["first_step_at_2pct_residual"] = next(
            (row["step"] for row in run["steps"] if row["residual"] <= .02), None)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, metric, title in zip(axes, ("image_relative_l2", "residual"),
                                 ("Reconstruction error", "Measurement residual")):
        ax.plot([row["step"] for row in cpu["steps"]],
                [100 * row[metric] for row in cpu["steps"]],
                color="#2467a3", linewidth=2, label="Exact Euler D-Flow · CPU")
        ax.plot([row["step"] for row in fast["steps"]],
                [100 * row[metric] for row in fast["steps"]],
                color="#d66b29", linewidth=2, label="Fast DLR Euler · historical GPU")
        ax.scatter([row["step"] for row in markers],
                   [100 * row[metric] for row in markers], marker="x", s=65,
                   linewidths=2, color="#202124", zorder=5, label="Exact Euler · cached GPU")
        ax.set(title=title, xlabel="Optimizer step", ylabel="Relative L2 (%)", yscale="log")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(True, which="both", alpha=.18)
    axes[1].axhline(1, color="#888888", linestyle=":", linewidth=1)
    axes[1].text(.3, 1.06, "1% measurement noise", color="#666666", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, .915))
    fig.suptitle("250 × 250 no-attention · sample 0 · 1% noise", y=.98)
    fig.text(.5, .025,
        "Same model, image/noise seeds, source and Armijo settings; fast factors rebuilt at every gradient.\n"
        "CPU/GPU forward and Radon roundoff remain: this comparison does not establish asymptotic accuracy.",
        ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .13, 1, .82))
    fig.savefig(out / "sample0_accuracy.png", dpi=180)
    fig.savefig(out / "sample0_accuracy.pdf")
    plt.close(fig)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

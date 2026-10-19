"""Plot the paired inverse-solve audit and CPU fixed-source refinement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/rk4_accuracy_2026_09_27")
    args = parser.parse_args()
    audit = json.loads((args.root / "completed_runs.json").read_text())
    refinement = json.loads((args.root / "refinement/results.json").read_text())
    pairs = audit["groups"]["armijo_local"]["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    ax = axes[0]
    euler = np.array([100 * float(p["euler"]["image_relative_l2"]) for p in pairs])
    rk4 = np.array([100 * float(p["rk4"]["image_relative_l2"]) for p in pairs])
    failures = np.maximum(euler, rk4) > 50
    ax.loglog([.7, 160], [.7, 160], color="0.55", linestyle="--", linewidth=1, label="Equal image error")
    ax.scatter(euler[~failures], rk4[~failures], c="#277F8E", s=45,
               label="Both image errors ≤50%", zorder=3)
    ax.scatter(euler[failures], rk4[failures], c="#B64C34", marker="x", s=60,
               label="At least one error >50%", zorder=3)
    for p, x, y in zip(pairs, euler, rk4):
        if p["sample"] in (1, 20, 25, 26):
            offset = (-28, 7) if p["sample"] == 26 else (6, 3)
            ax.annotate(f"s{p['sample']}", (x, y), xytext=offset, textcoords="offset points", fontsize=9)
    ax.text(.03, .96, "Above diagonal: Euler better", transform=ax.transAxes, va="top", fontsize=10)
    ax.set(xlim=(.7,160), ylim=(.7,160), xlabel="Euler image relative L2 (%)",
           ylabel="RK4 image relative L2 (%)", title=f"Noisy reconstruction: {len(pairs)} completed local pairs")
    ax.grid(which="major", alpha=.2)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    for method, label, color, marker in (("forward_euler", "Euler", "#277F8E", "o"),
                                         ("rk4", "RK4", "#A46A2B", "s")):
        rows = [v for k,v in refinement["runs"].items() if k.startswith(method+"_")
                and v.get("relative_to_reference", 0)>0]
        rows.sort(key=lambda row:row["velocity_evaluations"])
        ax.loglog([r["velocity_evaluations"] for r in rows],
                  [100*r["relative_to_reference"] for r in rows],
                  marker=marker, color=color, label=label)
        for row in rows:
            offset = (5, 5 if method=="forward_euler" else -13)
            if method == "forward_euler" and row["steps"] == 200:
                offset = (-58, -15)
            elif method == "rk4" and row["steps"] == 50:
                offset = (5, 5)
            elif method == "rk4" and row["steps"] == 400:
                offset = (-60, 6)
            ax.annotate(f"{row['steps']} steps", (row["velocity_evaluations"],100*row["relative_to_reference"]),
                        xytext=offset, textcoords="offset points", fontsize=8)
    reference = refinement["protocol"]["reference"].replace("_", "-").upper()
    ax.set(xlabel="Velocity evaluations per forward solve",
           ylabel=f"Terminal-state error vs {reference} (%)",
           title="Fixed source: sample 0, Euler iterate 10")
    ax.grid(which="both", alpha=.2)
    ax.legend()
    fig.suptitle("250 × 250 attention-free D-Flow: reconstruction and ODE accuracy", fontsize=13)
    for suffix in ("png", "pdf"):
        path = args.root / f"rk4_accuracy_comparison.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

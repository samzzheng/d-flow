"""Summarize actual first-crossing states from the CPU Euler/RK4 pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/rk4_noise_floor_2026_09_27")
    parser.add_argument("--sample", type=int, default=1)
    args = parser.parse_args()
    runs = {}
    for method, directory in (("Euler","euler"),("RK4","rk4")):
        path = args.root / f"sample{args.sample}_{directory}" / "results.json"
        if path.exists():
            runs[method] = json.loads(path.read_text())
    summary = {"sample":args.sample,"methods":{},"comparisons":{},
        "comparisons_clamped_stopping":{},
        "caveat":("One pilot sample selected because both historical methods reached strict noise floor; no population claim"
                  if args.sample == 1 else "One sample specified by CLI; no population claim"),
        "integrator_budget":"100 flow steps for both: 100 Euler versus 400 RK4 forward network calls"}
    for method,run in runs.items():
        clamped_crossings = {}
        for tau in (1.,1.05,1.1):
            crossing = next((row for row in run["history"] if row["clamped_noisy_residual"]
                            <= tau*run["protocol"]["noise_relative_observed"]),None)
            if crossing is not None:
                clamped_crossings[str(tau)] = crossing
        summary["methods"][method] = {"protocol":run["protocol"],"stop_reason":run["stop_reason"],
            "first_crossings":run["first_crossings"],"first_clamped_crossings":clamped_crossings,
            "latest":run["history"][-1]}
    if len(runs)==2:
        for tau in (1.,1.05,1.1):
            if all(str(tau) in run["first_crossings"] for run in runs.values()):
                euler, rk4 = [runs[m]["first_crossings"][str(tau)] for m in ("Euler","RK4")]
                summary["comparisons"][str(tau)] = {
                    "rk4_minus_euler_image_error_percentage_points":100*(rk4["image_relative_l2"]-euler["image_relative_l2"]),
                    "rk4_minus_euler_psnr_db":rk4["psnr_db"]-euler["psnr_db"],
                    "rk4_image_error_over_euler":rk4["image_relative_l2"]/euler["image_relative_l2"],
                    "euler_accepted_index":euler["accepted_index"],"rk4_accepted_index":rk4["accepted_index"]}
            clamped = [summary["methods"][m]["first_clamped_crossings"].get(str(tau)) for m in ("Euler","RK4")]
            if all(row is not None for row in clamped):
                euler,rk4=clamped
                summary["comparisons_clamped_stopping"][str(tau)] = {
                    "rk4_minus_euler_image_error_percentage_points":100*(rk4["image_relative_l2"]-euler["image_relative_l2"]),
                    "rk4_minus_euler_psnr_db":rk4["psnr_db"]-euler["psnr_db"],
                    "euler_accepted_index":euler["accepted_index"],"rk4_accepted_index":rk4["accepted_index"]}
    (args.root / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(11.0,4.6),constrained_layout=True)
    for method,color in (("Euler","#277F8E"),("RK4","#A46A2B")):
        if method not in runs:continue
        rows=runs[method]["history"]
        steps=[r["accepted_index"] for r in rows]
        axes[0].semilogy(steps,[r["residual_over_noise"] for r in rows],color=color,label=method)
        axes[0].semilogy(steps,[r["clamped_noisy_residual"] / runs[method]["protocol"]["noise_relative_observed"]
                               for r in rows],color=color,linestyle=":",alpha=.75,label=method+" after clamp")
        axes[1].semilogy(steps,[100*r["image_relative_l2"] for r in rows],color=color,label=method)
        for tau,marker in ((1.,"o"),(1.05,"s"),(1.1,"^")):
            row=runs[method]["first_crossings"].get(str(tau))
            if row:
                for axis,value in ((axes[0],row["residual_over_noise"]),(axes[1],100*row["image_relative_l2"])):
                    axis.scatter(row["accepted_index"],value,c=color,marker=marker,s=48,zorder=3)
    axes[0].axhline(1.,color="0.25",linestyle="--",linewidth=1,label="Exact noise norm")
    axes[0].set(ylabel="Measurement residual / actual noise norm",xlabel="Accepted optimizer steps",
                title="Stop at the first accepted crossing")
    axes[1].set(ylabel="Clamped image relative L2 (%)",xlabel="Accepted optimizer steps",
                title="Image accuracy at recorded source states")
    for axis in axes:
        axis.grid(which="both",alpha=.2)
        axis.legend()
    handles, labels = axes[1].get_legend_handles_labels()
    for tau, marker in ((1., "o"), (1.05, "s"), (1.1, "^")):
        handles.append(Line2D([], [], color="0.4", linestyle="none", marker=marker))
        labels.append(f"First raw crossing: {tau:g} × noise")
    axes[1].legend(handles, labels, fontsize=8)
    fig.suptitle(f"250 × 250 non-attention D-Flow, sample {args.sample}: 100 steps per integrator\n"
                 "First-only L-BFGS initialization; capped Armijo; 1% measurement noise",fontsize=12)
    for extension in ("png","pdf"):
        fig.savefig(args.root / f"noise_floor_curves.{extension}",dpi=180,bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    for method, color in (("Euler", "#277F8E"), ("RK4", "#A46A2B")):
        if method not in runs:
            continue
        rows = [row for row in runs[method]["history"] if row["accepted_index"] >= 35]
        steps = [row["accepted_index"] for row in rows]
        axes[0].plot(steps, [row["residual_over_noise"] for row in rows], color=color, label=method)
        axes[1].plot(steps, [100*row["image_relative_l2"] for row in rows], color=color, label=method)
        for tau, marker in ((1., "o"), (1.05, "s")):
            row = runs[method]["first_crossings"].get(str(tau))
            if row is not None:
                for axis, value in ((axes[0], row["residual_over_noise"]),
                                    (axes[1], 100*row["image_relative_l2"])):
                    axis.scatter(row["accepted_index"], value, c=color, marker=marker, s=48, zorder=3)
    axes[0].axhline(1., color="0.25", linestyle="--", linewidth=1, label="Primary: 1 × noise")
    axes[0].axhline(1.05, color="0.55", linestyle=":", linewidth=1, label="Secondary: 1.05 × noise")
    axes[0].set(ylim=(.995, 1.06), ylabel="Raw residual / actual noise norm",
                title="Approach to the strict noise threshold")
    axes[1].set(ylim=(.8, 1.8), ylabel="Clamped image relative L2 (%)",
                title="Late image accuracy at the same saved states")
    for axis in axes:
        axis.set(xlim=(35, max(40, max(run["history"][-1]["accepted_index"] for run in runs.values()))+2),
                 xlabel="Accepted optimizer steps")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    for tau, marker in ((1., "o"), (1.05, "s")):
        handles.append(Line2D([], [], color="0.4", linestyle="none", marker=marker))
        labels.append(f"First raw crossing: {tau:g} × noise")
    axes[1].legend(handles, labels, fontsize=8)
    fig.suptitle(f"250 × 250 sample {args.sample}: late convergence, 100 steps per integrator\n"
                 "Display zoom only; points outside these ranges are clipped, stopping thresholds unchanged",
                 fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(args.root / f"noise_floor_zoom.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["comparisons"],indent=2))


if __name__=="__main__":
    main()

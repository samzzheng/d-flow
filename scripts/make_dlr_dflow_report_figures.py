"""Figures for the 2026-09-01 DLR-D-Flow report."""
from __future__ import annotations
import json, re, glob, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "outputs/reports/2026-09-01_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
S1, S2, S3 = "#245b78", "#7a4b2d", "#4d7358"

# ── parse the no-line-search 1200-step noise sweep ────────────────────────
rows = []
for f in sorted(glob.glob(str(ROOT / "outputs/dflow_sparse_ct_no_line_search_1200_noise_sweep/noise_*/res*.log"))):
    noise = int(re.search(r"noise_(\d+)", f).group(1))
    for line in open(f):
        m = re.match(r"Completed res=(\d+) sample=(\d+) method=(\S+): "
                     r"image_rel=(\S+) measurement_rel=(\S+) time=(\S+)s", line)
        if m:
            rows.append(dict(noise=noise, res=int(m.group(1)), sample=int(m.group(2)),
                             method=m.group(3), image_rel=float(m.group(4)),
                             meas_rel=float(m.group(5)), time=float(m.group(6))))

NOISES = [1, 5, 10]

# Figures 1 and 2 (noise error / discrepancy) were removed at review: not needed.

# Figure 3 — DLR-D-Flow per-iteration speed and memory (from the speed benchmark)
speed_path = ROOT / "outputs/dlr_dflow_noise_benchmark/iteration_speed.json"
if speed_path.exists():
    speed = json.load(open(speed_path))
    res = [s["resolution"] for s in speed]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, colour, label in [("dflow", S2, "D-Flow (backprop through ODE)"),
                               ("dlr", S1, "DLR-D-Flow (rank 8)")]:
        t = [s.get(f"{key}_sec_per_iter") for s in speed]
        m = [s.get(f"{key}_peak_gib") for s in speed]
        xs = [r for r, y in zip(res, t) if y is not None]
        ax[0].plot(xs, [y for y in t if y is not None], "o-", lw=2, ms=7,
                   color=colour, label=label, mec="white", mew=1.2)
        xs = [r for r, y in zip(res, m) if y is not None]
        ax[1].plot(xs, [y for y in m if y is not None], "o-", lw=2, ms=7,
                   color=colour, label=label, mec="white", mew=1.2)
        for r, y in zip(res, t):
            if y is None:
                ax[0].annotate("OOM", (r, ax[0].get_ylim()[1]), ha="center",
                               fontsize=9, color=S2, fontweight="bold")
    ax[0].set_ylabel("seconds per outer iteration"); ax[0].set_title("Per-iteration wall-clock", fontsize=11)
    ax[1].set_ylabel("peak GPU memory (GiB)"); ax[1].set_title("Peak memory", fontsize=11)
    for a in ax:
        a.set_xlabel("resolution"); a.set_xticks(res); a.grid(alpha=.3); a.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Cost per outer iteration, 100-step flow, single inverse solve", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "iteration_speed.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print("saved iteration_speed.png")

json.dump(rows, open(ASSETS / "noise_rows.json", "w"), indent=1)
print("saved noise figures to", ASSETS)

# ── Best / median / worst DLR-D-Flow reconstructions per resolution ──────
import torch, torch.nn.functional as F, math, statistics as stat
import cmocean
DENSE, BALANCE = cmocean.cm.dense_r, cmocean.cm.balance
BENCH = ROOT / "outputs/dlr_dflow_sparse_ct_benchmark"
rel = json.load(open(BENCH / "results_rel_l2.json"))
angles = torch.linspace(0, math.pi, 16)[:-1]

def radon(x, a=angles):
    b, c, h, w = x.shape
    k = a.numel()
    rep = x[:, None].expand(b, k, c, h, w).reshape(b * k, c, h, w)
    ab = a.repeat(b)
    cs, sn = torch.cos(ab), torch.sin(ab)
    th = torch.zeros(b * k, 2, 3, dtype=x.dtype)
    th[:, 0, 0] = cs; th[:, 0, 1] = -sn
    th[:, 1, 0] = sn; th[:, 1, 1] = cs
    g = F.affine_grid(th, rep.shape, align_corners=False)
    r = F.grid_sample(rep, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return r.mean(-2).reshape(b, k, c, w).permute(0, 2, 1, 3)

np_ = lambda t: t.detach().squeeze().float().cpu().numpy()

for res in RESOLUTIONS if 'RESOLUTIONS' in dir() else [64, 96, 128, 160]:
    g = sorted((r for r in rel if r["resolution"] == res), key=lambda r: r["image_rel_l2"])
    picks = {"best": g[0], "median": g[len(g) // 2], "worst": g[-1]}
    for label, row in picks.items():
        blob = torch.load(BENCH / f"sample_{res}_{row['test_index']}.pt", weights_only=False)
        xs, xh, yo = blob["x_star"], blob["x_hat"], blob["y_obs"]
        yh = radon(xh)
        panels = [
            (np_(xs), "Ground truth", DENSE, 0.0, 1.0),
            (np_(xh), "DLR-D-Flow reconstruction", DENSE, 0.0, 1.0),
            (np_(xh) - np_(xs), "Image error", BALANCE, -1.0, 1.0),
            (np_(yo), "Observed Ax", DENSE, None, None),
            (np_(yh), "Reconstructed Ax", DENSE, None, None),
            (np_(yh) - np_(yo), "Measurement residual", BALANCE, -1.0, 1.0),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.6))
        for ax, (im, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
            r_ = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if cmap is BALANCE:
                cb = fig.colorbar(r_, ax=ax, orientation="vertical",
                                  ticks=[-1.0, 0.0, 1.0], fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=7)
        fig.suptitle(
            f"{label.capitalize()} of 5 — circles-{res} n1, test image {row['test_index']}\n"
            f"solution rel. $L_2$ = {100*row['image_rel_l2']:.2f}%     "
            f"measurement rel. $L_2$ = {100*row['meas_rel_l2']:.2f}%",
            fontsize=10.5)
        fig.tight_layout()
        out = ASSETS / f"recon_res{res}_{label}.png"
        fig.subplots_adjust(top=0.86)
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out.name}  ({100*row['image_rel_l2']:.2f}%)")

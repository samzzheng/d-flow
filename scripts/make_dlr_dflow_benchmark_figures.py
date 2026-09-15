"""Figures for the DLR-D-Flow sparse-CT resolution benchmark.

Reconstruction panels follow the repo convention in
scripts/benchmark_dflow_sparse_ct.py:save_figure. DLR-D-Flow solutions only.
"""
from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import cmocean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dlr_dflow_sparse_ct_benchmark"
DENSE, BALANCE = cmocean.cm.dense_r, cmocean.cm.balance
RESOLUTIONS = [64, 96, 128, 160]

rows = list(csv.DictReader(open(OUT / "results.csv")))
to_np = lambda t: t.detach().squeeze().float().cpu().numpy()


def save_reconstruction(res: int, index: int, meta: dict) -> Path:
    blob = torch.load(OUT / f"sample_{res}_{index}.pt", weights_only=False)
    x_star, x_hat, y_obs = blob["x_star"], blob["x_hat"], blob["y_obs"]
    angles = torch.linspace(0, math.pi, int(meta["angles"]) + 1)[:-1]
    import torch.nn.functional as F

    def radon(x):
        b, c, h, w = x.shape
        k = angles.numel()
        rep = x[:, None].expand(b, k, c, h, w).reshape(b * k, c, h, w)
        ab = angles.repeat(b)
        cos, sin = torch.cos(ab), torch.sin(ab)
        th = torch.zeros(b * k, 2, 3, dtype=x.dtype)
        th[:, 0, 0] = cos; th[:, 0, 1] = -sin
        th[:, 1, 0] = sin; th[:, 1, 1] = cos
        grid = F.affine_grid(th, rep.shape, align_corners=False)
        rot = F.grid_sample(rep, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        return rot.mean(-2).reshape(b, k, c, w).permute(0, 2, 1, 3)

    y_hat = radon(x_hat)
    panels = [
        (to_np(x_star), "Ground truth", DENSE, 0.0, 1.0),
        (to_np(x_hat), f"DLR-D-Flow reconstruction\nMSE={float(meta['image_mse']):.3e}   "
                       f"PSNR={float(meta['psnr_db']):.1f} dB", DENSE, 0.0, 1.0),
        (to_np(x_hat) - to_np(x_star), "Image error", BALANCE, -1.0, 1.0),
        (to_np(y_obs), "Observed Ax", DENSE, None, None),
        (to_np(y_hat), f"Reconstructed Ax\nMSE={float(meta['measurement_mse']):.3e}",
         DENSE, None, None),
        (to_np(y_hat) - to_np(y_obs), "Measurement residual", BALANCE, -1.0, 1.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
    for axis, (image, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        rendered = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.set_xticks([]); axis.set_yticks([])
        if cmap is BALANCE:
            cb = fig.colorbar(rendered, ax=axis, orientation="vertical",
                              ticks=[-1.0, 0.0, 1.0], fraction=0.046, pad=0.04)
            cb.set_label("Error")
    fig.suptitle(f"DLR-D-Flow, sparse-view CT ({meta['angles']} angles), "
                 f"circles-{res} n1 test image {index}  ·  "
                 f"rank {meta['rank']}, {meta['flow_steps']}-step flow, "
                 f"{meta['outer_steps']} outer steps", fontsize=11)
    fig.tight_layout()
    path = OUT / f"reconstruction_res{res}_idx{index}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# one reconstruction figure per sample, every resolution
for res in RESOLUTIONS:
    group = sorted((r for r in rows if int(r["resolution"]) == res),
                   key=lambda r: int(r["test_index"]))
    for meta in group:
        print("saved", save_reconstruction(res, int(meta["test_index"]), meta))

# summary: per-sample quality, and the misfit/PSNR relation
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for res in RESOLUTIONS:
    vals = sorted(float(r["psnr_db"]) for r in rows if int(r["resolution"]) == res)
    ax[0].scatter([res] * len(vals), vals, s=55, zorder=3, edgecolor="white", linewidth=1.2)
    ax[0].scatter([res], [st.median(vals)], marker="_", s=520, color="k", zorder=4)
ax[0].axhline(25, ls="--", lw=1.4, color="0.45")
ax[0].text(63, 25.7, "25 dB success threshold", fontsize=8, color="0.35")
ax[0].set_xticks(RESOLUTIONS); ax[0].set_xlabel("resolution")
ax[0].set_ylabel("PSNR (dB)")
ax[0].set_title("Per-sample reconstruction quality\n(black bar = median of 5)", fontsize=10)
ax[0].grid(alpha=0.3, zorder=0)

for res in RESOLUTIONS:
    g = [r for r in rows if int(r["resolution"]) == res]
    ax[1].scatter([float(r["measurement_mse"]) for r in g],
                  [float(r["psnr_db"]) for r in g],
                  s=55, label=f"{res}", zorder=3, edgecolor="white", linewidth=1.2)
ax[1].set_xscale("log")
ax[1].axhline(25, ls="--", lw=1.4, color="0.45")
ax[1].axvline(1e-4, ls=":", lw=1.4, color="0.45")
ax[1].set_xlabel("measurement MSE  (observable without ground truth)")
ax[1].set_ylabel("PSNR (dB)")
ax[1].set_title("Failures are detectable from the data misfit alone", fontsize=10)
ax[1].legend(title="resolution", fontsize=8, title_fontsize=8, frameon=False)
ax[1].grid(alpha=0.3, zorder=0)
fig.tight_layout()
path = OUT / "summary.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print("saved", path)

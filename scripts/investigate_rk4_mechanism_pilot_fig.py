"""Sample-1 figure: pilot Euler-100 vs RK4-100 semi-convergence with the
visible/invisible split, plus the warm-start swaps."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path("outputs/rk4_mechanism_2026_09_27")
dec = {m: json.load(open(OUT / f"decompose/pilot_s1_{m}.json"))["rows"] for m in ("euler", "rk4")}
swap = {k: json.load(open(OUT / f"swap/{k}/results.json"))["history"] for k in ("s1_euler100_from_R94", "s1_rk4100_from_E96")}
col = {"euler": "#1f77b4", "rk4": "#d62728"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for m, rows in dec.items():
    r = np.array([x["residual_over_noise"] for x in rows]); keep = r < 2.5
    img = 100 * np.array([x["clamped_image_rel"] for x in rows])
    vis = 100 * np.array([x["clamped_visible_rel"] for x in rows])
    inv = 100 * np.array([x["clamped_invisible_rel"] for x in rows])
    lab = {"euler": "Euler-100", "rk4": "RK4-100"}[m]
    axes[0].plot(r[keep], img[keep], "-", color=col[m], lw=1.8, label=f"{lab}: total")
    axes[0].plot(r[keep], inv[keep], "--", color=col[m], lw=1.2, label=f"{lab}: null-space part")
    axes[0].plot(r[keep], vis[keep], ":", color=col[m], lw=1.2, label=f"{lab}: visible part")
    steps = np.arange(len(rows))
    axes[1].plot(steps, img, "-", color=col[m], lw=1.8, label=lab)
for k, h in swap.items():
    r = np.array([x["residual_over_noise"] for x in h]); img = 100 * np.array([x["image_relative_l2"] for x in h])
    m = "euler" if "euler100_from" in k else "rk4"
    lab = "Euler-100 warm-started from RK4's solution" if m == "euler" else "RK4-100 warm-started from Euler's solution"
    axes[2].plot(r, img, "-o", ms=3, color=col[m], label=lab)
    axes[2].annotate("start", (r[0], img[0]), textcoords="offset points", xytext=(5, 5), fontsize=8)
for ax in (axes[0], axes[2]):
    ax.invert_xaxis(); ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("residual / noise norm  (optimization runs right to left)")
axes[0].set_xlim(2.5, 0.97); axes[0].set_ylim(0, 6)
axes[0].set_ylabel("image relative L2 error [%]"); axes[0].set_title("250px sample 1: error split at matched data misfit")
axes[1].set_xlabel("accepted step"); axes[1].set_ylabel("image relative L2 error [%]"); axes[1].set_ylim(0, 6)
axes[1].set_title("same runs vs iteration (both stop at residual = noise)")
axes[2].set_xlim(1.12, 0.99); axes[2].set_ylim(0.8, 2.2)
axes[2].set_ylabel("image relative L2 error [%]"); axes[2].set_title("warm-start swaps: each generator returns to its own level")
for ax in axes: ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "pilot_s1_mechanism.png", dpi=130)
print("saved")

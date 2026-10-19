"""MP4 of Jacobian singular values along the flow: D-Flow vs DLR-D-Flow.

For one source z on circles-64, both sensitivities are followed through the
same N=100 forward-Euler steps:

  D-Flow      the exact dense J_t = dx_t/dz, J_{i+1} = J_i + h A_{t_i} J_i,
              J_0 = I (what backpropagation through the flow differentiates)
  DLR-D-Flow  the low-rank model J_t ~= alpha_t I + U_t S_t V_t^T evolved by
              dlr_dflow.enriched (rank 8, 32 trace probes, 4 enrichment
              probes -- the benchmark settings)

and every one of the N+1 sorted spectra is recorded. The video has three
panels per frame: D-Flow alone, DLR-D-Flow alone, and both overlaid.

DLR's spectrum has at most 2r values off the plateau: every direction
orthogonal to span(U_t, V_t) is scaled by exactly alpha_t, so the other d - 2r
singular values all equal |alpha_t|. The benchmarks then set alpha_1 = 0 in
the gradient (zero_terminal_baseline=True); the last frame shows the evolved
alpha_1 before that override.

Usage: python scripts/jacobian_singular_values_dflow_dlr_video.py [--device cuda:1]
       [--sample-seed 0] [--reuse]   # --reuse skips recomputation if the .npz exists
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dlr_dflow.dlr_dflow as dd  # noqa: E402
import dlr_dflow.tangent_unet as tangent  # noqa: E402
from dlr_dflow.enriched import evolve_trace_centered_enriched  # noqa: E402
from dlr_dflow.graphed_products import GraphedAProducts  # noqa: E402
from dlr_dflow.tangent_unet import unet_jvp  # noqa: E402

import imageio_ffmpeg  # noqa: E402
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

OUT = ROOT / "outputs" / "jacobian_singular_values"
RESOLUTION, N_CIRCLES, EPOCH, N_STEPS, CHUNK = 64, 1, 50, 100, 512
RANK, TRACE_PROBES, OVERSAMPLING, ENRICHMENT = 8, 32, 8, 4
COLOURS = {"dflow": "C0", "dlr": "C3"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--reuse", action="store_true")
    return p.parse_args()


def configure_kernels():
    # Same resolution-64 kernel choices as the benchmarks.
    dd.HAND_JVP = True
    dd.HAND_VJP = True
    tangent.ATTENTION_MODE = "moments"
    tangent.ATTENTION_JVP_MODE = "chunked"
    tangent.ATTENTION_VJP_MODE = "chunked"
    tangent.GN_STATS_MODE = "native"


def exact_spectra(model, z, device):
    """Sorted singular values of the exact J_t at every step 0..N."""
    H = W = RESOLUTION
    d = H * W
    x = z.clone()
    J = torch.eye(d, device=device)
    h = 1.0 / N_STEPS
    sv = np.zeros((N_STEPS + 1, d), dtype=np.float64)
    for i in range(N_STEPS):
        sv[i] = torch.linalg.svdvals(J).cpu().numpy()
        cols, primal = [], None
        for s in range(0, d, CHUNK):
            with torch.no_grad():
                p, out = unet_jvp(model, i * h, x, J[:, s:s + CHUNK].T.reshape(-1, 1, H, W))
            primal = p if primal is None else primal
            cols.append(out.reshape(-1, d).T)
        J = J + h * torch.cat(cols, dim=1)
        x = x + h * primal.reshape(1, 1, H, W)
    sv[N_STEPS] = torch.linalg.svdvals(J).cpu().numpy()
    return sv, x


def dlr_spectra(model, z, device):
    """Sorted singular values of alpha_t I + U_t S_t V_t^T at every step 0..N."""
    d = RESOLUTION * RESOLUTION
    sv = np.zeros((N_STEPS + 1, d), dtype=np.float64)
    alphas = np.zeros(N_STEPS + 1)
    identity = torch.eye(d, device=device)

    def record(step, alpha, factors):
        U, S, V = factors
        M = alpha * identity + U @ S @ V.T
        sv[step] = torch.linalg.svdvals(M).cpu().numpy()
        alphas[step] = alpha

    products = GraphedAProducts(model, torch.zeros_like(z))
    x1, _, _ = evolve_trace_centered_enriched(
        model, z, N_STEPS, rank=RANK, trace_probes=TRACE_PROBES,
        initialization_oversampling=OVERSAMPLING, enrichment_probes=ENRICHMENT,
        factor_stride=1, products=products, zero_terminal_baseline=False,
        factor_callback=record)
    return sv, alphas, x1


def render(sv_exact, sv_dlr, alphas, seed, path):
    d = sv_exact.shape[1]
    idx = np.arange(1, d + 1)
    floor = 1e-9
    positive = np.concatenate([sv_exact[sv_exact > floor], sv_dlr[sv_dlr > floor]])
    ylim = (positive.min() * 0.5, positive.max() * 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True)
    panels = [("D-Flow: exact $J_t$", ("dflow",)),
              (r"DLR-D-Flow: $\alpha_t I + U_t S_t V_t^\top$", ("dlr",)),
              ("D-Flow and DLR-D-Flow", ("dflow", "dlr"))]
    labels = {"dflow": "D-Flow (exact)", "dlr": f"DLR-D-Flow (rank {RANK})"}
    lines = []
    for ax, (name, keys) in zip(axes, panels):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, d)
        ax.set_ylim(*ylim)
        ax.set_xlabel("index $i$ (sorted, log scale)")
        ax.axvline(RANK, color="0.6", lw=0.9, ls=":")
        ax.grid(True, which="major", alpha=0.3)
        ax.set_title(name, fontsize=11)
        for key in keys:
            (line,) = ax.plot([], [], "-", color=COLOURS[key], lw=1.5, label=labels[key])
            lines.append((line, key))
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    axes[0].set_ylabel(r"singular value $\sigma_i$")
    axes[1].text(RANK * 1.08, (ylim[0] * ylim[1]) ** 0.5, f"r = {RANK}", fontsize=8,
                 color="0.45")
    note = axes[1].text(0.97, 0.95, "", transform=axes[1].transAxes, ha="right", va="top",
                        fontsize=9, color=COLOURS["dlr"])
    header = fig.suptitle("", fontsize=12)
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.06, right=0.99, wspace=0.08)
    data = {"dflow": sv_exact, "dlr": sv_dlr}

    def update(frame):
        for line, key in lines:
            line.set_data(idx, np.maximum(data[key][frame], floor))
        note.set_text(rf"$\alpha_t$ = {alphas[frame]:.4g}"
                      + ("\n(benchmarks use $\\alpha_1 = 0$)" if frame == N_STEPS else ""))
        header.set_text(
            f"Jacobian singular values along the flow — circles-64, source seed {seed}, "
            f"{N_STEPS} Euler steps\nstep {frame}/{N_STEPS},  t = {frame / N_STEPS:.2f}")
        return [line for line, _ in lines] + [note, header]

    ani = animation.FuncAnimation(fig, update, frames=N_STEPS + 1, blit=False)
    ani.save(path, writer=animation.FFMpegWriter(fps=12, bitrate=2400))
    plt.close(fig)


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"singular_values_dflow_vs_dlr_res{RESOLUTION}_n{N_STEPS}_sample{args.sample_seed}"
    cache = OUT / f"{stem}.npz"
    if args.reuse and cache.exists():
        saved = np.load(cache)
        sv_exact, sv_dlr, alphas = saved["dflow"], saved["dlr"], saved["alpha"]
    else:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        configure_kernels()
        model, _ = dd.load_flow_model(RESOLUTION, N_CIRCLES, EPOCH, str(device))
        g = torch.Generator(device=device).manual_seed(args.sample_seed)
        z = torch.randn(1, 1, RESOLUTION, RESOLUTION, generator=g, device=device)
        t0 = time.time()
        sv_dlr, alphas, x_dlr = dlr_spectra(model, z, device)
        print(f"DLR spectra: {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        sv_exact, x_exact = exact_spectra(model, z, device)
        print(f"exact spectra: {time.time() - t0:.0f}s", flush=True)
        drift = (x_dlr - x_exact).norm().item() / x_exact.norm().item()
        print(f"terminal state mismatch between the two passes: {drift:.2e}", flush=True)
        np.savez_compressed(cache, dflow=sv_exact, dlr=sv_dlr, alpha=alphas)
        print("wrote", cache)
    for label, sv in (("D-Flow", sv_exact), ("DLR", sv_dlr)):
        print(f"{label} at t=1: top singular values {np.round(sv[-1][:6], 4).tolist()}")
    path = OUT / f"{stem}.mp4"
    render(sv_exact, sv_dlr, alphas, args.sample_seed, path)
    print("wrote", path)


if __name__ == "__main__":
    main()

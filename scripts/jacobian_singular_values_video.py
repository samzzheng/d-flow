"""Full Jacobian singular-value evolution along a 100-step Euler flow.

For a few samples, propagates the *exact*, dense flow sensitivity
J_t = dx_t/dz (d x d, d = H*W) alongside the state x_t through the same
forward-Euler discretization used everywhere else in this repo:

    J_{i+1} = J_i + dt * A_{t_i} J_i,      J_0 = I,      A_t = d_x v_theta(t, x_t)

A_t J is computed by chunked forward-mode JVPs through the hand-written
tangent UNet pass (dlr_dflow.tangent_unet.unet_jvp), exactly like the JVP
products used elsewhere for the low-rank residual, just applied to all d
columns instead of a skinny r. Sorted singular values of J are recorded at
every one of the N=100 steps (not just a coarse diagnostic subset), then
animated as a GIF: one panel per sample, the sorted spectrum sliding from
J_0 = I (flat at 1) down to the numerically low-rank J_1.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlr_dflow.dlr_dflow import load_flow_model  # noqa: E402
from dlr_dflow.tangent_unet import unet_jvp  # noqa: E402
OUT = ROOT / "outputs" / "jacobian_singular_values"
OUT.mkdir(parents=True, exist_ok=True)

RESOLUTION = 64
N_CIRCLES = 1
EPOCH = 50
N_STEPS = 100
SAMPLE_SEED = 0
CHUNK = 512
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

import imageio_ffmpeg  # noqa: E402
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def full_jacobian_action(model, t: float, x_img: torch.Tensor, J: torch.Tensor, chunk: int):
    """A_t @ J via chunked forward-mode JVPs. J: (d, d). Returns (primal_v, A_t J)."""
    H, W = x_img.shape[-2:]
    d = H * W
    x1 = x_img.reshape(1, 1, H, W)
    primal = None
    cols = []
    for s in range(0, d, chunk):
        T = J[:, s:s + chunk].T.reshape(-1, 1, H, W)
        with torch.no_grad():
            p, out = unet_jvp(model, t, x1, T)
        if primal is None:
            primal = p
        cols.append(out.reshape(T.shape[0], d).T)
    return primal, torch.cat(cols, dim=1)


def run_sample(model, seed: int, n_steps: int, device: str, chunk: int) -> np.ndarray:
    H = W = RESOLUTION
    d = H * W
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 1, H, W, generator=g, device=device)
    J = torch.eye(d, device=device)
    dt = 1.0 / n_steps
    sv = np.zeros((n_steps + 1, d), dtype=np.float32)
    for i in range(n_steps):
        t = i * dt
        sv[i] = torch.linalg.svdvals(J).cpu().numpy()
        primal, AJ = full_jacobian_action(model, t, x, J, chunk)
        J = J + dt * AJ
        x = x + dt * primal.reshape(1, 1, H, W)
    sv[n_steps] = torch.linalg.svdvals(J).cpu().numpy()
    return sv


def main():
    model, _ = load_flow_model(resolution=RESOLUTION, n_circles=N_CIRCLES, epoch=EPOCH, device=DEVICE)

    sv_path = OUT / f"singular_values_res{RESOLUTION}_n{N_STEPS}_sample{SAMPLE_SEED}.npy"
    t0 = time.time()
    sv = run_sample(model, SAMPLE_SEED, N_STEPS, DEVICE, CHUNK)
    print(f"sample {SAMPLE_SEED}: {time.time() - t0:.1f}s", flush=True)
    np.save(sv_path, sv)
    print("wrote", sv_path)

    # ---- video ----
    d = sv.shape[-1]
    idx = np.arange(d)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ymin = max(sv.min(), 1e-8) * 0.5
    ax.set_yscale("log")
    ax.set_xlabel("index i")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_ylim(ymin, 2.0)
    ax.set_xlim(0, d)
    ax.grid(True, alpha=0.3)
    (line,) = ax.plot([], [], "-", color="C0", lw=1.3)
    title = ax.set_title("", fontsize=11)
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.14, right=0.96)

    def update(frame):
        line.set_data(idx, np.maximum(sv[frame], 1e-12))
        title.set_text(f"Full Jacobian singular values, N={N_STEPS} Euler steps, sample {SAMPLE_SEED}\n"
                        f"step {frame}/{N_STEPS}, t={frame / N_STEPS:.2f}")
        return [line, title]

    ani = animation.FuncAnimation(fig, update, frames=N_STEPS + 1, blit=False, interval=80)
    mp4_path = OUT / f"singular_values_res{RESOLUTION}_n{N_STEPS}_sample{SAMPLE_SEED}.mp4"
    ani.save(mp4_path, writer=animation.FFMpegWriter(fps=12, bitrate=1800))
    plt.close(fig)
    print("wrote", mp4_path)


if __name__ == "__main__":
    main()

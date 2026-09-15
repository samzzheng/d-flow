#!/usr/bin/env python3
"""Fast controls for the 64x64 circle sparse-CT DLR + D-Flow experiment."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

import dlr_dflow.dlr_dflow as dd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/dlr_dflow_sparse_ct_circle64_rank8_controls"
DEVICE = torch.device("cpu")
CPU_THREADS = 16
FLOW_STEPS = 10
OUTER_STEPS = 8
SOURCE_STEP_RMS = 0.003
N_ANGLES = 15
TRACE_PROBES = 32
RANDOMIZED_DIM = 16
SEED = 20260819


def sparse_radon(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    count = angles.numel()
    repeated = x[:, None].expand(batch, count, channels, height, width)
    repeated = repeated.reshape(batch * count, channels, height, width)
    angle_batch = angles.repeat(batch)
    cosine, sine = torch.cos(angle_batch), torch.sin(angle_batch)
    theta = torch.zeros(batch * count, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    grid = F.affine_grid(theta, repeated.shape, align_corners=False)
    rotated = F.grid_sample(
        repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    projections = rotated.mean(dim=-2)
    return projections.reshape(batch, count, channels, width).permute(0, 2, 1, 3)


def relative_l2(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(estimate - reference) /
            (torch.linalg.vector_norm(reference) + 1e-30)).item()


def exact_gradient_control(model, z0, x_true, measurement, A_op):
    z = z0.detach().clone()
    denominator = measurement.square().sum() + 1e-30
    history = []
    best = None
    start_total = time.perf_counter()

    for step in range(1, OUTER_STEPS + 1):
        start = time.perf_counter()
        zl = z.detach().requires_grad_(True)
        x1 = dd.euler_flow(model, zl, FLOW_STEPS)
        loss = 0.5 * (A_op(x1) - measurement).square().sum() / denominator
        (gradient,) = torch.autograd.grad(loss, zl)
        elapsed = time.perf_counter() - start

        with torch.no_grad():
            reconstruction = x1.clamp(0, 1)
            predicted = A_op(reconstruction)
            row = {
                "step": step,
                "measurement_relative_l2": relative_l2(predicted, measurement),
                "image_relative_l2": relative_l2(reconstruction, x_true),
                "ssim": float(structural_similarity(
                    x_true[0, 0].numpy(), reconstruction[0, 0].numpy(),
                    data_range=1.0)),
                "gradient_norm": gradient.norm().item(),
                "elapsed_s": elapsed,
            }
            if best is None or row["measurement_relative_l2"] < best["measurement_relative_l2"]:
                best = {**row, "reconstruction": reconstruction.detach().clone()}
            gradient_rms = gradient.square().mean().sqrt()
            z = (z - SOURCE_STEP_RMS * gradient / (gradient_rms + 1e-30)).detach()
        history.append(row)
        print(
            f"exact step={step:02d} measurement_rel={row['measurement_relative_l2']:.5f} "
            f"image_rel={row['image_relative_l2']:.5f} SSIM={row['ssim']:.4f} "
            f"time={elapsed:.2f}s",
            flush=True,
        )

    best_serializable = {key: value for key, value in best.items()
                         if key != "reconstruction"}
    return best_serializable, history, best["reconstruction"], time.perf_counter() - start_total


def hutchinson_alpha(model, z):
    height, width = z.shape[-2:]
    dimension = height * width
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    probes = (2 * torch.randint(
        0, 2, (dimension, TRACE_PROBES), generator=generator,
        device=DEVICE, dtype=torch.int64) - 1).to(dtype=z.dtype)
    x = z.detach()
    log_alpha = 0.0
    dt = 1.0 / FLOW_STEPS
    for index in range(FLOW_STEPS):
        t = index * dt
        A, _ = dd.make_A_products(model, t, x)
        trace_estimate = (probes * A(probes)).sum().item() / TRACE_PROBES
        log_alpha += dt * trace_estimate / dimension
        with torch.no_grad():
            x = x + dt * dd.velocity(model, t, x)
    return math.exp(log_alpha)


def make_terminal_jacobian_products(model, z):
    height, width = z.shape[-2:]
    dimension = height * width

    def batched_inputs(Q):
        count = Q.shape[1]
        source_batch = z.expand(count, -1, -1, -1).contiguous()
        tangent_batch = Q.T.reshape(count, 1, height, width).contiguous()
        return source_batch, tangent_batch

    def J(Q):
        source_batch, tangent_batch = batched_inputs(Q)
        _, product = torch.func.jvp(
            lambda source: dd.euler_flow(model, source, FLOW_STEPS),
            (source_batch,), (tangent_batch,))
        return product.reshape(Q.shape[1], dimension).T

    def JT(Q):
        source_batch, cotangent_batch = batched_inputs(Q)
        _, vjp_function = torch.func.vjp(
            lambda source: dd.euler_flow(model, source, FLOW_STEPS), source_batch)
        (product,) = vjp_function(cotangent_batch)
        return product.reshape(Q.shape[1], dimension).T

    return J, JT


def randomized_rank_diagnostic(model, z0, measurement, A_op):
    start = time.perf_counter()
    alpha = hutchinson_alpha(model, z0)
    x1 = dd.euler_flow(model, z0, FLOW_STEPS)
    g_x = dd.terminal_gradient(x1, A_op, measurement)
    g_x = g_x / (measurement.square().sum() + 1e-30)
    exact_gradient = dd.exact_source_gradient(model, z0, FLOW_STEPS, g_x)

    dimension = z0.numel()
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 20)
    omega = (2 * torch.randint(
        0, 2, (dimension, RANDOMIZED_DIM), generator=generator,
        device=DEVICE, dtype=torch.int64) - 1).to(dtype=z0.dtype)
    J, JT = make_terminal_jacobian_products(model, z0)

    B_omega = J(omega) - alpha * omega
    range_basis, _ = torch.linalg.qr(B_omega)
    BT_Q = JT(range_basis) - alpha * range_basis
    right_vectors, singular_values, small_left_t = torch.linalg.svd(
        BT_Q, full_matrices=False)
    left_vectors = range_basis @ small_left_t.T

    rank = 8
    oracle_residual_action = (
        right_vectors[:, :rank]
        @ (singular_values[:rank, None]
           * (left_vectors[:, :rank].T @ g_x.reshape(-1, 1))))
    oracle_gradient = alpha * g_x.reshape(-1, 1) + oracle_residual_action
    exact_flat = exact_gradient.reshape(-1, 1)

    frobenius_squared = B_omega.square().sum(dim=0).mean().item()
    captured_squared = singular_values[:rank].square().sum().item()
    energy_fraction = min(captured_squared / max(frobenius_squared, 1e-30), 1.0)
    result = {
        "alpha1": alpha,
        "randomized_subspace_dimension": RANDOMIZED_DIM,
        "estimated_top_singular_values": singular_values.cpu().tolist(),
        "estimated_rank8_frobenius_energy_fraction": energy_fraction,
        "estimated_best_rank8_frobenius_relative_error": math.sqrt(max(1.0 - energy_fraction, 0.0)),
        "oracle_rank8_action_relative_error": relative_l2(oracle_gradient, exact_flat),
        "oracle_rank8_action_cosine": F.cosine_similarity(
            oracle_gradient.T, exact_flat.T).item(),
        "elapsed_s": time.perf_counter() - start,
    }
    return result


def main():
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, _ = dd.load_flow_model(64, 1, 50, str(DEVICE))
    bundle = torch.load(
        ROOT / "data/circles-64-n1.pt", map_location="cpu",
        weights_only=True, mmap=True)
    x_true = bundle["test"][0:1].to(dtype=torch.float32)
    angles = torch.linspace(0, math.pi, N_ANGLES + 1)[:-1]
    A_op = lambda image: sparse_radon(image, angles)
    with torch.no_grad():
        measurement = A_op(x_true)
    source_generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1_000_000)
    z0 = torch.randn(x_true.shape, generator=source_generator)

    print("Running exact-gradient control", flush=True)
    best, history, reconstruction, exact_elapsed = exact_gradient_control(
        model, z0, x_true, measurement, A_op)
    print("Running randomized rank-8 diagnostic", flush=True)
    rank_result = randomized_rank_diagnostic(model, z0, measurement, A_op)

    result = {
        "configuration": {
            "resolution": 64,
            "sample_index": 0,
            "angles": N_ANGLES,
            "noise_level": 0.0,
            "flow_steps": FLOW_STEPS,
            "outer_steps": OUTER_STEPS,
            "source_step_rms": SOURCE_STEP_RMS,
            "cpu_threads": CPU_THREADS,
        },
        "exact_gradient_control": {
            "best": best,
            "history": history,
            "total_elapsed_s": exact_elapsed,
        },
        "rank8_diagnostic": rank_result,
    }
    result_path = OUT_DIR / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    truth = x_true[0, 0].numpy()
    estimate = reconstruction[0, 0].numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].imshow(truth, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground truth")
    axes[1].imshow(estimate, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Exact-gradient solution")
    error_plot = axes[2].imshow(estimate - truth, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("Signed error")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(error_plot, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exact_gradient_reconstruction.png", dpi=180)
    plt.close(fig)

    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare enriched dynamic rank-8 and direct terminal rank-8 D-Flow gradients."""

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

from dlr_dflow import dlr_dflow as dd
from dlr_dflow import enriched as de
from scripts.test_dlr_rank8_adequacy_circle64 import sparse_radon


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/dlr_dflow_sparse_ct_circle64_gradient_improvements"
DEVICE = torch.device("cpu")
SEED = 20260819
FLOW_STEPS = 10
OUTER_STEPS = 8
RANK = 8
TRACE_PROBES = 32
GRADIENT_SCALE = 1000.0
MAX_UPDATE_RMS = 0.003
N_ANGLES = 15


def relative_l2(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(estimate - reference) /
            (torch.linalg.vector_norm(reference) + 1e-30)).item()


def capped_update(z: torch.Tensor, gradient: torch.Tensor):
    raw_update = GRADIENT_SCALE * gradient
    raw_rms = raw_update.square().mean().sqrt()
    clip_factor = torch.clamp(MAX_UPDATE_RMS / (raw_rms + 1e-30), max=1.0)
    update = clip_factor * raw_update
    return (z - update).detach(), {
        "raw_update_rms": raw_rms.item(),
        "applied_update_rms": update.square().mean().sqrt().item(),
        "update_was_capped": bool(clip_factor.item() < 1.0),
    }


def run_solver(name, model, z0, x_true, measurement, A_op):
    z = z0.detach().clone()
    history = []
    best = None
    total_start = time.perf_counter()

    for step in range(1, OUTER_STEPS + 1):
        start = time.perf_counter()
        if name == "enriched_dlr":
            x1, factors, alpha = de.evolve_trace_centered_enriched(
                model, z, FLOW_STEPS, rank=RANK, trace_probes=TRACE_PROBES,
                initialization_oversampling=8, enrichment_probes=4, seed=SEED)
        elif name == "direct_terminal_rank8":
            x1, alpha = de.trace_centered_flow(
                model, z, FLOW_STEPS, trace_probes=TRACE_PROBES, seed=SEED)
            factors = de.direct_terminal_rank_factors(
                model, z, FLOW_STEPS, RANK, alpha,
                oversampling=8, seed=SEED + 100)
        else:
            raise ValueError(name)

        g_x = dd.terminal_gradient(x1, A_op, measurement)
        g_x = g_x / (measurement.square().sum() + 1e-30)
        gradient = dd.approx_source_gradient(factors, g_x, alpha)
        exact_gradient = dd.exact_source_gradient(model, z, FLOW_STEPS, g_x)
        cosine = F.cosine_similarity(
            gradient.reshape(1, -1), exact_gradient.reshape(1, -1)).item()
        action_error = relative_l2(gradient, exact_gradient)
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
                "alpha": alpha,
                "gradient_cosine": cosine,
                "gradient_relative_error": action_error,
                "gradient_norm": gradient.norm().item(),
                "elapsed_s": elapsed,
            }
            if best is None or row["measurement_relative_l2"] < best["measurement_relative_l2"]:
                best = {
                    **row,
                    "reconstruction": reconstruction.detach().clone(),
                    "predicted": predicted.detach().clone(),
                }
            z, update_metrics = capped_update(z, gradient)
            row.update(update_metrics)
        history.append(row)
        print(
            f"{name} step={step:02d} measurement={row['measurement_relative_l2']:.5f} "
            f"image={row['image_relative_l2']:.5f} SSIM={row['ssim']:.4f} "
            f"cos={cosine:.4f} action_rel={action_error:.4f} "
            f"update={row['applied_update_rms']:.3e} time={elapsed:.2f}s",
            flush=True,
        )

    serializable_best = {
        key: value for key, value in best.items()
        if key not in {"reconstruction", "predicted"}
    }
    summary = {
        "best": serializable_best,
        "mean_gradient_cosine": float(np.mean(
            [row["gradient_cosine"] for row in history])),
        "minimum_gradient_cosine": float(np.min(
            [row["gradient_cosine"] for row in history])),
        "mean_gradient_relative_error": float(np.mean(
            [row["gradient_relative_error"] for row in history])),
        "mean_outer_time_s": float(np.mean(
            [row["elapsed_s"] for row in history])),
        "total_elapsed_s": time.perf_counter() - total_start,
        "history": history,
    }
    return summary, best["reconstruction"]


def main():
    torch.set_num_threads(16)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, _ = dd.load_flow_model(64, 1, 50, str(DEVICE))
    bundle = torch.load(
        ROOT / "data/circles-64-n1.pt", map_location="cpu",
        weights_only=True, mmap=True)
    x_true = bundle["test"][0:1].float()
    angles = torch.linspace(0, math.pi, N_ANGLES + 1)[:-1]
    A_op = lambda image: sparse_radon(image, angles)
    with torch.no_grad():
        measurement = A_op(x_true)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1_000_000)
    z0 = torch.randn(x_true.shape, generator=generator)

    results = {
        "configuration": {
            "resolution": 64,
            "sample_index": 0,
            "angles": N_ANGLES,
            "noise_level": 0.0,
            "flow_steps": FLOW_STEPS,
            "outer_steps": OUTER_STEPS,
            "rank": RANK,
            "trace_probes": TRACE_PROBES,
            "gradient_scale": GRADIENT_SCALE,
            "maximum_update_rms": MAX_UPDATE_RMS,
            "cpu_threads": 16,
        }
    }
    reconstructions = {}
    for solver in ("enriched_dlr", "direct_terminal_rank8"):
        print(f"Running {solver}", flush=True)
        results[solver], reconstructions[solver] = run_solver(
            solver, model, z0, x_true, measurement, A_op)

    result_path = OUT_DIR / "results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")

    truth = x_true[0, 0].numpy()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for row_index, solver in enumerate(("enriched_dlr", "direct_terminal_rank8")):
        estimate = reconstructions[solver][0, 0].numpy()
        history = results[solver]["history"]
        axes[row_index, 0].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 0].set_title("Ground truth")
        axes[row_index, 1].imshow(estimate, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title(solver.replace("_", " "))
        error_artist = axes[row_index, 2].imshow(
            estimate - truth, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row_index, 2].set_title("Signed image error")
        axes[row_index, 3].plot(
            [entry["step"] for entry in history],
            [entry["measurement_relative_l2"] for entry in history],
            marker="o", label="Measurement rel. L2")
        axes[row_index, 3].plot(
            [entry["step"] for entry in history],
            [entry["gradient_cosine"] for entry in history],
            marker="s", label="Gradient cosine")
        axes[row_index, 3].set_xlabel("Outer iteration")
        axes[row_index, 3].grid(alpha=0.25)
        axes[row_index, 3].legend()
        for column in range(3):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        fig.colorbar(error_artist, ax=axes[row_index, 2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    figure_path = OUT_DIR / "gradient_improvements_comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(results, indent=2), flush=True)
    print(f"Saved {result_path}", flush=True)
    print(f"Saved {figure_path}", flush=True)


if __name__ == "__main__":
    main()

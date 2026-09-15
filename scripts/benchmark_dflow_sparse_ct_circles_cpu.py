"""D-Flow sparse-CT comparison of Forward Euler and RK4 on circles.

The benchmark adds Gaussian measurement noise whose L2 norm is exactly 1% of
the clean sinogram norm. Relative L2 error is used for convergence and primary
reporting so results are comparable across image and measurement dimensions.
By default, 20 Euler steps and 5 RK4 steps give both methods 20 velocity-model
evaluations per flow solve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cmocean
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unet.default import Unet


METHODS = ("forward_euler", "rk4")


class ClosureBudgetReached(RuntimeError):
    """Raised before an LBFGS closure would exceed the exact run budget."""


@dataclass
class SampleResult:
    resolution: int
    sample_index: int
    method: str
    flow_steps: int
    velocity_evaluations_per_flow: int
    noise_relative_l2: float
    steps_run: int
    closure_evaluations: int
    closure_budget: int | None
    closure_budget_reached: bool
    best_step: int
    best_closure_evaluation: int
    image_relative_l2: float
    measurement_relative_l2_noisy: float
    measurement_relative_l2_clean: float
    elapsed_s: float
    mean_outer_time_s: float
    lbfgs_rollbacks: int
    device: str
    device_name: str
    peak_gpu_memory_mb: float
    checkpoint_sha256: str
    diagnostic_figure: str


@dataclass
class IterationResult:
    resolution: int
    sample_index: int
    method: str
    outer_step: int
    measurement_relative_l2: float
    best_measurement_relative_l2: float
    elapsed_s: float
    closure_evaluations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 96, 128, 160])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--angles", type=int, default=15)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--euler-flow-steps", type=int, default=20)
    parser.add_argument("--rk4-flow-steps", type=int, default=5)
    parser.add_argument("--outer-steps", type=int, default=100)
    parser.add_argument(
        "--closure-budget", type=int, default=None,
        help=(
            "Exact maximum number of objective/gradient evaluations per method. "
            "When set, LBFGS advances one accepted iteration per outer call."
        ),
    )
    parser.add_argument("--lbfgs-lr", type=float, default=0.3)
    parser.add_argument("--lbfgs-max-iter", type=int, default=10)
    parser.add_argument(
        "--lbfgs-max-eval", type=int, default=None,
        help=(
            "Maximum closure evaluations available to one LBFGS step. "
            "Set this explicitly when --closure-budget forces max_iter=1."
        ),
    )
    parser.add_argument("--history-size", type=int, default=100)
    parser.add_argument(
        "--lbfgs-line-search", choices=("none", "strong_wolfe"), default="none",
    )
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-10)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-14)
    parser.add_argument(
        "--objective", choices=("relative_scaled", "measurement_mse"),
        default="relative_scaled",
        help=(
            "relative_scaled preserves the historical benchmark objective; "
            "measurement_mse matches the repository D-Flow data cost"
        ),
    )
    parser.add_argument(
        "--solution-state", choices=("best", "final"), default="best",
        help="Report either the lowest-cost closure or the final optimizer state.",
    )
    parser.add_argument("--rollback-factor", type=float, default=1.5)
    parser.add_argument(
        "--rollback-policy", choices=("factor", "nonfinite", "none"), default="factor",
        help=(
            "factor restores the best latent and clears LBFGS history after a "
            "large loss increase; nonfinite preserves history unless the state "
            "or objective becomes nonfinite"
        ),
    )
    parser.add_argument("--base-seed", type=int, default=20260807)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False,
        help="Require deterministic PyTorch kernels and deterministic SDP attention.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint velocity-model calls (defaults on for CUDA).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/dflow_sparse_ct_circles_cpu_euler_rk4"),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_model(resolution: int, epoch: int, device: torch.device) -> torch.nn.Module:
    run = ROOT / f"outputs/fm_unet/circles-{resolution}/n1"
    config = json.loads((run / "config.json").read_text())
    arch = config["arch"]
    model = Unet(
        ch=arch["ch"], ch_mul=arch["ch_mul"],
        att_channels=arch["att_channels"], groups=arch["groups"], dropout=0.0,
    )
    state = torch.load(
        run / f"checkpoints/epoch_{epoch:04d}.pt", map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("module._orig_mod.").removeprefix("module."): value
        for key, value in state.items() if key != "n_averaged"
    }
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    return model


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


def velocity(
    model: torch.nn.Module,
    t: float,
    x: torch.Tensor,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    time_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
    if activation_checkpointing and torch.is_grad_enabled() and x.requires_grad:
        return checkpoint(model, x, time_batch, use_reentrant=False)
    return model(x, time_batch)


def solve_flow(
    model: torch.nn.Module, source: torch.Tensor, steps: int, method: str,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    h = 1.0 / steps
    x = source
    for index in range(steps):
        t = index * h
        if method == "forward_euler":
            x = x + h * velocity(model, t, x, activation_checkpointing)
        elif method == "rk4":
            k1 = velocity(model, t, x, activation_checkpointing)
            k2 = velocity(
                model, t + 0.5 * h, x + 0.5 * h * k1,
                activation_checkpointing,
            )
            k3 = velocity(
                model, t + 0.5 * h, x + 0.5 * h * k2,
                activation_checkpointing,
            )
            k4 = velocity(
                model, t + h, x + h * k3, activation_checkpointing,
            )
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"unknown flow solver: {method}")
    return x


def flow_steps_for_method(args: argparse.Namespace, method: str) -> int:
    if method == "forward_euler":
        return args.euler_flow_steps
    if method == "rk4":
        return args.rk4_flow_steps
    raise ValueError(f"unknown flow solver: {method}")


def velocity_evaluations(method: str, steps: int) -> int:
    return steps if method == "forward_euler" else 4 * steps


def relative_l2(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(estimate - reference) / (
        torch.linalg.vector_norm(reference) + 1e-30)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Convert stored circle tensors to float32 intensities in [0, 1]."""
    if images.dtype == torch.uint8:
        normalized = images.float().div_(255.0)
    elif images.is_floating_point():
        normalized = images.float()
    else:
        raise TypeError(f"Unsupported image dtype: {images.dtype}")
    if normalized.numel() and (
        normalized.min().item() < 0.0 or normalized.max().item() > 1.0
    ):
        raise ValueError("Circle images must have intensities in [0, 1]")
    return normalized


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def add_relative_noise(
    clean: torch.Tensor, level: float, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    noise = noise / (noise.norm() + 1e-30) * (level * clean.norm())
    noisy = clean + noise
    realized = relative_l2(noisy, clean).item()
    return noisy, noise, realized


def save_diagnostic_figure(
    path: Path, reconstruction: torch.Tensor, ground_truth: torch.Tensor,
    clean_measurement: torch.Tensor, noisy_measurement: torch.Tensor,
    predicted_measurement: torch.Tensor,
    history: list[float], best_history: list[float], method: str,
    image_relative_error: float, measurement_relative_error: float,
    noise_level: float,
    flow_steps: int,
) -> None:
    estimate = reconstruction[0, 0].detach().cpu().numpy()
    truth = ground_truth[0, 0].detach().cpu().numpy()
    clean_sinogram = clean_measurement[0, 0].detach().cpu().numpy()
    noisy_sinogram = noisy_measurement[0, 0].detach().cpu().numpy()
    predicted_sinogram = predicted_measurement[0, 0].detach().cpu().numpy()
    image_error = estimate - truth
    clean_measurement_error = predicted_sinogram - clean_sinogram
    noisy_measurement_residual = predicted_sinogram - noisy_sinogram
    label = "Forward Euler" if method == "forward_euler" else "RK4"
    noise_percent = 100.0 * noise_level
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 12.2))

    def image_panel(axis, array, title, cmap, vmin=None, vmax=None, label_text="Value"):
        artist = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(label_text)

    sinogram_min = min(float(clean_sinogram.min()), float(noisy_sinogram.min()))
    sinogram_max = max(float(clean_sinogram.max()), float(noisy_sinogram.max()))
    image_panel(axes[0, 0], truth, "Ground truth", cmocean.cm.dense_r, 0, 1, "Intensity")
    image_panel(
        axes[1, 0], estimate, f"Proposed solution ({label})", cmocean.cm.dense_r,
        0, 1, "Intensity")
    image_panel(
        axes[2, 0], image_error, "Signed image error: proposed − GT",
        cmocean.cm.balance, -1, 1, "Signed error")

    image_panel(
        axes[0, 1], clean_sinogram, "Sinogram (clean)", cmocean.cm.dense_r,
        sinogram_min, sinogram_max, "Projection value")
    image_panel(
        axes[1, 1], predicted_sinogram, "Radon transform of proposed solution",
        cmocean.cm.dense_r, sinogram_min, sinogram_max, "Projection value")
    image_panel(
        axes[2, 1], clean_measurement_error,
        "Signed sinogram error: proposed − clean",
        cmocean.cm.balance, -1, 1, "Signed error")

    image_panel(
        axes[0, 2], noisy_sinogram,
        f"Sinogram with {noise_percent:g}% noise", cmocean.cm.dense_r,
        sinogram_min, sinogram_max, "Projection value")
    image_panel(
        axes[1, 2], noisy_measurement_residual,
        "Signed residual: proposed − noisy",
        cmocean.cm.balance, -1, 1, "Signed residual")

    steps = range(1, len(history) + 1)
    axes[2, 2].semilogy(steps, best_history, linewidth=2, label="best so far")
    if noise_level > 0:
        axes[2, 2].axhline(
            noise_level, color="crimson", linestyle="--",
            label=f"{noise_percent:g}% noise level")
    axes[2, 2].set_title("Measurement relative ℓ₂ error")
    axes[2, 2].set_xlabel("Outer iteration")
    axes[2, 2].set_ylabel(r"$\|Ax-y_{noisy}\|_2/\|y_{noisy}\|_2$")
    axes[2, 2].grid(alpha=0.25)
    axes[2, 2].legend()
    fig.suptitle(
        f"{label}: {flow_steps} steps, "
        f"{velocity_evaluations(method, flow_steps)} velocity evaluations | "
        f"image relative ℓ₂={image_relative_error:.3e} | "
        f"measurement relative ℓ₂={measurement_relative_error:.3e}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def data_objective(
    predicted: torch.Tensor,
    measurement: torch.Tensor,
    resolution: int,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the optimization objective and squared relative residual."""
    residual_sq = (predicted - measurement).square().sum()
    relative_sq = residual_sq / (measurement.square().sum() + 1e-30)
    if objective == "relative_scaled":
        return relative_sq * float(resolution * resolution), relative_sq
    if objective == "measurement_mse":
        return residual_sq / predicted.numel(), relative_sq
    raise ValueError(f"unknown objective: {objective}")


def accepted_trial_record(
    accepted_source: torch.Tensor,
    trial_records: list[tuple[torch.Tensor, float, int]],
) -> tuple[float, int]:
    """Return the objective diagnostic evaluated at the LBFGS-accepted source.

    Strong-Wolfe may return an earlier point from its bracket rather than the
    final trial it evaluated.  PyTorch then reconstructs that accepted point
    exactly from the saved line-search origin, step, and direction.
    """
    for trial_source, relative, closure_index in reversed(trial_records):
        if torch.equal(trial_source, accepted_source):
            return relative, closure_index
    raise RuntimeError(
        "LBFGS accepted a source that was not evaluated by its closure; "
        "cannot report the accepted-iterate objective without spending an "
        "additional closure evaluation."
    )


def run_inverse(
    model: torch.nn.Module,
    ground_truth: torch.Tensor,
    clean_measurement: torch.Tensor,
    noisy_measurement: torch.Tensor,
    angles: torch.Tensor,
    resolution: int,
    sample_index: int,
    method: str,
    source_initial: torch.Tensor,
    checkpoint_sha256: str,
    device_name: str,
    args: argparse.Namespace,
) -> tuple[SampleResult, list[IterationResult]]:
    flow_steps = flow_steps_for_method(args, method)
    source = source_initial.detach().clone().requires_grad_(True)
    effective_max_iter = 1 if args.closure_budget is not None else args.lbfgs_max_iter
    optimizer = torch.optim.LBFGS(
        [source], lr=args.lbfgs_lr, max_iter=effective_max_iter,
        max_eval=args.lbfgs_max_eval,
        history_size=args.history_size, line_search_fn=None,
        tolerance_grad=args.lbfgs_tolerance_grad,
        tolerance_change=args.lbfgs_tolerance_change,
    )
    optimizer.param_groups[0]["line_search_fn"] = (
        None if args.lbfgs_line_search == "none" else args.lbfgs_line_search
    )
    current_step = 0
    closure_count = 0
    best_relative = float("inf")
    best_source = source.detach().clone()
    best_step = 0
    best_closure = 0
    last_relative = float("inf")
    history: list[float] = []
    best_history: list[float] = []
    iteration_rows: list[IterationResult] = []
    rollback_count = 0
    closure_budget_reached = False
    step_trial_records: list[tuple[torch.Tensor, float, int]] = []

    def closure():
        nonlocal closure_count, best_relative, best_source, best_step
        nonlocal best_closure, last_relative
        if args.closure_budget is not None and closure_count >= args.closure_budget:
            raise ClosureBudgetReached
        closure_count += 1
        optimizer.zero_grad()
        reconstruction = solve_flow(
            model, source, flow_steps, method, args.activation_checkpointing,
        )
        predicted = sparse_radon(reconstruction.to(angles.device), angles)
        optimization_objective, relative_sq = data_objective(
            predicted, noisy_measurement, resolution, args.objective,
        )
        optimization_objective.backward()
        last_relative = math.sqrt(max(relative_sq.item(), 0.0))
        step_trial_records.append(
            (source.detach().clone(), last_relative, closure_count)
        )
        # Without a line search, every closure is evaluated at the current
        # accepted iterate before the next fixed L-BFGS update is applied.
        if (
            args.lbfgs_line_search == "none"
            and math.isfinite(last_relative)
            and last_relative < best_relative
        ):
            best_relative = last_relative
            best_source = source.detach().clone()
            best_step = current_step
            best_closure = closure_count
        return optimization_objective

    device = source.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    total_start = time.perf_counter()
    outer_times: list[float] = []
    maximum_outer_steps = (
        args.closure_budget if args.closure_budget is not None else args.outer_steps
    )
    for step in range(1, maximum_outer_steps + 1):
        if args.closure_budget is not None and closure_count >= args.closure_budget:
            closure_budget_reached = True
            break
        current_step = step
        closures_before = closure_count
        accepted_source_before_step = source.detach().clone()
        accepted_relative_before_step = last_relative
        step_trial_records.clear()
        synchronize(device)
        start = time.perf_counter()
        budget_interrupted_step = False
        try:
            optimizer.step(closure)
        except ClosureBudgetReached:
            # A line search requested another trial after consuming the budget.
            # Restore the last accepted iterate so rejected trials are never reported.
            with torch.no_grad():
                source.copy_(accepted_source_before_step)
            last_relative = accepted_relative_before_step
            closure_budget_reached = True
            budget_interrupted_step = True
        synchronize(device)
        elapsed = time.perf_counter() - start
        outer_times.append(elapsed)
        used_closures = closure_count - closures_before
        if budget_interrupted_step:
            print(
                f"res={resolution} sample={sample_index} method={method} "
                f"closure_budget={closure_count} reached during line search",
                flush=True,
            )
            break
        if args.lbfgs_line_search == "strong_wolfe":
            last_relative, accepted_closure = accepted_trial_record(
                source.detach(), step_trial_records,
            )
            if math.isfinite(last_relative) and last_relative < best_relative:
                best_relative = last_relative
                best_source = source.detach().clone()
                best_step = current_step
                best_closure = accepted_closure
        source_is_finite = bool(torch.isfinite(source).all().item())
        should_restore = False
        if args.rollback_policy in {"factor", "nonfinite"}:
            should_restore = not math.isfinite(last_relative) or not source_is_finite
        if args.rollback_policy == "factor":
            should_restore = (
                should_restore
                or last_relative > args.rollback_factor * best_relative
            )
        if should_restore:
            with torch.no_grad():
                source.copy_(best_source)
            optimizer.state.clear()
            last_relative = best_relative
            rollback_count += 1
        history.append(last_relative)
        best_history.append(best_relative)
        iteration_rows.append(IterationResult(
            resolution=resolution, sample_index=sample_index, method=method,
            outer_step=step, measurement_relative_l2=last_relative,
            best_measurement_relative_l2=best_relative,
            elapsed_s=elapsed, closure_evaluations=used_closures,
        ))
        print(
            f"res={resolution} sample={sample_index} method={method} step={step} "
            f"relative={last_relative:.3e} closures={used_closures} time={elapsed:.2f}s",
            flush=True,
        )
        if not math.isfinite(last_relative):
            break
    if args.closure_budget is not None and closure_count >= args.closure_budget:
        closure_budget_reached = True
    synchronize(device)
    total_elapsed = time.perf_counter() - total_start

    with torch.no_grad():
        selected_source = best_source if args.solution_state == "best" else source.detach()
        raw = solve_flow(model, selected_source, flow_steps, method)
        reconstruction = raw.clamp(0, 1)
        predicted_raw = sparse_radon(raw.to(angles.device), angles)
        predicted_clamped = sparse_radon(reconstruction.to(angles.device), angles)
        image_relative_error = relative_l2(reconstruction, ground_truth).item()
        noisy_relative_error = relative_l2(predicted_raw, noisy_measurement).item()
        clean_relative_error = relative_l2(predicted_clamped, clean_measurement).item()
    diagnostic_path = (
        args.out_dir / "figures" / f"res_{resolution}" / f"sample_{sample_index:03d}"
        / f"{method}.png"
    )
    save_diagnostic_figure(
        diagnostic_path, reconstruction, ground_truth, clean_measurement,
        noisy_measurement, predicted_clamped, history, best_history, method,
        image_relative_error, noisy_relative_error, args.noise_level,
        flow_steps,
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    result = SampleResult(
        resolution=resolution, sample_index=sample_index, method=method,
        flow_steps=flow_steps,
        velocity_evaluations_per_flow=velocity_evaluations(method, flow_steps),
        noise_relative_l2=relative_l2(noisy_measurement, clean_measurement).item(),
        steps_run=len(history), closure_evaluations=closure_count, best_step=best_step,
        closure_budget=args.closure_budget,
        closure_budget_reached=closure_budget_reached,
        best_closure_evaluation=best_closure,
        image_relative_l2=image_relative_error,
        measurement_relative_l2_noisy=noisy_relative_error,
        measurement_relative_l2_clean=clean_relative_error,
        elapsed_s=total_elapsed,
        mean_outer_time_s=statistics.mean(outer_times),
        lbfgs_rollbacks=rollback_count,
        device=str(device), device_name=device_name,
        peak_gpu_memory_mb=peak_memory_mb,
        checkpoint_sha256=checkpoint_sha256,
        diagnostic_figure=str(diagnostic_path),
    )
    return result, iteration_rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for resolution in sorted({int(row["resolution"]) for row in rows}):
        for method in METHODS:
            group = [
                row for row in rows
                if int(row["resolution"]) == resolution and row["method"] == method
            ]
            if not group:
                continue
            values = lambda key: [float(row[key]) for row in group]
            summary.append({
                "resolution": resolution,
                "method": method,
                "samples": len(group),
                "mean_image_relative_l2": statistics.mean(values("image_relative_l2")),
                "median_image_relative_l2": statistics.median(values("image_relative_l2")),
                "mean_measurement_relative_l2_noisy": statistics.mean(values("measurement_relative_l2_noisy")),
                "mean_measurement_relative_l2_clean": statistics.mean(values("measurement_relative_l2_clean")),
                "mean_elapsed_s": statistics.mean(values("elapsed_s")),
                "median_elapsed_s": statistics.median(values("elapsed_s")),
                "mean_outer_steps": statistics.mean(values("steps_run")),
                "mean_closure_evaluations": statistics.mean(values("closure_evaluations")),
                "closure_budget_reached_count": sum(
                    str(row.get("closure_budget_reached", "")).lower() == "true"
                    for row in group
                ),
                "mean_peak_gpu_memory_mb": statistics.mean(values("peak_gpu_memory_mb")),
            })
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resume_settings(existing: dict, current: dict) -> None:
    immutable = (
        "epoch", "angles", "noise_level", "euler_flow_steps",
        "rk4_flow_steps", "outer_steps", "closure_budget", "lbfgs_lr", "lbfgs_max_iter",
        "lbfgs_max_eval",
        "history_size", "lbfgs_line_search", "lbfgs_tolerance_grad",
        "lbfgs_tolerance_change", "objective", "solution_state",
        "rollback_factor", "base_seed",
        "rollback_policy", "activation_checkpointing", "deterministic",
    )
    mismatches = {
        key: (existing.get(key), current.get(key))
        for key in immutable if existing.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Incompatible existing output settings: {mismatches}")
    existing_hashes = existing.get("checkpoint_sha256", {})
    current_hashes = current.get("checkpoint_sha256", {})
    hash_mismatches = {
        resolution: (digest, current_hashes.get(resolution))
        for resolution, digest in existing_hashes.items()
        if resolution in current_hashes and current_hashes[resolution] != digest
    }
    if hash_mismatches:
        raise RuntimeError(f"Checkpoint identity changed: {hash_mismatches}")


def main() -> None:
    args = parse_args()
    if args.closure_budget is not None and args.closure_budget <= 0:
        raise ValueError("--closure-budget must be positive")
    if args.lbfgs_max_eval is not None and args.lbfgs_max_eval <= 0:
        raise ValueError("--lbfgs-max-eval must be positive")
    if (
        args.closure_budget is not None
        and args.lbfgs_line_search == "strong_wolfe"
        and args.lbfgs_max_eval is None
    ):
        raise ValueError(
            "Fixed-budget strong-Wolfe LBFGS requires an explicit "
            "--lbfgs-max-eval; use 25 to match the reference allowance."
        )
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    if args.activation_checkpointing is None:
        args.activation_checkpointing = device.type == "cuda"
    euler_evaluations = velocity_evaluations("forward_euler", args.euler_flow_steps)
    rk4_evaluations = velocity_evaluations("rk4", args.rk4_flow_steps)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(min(4, args.cpu_threads))
    except RuntimeError:
        pass
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["out_dir"] = str(args.out_dir)
    settings["device"] = str(device)
    settings["device_name"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    settings["methods"] = list(METHODS)
    settings["effective_lbfgs_max_iter"] = (
        1 if args.closure_budget is not None else args.lbfgs_max_iter
    )
    settings["effective_lbfgs_max_eval"] = (
        args.lbfgs_max_eval
        if args.lbfgs_max_eval is not None
        else int(1.25 * settings["effective_lbfgs_max_iter"])
    )
    settings["analysis_policy"] = (
        "intention_to_treat_all_runs_included"
        if args.closure_budget is not None else "standard"
    )
    settings["velocity_evaluations_per_flow"] = {
        "forward_euler": euler_evaluations,
        "rk4": rk4_evaluations,
    }
    settings["noise_definition"] = "||noise||_2 / ||clean measurement||_2 = noise_level"
    settings["primary_metric"] = "relative L2 error"
    checkpoint_paths = {
        resolution: (
            ROOT / f"outputs/fm_unet/circles-{resolution}/n1/checkpoints"
            / f"epoch_{args.epoch:04d}.pt"
        )
        for resolution in args.resolutions
    }
    settings["checkpoint_sha256"] = {
        str(resolution): sha256_file(path)
        for resolution, path in checkpoint_paths.items()
    }
    settings_path = args.out_dir / "settings.json"
    if settings_path.exists():
        validate_resume_settings(json.loads(settings_path.read_text()), settings)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    samples_path = args.out_dir / "samples.csv"
    iterations_path = args.out_dir / "iterations.csv"
    sample_rows = read_csv(samples_path)
    iteration_rows = read_csv(iterations_path)
    completed = {
        (int(row["resolution"]), int(row["sample_index"]), row["method"])
        for row in sample_rows
    }

    for resolution in args.resolutions:
        checkpoint_sha256 = settings["checkpoint_sha256"][str(resolution)]
        matching_rows = [
            row for row in sample_rows if int(row["resolution"]) == resolution
        ]
        wrong_hashes = {
            row.get("checkpoint_sha256") for row in matching_rows
            if row.get("checkpoint_sha256") != checkpoint_sha256
        }
        if wrong_hashes:
            raise RuntimeError(
                f"Existing rows for resolution {resolution} use different checkpoints: "
                f"{wrong_hashes}"
            )
        model = load_model(resolution, args.epoch, device)
        payload = torch.load(
            ROOT / f"data/circles-{resolution}-n1.pt",
            map_location="cpu", weights_only=True, mmap=True,
        )
        dataset = payload["test"]
        metadata = payload.get("metadata", {})
        if metadata.get("background_value") != 0 or metadata.get("circle_value") != 1:
            raise RuntimeError(
                f"Unexpected dataset polarity for resolution {resolution}: {metadata}"
            )
        radon_device = torch.device("cpu") if args.deterministic else device
        angles = torch.linspace(
            0, math.pi, args.angles + 1, device=radon_device,
        )[:-1]
        for local_index in range(args.samples):
            sample_index = args.sample_offset + local_index
            if all((resolution, sample_index, method) in completed for method in METHODS):
                print(f"Skipping completed res={resolution} sample={sample_index}", flush=True)
                continue
            ground_truth = normalize_images(
                dataset[sample_index:sample_index + 1]
            ).to(device)
            with torch.no_grad():
                clean_measurement = sparse_radon(
                    ground_truth.to(radon_device), angles,
                )
            noise_generator = torch.Generator(device="cpu").manual_seed(
                args.base_seed + resolution * 1000 + sample_index)
            clean_measurement_cpu = clean_measurement.cpu()
            noisy_measurement_cpu, _, realized_noise = add_relative_noise(
                clean_measurement_cpu, args.noise_level, noise_generator)
            if not math.isclose(
                realized_noise, args.noise_level, rel_tol=1e-5, abs_tol=1e-7,
            ):
                raise RuntimeError(
                    f"Realized noise {realized_noise} != requested {args.noise_level}"
                )
            noisy_measurement = noisy_measurement_cpu.to(radon_device)
            source_generator = torch.Generator(device="cpu").manual_seed(
                args.base_seed + 10_000_000 + resolution * 1000 + sample_index)
            source_initial = torch.randn(
                ground_truth.shape, generator=source_generator,
            ).to(device)

            for method in METHODS:
                key = (resolution, sample_index, method)
                if key in completed:
                    print(f"Skipping completed {key}", flush=True)
                    continue
                result, timed_rows = run_inverse(
                    model, ground_truth, clean_measurement, noisy_measurement,
                    angles, resolution, sample_index, method, source_initial,
                    checkpoint_sha256, settings["device_name"], args,
                )
                sample_rows.append(asdict(result))
                iteration_rows.extend(asdict(row) for row in timed_rows)
                write_csv(samples_path, sample_rows)
                write_csv(iterations_path, iteration_rows)
                write_csv(args.out_dir / "summary.csv", summarize(sample_rows))
                completed.add(key)
                print(
                    f"Completed res={resolution} sample={sample_index} method={method}: "
                    f"image_rel={result.image_relative_l2:.3e} "
                    f"measurement_rel={result.measurement_relative_l2_noisy:.3e} "
                    f"time={result.elapsed_s:.1f}s",
                    flush=True,
                )
        del model, dataset


if __name__ == "__main__":
    main()

"""Benchmark D-Flow on sparse-view CT with trained Shepp-Logan flows."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cmocean
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unet.default import Unet


@dataclass
class SampleResult:
    resolution: int
    sample_idx: int
    steps_run: int
    closure_evals: int
    elapsed_s: float
    best_step: int
    measurement_mse: float
    image_mse: float
    reconstruction_tv: float
    figure_path: str


@dataclass
class IterationResult:
    resolution: int
    sample_idx: int
    outer_step: int
    elapsed_s: float
    closure_evals: int
    elapsed_per_closure_s: float
    measurement_mse: float
    objective: float
    reconstruction_tv: float
    grad_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--angles", type=int, default=15)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--outer-steps", type=int, default=100)
    parser.add_argument("--lbfgs-lr", type=float, default=0.3)
    parser.add_argument("--lbfgs-max-iter", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=100)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--tv-eps", type=float, default=1e-6)
    parser.add_argument("--stop-mse", type=float, default=1e-12)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/dflow_sparse_ct_shepp_logan")
    )
    return parser.parse_args()


def load_model(resolution: int, epoch: int, device: torch.device) -> torch.nn.Module:
    run_dir = Path(f"outputs/fm_unet/shepp-logan-{resolution}")
    with (run_dir / "config.json").open() as handle:
        architecture = json.load(handle)["arch"]
    model = Unet(
        ch=architecture["ch"],
        ch_mul=architecture["ch_mul"],
        att_channels=architecture["att_channels"],
        groups=architecture["groups"],
        dropout=0.0,
    ).to(device)
    checkpoint = run_dir / f"checkpoints/epoch_{epoch:04d}.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    state = {
        key.removeprefix("module._orig_mod.").removeprefix("module."): value
        for key, value in state.items()
        if key != "n_averaged"
    }
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model


def sparse_ct(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    count = angles.numel()
    repeated = x[:, None].expand(batch, count, channels, height, width)
    repeated = repeated.reshape(batch * count, channels, height, width)
    angle_batch = angles.repeat(batch)
    cosine, sine = torch.cos(angle_batch), torch.sin(angle_batch)
    theta = torch.zeros(batch * count, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, repeated.shape, align_corners=False)
    rotated = F.grid_sample(
        repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    projections = rotated.mean(dim=-2)
    return projections.reshape(batch, count, channels, width).permute(0, 2, 1, 3)


def euler_flow(model: torch.nn.Module, source: torch.Tensor, steps: int) -> torch.Tensor:
    x = source
    step_size = 1.0 / steps
    for step in range(steps):
        t = torch.full(
            (x.shape[0],), step * step_size, device=x.device, dtype=x.dtype
        )
        x = x + step_size * model(x, t)
    return x


def total_variation(x: torch.Tensor, eps: float) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return (
        torch.sqrt(dx.square() + eps * eps).mean()
        + torch.sqrt(dy.square() + eps * eps).mean()
    )


def save_figure(
    path: Path,
    x_gt: torch.Tensor,
    observation: torch.Tensor,
    x_hat: torch.Tensor,
    predicted_measurement: torch.Tensor,
    image_mse: float,
    measurement_mse: float,
) -> None:
    ground_truth = x_gt[0, 0].detach().cpu().numpy()
    reconstruction = x_hat[0, 0].detach().cpu().numpy()
    observed = observation[0, 0].detach().cpu().numpy()
    predicted = predicted_measurement[0, 0].detach().cpu().numpy()
    image_error = reconstruction - ground_truth
    measurement_error = predicted - observed
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
    panels = [
        (axes[0, 0], ground_truth, "Ground truth", cmocean.cm.dense_r, 0.0, 1.0),
        (axes[0, 1], reconstruction, f"D-Flow reconstruction\nMSE={image_mse:.3e}", cmocean.cm.dense_r, 0.0, 1.0),
        (axes[0, 2], image_error, "Image error", cmocean.cm.balance, -1.0, 1.0),
        (axes[1, 0], observed, "Observed Ax", cmocean.cm.dense_r, None, None),
        (axes[1, 1], predicted, f"Reconstructed Ax\nMSE={measurement_mse:.3e}", cmocean.cm.dense_r, None, None),
        (axes[1, 2], measurement_error, "Measurement residual", cmocean.cm.balance, -1.0, 1.0),
    ]
    for axis, image, title, cmap, vmin, vmax in panels:
        rendered = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if cmap == cmocean.cm.balance:
            colorbar = fig.colorbar(
                rendered,
                ax=axis,
                orientation="vertical",
                ticks=[-1.0, 0.0, 1.0],
                fraction=0.046,
                pad=0.04,
            )
            colorbar.set_label("Error")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_sample(
    model: torch.nn.Module,
    x_gt: torch.Tensor,
    resolution: int,
    sample_idx: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SampleResult, list[IterationResult]]:
    angles = torch.linspace(0, math.pi, args.angles + 1, device=device)[:-1]
    with torch.no_grad():
        observation = sparse_ct(x_gt, angles)
    generator = torch.Generator(device=device).manual_seed(
        args.base_seed + resolution * 1000 + sample_idx
    )
    source = torch.randn(
        x_gt.shape, generator=generator, device=device, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [source],
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.history_size,
        line_search_fn=None,
        tolerance_grad=1e-12,
        tolerance_change=1e-16,
    )
    scale = float(resolution * resolution)
    current_step = 0
    closure_count = 0
    best_objective = float("inf")
    best_source = source.detach().clone()
    best_step = -1
    last = {"mse": float("inf"), "objective": float("inf"), "tv": 0.0, "grad": 0.0}

    def closure():
        nonlocal closure_count, best_objective, best_source, best_step
        closure_count += 1
        optimizer.zero_grad()
        reconstruction = euler_flow(model, source, args.flow_steps)
        Ax = sparse_ct(reconstruction, angles)
        measurement_mse = (Ax - observation).square().mean()
        tv = total_variation(reconstruction, args.tv_eps)
        objective = measurement_mse + args.tv_weight * tv
        (objective * scale).backward()
        last.update(
            mse=measurement_mse.item(),
            objective=objective.item(),
            tv=tv.item(),
            grad=source.grad.norm().item(),
        )
        if math.isfinite(last["objective"]) and last["objective"] < best_objective:
            best_objective = last["objective"]
            best_source = source.detach().clone()
            best_step = current_step
        return objective * scale

    rows = []
    torch.cuda.synchronize(device)
    sample_start = time.perf_counter()
    for step in range(args.outer_steps):
        current_step = step
        closures_before = closure_count
        torch.cuda.synchronize(device)
        iteration_start = time.perf_counter()
        optimizer.step(closure)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - iteration_start
        closures = closure_count - closures_before
        rows.append(
            IterationResult(
                resolution=resolution,
                sample_idx=sample_idx,
                outer_step=step,
                elapsed_s=elapsed,
                closure_evals=closures,
                elapsed_per_closure_s=elapsed / closures if closures else float("nan"),
                measurement_mse=last["mse"],
                objective=last["objective"],
                reconstruction_tv=last["tv"],
                grad_norm=last["grad"],
            )
        )
        if step % 10 == 0:
            print(
                f"res={resolution} sample={sample_idx} step={step} "
                f"mse={last['mse']:.3e} time={elapsed:.3f}s closures={closures}",
                flush=True,
            )
        if last["mse"] <= args.stop_mse or not math.isfinite(last["objective"]):
            break
    torch.cuda.synchronize(device)
    sample_elapsed = time.perf_counter() - sample_start
    with torch.no_grad():
        raw = euler_flow(model, best_source, args.flow_steps)
        x_hat = raw.clamp(0, 1)
        predicted = sparse_ct(raw, angles)
        image_mse = (x_hat - x_gt).square().mean().item()
        measurement_mse = (predicted - observation).square().mean().item()
        reconstruction_tv = total_variation(x_hat, args.tv_eps).item()
    figure_path = args.out_dir / "figures" / f"res_{resolution}" / f"sample_{sample_idx:03d}.png"
    save_figure(
        figure_path, x_gt, observation, x_hat, predicted, image_mse, measurement_mse
    )
    result = SampleResult(
        resolution=resolution,
        sample_idx=sample_idx,
        steps_run=len(rows),
        closure_evals=closure_count,
        elapsed_s=sample_elapsed,
        best_step=best_step,
        measurement_mse=measurement_mse,
        image_mse=image_mse,
        reconstruction_tv=reconstruction_tv,
        figure_path=str(figure_path),
    )
    return result, rows


def read_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_stats(sample_rows: list[dict], iteration_rows: list[dict], path: Path) -> None:
    stats = {}
    for resolution in sorted({int(row["resolution"]) for row in sample_rows}):
        samples = [row for row in sample_rows if int(row["resolution"]) == resolution]
        iterations = [row for row in iteration_rows if int(row["resolution"]) == resolution]
        times = [float(row["elapsed_s"]) for row in iterations]
        stats[str(resolution)] = {
            "completed_samples": len(samples),
            "mean_image_mse": statistics.mean(float(row["image_mse"]) for row in samples),
            "median_image_mse": statistics.median(float(row["image_mse"]) for row in samples),
            "mean_measurement_mse": statistics.mean(float(row["measurement_mse"]) for row in samples),
            "median_iteration_time_s": statistics.median(times),
            "mean_iteration_time_s": statistics.mean(times),
            "minimum_iteration_time_s": min(times),
            "maximum_iteration_time_s": max(times),
        }
    path.write_text(json.dumps(stats, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This timing benchmark requires a CUDA device")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["out_dir"] = str(args.out_dir)
    settings["line_search"] = None
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    samples_path = args.out_dir / "sparse_ct_samples.csv"
    iterations_path = args.out_dir / "sparse_ct_iterations.csv"
    sample_rows = read_completed(samples_path)
    iteration_rows = read_completed(iterations_path)
    completed = {(int(row["resolution"]), int(row["sample_idx"])) for row in sample_rows}

    for resolution in args.resolutions:
        model = load_model(resolution, args.epoch, device)
        dataset = torch.load(
            f"data/shepp-logan-{resolution}.pt", map_location="cpu", weights_only=True
        )["test"]
        for sample_idx in range(args.samples):
            if (resolution, sample_idx) in completed:
                print(f"Skipping completed res={resolution} sample={sample_idx}", flush=True)
                continue
            x_gt = dataset[sample_idx:sample_idx + 1].float().div(255.0).to(device)
            result, timed = run_sample(
                model, x_gt, resolution, sample_idx, args, device
            )
            sample_rows.append(asdict(result))
            iteration_rows.extend(asdict(row) for row in timed)
            write_csv(samples_path, sample_rows)
            write_csv(iterations_path, iteration_rows)
            save_stats(sample_rows, iteration_rows, args.out_dir / "sparse_ct_stats.json")
            print(
                f"Completed res={resolution} sample={sample_idx}: "
                f"image_mse={result.image_mse:.3e}, elapsed={result.elapsed_s:.1f}s",
                flush=True,
            )
        del model, dataset
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Compare exponential and Hutchinson-trace baselines for DLR + D-Flow.

Runs one held-out inverse problem per requested trained resolution. Single-circle
models use Gaussian blur; Shepp-Logan models use sparse-view Radon measurements.
The benchmark records the error of the approximate sensitivity action against
autograd through the same discrete Euler flow, inversion quality, and runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dlr_dflow.dlr_dflow as dd
from models.unet.default import Unet


CASES = (
    ("circle", 64),
    ("circle", 96),
    ("circle", 128),
    ("circle", 160),
    ("shepp_logan", 64),
    ("shepp_logan", 128),
)
BASELINES = ("exponential", "hutchinson_trace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--outer-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--trace-probes", type=int, default=8)
    parser.add_argument("--alpha-eps", type=float, default=0.05)
    parser.add_argument("--residual-eps", type=float, default=1e-4)
    parser.add_argument("--blur-sigma", type=float, default=5.0)
    parser.add_argument("--angles", type=int, default=15)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--families", nargs="+", choices=["circle", "shepp_logan"],
        default=["circle", "shepp_logan"],
    )
    parser.add_argument("--resolutions", nargs="+", type=int)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/dlr_dflow_baseline_comparison"),
    )
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_dir(family: str, resolution: int) -> Path:
    if family == "circle":
        return ROOT / f"outputs/fm_unet/circles-{resolution}/n1"
    return ROOT / f"outputs/fm_unet/shepp-logan-{resolution}"


def data_path(family: str, resolution: int) -> Path:
    stem = f"circles-{resolution}-n1" if family == "circle" else f"shepp-logan-{resolution}"
    return ROOT / f"data/{stem}.pt"


def load_model(family: str, resolution: int, device: torch.device) -> torch.nn.Module:
    run = model_dir(family, resolution)
    config = json.loads((run / "config.json").read_text())
    arch = config["arch"]
    model = Unet(
        ch=arch["ch"], ch_mul=arch["ch_mul"],
        att_channels=arch["att_channels"], groups=arch["groups"], dropout=0.0,
    ).to(device)
    state = torch.load(
        run / "checkpoints/epoch_0050.pt", map_location=device, weights_only=True)
    state = {
        key.removeprefix("module._orig_mod.").removeprefix("module."): value
        for key, value in state.items() if key != "n_averaged"
    }
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model


def load_sample(family: str, resolution: int, index: int, device: torch.device) -> torch.Tensor:
    dataset = torch.load(data_path(family, resolution), map_location="cpu", weights_only=True, mmap=True)
    sample = dataset["test"][index:index + 1].to(device=device, dtype=torch.float32)
    if dataset["test"].dtype == torch.uint8:
        sample = sample.div(255.0)
    return sample


def sparse_radon(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Differentiable parallel-beam Radon transform used by the sparse-CT benchmark."""
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


def make_operator(family: str, args: argparse.Namespace, device: torch.device):
    if family == "circle":
        return dd.make_blur(args.blur_sigma, str(device)), "Gaussian blur"
    angles = torch.linspace(0, math.pi, args.angles + 1, device=device)[:-1]
    return lambda x: sparse_radon(x, angles), f"Radon transform ({args.angles} views)"


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def exact_action(model, z, flow_steps: int, g_x: torch.Tensor, device: torch.device):
    sync(device)
    start = time.perf_counter()
    result = dd.exact_source_gradient(model, z, flow_steps, g_x)
    sync(device)
    return result, time.perf_counter() - start


def dlr_action(model, z, g_x, args, baseline: str, device: torch.device):
    direction = g_x.reshape(-1, 1) / (g_x.norm() + 1e-12)
    sync(device)
    start = time.perf_counter()
    x1, factors, alpha1, alpha_path = dd.evolve_sensitivity(
        model, z, args.flow_steps, rank=args.rank, eps=args.residual_eps,
        seed=args.seed, seed_dir=direction, baseline=baseline,
        alpha_eps=args.alpha_eps, trace_probes=args.trace_probes,
        return_alpha_path=True,
    )
    gradient = dd.approx_source_gradient(factors, g_x, alpha1)
    sync(device)
    elapsed = time.perf_counter() - start
    return x1, gradient, alpha1, alpha_path, elapsed


def run_inverse(model, z0, x_true, observation, operator, args, baseline, device):
    z = z0.clone()
    first_moment = torch.zeros_like(z)
    second_moment = torch.zeros_like(z)
    previous_direction = None
    best = {
        "measurement_mse": float("inf"), "image_mse": float("inf"),
        "step": 0, "source": z.detach().clone(), "alpha1": float("nan"),
    }
    outer_times = []
    alpha_path = None

    sync(device)
    total_start = time.perf_counter()
    for step in range(1, args.outer_steps + 1):
        sync(device)
        outer_start = time.perf_counter()
        x1, factors, alpha1, alpha_path = dd.evolve_sensitivity(
            model, z, args.flow_steps, rank=args.rank, eps=args.residual_eps,
            seed=args.seed, seed_dir=previous_direction, baseline=baseline,
            alpha_eps=args.alpha_eps, trace_probes=args.trace_probes,
            return_alpha_path=True,
        )
        g_x = dd.terminal_gradient(x1, operator, observation)
        gradient = dd.approx_source_gradient(factors, g_x, alpha1)
        sync(device)
        outer_times.append(time.perf_counter() - outer_start)

        with torch.no_grad():
            measurement_mse = (operator(x1) - observation).square().mean().item()
            image_mse = (x1.clamp(0, 1) - x_true).square().mean().item()
        if math.isfinite(measurement_mse) and measurement_mse < best["measurement_mse"]:
            best.update(
                measurement_mse=measurement_mse, image_mse=image_mse, step=step,
                source=z.detach().clone(), alpha1=alpha1,
            )

        previous_direction = gradient.detach().reshape(-1, 1)
        previous_direction = previous_direction / (previous_direction.norm() + 1e-12)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient.square()
        first_hat = first_moment / (1.0 - 0.9 ** step)
        second_hat = second_moment / (1.0 - 0.999 ** step)
        z = (z - args.lr * first_hat / (second_hat.sqrt() + 1e-8)).detach()

    sync(device)
    total_time = time.perf_counter() - total_start
    return best, total_time, sum(outer_times) / len(outer_times), alpha_path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["device"] = str(device)
    settings["out_dir"] = str(args.out_dir)
    settings["baselines"] = list(BASELINES)
    settings["cases"] = [list(case) for case in CASES]
    settings["cpu_threads"] = torch.get_num_threads()
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    results_path = args.out_dir / "results.csv"
    alpha_path_file = args.out_dir / "alpha_paths.csv"
    rows = read_rows(results_path)
    alpha_rows = read_rows(alpha_path_file)
    completed = {(row["family"], int(row["resolution"]), row["baseline"]) for row in rows}

    selected = [
        case for case in CASES
        if case[0] in args.families
        and (args.resolutions is None or case[1] in args.resolutions)
    ]
    for case_index, (family, resolution) in enumerate(selected):
        if all((family, resolution, baseline) in completed for baseline in BASELINES):
            print(f"Skipping completed case {family} {resolution}x{resolution}", flush=True)
            continue
        print(f"Loading {family} {resolution}x{resolution}", flush=True)
        model = load_model(family, resolution, device)
        x_true = load_sample(family, resolution, args.sample_index, device)
        operator, operator_name = make_operator(family, args, device)
        with torch.no_grad():
            observation = operator(x_true)
        generator = torch.Generator(device=device).manual_seed(args.seed + case_index)
        z0 = torch.randn(x_true.shape, generator=generator, device=device)

        x1_reference = dd.euler_flow(model, z0, args.flow_steps)
        g_x = dd.terminal_gradient(x1_reference, operator, observation)
        exact_gradient, exact_time = exact_action(model, z0, args.flow_steps, g_x, device)
        exact_norm = exact_gradient.norm().item()

        for baseline in BASELINES:
            key = (family, resolution, baseline)
            if key in completed:
                print(f"Skipping completed {key}", flush=True)
                continue
            _, approximate_gradient, alpha1_diag, diagnostic_path, dlr_time = dlr_action(
                model, z0, g_x, args, baseline, device)
            relative_error = (
                (approximate_gradient - exact_gradient).norm().item() / (exact_norm + 1e-30))
            cosine = torch.nn.functional.cosine_similarity(
                approximate_gradient.reshape(1, -1), exact_gradient.reshape(1, -1)).item()

            best, total_time, mean_outer_time, final_alpha_path = run_inverse(
                model, z0, x_true, observation, operator, args, baseline, device)
            row = {
                "family": family,
                "resolution": resolution,
                "operator": operator_name,
                "sample_index": args.sample_index,
                "baseline": baseline,
                "rank": args.rank,
                "flow_steps": args.flow_steps,
                "outer_steps": args.outer_steps,
                "alpha1_diagnostic": alpha1_diag,
                "gradient_rel_error": relative_error,
                "gradient_cosine": cosine,
                "exact_action_time_s": exact_time,
                "dlr_action_time_s": dlr_time,
                "dlr_over_exact_time": dlr_time / exact_time,
                "best_step": best["step"],
                "best_measurement_mse": best["measurement_mse"],
                "best_image_mse": best["image_mse"],
                "best_alpha1": best["alpha1"],
                "inverse_total_time_s": total_time,
                "mean_outer_time_s": mean_outer_time,
            }
            rows.append(row)
            for path_kind, values in (("diagnostic", diagnostic_path), ("final_inverse", final_alpha_path)):
                for time_index, value in enumerate(values):
                    alpha_rows.append({
                        "family": family, "resolution": resolution,
                        "baseline": baseline, "path": path_kind,
                        "time": time_index / args.flow_steps, "alpha": value,
                    })
            write_rows(results_path, rows)
            write_rows(alpha_path_file, alpha_rows)
            print(
                f"Completed {family} {resolution} {baseline}: "
                f"action_err={relative_error:.3e} cos={cosine:.3f} "
                f"image_mse={best['image_mse']:.3e} time={total_time:.1f}s",
                flush=True,
            )

        del model, x_true, observation, z0, exact_gradient
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

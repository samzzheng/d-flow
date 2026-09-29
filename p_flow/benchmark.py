"""CPU pilot: supplied P-Flow versus matched projected D-Flow and fast DLR.

Calibrate a fixed learning rate on validation image zero, then compare on
held-out images with identical sources, noise, ODE grids and update counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors, terminal_gradient
from sampler.ode_samplers import euler_flow
from .p_flow import ForwardOperator, GradientEvaluation, project_source, proxy_gradient, step

METHODS = ("pflow", "dflow", "fast_dlr")


def exact_gradient(model: torch.nn.Module, source: torch.Tensor,
                   operator: ForwardOperator, measurement: torch.Tensor,
                   n_steps: int) -> GradientEvaluation:
    z = source.detach().requires_grad_(True)
    endpoint = euler_flow(model, z, n_steps)
    loss = .5 * (operator(endpoint) - measurement).square().sum()
    gradient, = torch.autograd.grad(loss, z)
    return endpoint.detach(), loss.detach(), gradient.detach()


def metrics(endpoint: torch.Tensor, truth: torch.Tensor, operator: ForwardOperator,
            measurement: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        mse = float((endpoint - truth).square().mean())
        clipped_mse = float((endpoint.clamp(0, 1) - truth).square().mean())
        return {"mse": mse, "clipped_mse": clipped_mse,
                "psnr": -10 * math.log10(max(mse, 1e-30)),
                "clipped_psnr": -10 * math.log10(max(clipped_mse, 1e-30)),
                "measurement_mse": float((operator(endpoint) - measurement).square().mean())}


def run_trial(model: torch.nn.Module, source: torch.Tensor, truth: torch.Tensor,
              operator: ForwardOperator, measurement: torch.Tensor, method: str,
              learning_rate: float, iterations: int, args: argparse.Namespace
              ) -> tuple[dict[str, Any], torch.Tensor]:
    z = source.clone()
    history = []
    factors = None
    alpha = 0.
    for index in range(iterations):
        started = time.perf_counter()
        if method == "pflow":
            updated, (endpoint, loss, gradient) = step(
                model, z, operator, measurement, learning_rate, args.steps)
        else:
            if method == "dflow":
                endpoint, loss, gradient = exact_gradient(model, z, operator, measurement, args.steps)
            else:
                if index % args.refresh_every == 0:
                    endpoint, factors, alpha = sensitivity_factors(
                        model, z, args.steps, rank=args.rank, speed_profile="fast", seed=args.seed)
                else:
                    with torch.no_grad():
                        endpoint = euler_flow(model, z, args.steps)
                gx = terminal_gradient(endpoint, operator, measurement)
                gradient = approx_source_gradient(factors, gx, alpha)
                with torch.no_grad():
                    loss = .5 * (operator(endpoint) - measurement).square().sum()
            updated = project_source(z - learning_rate * gradient)
        elapsed = time.perf_counter() - started
        history.append({"step": index, "seconds": elapsed, "phi": float(loss),
                        "source_norm": float(z.norm()), **metrics(endpoint, truth, operator, measurement)})
        z = updated.detach()
    with torch.no_grad():
        final_endpoint = euler_flow(model, z, args.steps)
    return {"method": method, "learning_rate": learning_rate, "history": history,
            "mean_iteration_seconds": statistics.mean(row["seconds"] for row in history),
            "median_iteration_seconds": statistics.median(row["seconds"] for row in history),
            "final_source_norm": float(z.norm()),
            "final": metrics(final_endpoint, truth, operator, measurement)}, final_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/fm_unet/circles-64-noattn-ch16-b8/n1"))
    parser.add_argument("--data", type=Path, default=Path("data/circles-64-n1.pt"))
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--calibration-iterations", type=int, default=5)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[.001, .01, .1, 1.])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--refresh-every", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=.05)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2909)
    parser.add_argument("--out", type=Path, default=Path("p_flow/outputs/pilot_64"))
    args = parser.parse_args()
    if min(args.steps, args.iterations, args.calibration_iterations, args.samples, args.refresh_every) < 1:
        parser.error("step, sample, and refresh counts must be positive")
    if args.out.exists():
        parser.error("Choose a new --out directory; existing results are not overwritten")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    config = json.loads((args.run_dir / "config.json").read_text())
    if config["arch"].get("mid_attn", True) or any(config["arch"]["att_channels"]):
        raise ValueError("This pilot requires a non-attention model")
    model, _ = load_flow_model(args.resolution, epoch=args.epoch, run_dir=args.run_dir, device="cpu")
    data = torch.load(args.data, map_location="cpu", weights_only=True, mmap=True)
    if data["test"].shape[-2:] != (args.resolution, args.resolution):
        raise ValueError("Dataset resolution does not match model configuration")
    coordinate = torch.arange(-4, 5, dtype=torch.float32)
    kernel1d = torch.exp(-.5 * (coordinate / 1.5).square())
    kernel1d /= kernel1d.sum()
    kernel = (kernel1d[:, None] * kernel1d[None, :]).reshape(1, 1, 9, 9)
    operator = lambda image: F.conv2d(image, kernel, padding=4)
    checkpoint = args.run_dir / f"checkpoints/epoch_{args.epoch:04d}.pt"
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "protocol": {
            "device": "cpu", "integrator": "Euler", "projection": "sqrt(d), all methods",
            "optimizer": "fixed-rate projected gradient descent, no Armijo or first-step normalization",
            "task": "9x9 Gaussian blur, sigma=1.5 pixels, zero padding, additive Gaussian noise",
            "loss": "0.5 * squared residual sum; no image clipping in optimization",
            "calibration": "validation sample 0, best final raw image MSE; equal candidate rates/budget",
            "test": "test split first samples, no test-set learning-rate selection",
            "dlr": "fast profile rank 8 by default, terminal alpha zero, refresh_every as configured",
            "timing": "forward + gradient + source update; excludes diagnostic metrics and final reporting solve",
            "caveat": "small pilot; matched projected-GD D-Flow baseline, not tuned LBFGS D-Flow",
        }, "calibration": [], "runs": [], "gradient_diagnostics": []}
    args.out.mkdir(parents=True)
    calibration_truth = data["val"][:1].float() / 255
    generator = torch.Generator().manual_seed(args.seed)
    calibration_source = project_source(torch.randn(calibration_truth.shape, generator=generator))
    calibration_measurement = operator(calibration_truth) + args.noise_std * torch.randn(calibration_truth.shape, generator=generator)
    with torch.no_grad():
        euler_flow(model, calibration_source, 5)
    selected = {}
    for method in METHODS:
        candidates = []
        for learning_rate in args.learning_rates:
            trial, _ = run_trial(model, calibration_source, calibration_truth, operator,
                                 calibration_measurement, method, learning_rate, args.calibration_iterations, args)
            candidates.append(trial)
            report["calibration"].append(trial)
            (args.out / "results.json").write_text(json.dumps(report, indent=2))
            print(f"calibration {method} lr={learning_rate:g}: PSNR={trial['final']['psnr']:.3f}, iteration={trial['mean_iteration_seconds']:.3f}s", flush=True)
        selected[method] = min(candidates, key=lambda trial: trial["final"]["mse"])["learning_rate"]
    report["selected_learning_rates"] = selected
    images = []
    for index in range(args.samples):
        truth = data["test"][index:index + 1].float() / 255
        generator = torch.Generator().manual_seed(args.seed + index + 1)
        source = project_source(torch.randn(truth.shape, generator=generator))
        measurement = operator(truth) + args.noise_std * torch.randn(truth.shape, generator=generator)
        # Shared-source diagnostics are separate from all solver timings.
        _, _, exact = exact_gradient(model, source, operator, measurement, args.steps)
        endpoint, _, proxy = proxy_gradient(model, source, operator, measurement, args.steps)
        terminal, factors, alpha = sensitivity_factors(model, source, args.steps, rank=args.rank, speed_profile="fast", seed=args.seed)
        dlr = approx_source_gradient(factors, terminal_gradient(terminal, operator, measurement), alpha)
        exact_tangent = exact - source * (source * exact).sum() / source.square().sum()
        for method, gradient in (("pflow", proxy), ("fast_dlr", dlr)):
            tangent = gradient - source * (source * gradient).sum() / source.square().sum()
            report["gradient_diagnostics"].append({
                "sample": index, "method": method,
                "cosine": float(F.cosine_similarity(gradient.flatten(), exact.flatten(), dim=0)),
                "relative_error": float((gradient - exact).norm() / exact.norm()),
                "norm_ratio": float(gradient.norm() / exact.norm()),
                "projected_cosine": float(F.cosine_similarity(tangent.flatten(), exact_tangent.flatten(), dim=0)),
                "projected_directional_derivative": -float((exact * tangent).sum()),
            })
        image_row = {"truth": truth, "measurement": measurement, "initial": endpoint}
        report.setdefault("initial_metrics", []).append({"sample": index, **metrics(endpoint, truth, operator, measurement)})
        for method in METHODS:
            trial, result = run_trial(model, source, truth, operator, measurement, method,
                                      selected[method], args.iterations, args)
            trial["sample"] = index
            report["runs"].append(trial)
            image_row[method] = result
            (args.out / "results.json").write_text(json.dumps(report, indent=2))
            print(f"test {index} {method}: PSNR={trial['final']['psnr']:.3f}, iteration={trial['mean_iteration_seconds']:.3f}s", flush=True)
        images.append(image_row)
        torch.save(images, args.out / "images.pt")
    summary = {}
    for method in METHODS:
        runs = [run for run in report["runs"] if run["method"] == method]
        summary[method] = {key: statistics.mean(run["final"][key] for run in runs) for key in runs[0]["final"]}
        summary[method]["mean_iteration_seconds"] = statistics.mean(run["mean_iteration_seconds"] for run in runs)
    report["summary"] = summary
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Repeat a saved P-Flow pilot using shared projected LBFGS + Armijo."""

from __future__ import annotations

import argparse
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
from .armijo import ArmijoLBFGS
from .benchmark import METHODS, exact_gradient, metrics
from .p_flow import ForwardOperator, project_source


def run_trial(model: torch.nn.Module, initial: torch.Tensor, truth: torch.Tensor,
              operator: ForwardOperator, measurement: torch.Tensor, method: str,
              settings: dict[str, Any], args: argparse.Namespace
              ) -> tuple[dict[str, Any], torch.Tensor]:
    z = initial.detach().clone().requires_grad_(True)
    factors, alpha = None, 0.
    cached_source, cached_endpoint, cached_seconds = None, None, 0.
    last_endpoint = None
    trial_seconds, reused_trial_seconds, gradient_seconds = 0., 0., 0.
    gradient_evaluations = 0

    def objective(candidate: torch.Tensor) -> float:
        nonlocal cached_source, cached_endpoint, cached_seconds, trial_seconds
        started = time.perf_counter()
        with torch.no_grad():
            cached_endpoint = euler_flow(model, candidate, settings["steps"])
            value = .5 * (operator(cached_endpoint) - measurement).square().sum()
        cached_source = candidate.detach().clone()
        cached_seconds = time.perf_counter() - started
        trial_seconds += cached_seconds
        return float(value)

    def closure() -> torch.Tensor:
        nonlocal cached_source, cached_endpoint, cached_seconds, last_endpoint
        nonlocal reused_trial_seconds, gradient_seconds, gradient_evaluations
        started = time.perf_counter()
        if method == "dflow":
            endpoint, value, gradient = exact_gradient(model, z, operator, measurement, settings["steps"])
        else:
            if cached_source is not None and torch.equal(cached_source, z.detach()):
                endpoint = cached_endpoint
                reused_trial_seconds += cached_seconds
            else:
                with torch.no_grad():
                    endpoint = euler_flow(model, z.detach(), settings["steps"])
            gx = terminal_gradient(endpoint, operator, measurement)
            gradient = gx if method == "pflow" else approx_source_gradient(factors, gx, alpha)
            with torch.no_grad():
                value = .5 * (operator(endpoint) - measurement).square().sum()
        z.grad = gradient.detach()
        last_endpoint = endpoint.detach()
        cached_source, cached_endpoint, cached_seconds = None, None, 0.
        gradient_seconds += time.perf_counter() - started
        gradient_evaluations += 1
        return value.detach()

    optimizer = ArmijoLBFGS(z, objective, lr=args.lr, history_size=100,
                            max_ls=args.max_ls, step_cap=args.step_cap * math.sqrt(z.numel()),
                            project=args.project)
    history = []
    for index in range(args.iterations):
        started = time.perf_counter()
        trial_seconds, reused_trial_seconds, gradient_seconds = 0., 0., 0.
        before_evaluations = gradient_evaluations
        refresh_seconds = 0.
        if method == "fast_dlr" and index % settings["refresh_every"] == 0:
            refreshed = time.perf_counter()
            cached_endpoint, factors, alpha = sensitivity_factors(
                model, z.detach(), settings["steps"], rank=settings["rank"],
                speed_profile="fast", seed=settings["seed"])
            cached_source, cached_seconds = z.detach().clone(), 0.
            refresh_seconds = time.perf_counter() - refreshed
            # Curvature pairs cannot straddle different frozen DLR gradients.
            # Clearing history preserves lifetime first-step scaling.
            optimizer.state.clear()
        phi = optimizer.step(closure)
        elapsed = time.perf_counter() - started
        armijo_overhead = trial_seconds - reused_trial_seconds
        row = {"step": index + 1, "seconds": elapsed,
               "seconds_excluding_extra_trial_flows": elapsed - armijo_overhead,
               "trial_flow_seconds": trial_seconds, "reused_trial_flow_seconds": reused_trial_seconds,
               "gradient_seconds": gradient_seconds, "refresh_seconds": refresh_seconds,
               "phi": phi, "gradient_evaluations": gradient_evaluations - before_evaluations,
               "trial_evaluations": optimizer.last_function_evaluations,
               "source_norm": float(z.detach().norm()),
               **metrics(last_endpoint, truth, operator, measurement)}
        history.append(row)
        if (index + 1) % 5 == 0 or optimizer.consecutive_failures >= 2:
            print(f"{method} step={index + 1} relative_error={100 * row['relative_l2']:.2f}% trials={row['trial_evaluations']} seconds={elapsed:.2f}", flush=True)
        if optimizer.consecutive_failures >= 2:
            break
    return {"method": method, "learning_rate": args.lr, "history": history,
            "mean_iteration_seconds": statistics.mean(row["seconds"] for row in history),
            "mean_seconds_excluding_extra_trial_flows": statistics.mean(row["seconds_excluding_extra_trial_flows"] for row in history),
            "total_seconds": sum(row["seconds"] for row in history),
            "failed_searches": optimizer.failed_searches, "capped_steps": optimizer.capped_steps,
            "steps_run": len(history), "gradient_evaluations": gradient_evaluations,
            "trial_evaluations": sum(row["trial_evaluations"] for row in history),
            "final_source_norm": float(z.detach().norm()),
            "final": metrics(last_endpoint, truth, operator, measurement)}, last_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("p_flow/outputs/pilot_64_2026_09_29"))
    parser.add_argument("--out", type=Path, default=Path("p_flow/outputs/lbfgs_64_2026_09_29"))
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1.)
    parser.add_argument("--step-cap", type=float, default=.01)
    parser.add_argument("--max-ls", type=int, default=30)
    parser.add_argument("--project", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory")
    baseline = json.loads((args.baseline / "results.json").read_text())
    settings = baseline["settings"]
    images = torch.load(args.baseline / "images.pt", map_location="cpu", weights_only=True)
    if "summary" not in baseline or len(images) != settings["samples"]:
        parser.error("Baseline pilot must be complete")
    torch.set_num_threads(settings["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(settings["seed"])
    model, _ = load_flow_model(settings["resolution"], epoch=settings["epoch"], run_dir=settings["run_dir"], device="cpu")
    coordinate = torch.arange(-4, 5, dtype=torch.float32)
    kernel1d = torch.exp(-.5 * (coordinate / 1.5).square())
    kernel1d /= kernel1d.sum()
    kernel = (kernel1d[:, None] * kernel1d[None, :]).reshape(1, 1, 9, 9)
    operator = lambda image: F.conv2d(image, kernel, padding=4)
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline": str(args.baseline), "settings": {**settings, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "protocol": {"optimizer": "projected ambient LBFGS + Armijo" if args.project else "LBFGS + Armijo",
                     "initial_step_policy": ArmijoLBFGS.initial_step_policy,
                     "progress_policy": ArmijoLBFGS.progress_policy,
                     "step_cap": "fraction of sqrt(d)", "c1": 1e-4, "history_size": 100,
                     "dlr_history_reset": "on scheduled factor refresh; preserves lifetime step count",
                     "timing": "Includes line search; separate column excludes trial flows not reused for accepted gradients",
                     "samples": "Identical observations and regenerated initial sources from baseline pilot",
                     "stopping": "iteration budget or two consecutive failed Armijo searches"},
        "runs": []}
    args.out.mkdir(parents=True)
    with torch.no_grad():
        euler_flow(model, torch.zeros_like(images[0]["truth"]), 5)
    results = []
    for index, image in enumerate(images):
        generator = torch.Generator().manual_seed(settings["seed"] + index + 1)
        source = project_source(torch.randn(image["truth"].shape, generator=generator))
        result_images = {key: image[key] for key in ("truth", "measurement", "initial")}
        for method in METHODS:
            trial, endpoint = run_trial(model, source, image["truth"], operator, image["measurement"], method, settings, args)
            trial["sample"] = index
            report["runs"].append(trial)
            result_images[method] = endpoint
            (args.out / "results.json").write_text(json.dumps(report, indent=2))
            print(f"test {index} {method}: error={100 * trial['final']['relative_l2']:.2f}% mean={trial['mean_iteration_seconds']:.2f}s", flush=True)
        results.append(result_images)
        torch.save(results, args.out / "images.pt")
    summary = {}
    for method in METHODS:
        runs = [run for run in report["runs"] if run["method"] == method]
        values = {key: statistics.mean(run["final"][key] for run in runs) for key in runs[0]["final"]}
        for key in ("mean_iteration_seconds", "mean_seconds_excluding_extra_trial_flows", "total_seconds", "steps_run"):
            values[key] = statistics.mean(run[key] for run in runs)
        summary[method] = values
    report["summary"] = summary
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

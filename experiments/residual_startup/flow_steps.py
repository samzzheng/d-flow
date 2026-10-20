"""Paired coarse-grid timing and reconstruction with two-step residual startup."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import Evaluator, solve, write_json
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow
from scripts.reconstruction_figures import plot_reconstruction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/residual_startup_20261020"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 4, 3, 2])
    args = parser.parse_args()
    if min(args.steps) < 2:
        parser.error("Two-step initialization requires at least two Euler steps")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    baseline = json.loads((args.reference / "circles_ct/gaussian_two.json").read_text())["protocol"]
    settings = baseline["settings"]
    run = Path(settings["run_template"].format(resolution=64, circles=1))
    assert hashlib.sha256((run / f"checkpoints/epoch_{settings['epoch']:04d}.pt").read_bytes()).hexdigest() == baseline["checkpoint_sha256"]
    model, _ = load_flow_model(run_dir=run, epoch=settings["epoch"], device=str(device))
    inputs = torch.load(args.reference / "circles_ct/paired_inputs.pt", weights_only=True)
    products = GraphedAProducts(model, inputs["initial"].to(device))
    rows = []
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for index, problem in enumerate(("ct", "mri")):
        path = args.reference / f"circles_{problem}/paired_inputs.pt"
        inputs = torch.load(path, weights_only=True)
        truth, initial, clean, observed = (
            inputs[name].to(device) for name in ("truth", "initial", "clean", "observed"))
        operator = CT(64, 10, device) if problem == "ct" else MRI(64, 4, 607, device)
        assert operator.metadata == inputs["operator"]
        torch.testing.assert_close(operator(truth), clean, rtol=0, atol=0)
        records = []
        for steps in args.steps:
            case = args.out / f"{problem}_steps{steps}"
            case.mkdir(exist_ok=True)
            config = deepcopy(settings)
            config["flow_steps"] = steps
            protocol = {**baseline, "settings": config,
                        "paired_inputs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            evaluator = Evaluator(model, operator, observed, config, 609, products)
            evaluator.method = "DLR D-Flow"
            approx = evaluator.gradient(initial, "DLR D-Flow")
            source = initial.clone().requires_grad_(True)
            image = euler_flow(model, source, steps)
            exact, = torch.autograd.grad(.5 * (operator(image) - observed).abs().square().sum(), source)
            options = {**config["dlr"], "seed": 609}
            eager_image, factors, alpha = sensitivity_factors(model, initial, steps, **options)
            _, gx = endpoint_gradient(image.detach(), operator, observed)
            eager_gradient = approx_source_gradient(factors, gx, alpha)
            graph_error = float((approx - eager_gradient).norm() / eager_gradient.norm())
            endpoint_error = float((eager_image - image.detach()).norm() / image.detach().norm())
            assert graph_error < .01 and endpoint_error < 1e-4
            timings = {}
            for name, operation in (("gradient", lambda: evaluator.gradient(initial, "DLR D-Flow")),
                                    ("objective", lambda: evaluator.objective(initial))):
                operation()
                batches = []
                for _ in range(5):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    for _ in range(10):
                        operation()
                    torch.cuda.synchronize(device)
                    batches.append((time.perf_counter() - started) / 10)
                timings[f"{name}_seconds"] = statistics.median(batches)
                timings[f"{name}_batch_seconds"] = batches
            evaluator = Evaluator(model, operator, observed, config, 609, products)
            result = solve(evaluator, "DLR D-Flow", initial, truth, clean,
                           case / "result.pt", protocol, [False])
            final = result["history"][-1]
            row = dict(problem=problem, flow_steps=steps, status=result["status"],
                       attempts=final["step"], image_error_percent=100 * final["image_relative_l2"],
                       clean_measurement_error_percent=100 * final["measurement_relative_l2_clean"],
                       observed_residual_percent=100 * final["measurement_relative_l2_noisy"],
                       elapsed_seconds=result["elapsed_seconds"],
                       gradient_relative_error=float((approx - exact).norm() / exact.norm()),
                       graph_vs_eager_relative=graph_error, endpoint_relative_error=endpoint_error,
                       evaluations=result["evaluations"], **timings)
            rows.append(row)
            records.append(result)
            write_json(args.out / "summary.json", rows)
            print(json.dumps(row), flush=True)
            panel = plot_reconstruction_comparison(
                truth.cpu(), clean.cpu(), {"DLR D-Flow": (result["image"], result["measurement"])},
                observation=observed.cpu(), measurement_log=problem == "mri",
                title=f"Resolution: 64², Inverse Problem: {problem.upper()}, Noise Level 5%, Sample: 0\n"
                      f"{steps} Euler steps; two-step initialization")
            panel.savefig(case / "reconstruction.png", dpi=120)
            plt.close(panel)
            history = result["history"]
            for ax, coordinate in zip(axes[index], ("step", "elapsed_seconds")):
                ax.plot([h[coordinate] for h in history],
                        [100 * h["image_relative_l2"] for h in history], label=f"{steps} Euler steps")
        common = min(result["history"][-1]["step"] for result in records)
        for row, result in zip(rows[-len(records):], records):
            row["common_attempt"] = common
            row["common_image_error_percent"] = 100 * result["history"][common]["image_relative_l2"]
        for ax, label in zip(axes[index], ("Optimizer attempts", "Solve wall time (s)")):
            ax.set_xlabel(label)
            ax.set_ylabel(f"{problem.upper()} image error (%)")
            ax.legend()
            ax.grid(alpha=.2)
    figure.savefig(args.out / "convergence.png", dpi=150)
    plt.close(figure)
    write_json(args.out / "summary.json", rows)
    with (args.out / "summary.csv").open("w") as stream:
        fields = [key for key in rows[0] if not key.endswith("batch_seconds") and key != "evaluations"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out / "complete.json", {"solves": len(rows), "status": "complete"})


if __name__ == "__main__":
    main()

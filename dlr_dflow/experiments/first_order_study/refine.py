"""Refine saved-source spectra within the original deadline.

Writes a separate artifact so a concurrent study process cannot overwrite it.
Two threads here plus two in the primary runner stay within the four-thread cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from datetime import datetime
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model, make_terminal_flow_products, terminal_gradient
from . import diagnostics
from .diagnostics import reference_svd
from .study import BudgetExpired, Problem, Radon, Record, Tensor, timeout_handler
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--both-constraints", action="store_true")
    parser.add_argument("--output-name", default="spectral_refinement.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM, timeout_handler)
    report = json.loads((args.out / "results.json").read_text())
    destination = args.out / args.output_name
    if destination.exists():
        parser.error("Existing spectral refinement is not overwritten")
    settings = report["settings"]
    remaining = datetime.fromisoformat(report["created_at"]).timestamp() + settings["hours"] * 3600 - 300 - time.time()
    deadline = time.monotonic() + remaining
    if remaining <= 0:
        parser.error("Original pilot deadline has passed")
    checkpoint = Path(settings["checkpoint"])
    model, _ = load_flow_model(64, epoch=100, device="cpu", run_dir=checkpoint.parents[1])
    product_model = model
    if args.device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        product_model, _ = load_flow_model(64, epoch=100, device=args.device, run_dir=checkpoint.parents[1])
    inputs = torch.load(args.out / "inputs.pt", weights_only=True)["test"][0]
    states = torch.load(args.out / "states.pt", weights_only=True)
    problem = Problem(model, inputs["truth"], inputs["source"], Radon(), inputs["clean"], inputs["noisy"], settings["seed"] + 21000)
    source_runs = [r for r in report["comparisons"] if r["sample"] == 0 and r["estimator"] == "exact"
                   and r["optimizer"] == "gd" and (args.both_constraints or not r["projected"])]
    result: Record = {"created_at": datetime.now().astimezone().isoformat(), "threads": 2, "device": args.device,
        "checkpoint_sha256": settings["checkpoint_sha256"],
        "diagnostics_sha256": hashlib.sha256(Path(diagnostics.__file__).read_bytes()).hexdigest(),
        "refine_cutoff": True, "points": []}
    saved_factors: Record = {}
    points = [(run, iteration) for run in source_runs for iteration in sorted({0, run["attempts"] // 2, run["attempts"]})]
    for source_run, iteration in points:
        if time.monotonic() >= deadline:
            break
        point: Record = {"projected": source_run["projected"], "iteration": iteration, "source_run": source_run["label"],
                         "status": "running", "spectral": {}}
        result["points"].append(point)
        started = time.perf_counter()
        try:
            signal.setitimer(signal.ITIMER_REAL, max(.001, min(900, deadline - time.monotonic())))
            z = states[source_run["label"]]["sources"][iteration]
            x, _, exact = problem.evaluate(z, "exact")
            gx = terminal_gradient(x, problem.operator, problem.noisy)
            raw_J, raw_JT = make_terminal_flow_products(product_model, z.to(args.device), 100)
            batch_size = 2 if args.device == "cpu" else 32

            def J(q: Tensor) -> Tensor:
                return torch.cat([raw_J(chunk.to(args.device)).cpu() for chunk in q.split(batch_size, dim=1)], dim=1)

            def JT(q: Tensor) -> Tensor:
                return torch.cat([raw_JT(chunk.to(args.device)).cpu() for chunk in q.split(batch_size, dim=1)], dim=1)

            if args.device != "cpu":
                with torch.no_grad():
                    gpu_x = euler_flow(product_model, z.to(args.device), 100).cpu()
                endpoint_error = float((gpu_x - x).norm() / x.norm())
                gradient_error = float((JT(gx.reshape(-1, 1)).reshape_as(exact) - exact).norm() / exact.norm())
                point["cpu_gpu_parity"] = {"endpoint_relative_error": endpoint_error, "gradient_relative_error": gradient_error}
                if endpoint_error > 1e-4 or gradient_error > 1e-3:
                    raise AssertionError(f"CPU/GPU map mismatch: {point['cpu_gpu_parity']}")
            factors = reference_svd(J, JT, gx, exact, 8, problem.seed + 40000,
                                    min(deadline, time.monotonic() + 890), point["spectral"],
                                    max_refinements=8, refine_cutoff=True)
            if factors is not None:
                u, s, v = factors
                saved_factors[f"{source_run['label']}_{iteration}"] = {"factors": factors, "gradient": v @ s.T @ u.T @ gx.reshape(-1, 1)}
                if args.device != "cpu":
                    cpu_J, cpu_JT = make_terminal_flow_products(model, z, 100)
                    left = torch.cat([cpu_J(chunk) for chunk in v.split(2, dim=1)], dim=1) - u @ s
                    right = torch.cat([cpu_JT(chunk) for chunk in u.split(2, dim=1)], dim=1) - v @ s.T
                    residual = max(float(left.norm(dim=0).max()), float(right.norm(dim=0).max())) / float(s[0, 0])
                    point["spectral"]["cpu_verified_triplet_residual"] = residual
                    gap = point["spectral"]["refinements"][-1]["relative_cutoff_gap"]
                    if residual >= 1e-3:
                        point["spectral"]["status"] = "uncertified_cpu_residual"
                    elif gap is not None and gap <= max(2 * residual, 1e-6):
                        point["spectral"]["status"] = "ambiguous_cutoff"
                    point["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
            point["status"] = "complete"
        except BudgetExpired:
            point["status"] = "budget_limited"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        point["seconds"] = time.perf_counter() - started
        point["finished_at"] = datetime.now().astimezone().isoformat()
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(destination)
        torch.save(saved_factors, destination.with_suffix(".pt"))
        print(f"Refined sphere={source_run['projected']} source {iteration}: {point['spectral'].get('status', 'incomplete')}; {point['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

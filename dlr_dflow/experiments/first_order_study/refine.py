"""Spend spare CPU capacity refining saved-source spectra within the original deadline.

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM, timeout_handler)
    report = json.loads((args.out / "results.json").read_text())
    destination = args.out / "spectral_refinement.json"
    if destination.exists():
        parser.error("Existing spectral refinement is not overwritten")
    settings = report["settings"]
    remaining = datetime.fromisoformat(report["created_at"]).timestamp() + settings["hours"] * 3600 - 300 - time.time()
    deadline = time.monotonic() + remaining
    if remaining <= 0:
        parser.error("Original pilot deadline has passed")
    checkpoint = Path(settings["checkpoint"])
    model, _ = load_flow_model(64, epoch=100, device="cpu", run_dir=checkpoint.parents[1])
    inputs = torch.load(args.out / "inputs.pt", weights_only=True)["test"][0]
    states = torch.load(args.out / "states.pt", weights_only=True)
    problem = Problem(model, inputs["truth"], inputs["source"], Radon(), inputs["clean"], inputs["noisy"], settings["seed"] + 21000)
    source_run = next(r for r in report["comparisons"] if r["label"] == "test0_exact_gd_False")
    result: Record = {"created_at": datetime.now().astimezone().isoformat(), "threads": 2,
        "checkpoint_sha256": settings["checkpoint_sha256"], "source_run": source_run["label"],
        "diagnostics_sha256": hashlib.sha256(Path(diagnostics.__file__).read_bytes()).hexdigest(),
        "refine_cutoff": True, "points": []}
    for iteration in sorted({0, source_run["attempts"] // 2, source_run["attempts"]}):
        if time.monotonic() >= deadline:
            break
        point: Record = {"projected": False, "iteration": iteration, "status": "running", "spectral": {}}
        result["points"].append(point)
        started = time.perf_counter()
        try:
            signal.setitimer(signal.ITIMER_REAL, max(.001, min(900, deadline - time.monotonic())))
            z = states[source_run["label"]]["sources"][iteration]
            x, _, exact = problem.evaluate(z, "exact")
            gx = terminal_gradient(x, problem.operator, problem.noisy)
            raw_J, raw_JT = make_terminal_flow_products(model, z, 100)

            def J(q: Tensor) -> Tensor:
                return torch.cat([raw_J(chunk) for chunk in q.split(2, dim=1)], dim=1)

            def JT(q: Tensor) -> Tensor:
                return torch.cat([raw_JT(chunk) for chunk in q.split(2, dim=1)], dim=1)

            reference_svd(J, JT, gx, exact, 8, problem.seed + 40000,
                          min(deadline, time.monotonic() + 890), point["spectral"],
                          max_refinements=8, refine_cutoff=True)
            point["status"] = "complete"
        except BudgetExpired:
            point["status"] = "budget_limited"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        point["seconds"] = time.perf_counter() - started
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(destination)
        print(f"Refined source {iteration}: {point['spectral'].get('status', 'incomplete')}; {point['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

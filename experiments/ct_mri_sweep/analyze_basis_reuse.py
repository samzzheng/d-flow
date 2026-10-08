"""Compare saved baseline/reused-basis gradients on matched CPU sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dlr_dflow.api import load_flow_model
from sampler.ode_samplers import euler_flow
from .operators import CT, MRI
from .run import write_json
from .test_basis_reuse import CASES


def gradient_metrics(approximate: torch.Tensor, exact: torch.Tensor,
                     basis: torch.Tensor) -> dict[str, float | None]:
    a, g = approximate.flatten(), exact.flatten()
    norm = float(g.norm())
    projected = basis @ (basis.T @ g)
    error, missing, coefficient = (a-g).norm(), (projected-g).norm(), (a-projected).norm()
    return {"exact_norm": norm, "approximate_norm": float(a.norm()), "absolute_error": float(error),
            "relative_error": float(error)/norm if norm else None,
            "cosine": float((a@g)/(a.norm()*g.norm())) if norm and float(a.norm()) else None,
            "subspace_relative_error": float(missing)/norm if norm else None,
            "coefficient_relative_error": float(coefficient)/norm if norm else None,
            "negative_gradient_true_slope": -float(a@g)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    study = json.loads((args.output / "study.json").read_text())
    root = Path(study["root"])
    model, _ = load_flow_model(64, 1, 200, "cpu", Path("outputs/fm_unet/circles-64-noattn-ch16-b8-e200/n1"))
    results: list[dict[str, Any]] = []
    for case in CASES:
        inputs = torch.load(root / case / "inputs.pt", weights_only=True, map_location="cpu")
        job, acquisition = inputs["protocol"]["job"], inputs["protocol"]["operator"]
        operator = (CT(64, job["level"], torch.device("cpu")) if job["problem"] == "ct" else
                    MRI(64, job["level"], acquisition["mask_seed"], torch.device("cpu")))
        first = {variant: torch.load(args.output / case / variant / "first_update.pt", weights_only=True)
                 for variant in ("random16", "previous_v8")}
        assert torch.equal(first["random16"]["source"], first["previous_v8"]["source"])
        source = first["random16"]["source"].requires_grad_(True)
        image = euler_flow(model, source, 100)
        phi = .5 * (operator(image) - inputs["observed"]).abs().square().sum(dtype=torch.float64)
        exact, = torch.autograd.grad(phi, source)
        result: dict[str, Any] = {"case": case, "first_shared_update": {
            variant: gradient_metrics(record["gradient"], exact, record["v"]) for variant, record in first.items()}, "terminal": {}}
        del image, phi, exact, source
        for variant in ("random16", "previous_v8"):
            state = torch.load(args.output / case / variant / "state.pt", weights_only=True)
            assert state["status"] in {"stalled", "step_limit"}
            source = state["source"].requires_grad_(True)
            image = euler_flow(model, source, 100)
            phi = .5 * (operator(image)-inputs["observed"]).abs().square().sum(dtype=torch.float64)
            exact, = torch.autograd.grad(phi, source)
            record = gradient_metrics(state["optimizer"]["g"], exact, state["previous_v"])
            record["relative_phi_replay_difference"] = abs(float(phi.detach()) - state["history"][-1]["phi"]) / max(abs(state["history"][-1]["phi"]), 1e-30)
            result["terminal"][variant] = record
            del source, image, phi, exact
        results.append(result)
        write_json(args.output / "gradient_diagnostics.json", results)
        print(json.dumps(result), flush=True)
    lines = ["\n## Same-source gradient comparison after the first accepted update", "",
             "Both runs have identical source, native endpoint, and terminal gradient here; all differences come from initialization.", "",
             "| Case | Relative gradient error %: random / reuse | Cosine: random / reuse |",
             "| --- | --- | --- |"]
    for result in results:
        a, b = (result["first_shared_update"][variant] for variant in ("random16", "previous_v8"))
        lines.append(f"| {result['case']} | {100*a['relative_error']:.3f} / {100*b['relative_error']:.3f} | {a['cosine']:.6f} / {b['cosine']:.6f} |")
    lines += ["", "Terminal-source diagnostics are saved in gradient_diagnostics.json. The two variants have different terminal sources."]
    with (args.output / "report.md").open("a") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""Paired CPU continuations from failed GPU sources, with spectral reweighting.

The median-cap variant clips singular values of the small terminal core to their
lower median before applying its transpose to the image gradient. This changes
gradient weighting, not the objective, rank, or model derivative product count.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from p_flow import endpoint_gradient
from p_flow.armijo import ArmijoLBFGS
from scripts.reconstruction_figures import plot_reconstruction_comparison
from .operators import CT, MRI
from .run import Evaluator, metrics, save_tensor, write_json

Record = dict[str, Any]
Variant = Literal["unchanged", "median_cap"]
CASES = ("r128_n3_ct10/sample_008", "r128_n2_mri4/sample_003")
VARIANTS: tuple[Variant, ...] = ("unchanged", "median_cap")


class ContinuationEvaluator(Evaluator):
    """Reuse right bases and optionally cap terminal singular-value weights."""

    def __init__(self, model: torch.nn.Module, operator: CT | MRI,
                 observed: torch.Tensor, settings: Record, seed: int,
                 variant: Variant) -> None:
        super().__init__(model, operator, observed, settings, seed)
        self.variant = variant
        self.diagnostic: Record = {}

    def gradient(self, source: torch.Tensor, method: str = "DLR D-Flow") -> torch.Tensor:
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        self.counts["gradients"] += 1
        _, terminal_gradient = endpoint_gradient(self.image, self.operator, self.observed)
        _, factors, alpha = sensitivity_factors(
            self.model, source.detach(), self.settings["flow_steps"],
            **self.settings["dlr"], seed=self.seed, seed_basis=self.previous_v)
        left, core, right = factors
        core_left, singular_values, core_right = torch.linalg.svd(core, full_matrices=False)
        median = singular_values.median()
        original_gradient = approx_source_gradient(factors, terminal_gradient, alpha)
        if self.variant == "median_cap":
            core = (core_left * singular_values.clamp(max=median)) @ core_right
        gradient = (original_gradient if self.variant == "unchanged" else
                    approx_source_gradient((left, core, right), terminal_gradient, alpha)).detach()
        if alpha != 0 or not math.isfinite(self.value) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite gradient/objective or nonzero terminal alpha")
        self.previous_v = right.detach().clone()
        self.diagnostic = {"singular_values": singular_values.tolist(),
                           "median_singular_value": float(median),
                           "original_gradient_norm": float(original_gradient.norm()),
                           "gradient_norm": float(gradient.norm())}
        return gradient


def run_case(snapshot_path: Path, output: Path, case: str,
             variant: Variant, max_steps: int) -> Record:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "native"
    tangent_unet.GN_PRIMAL_BACKEND = "native"
    snapshot = json.loads(snapshot_path.read_text())
    settings, root = snapshot["settings"], Path(snapshot["root"])
    state_path = root / case / "dlr_d_flow.pt"
    original = torch.load(state_path, map_location="cpu", weights_only=True)
    inputs = torch.load(root / case / "inputs.pt", map_location="cpu", weights_only=True)
    frozen = next(item for item in snapshot["records"] if item["case"] == str(root / case))
    assert original["history"] == frozen["methods"]["DLR D-Flow"]["history"]
    protocol, job = original["protocol"], original["protocol"]["job"]
    destination = output / case / variant
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    current_path = destination / "state.pt"
    study_protocol = {"parent": protocol, "variant": variant, "max_steps": max_steps,
                      "settings": settings, "device": "cpu", "threads": 1,
                      "source_checkpoint": str(state_path),
                      "source_checkpoint_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "optimizer_reset": True,
                      "gradient_weighting": "unchanged" if variant == "unchanged" else
                      "clip terminal core singular values to torch.median (lower middle value for rank 8)"}
    saved = None
    if current_path.exists():
        saved = torch.load(current_path, map_location="cpu", weights_only=True)
        if saved["protocol"] != study_protocol:
            raise ValueError(f"Resume protocol mismatch: {current_path}")
        if saved["status"] in {"stalled", "step_limit"}:
            return json.loads((destination / "result.json").read_text())
    model_dir = Path(settings["run_template"].format(**job))
    checkpoint = model_dir / f"checkpoints/epoch_{settings['epoch']:04d}.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    model, _ = load_flow_model(job["resolution"], job["circles"], settings["epoch"], "cpu", model_dir)
    operator = (CT(job["resolution"], job["level"], torch.device("cpu")) if job["problem"] == "ct" else
                MRI(job["resolution"], job["level"], protocol["operator"]["mask_seed"], torch.device("cpu")))
    if isinstance(operator, MRI):
        assert torch.equal(operator.mask, inputs["mask"])
    assert torch.equal(inputs["observed"], inputs["clean"])
    evaluator = ContinuationEvaluator(model, operator, inputs["observed"], settings, protocol["dlr_seed"], variant)
    evaluator.previous_v = original["previous_v"].clone()
    source = original["source"].clone().requires_grad_(True)
    optimizer = ArmijoLBFGS(source, evaluator.objective, **{
        **settings["optimizer"], "backtracking": "geometric",
        "step_cap": .01 * math.sqrt(source.numel()), "project": False})
    history: list[Record] = []
    prior_seconds = 0.0
    if saved is not None:
        with torch.no_grad():
            source.copy_(saved["source"])
        for name, value in saved["optimizer"].items():
            setattr(optimizer, name, value)
        evaluator.previous_v, evaluator.counts = saved["previous_v"], saved["evaluations"]
        evaluator.diagnostic = saved["gradient_diagnostic"]
        history, prior_seconds = saved["history"], saved["elapsed_seconds"]
    started = time.monotonic()
    evaluator.objective(source.detach())

    def closure() -> torch.Tensor:
        source.grad = evaluator.gradient(source)
        return source.new_tensor(evaluator.value, dtype=torch.float64)

    if not history:
        initial_gradient = evaluator.gradient(source)
        optimizer.f, optimizer.g = evaluator.value, initial_gradient.flatten().clone()
        optimizer.state["initialized"] = True
        history.append({"step": 0, "phi": evaluator.value, "source_step_norm": 0.0,
                        "line_search_trials": 0, "gradient_diagnostic": evaluator.diagnostic,
                        **metrics(evaluator.image, operator(evaluator.image), inputs["truth"], inputs["clean"], inputs["observed"])})
        save_tensor(checkpoints / "step_000.pt", {"source": source, "previous_v": evaluator.previous_v,
                    "gradient": optimizer.g, "seed_basis": original["previous_v"], "history": history[-1]})
        write_json(destination / "initial.json", {"protocol": study_protocol, "history": history,
                   "gpu_terminal_phi": original["history"][-1]["phi"],
                   "cpu_endpoint_relative_drift": float((evaluator.image-original["image"]).norm()/original["image"].norm())})
        print(f"{case} {variant} initialized phi={evaluator.value:.8g} image={100*history[-1]['image_relative_l2']:.3f}%", flush=True)
    status = "running"
    while status == "running":
        before = source.detach().clone()
        optimizer.step(closure)
        if not torch.equal(evaluator.source, source.detach()):
            evaluator.objective(source.detach())
        displacement = float((source.detach() - before).norm())
        assert displacement <= optimizer.step_cap * (1 + 1e-4)
        assert evaluator.value <= history[-1]["phi"]
        if displacement:
            assert evaluator.value < history[-1]["phi"]
        history.append({"step": optimizer.step_attempts, "phi": evaluator.value,
                        "source_step_norm": displacement, "line_search_trials": optimizer.last_function_evaluations,
                        "gradient_diagnostic": evaluator.diagnostic,
                        **metrics(evaluator.image, operator(evaluator.image), inputs["truth"], inputs["clean"], inputs["observed"])})
        if optimizer.consecutive_failures >= 2:
            status = "stalled"
        elif optimizer.step_attempts >= max_steps:
            status = "step_limit"
        state = {"protocol": study_protocol, "source": source, "status": status,
                 "image": evaluator.image, "measurement": operator(evaluator.image).detach(),
                 "previous_v": evaluator.previous_v, "history": history,
                 "gradient_diagnostic": evaluator.diagnostic,
                 "optimizer": {key: value for key, value in vars(optimizer).items() if key not in {"source", "objective"}},
                 "evaluations": evaluator.counts, "elapsed_seconds": prior_seconds + time.monotonic() - started}
        save_tensor(current_path, state)
        save_tensor(checkpoints / f"step_{optimizer.step_attempts:03d}.pt", {
            "source": source, "previous_v": evaluator.previous_v, "gradient": optimizer.g, "history": history[-1]})
        result = {key: state[key] for key in ("protocol", "status", "history", "evaluations", "elapsed_seconds")}
        result.update(case=case, variant=variant)
        write_json(destination / "result.json", result)
        print(f"{case} {variant} step={optimizer.step_attempts} {status} phi={evaluator.value:.8g} image={100*history[-1]['image_relative_l2']:.3f}%", flush=True)
    fig = plot_reconstruction_comparison(
        inputs["truth"], inputs["clean"], {"DLR D-Flow": (evaluator.image, operator(evaluator.image))},
        observation=inputs["observed"], histories={"DLR D-Flow": history}, measurement_log=job["problem"] == "mri",
        title=f"{case} | CPU continuation: {variant} | {status}\n100 Euler steps; D-Flow and P-Flow unavailable in this paired continuation")
    fig.savefig(destination / "comparison.png", dpi=100)
    plt.close(fig)
    return result


def summarize(output: Path) -> None:
    results = [json.loads(path.read_text()) for path in sorted(output.glob("r*/sample_*/*/result.json"))]
    fig, axes = plt.subplots(len(CASES), 2, figsize=(12, 8), layout="constrained")
    lines = ["# CPU continuation pilot", "", "Selected severe saved GPU sources; fresh native CPU32 objective and optimizer.",
             "Both variants reuse V, rank 8, 100 Euler steps, analytical alpha, geometric Armijo history 100, and cap 1.28.",
             "Median cap clips to torch.median (lower middle value at rank 8), without extra model derivative products.",
             "Compare completed matched budgets; CPU continuation cannot certify a GPU rescue.", "",
             "| Case | Variant | Status | Attempts | Final image % | Final residual % |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for row, case in enumerate(CASES):
        for result in filter(lambda item: item["case"] == case, results):
            history, variant = result["history"], result["variant"]
            for column, key in enumerate(("image_relative_l2", "measurement_relative_l2_clean")):
                axes[row, column].plot([entry["step"] for entry in history], [100*entry[key] for entry in history], label=variant)
            last = history[-1]
            lines.append(f"| {case} | {variant} | {result['status']} | {last['step']} | {100*last['image_relative_l2']:.3f} | {100*last['measurement_relative_l2_clean']:.3f} |")
        for column, label in enumerate(("Image error (%)", "Clean measurement residual (%)")):
            axes[row, column].set(title=case, xlabel="Continuation optimizer attempt", ylabel=label)
            axes[row, column].grid(alpha=.2)
            if axes[row, column].lines:
                axes[row, column].legend()
    fig.savefig(output / "convergence.png", dpi=140)
    plt.close(fig)
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("outputs/ct_mri_geometric_v_audit_20261008_1645/snapshot.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dlr_failure_tests_20261008/continuations"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        write_json(args.output / "study.json", {"cases": CASES, "variants": VARIANTS, "max_steps": args.max_steps,
                   "workers": args.workers, "threads_per_worker": 1, "snapshot": str(args.snapshot),
                   "pid": os.getpid(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_case, args.snapshot, args.output, case, variant, args.max_steps): (case, variant)
                       for case in CASES for variant in VARIANTS}
            for future in as_completed(futures):
                print(json.dumps({"finished": futures[future], "status": future.result()["status"]}), flush=True)
                summarize(args.output)
    summarize(args.output)


if __name__ == "__main__":
    main()

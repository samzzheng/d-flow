"""Isolated paired CPU reconstruction test of previous-terminal-V sketch reuse.

Each gradient rebuilds the sensitivity factors from t=0. The warm variant
replaces eight of the existing sixteen sketch columns with the previous
accepted source's terminal right basis. Neither rank nor product width grows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from p_flow import endpoint_gradient
from p_flow.armijo import ArmijoLBFGS
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison
from .operators import CT, MRI
from .run import Evaluator, metrics, save_tensor, write_json

CASES = ("r64_n1_ct10/sample_003", "r64_n1_ct10/sample_008",
         "r64_n1_ct10/sample_022", "r64_n1_ct10/sample_028",
         "r64_n1_mri4/sample_000", "r64_n1_mri4/sample_003",
         "r64_n1_mri4/sample_018", "r64_n1_mri4/sample_026")
Record = dict[str, Any]


class ReuseEvaluator(Evaluator):
    """Advance the reusable basis only when the optimizer requests a gradient."""

    def __init__(self, model: torch.nn.Module, operator: CT | MRI, observed: torch.Tensor,
                 settings: Record, seed: int, warm_columns: int) -> None:
        super().__init__(model, operator, observed, settings, seed)
        self.warm_columns = warm_columns
        self.previous_v: torch.Tensor | None = None
        self.diagnostic: Record | None = None

    def gradient(self, source: torch.Tensor, method: str = "DLR D-Flow") -> torch.Tensor:
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        seed_basis = None if self.previous_v is None or not self.warm_columns else self.previous_v[:, :self.warm_columns]
        self.counts["gradients"] += 1
        _, gx = endpoint_gradient(self.image, self.operator, self.observed)
        _, factors, alpha = sensitivity_factors(
            self.model, source.detach(), self.settings["flow_steps"],
            **self.settings["dlr"], seed=self.seed, seed_basis=seed_basis)
        gradient = approx_source_gradient(factors, gx, alpha).detach()
        if alpha != 0 or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite gradient or nonzero terminal alpha")
        self.previous_v = factors[2].detach().clone()
        assert torch.allclose(self.previous_v.T @ self.previous_v,
                              torch.eye(self.previous_v.shape[1]), atol=2e-5, rtol=2e-5)
        # Save one matched-source diagnostic after the first accepted update.
        if self.counts["gradients"] == 2:
            self.diagnostic = {"source": source.detach().clone(), "gradient": gradient,
                               "gx": gx.detach().clone(), "image": self.image.detach().clone(),
                               "v": self.previous_v.clone(), "seed_basis": None if seed_basis is None else seed_basis.clone()}
        return gradient


def run_case(root: Path, output: Path, case: str, warm_columns: int,
             max_steps: int, threads: int) -> Record:
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "native"
    tangent_unet.GN_PRIMAL_BACKEND = "native"
    manifest = json.loads((root / "manifest.json").read_text())
    settings = manifest["settings"]
    inputs = torch.load(root / case / "inputs.pt", weights_only=True, map_location="cpu")
    protocol = inputs["protocol"]
    job = protocol["job"]
    variant = "random16" if warm_columns == 0 else f"previous_v{warm_columns}"
    destination = output / case / variant
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "state.pt"
    study_protocol = {"parent": protocol, "warm_columns": warm_columns, "max_steps": max_steps,
                      "device": "cpu", "threads": threads, "settings": settings,
                      "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if state_path.exists():
        saved = torch.load(state_path, weights_only=True, map_location="cpu")
        if saved["protocol"] != study_protocol:
            raise ValueError(f"Resume protocol mismatch: {state_path}")
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
    evaluator = ReuseEvaluator(model, operator, inputs["observed"], settings, protocol["dlr_seed"], warm_columns)
    source = inputs["initial"].clone().requires_grad_(True)
    optimizer = ArmijoLBFGS(source, evaluator.objective, **{
        **settings["optimizer"], "step_cap": .01 * math.sqrt(source.numel()), "project": False})
    history: list[Record] = []
    prior_seconds = 0.0
    if state_path.exists():
        with torch.no_grad():
            source.copy_(saved["source"])
        for name, value in saved["optimizer"].items():
            setattr(optimizer, name, value)
        evaluator.previous_v = saved["previous_v"]
        evaluator.counts = saved["evaluations"]
        history = saved["history"]
        prior_seconds = saved["elapsed_seconds"]
    started = time.monotonic()
    evaluator.objective(source.detach())
    if not history:
        history.append({"step": 0, "phi": evaluator.value,
                        **metrics(evaluator.image, operator(evaluator.image), inputs["truth"], inputs["clean"], inputs["observed"])})

    def closure() -> torch.Tensor:
        source.grad = evaluator.gradient(source)
        if evaluator.counts["gradients"] == 1:
            save_tensor(destination / "initial_gradient.pt", source.grad)
        if evaluator.diagnostic is not None:
            save_tensor(destination / "first_update.pt", evaluator.diagnostic)
            evaluator.diagnostic = None
        return source.new_tensor(evaluator.value, dtype=torch.float64)

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
                        **metrics(evaluator.image, operator(evaluator.image), inputs["truth"], inputs["clean"], inputs["observed"])})
        if optimizer.consecutive_failures >= 2:
            status = "stalled"
        elif optimizer.step_attempts >= max_steps:
            status = "step_limit"
        if optimizer.step_attempts % 5 == 0 or status != "running":
            state = {"protocol": study_protocol, "source": source.detach(), "status": status,
                     "image": evaluator.image, "measurement": operator(evaluator.image).detach(),
                     "previous_v": evaluator.previous_v, "history": history,
                     "optimizer": {key: value for key, value in vars(optimizer).items() if key not in {"source", "objective"}},
                     "evaluations": evaluator.counts, "elapsed_seconds": prior_seconds + time.monotonic() - started}
            save_tensor(state_path, state)
            result = {key: state[key] for key in ("protocol", "status", "history", "evaluations", "elapsed_seconds")}
            result.update(case=case, variant=variant)
            write_json(destination / "result.json", result)
            print(f"{case} {variant} step={optimizer.step_attempts} {status} image={100*history[-1]['image_relative_l2']:.3f}%", flush=True)
    fig = plot_reconstruction_comparison(
        inputs["truth"], inputs["clean"], {"DLR D-Flow": (evaluator.image, operator(evaluator.image))},
        observation=inputs["observed"], histories={"DLR D-Flow": history}, measurement_log=job["problem"] == "mri",
        residual_vmax=RESIDUAL_VMAX[job["problem"]],
        title=f"{case} | CPU {variant} | rank 8 | 100 Euler steps | {status}\nD-Flow / P-Flow not rerun in this initialization ablation")
    fig.savefig(destination / "comparison.png", dpi=100)
    plt.close(fig)
    return result


def summarize(output: Path) -> None:
    results = [json.loads(path.read_text()) for path in sorted(output.glob("r*/sample_*/*/result.json"))]
    pairs = {case: {r["variant"]: r for r in results if r["case"] == case} for case in CASES}
    lines = ["# Previous-terminal-V initialization pilot", "",
             "Paired CPU runs from original Gaussian sources; selected diagnostic cases, not a random population sample.",
             "Both use 100 Euler steps, rank 8, analytical alpha=1-t, zero enrichment, the same LBFGS/Armijo and cap.",
             "Random16 uses 16 random initialization columns. Previous-V8 replaces eight with previous terminal right vectors.",
             "Factors always restart from t=0. All remaining random columns and seeds are paired.", "",
             "| Case | Final image %: random / reuse | Final residual %: random / reuse | Attempts: random / reuse | Status: random / reuse |",
             "| --- | --- | --- | --- | --- |"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), layout="constrained")
    finished = []
    for axis, (case, pair) in zip(axes.flat, pairs.items()):
        axis.set_title(case.replace("r64_n1_", ""))
        axis.set(xlabel="Optimizer attempt", ylabel="Relative L2 (%)")
        for variant, result in pair.items():
            h = result["history"]
            axis.plot([v["step"] for v in h], [100*v["image_relative_l2"] for v in h], label=variant)
        if len(pair) != 2 or any(v["status"] not in {"stalled", "step_limit"} for v in pair.values()):
            continue
        a, b = pair["random16"], pair["previous_v8"]
        za = torch.load(output / case / "random16/initial_gradient.pt", weights_only=True)
        zb = torch.load(output / case / "previous_v8/initial_gradient.pt", weights_only=True)
        assert torch.equal(za, zb), "Initial gradients must match before any basis reuse"
        fa, fb = a["history"][-1], b["history"][-1]
        assert a["history"][1] == b["history"][1], "First accepted update must match"
        lines.append(f"| {case} | {100*fa['image_relative_l2']:.3f} / {100*fb['image_relative_l2']:.3f} | "
                     f"{100*fa['measurement_relative_l2_clean']:.3f} / {100*fb['measurement_relative_l2_clean']:.3f} | "
                     f"{fa['step']} / {fb['step']} | {a['status']} / {b['status']} |")
        finished.append((fa["image_relative_l2"], fb["image_relative_l2"]))
        axis.legend()
        axis.grid(alpha=.2)
    if finished:
        array = np.array(finished)
        lines += ["", f"Completed paired cases: {len(finished)}/{len(CASES)}.",
                  f"Final image error <=5%: random {(array[:, 0] <= .05).sum()}, reuse {(array[:, 1] <= .05).sum()}.",
                  f"Mean final image error (%): random {100*array[:, 0].mean():.3f}, reuse {100*array[:, 1].mean():.3f}.",
                  f"Median final image error (%): random {100*np.median(array[:, 0]):.3f}, reuse {100*np.median(array[:, 1]):.3f}.",
                  f"Reuse lower final image error: {(array[:, 1] < array[:, 0]).sum()}/{len(finished)}.",
                  "CPU results must not be treated as exact replays of the historical GPU trajectories."]
    fig.savefig(output / "convergence.png", dpi=140)
    plt.close(fig)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ct_mri_sweep_100steps_noiseless_20261007"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        tasks = [(case, warm) for case in CASES for warm in (0, 8)]
        write_json(args.output / "study.json", {"root": str(args.root), "cases": CASES, "warm_columns": [0, 8],
                   "max_steps": args.max_steps, "workers": args.workers, "threads_per_worker": args.threads,
                   "selection": "Diagnostic successes, failures and reversals from the earlier GPU sweep; not population-unbiased",
                   "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_case, args.root, args.output, case, warm, args.max_steps, args.threads) for case, warm in tasks]
            for future in as_completed(futures):
                future.result()
        summarize(args.output)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()

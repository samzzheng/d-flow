"""Paired CPU reconstructions isolating LBFGS history and Armijo backtracking."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import signal
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison
from .operators import CT, MRI
from .optimizer_ablation import ExperimentalArmijo, VARIANTS
from .queue import source_hashes
from .run import Evaluator, metrics, save_tensor, write_json
from .test_basis_reuse import CASES

Record = dict[str, Any]
TERMINAL = {"stalled", "step_limit", "objective_plateau"}


def run_case(root: Path, output: Path, case: str, variant: str, max_steps: int) -> Record:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "native"
    tangent_unet.GN_PRIMAL_BACKEND = "native"
    manifest = json.loads((root / "manifest.json").read_text())
    study = json.loads((output / "study.json").read_text())
    assert source_hashes() == study["source_sha256"] == manifest["source_sha256"]
    for name, expected in study["experiment_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    settings = manifest["settings"]
    inputs = torch.load(root / case / "inputs.pt", weights_only=True, map_location="cpu")
    protocol = inputs["protocol"]
    job = protocol["job"]
    destination = output / case / variant
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "state.pt"
    study_protocol = {"parent": protocol, "variant": variant, "policy": VARIANTS[variant],
                      "max_steps": max_steps, "device": "cpu", "threads": 1, "settings": settings,
                      "experiment_sha256": study["experiment_sha256"]}
    saved = None
    if state_path.exists():
        saved = torch.load(state_path, weights_only=True, map_location="cpu")
        if saved["protocol"] != study_protocol:
            raise ValueError(f"Resume protocol mismatch: {state_path}")
        if saved["status"] in TERMINAL:
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
    evaluator = Evaluator(model, operator, inputs["observed"], settings, protocol["dlr_seed"])
    source = inputs["initial"].clone().requires_grad_(True)
    optimizer_settings = {key: value for key, value in settings["optimizer"].items() if key != "history_size"}
    optimizer_settings["step_cap"] *= math.sqrt(source.numel())
    optimizer = ExperimentalArmijo(source, evaluator.objective, variant, **optimizer_settings, project=False)
    history: list[Record] = []
    prior_seconds = 0.
    if saved is not None:
        with torch.no_grad():
            source.copy_(saved["source"])
        for name, value in saved["optimizer"].items():
            setattr(optimizer, name, value)
        evaluator.counts = saved["evaluations"]
        history, prior_seconds = saved["history"], saved["elapsed_seconds"]
    interrupted = [False]

    def stop(signum: int, frame: Any) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    evaluator.objective(source.detach())
    if not history:
        history.append({"step": 0, "phi": evaluator.value,
                        **metrics(evaluator.image, operator(evaluator.image), inputs["truth"], inputs["clean"], inputs["observed"])})

    def closure() -> torch.Tensor:
        source.grad = evaluator.gradient(source, "DLR D-Flow")
        if evaluator.counts["gradients"] == 1:
            save_tensor(destination / "initial_gradient.pt", source.grad)
        return source.new_tensor(evaluator.value, dtype=torch.float64)

    status = "running"
    while status == "running":
        if interrupted[0]:
            status = "paused"
        else:
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
                            "optimizer": optimizer.last_diagnostics,
                            **metrics(evaluator.image, operator(evaluator.image), inputs["truth"], inputs["clean"], inputs["observed"])})
            if optimizer.consecutive_failures >= 2:
                status = "stalled"
            elif optimizer.step_attempts >= max_steps:
                status = "step_limit"
            elif optimizer.step_attempts >= 100 and optimizer.step_attempts % 20 == 0:
                values = [history[index]["phi"] for index in (-41, -21, -1)]
                if max((a - b) / max(abs(a), 1e-30) for a, b in zip(values, values[1:])) < 1e-4:
                    status = "objective_plateau"
        if optimizer.step_attempts % 5 == 0 or status != "running":
            state = {"protocol": study_protocol, "source": source.detach(), "status": status,
                     "image": evaluator.image, "measurement": operator(evaluator.image).detach(), "history": history,
                     "optimizer": {key: value for key, value in vars(optimizer).items() if key not in {"source", "objective"}},
                     "evaluations": evaluator.counts, "elapsed_seconds": prior_seconds + time.monotonic() - started}
            save_tensor(state_path, state)
            result = {key: state[key] for key in ("protocol", "status", "history", "evaluations", "elapsed_seconds")}
            result.update(case=case, variant=variant)
            write_json(destination / "result.json", result)
            print(f"{case} {variant} step={optimizer.step_attempts} {status} image={100*history[-1]['image_relative_l2']:.3f}%", flush=True)
    figure = plot_reconstruction_comparison(
        inputs["truth"], inputs["clean"], {"DLR D-Flow": (evaluator.image, operator(evaluator.image))},
        observation=inputs["observed"], histories={"DLR D-Flow": history}, measurement_log=job["problem"] == "mri",
        residual_vmax=RESIDUAL_VMAX[job["problem"]],
        title=f"{case} | CPU {variant} | {status} | rank 8 | 100 Euler steps\nDLR optimizer ablation; D-Flow/P-Flow unavailable")
    figure.savefig(destination / "comparison.png", dpi=100)
    plt.close(figure)
    return result


def summarize(output: Path) -> None:
    study = json.loads((output / "study.json").read_text())
    results = [json.loads(path.read_text()) for path in sorted(output.glob("r*/sample_*/*/result.json"))]
    variants, cases = study["variants"], study["cases"]
    lines = ["# LBFGS history and backtracking pilot", "",
             "Selected 64x64 one-circle CPU cases, noiseless measurements, 100 Euler steps, rank 8.",
             "Same sources, masks, fixed random sketches and first-attempt-only scaling; no V reuse.",
             f"Up to {study['max_steps']} attempts; two failures or sustained objective plateau may stop earlier.",
             "Filtered: history 10 and s.T@y > max(1e-10, 0.01*||s||*||y||). Geometric: halve rejected rates.",
             "All variants keep lr=1, Armijo c1=1e-4, 30 trials, and displacement cap 0.64.", "",
             "| Case | " + " | ".join(variants) + " |", "| --- |" + " ---: |" * len(variants)]
    figure, axes = plt.subplots(len(cases), 2, figsize=(13, 3 * len(cases)), squeeze=False, layout="constrained")
    complete = []
    for index, case in enumerate(cases):
        pair = {row["variant"]: row for row in results if row["case"] == case}
        cells = []
        for variant in variants:
            row = pair.get(variant)
            cells.append("pending" if row is None else f"{100*row['history'][-1]['image_relative_l2']:.3f}% ({row['status']}, {row['history'][-1]['step']})")
            if row is None:
                continue
            for axis, field in zip(axes[index], ("image_relative_l2", "measurement_relative_l2_clean")):
                axis.plot([v["step"] for v in row["history"]], [100*v[field] for v in row["history"]], label=variant)
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
        for axis, label in zip(axes[index], ("Image error (%)", "Clean measurement residual (%)")):
            axis.set(title=case.replace("r64_n1_", ""), xlabel="Optimizer attempt", ylabel=label)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        if len(pair) == len(variants) and all(row["status"] in TERMINAL for row in pair.values()):
            initial = [torch.load(output / case / variant / "initial_gradient.pt", weights_only=True) for variant in variants]
            assert all(torch.equal(initial[0], gradient) for gradient in initial[1:])
            assert all(row["history"][0] == next(iter(pair.values()))["history"][0] for row in pair.values())
            complete.append(pair)
    lines += ["", f"Completed matched cases: {len(complete)}/{len(cases)}.", "",
              "| Variant | Mean image % | <=5% image | Lower image vs baseline | Mean residual % | Stalls | Filter rejections |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in variants:
        if not complete:
            break
        rows = [pair[variant] for pair in complete]
        lines.append(f"| {variant} | {100*np.mean([r['history'][-1]['image_relative_l2'] for r in rows]):.3f} | "
                     f"{sum(r['history'][-1]['image_relative_l2'] <= .05 for r in rows)}/{len(rows)} | "
                     f"{sum(pair[variant]['history'][-1]['image_relative_l2'] < pair['baseline']['history'][-1]['image_relative_l2'] for pair in complete)}/{len(rows)} | "
                     f"{100*np.mean([r['history'][-1]['measurement_relative_l2_clean'] for r in rows]):.3f} | "
                     f"{sum(r['status'] == 'stalled' for r in rows)} | "
                     f"{sum(h.get('optimizer', {}).get('filter_rejected', False) for r in rows for h in r['history'])} |")
    lines += ["", "A stall does not certify convergence. These selected single-seed CPU cases do not establish GPU or population gains.",
              "Geometric halving changes the explored trial grid and forward-evaluation count, while keeping gradient products unchanged."]
    lines += ["", "## At 100 attempts (earlier terminal values carried forward)", "",
              "| Case | " + " | ".join(variants) + " |", "| --- |" + " ---: |" * len(variants)]
    for pair in complete:
        lines.append("| " + pair[variants[0]]["case"] + " | " + " | ".join(
            f"{100*pair[variant]['history'][min(100, len(pair[variant]['history']) - 1)]['image_relative_l2']:.3f}%"
            for variant in variants) + " |")
    figure.savefig(output / "convergence.png", dpi=120)
    plt.close(figure)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ct_mri_sweep_100steps_noiseless_20261007"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        study = {"root": str(args.root), "cases": args.cases, "variants": args.variants, "policies": VARIANTS,
                 "max_steps": args.max_steps, "workers": args.workers, "threads_per_worker": 1,
                 "selection": "Same eight diagnostic cases as the previous-V pilot; selected, not population-unbiased",
                 "source_sha256": source_hashes(),
                 "experiment_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                     (Path(__file__).relative_to(Path.cwd()), Path(__file__).with_name("optimizer_ablation.py").relative_to(Path.cwd()))}}
        if (args.output / "study.json").exists():
            assert json.loads((args.output / "study.json").read_text()) == study
        write_json(args.output / "study.json", study)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_case, args.root, args.output, case, variant, args.max_steps)
                       for case in args.cases for variant in args.variants]
            for future in as_completed(futures):
                future.result()
        summarize(args.output)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()

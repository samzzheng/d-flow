"""Paired pure and alternating P/DLR reconstructions; isolated from the sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import (
    Evaluator, metrics, reviewed_source_hashes, save_tensor, tensor_tree, write_json,
)
from p_flow.armijo import ArmijoLBFGS
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison

Record = dict[str, Any]
SCHEDULES = {"p_flow": ("P-Flow",), "dlr_d_flow": ("DLR D-Flow",),
             "p1_dlr1": ("P-Flow", "DLR D-Flow"),
             "p3_dlr1": ("P-Flow", "P-Flow", "P-Flow", "DLR D-Flow")}


class AlternatingLBFGS:
    """Hold one estimator fixed for both closures within an optimizer step.

    Clear curvature and the cached gradient at switches, retaining the global
    attempt count and failure accounting. Therefore first-step scaling is not
    repeated, and both endpoints of every stored curvature pair use the same
    estimator. A 1:1 schedule never reuses a curvature pair on its next step.
    """

    def __init__(self, optimizer: ArmijoLBFGS, schedule: tuple[str, ...]) -> None:
        self.optimizer, self.schedule = optimizer, schedule
        self.previous: str | None = None

    def step(self, gradient: Callable[[str], torch.Tensor]) -> tuple[str, bool, int]:
        method = self.schedule[self.optimizer.step_attempts % len(self.schedule)]
        switched = self.previous is not None and method != self.previous
        if switched:
            self.optimizer.S, self.optimizer.Y = [], []
            self.optimizer.f, self.optimizer.g = None, None
        pairs_used = len(self.optimizer.S)
        self.optimizer.step(lambda: gradient(method))
        self.previous = method
        return method, switched, pairs_used


def solve(evaluator: Evaluator, variant: str, inputs: Record, path: Path,
          protocol: Record, interrupted: list[bool]) -> Record:
    source = inputs["initial"].detach().clone().requires_grad_(True)
    settings = evaluator.settings
    optimizer = ArmijoLBFGS(source, evaluator.objective, **{
        **settings["optimizer"], "step_cap": .01 * math.sqrt(source.numel()), "project": False})
    alternating = AlternatingLBFGS(optimizer, SCHEDULES[variant])
    history: list[Record] = []
    previous_seconds = 0.
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["protocol"] != protocol:
            raise ValueError(f"Resume protocol mismatch: {path}")
        if saved["status"] in {"stalled", "step_limit"}:
            return saved
        with torch.no_grad():
            source.copy_(saved["source"].to(source.device))
        for key, value in saved["optimizer"].items():
            setattr(optimizer, key, tensor_tree(value, source.device))
        alternating.previous = saved["previous_method"]
        history, previous_seconds = saved["history"], saved["elapsed_seconds"]
        evaluator.previous_v = tensor_tree(saved["previous_v"], source.device)
        evaluator.counts = saved["evaluations"]
    started = time.monotonic()
    evaluator.objective(source.detach())

    def closure(method: str) -> torch.Tensor:
        source.grad = evaluator.gradient(source, method)
        return source.new_tensor(evaluator.value, dtype=torch.float64)

    if not history:
        history.append({"step": 0, "phi": evaluator.value,
                        **metrics(evaluator.image, evaluator.operator(evaluator.image),
                                  inputs["truth"], inputs["clean"], inputs["observed"])})
    status = "running"
    while status == "running":
        if interrupted[0]:
            status = "paused"
        else:
            before = source.detach().clone()
            method, switched, pairs = alternating.step(closure)
            if not torch.equal(evaluator.source, source.detach()):
                evaluator.objective(source.detach())
            displacement = float((source.detach() - before).norm())
            if displacement > optimizer.step_cap * (1 + 1e-4):
                raise AssertionError("Displacement cap violated")
            if evaluator.value > history[-1]["phi"] or (displacement > 0 and evaluator.value == history[-1]["phi"]):
                raise AssertionError("Objective acceptance invariant violated")
            if switched and pairs:
                raise AssertionError("Curvature history crossed an estimator switch")
            history.append({"step": optimizer.step_attempts, "phi": evaluator.value,
                            "estimator": method, "switched": switched, "curvature_pairs_used": pairs,
                            "source_step_norm": displacement,
                            "line_search_trials": optimizer.last_function_evaluations,
                            **metrics(evaluator.image, evaluator.operator(evaluator.image),
                                      inputs["truth"], inputs["clean"], inputs["observed"])})
            # A failed step still advances the schedule; give both estimators a
            # chance before declaring a hybrid stalled (two whole failed cycles).
            if optimizer.consecutive_failures >= 2 * len(alternating.schedule):
                status = "stalled"
            elif optimizer.step_attempts >= settings["stopping"]["max_updates"]:
                status = "step_limit"
        if status != "running" or optimizer.step_attempts % 10 == 0:
            saved = {"protocol": protocol, "variant": variant, "status": status,
                     "source": source.detach(), "image": evaluator.image,
                     "measurement": evaluator.operator(evaluator.image).detach(),
                     "previous_v": evaluator.previous_v, "previous_method": alternating.previous,
                     "optimizer": {key: value for key, value in vars(optimizer).items()
                                   if key not in {"source", "objective"}},
                     "history": history, "evaluations": evaluator.counts,
                     "elapsed_seconds": previous_seconds + time.monotonic() - started}
            save_tensor(path, saved)
            write_json(path.with_suffix(".json"), {key: saved[key] for key in
                       ("protocol", "variant", "status", "history", "evaluations", "elapsed_seconds")})
            print(f"{path.parent.name} {variant} {optimizer.step_attempts} {status} "
                  f"image={100 * history[-1]['image_relative_l2']:.4f}%", flush=True)
    return tensor_tree(saved, "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ct_mri_sweep_fp32_20261009"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100, help="Optimizer attempts; Euler steps remain 100")
    parser.add_argument("--variants", nargs="+", choices=list(SCHEDULES), default=list(SCHEDULES))
    parser.add_argument("--memory-fraction", type=float, default=.65)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.root / "manifest.json").read_text())
    for name, expected in reviewed_source_hashes(args.root / "manifest.json", manifest).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Production source changed: {name}")
    settings = manifest["settings"]
    settings["stopping"]["max_updates"] = args.steps
    assert settings["flow_steps"] == 100 and settings["noise_relative"] == 0
    input_path = args.root / args.case / "inputs.pt"
    inputs = torch.load(input_path, weights_only=True, map_location="cpu")
    parent = inputs["protocol"]
    job = parent["job"]
    run = Path(settings["run_template"].format(**job))
    if hashlib.sha256((run / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest() != parent["checkpoint_sha256"]:
        raise ValueError("Model checkpoint changed")
    model, _ = load_flow_model(job["resolution"], job["circles"], 200, args.device, run)
    device = torch.device(args.device)
    operator = (CT(job["resolution"], job["level"], device) if job["problem"] == "ct" else
                MRI(job["resolution"], job["level"], parent["operator"]["mask_seed"], device,
                    settings["mri_pattern"]))
    if isinstance(operator, MRI):
        assert torch.equal(operator.mask.cpu(), inputs["mask"])
    torch.testing.assert_close(operator(inputs["truth"].to(device)).cpu(), inputs["clean"], rtol=1e-6, atol=1e-7)
    assert torch.equal(inputs["clean"], inputs["observed"])
    tangent_unet.GN_STATS_MODE = "native" if job["resolution"] <= 96 else "manual"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tensors = tensor_tree(inputs, device)
    products = GraphedAProducts(model, tensors["initial"], share_linearization=True,
                               full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
                               max_step_jvp_columns=6, max_step_vjp_columns=4)
    output = args.output / args.case
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"parent": parent, "settings": settings,
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "schedule_unit": "attempted updates", "switch_policy": "clear pairs and cached gradient",
                "stall_policy": "two completely failed schedule cycles",
                "device": args.device, "rank_policy": "fixed existing rank 8, not selected by spectral study"}
    save_tensor(output / "inputs.pt", inputs)
    interrupted = [False]

    def stop(signum: int, frame: Any) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    results = {}
    # Paired pure controls are rerun here rather than silently reusing a
    # different execution or stopping protocol from the production sweep.
    for variant in args.variants:
        evaluator = Evaluator(model, operator, tensors["observed"], settings, parent["dlr_seed"], products)
        results[variant] = solve(evaluator, variant, tensors, output / f"{variant}.pt", protocol, interrupted)
        if interrupted[0]:
            return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for variant, result in results.items():
        for ax, key in zip(axes, ("image_relative_l2", "measurement_relative_l2_clean")):
            ax.plot([row["step"] for row in result["history"]],
                    [100 * row[key] for row in result["history"]], label=variant)
    for ax, label in zip(axes, ("Relative image L2 error (%)", "Relative measurement residual (%)")):
        ax.set(xlabel="Optimization attempts", ylabel=label, yscale="log")
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(args.case + " | noiseless, FP32, 100 Euler steps")
    fig.savefig(output / "convergence.png", dpi=150)
    plt.close(fig)
    for variant in args.variants:
        if variant == "p_flow":
            continue
        result = results[variant]
        reconstructions = {"DLR D-Flow": (result["image"], result["measurement"])}
        histories = {"DLR D-Flow": result["history"]}
        if "p_flow" in results:
            pure = results["p_flow"]
            reconstructions["P-Flow"] = (pure["image"], pure["measurement"])
            histories["P-Flow"] = pure["history"]
        fig = plot_reconstruction_comparison(inputs["truth"], inputs["clean"], reconstructions,
                observation=inputs["observed"], histories=histories, measurement_log=job["problem"] == "mri",
                residual_vmax=RESIDUAL_VMAX[job["problem"]],
                title=f"{args.case} | DLR column variant: {variant} | FP32, noiseless, 100 Euler steps")
        fig.savefig(output / f"{variant}_comparison.png", dpi=140)
        plt.close(fig)
    write_json(output / "complete.json", {"protocol": protocol, "variants": args.variants,
                                          "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})


if __name__ == "__main__":
    main()

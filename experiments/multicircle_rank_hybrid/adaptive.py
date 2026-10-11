"""Between-update rank adaptation using same-source gradient refinement checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import (
    Evaluator, metrics, reviewed_source_hashes, save_tensor, tensor_tree, write_json,
)
from p_flow import endpoint_gradient
from p_flow.armijo import ArmijoLBFGS
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison

Record = dict[str, Any]
LADDER = (4, 8, 16, 32, 64)
VARIANTS = {"fixed8": (8, None), "fixed32": (32, None),
            "adaptive05": (8, .05), "adaptive01": (8, .01)}


def select_rank(current: int, tolerance: float,
                evaluate: Callable[[int], torch.Tensor]) -> tuple[int, Record]:
    """Refinement agreement is a heuristic, not an exact-gradient certificate.

    Probe smaller rank first; require twice as close agreement to shrink as to
    retain/grow. The cap is explicitly unvalidated when no larger check exists.
    """
    gradients: dict[int, torch.Tensor] = {}
    comparisons: list[Record] = []
    index = LADDER.index(current)
    low = max(0, index - 1)
    for position in range(low, len(LADDER) - 1):
        small, large = LADDER[position:position + 2]
        for rank in (small, large):
            if rank not in gradients:
                gradients[rank] = evaluate(rank).detach().flatten().double()
        numerator = float((gradients[small] - gradients[large]).norm())
        denominator = float(gradients[large].norm())
        difference = numerator / max(denominator, 1e-12)
        threshold = tolerance * (.5 if small < current else 1.)
        comparisons.append({"small": small, "large": large, "relative_difference": difference,
                            "higher_gradient_norm": denominator, "threshold": threshold})
        if difference <= threshold:
            return small, {"selected": small, "cap_unvalidated": False, "comparisons": comparisons}
    return LADDER[-1], {"selected": LADDER[-1], "cap_unvalidated": True, "comparisons": comparisons}


class RankEvaluator(Evaluator):
    """Keep diagnostic rank probes from changing the reconstruction's warm V."""

    def rank_gradient(self, source: torch.Tensor, rank: int, commit_basis: bool) -> torch.Tensor:
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        _, gx = endpoint_gradient(self.image, self.operator, self.observed)
        _, factors, alpha = sensitivity_factors(
            self.model, source.detach(), self.settings["flow_steps"],
            **{**self.settings["dlr"], "rank": rank}, seed=self.seed,
            products=self.products, seed_basis=self.previous_v)
        gradient = approx_source_gradient(factors, gx, alpha).detach()
        if alpha != 0 or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite gradient or nonzero terminal baseline")
        self.counts["factor_evaluations"] = self.counts.get("factor_evaluations", 0) + 1
        if commit_basis:
            self.previous_v = factors[2].detach().clone()
            self.counts["gradients"] += 1
        return gradient


def solve(evaluator: RankEvaluator, inputs: Record, variant: str, path: Path,
          protocol: Record, interrupted: list[bool]) -> Record:
    rank, tolerance = VARIANTS[variant]
    source = inputs["initial"].detach().clone().requires_grad_(True)
    optimizer = ArmijoLBFGS(source, evaluator.objective, **{
        **evaluator.settings["optimizer"], "step_cap": .01 * math.sqrt(source.numel()), "project": False})
    history: list[Record] = []
    rank_checks: list[Record] = []
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["protocol"] != protocol:
            raise ValueError(f"Resume protocol mismatch: {path}")
        if saved["status"] in {"stalled", "step_limit"}:
            return saved
        with torch.no_grad():
            source.copy_(saved["source"].to(source.device))
        for name, value in saved["optimizer"].items():
            setattr(optimizer, name, tensor_tree(value, source.device))
        history, rank_checks, rank = saved["history"], saved["rank_checks"], saved["rank"]
        evaluator.previous_v = tensor_tree(saved["previous_v"], source.device)
        evaluator.counts = saved["evaluations"]
    evaluator.objective(source.detach())

    def closure() -> torch.Tensor:
        source.grad = evaluator.rank_gradient(source, rank, commit_basis=True)
        return source.new_tensor(evaluator.value, dtype=torch.float64)

    if not history:
        history.append({"step": 0, "phi": evaluator.value, "rank": rank,
                        **metrics(evaluator.image, evaluator.operator(evaluator.image),
                                  inputs["truth"], inputs["clean"], inputs["observed"])})
    status = "running"
    while status == "running":
        if interrupted[0]:
            status = "paused"
        else:
            changed = False
            if tolerance is not None and (optimizer.step_attempts % 5 == 0 or optimizer.consecutive_failures):
                before_rank = rank
                rank, diagnostic = select_rank(rank, tolerance,
                    lambda candidate: evaluator.rank_gradient(source, candidate, commit_basis=False))
                changed = rank != before_rank
                rank_checks.append({"step": optimizer.step_attempts, "previous_rank": before_rank,
                                    "reason": "failed_search" if optimizer.consecutive_failures else "periodic",
                                    **diagnostic})
                if changed:
                    optimizer.S, optimizer.Y = [], []
                    optimizer.f, optimizer.g = None, None
            # Both closures in this update use the selected rank. Global
            # first-step scaling and consecutive-failure counts are retained.
            pairs = len(optimizer.S)
            before = source.detach().clone()
            optimizer.step(closure)
            if not torch.equal(evaluator.source, source.detach()):
                evaluator.objective(source.detach())
            displacement = float((source.detach() - before).norm())
            if displacement > optimizer.step_cap * (1 + 1e-4):
                raise AssertionError("Displacement cap violated")
            if evaluator.value > history[-1]["phi"] or (displacement > 0 and evaluator.value == history[-1]["phi"]):
                raise AssertionError("Objective acceptance invariant violated")
            if changed and pairs:
                raise AssertionError("Curvature pairs crossed a rank change")
            history.append({"step": optimizer.step_attempts, "phi": evaluator.value, "rank": rank,
                            "rank_changed": changed, "curvature_pairs_used": pairs,
                            "source_step_norm": displacement, "line_search_trials": optimizer.last_function_evaluations,
                            **metrics(evaluator.image, evaluator.operator(evaluator.image),
                                      inputs["truth"], inputs["clean"], inputs["observed"])})
            if optimizer.consecutive_failures >= 2:
                status = "stalled"
            elif optimizer.step_attempts >= evaluator.settings["stopping"]["max_updates"]:
                status = "step_limit"
        if status != "running" or optimizer.step_attempts % 5 == 0:
            saved = {"protocol": protocol, "variant": variant, "status": status, "rank": rank,
                     "source": source.detach(), "image": evaluator.image,
                     "measurement": evaluator.operator(evaluator.image).detach(), "previous_v": evaluator.previous_v,
                     "optimizer": {key: value for key, value in vars(optimizer).items() if key not in {"source", "objective"}},
                     "history": history, "rank_checks": rank_checks, "evaluations": evaluator.counts}
            save_tensor(path, saved)
            write_json(path.with_suffix(".json"), {key: saved[key] for key in
                       ("protocol", "variant", "status", "rank", "history", "rank_checks", "evaluations")})
            print(f"{path.parent.name} {variant} {optimizer.step_attempts} {status} rank={rank} "
                  f"image={100 * history[-1]['image_relative_l2']:.4f}%", flush=True)
    return tensor_tree(saved, "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ct_mri_sweep_fp32_20261009"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(4000 / 49140, device)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.root / "manifest.json").read_text())
    for name, expected in reviewed_source_hashes(args.root / "manifest.json", manifest).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Production source changed: {name}")
    settings = manifest["settings"]
    settings["stopping"]["max_updates"] = args.steps
    input_path = args.root / args.case / "inputs.pt"
    inputs = torch.load(input_path, weights_only=True, map_location="cpu")
    parent = inputs["protocol"]
    job = parent["job"]
    # This bounded pilot has an explicit GPU budget and only 64px models.
    if job["resolution"] != 64:
        raise ValueError("The first adaptive pilot is restricted to 64px")
    run = Path(settings["run_template"].format(**job))
    if hashlib.sha256((run / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest() != parent["checkpoint_sha256"]:
        raise ValueError("Model checkpoint changed")
    model, _ = load_flow_model(64, job["circles"], 200, str(device), run)
    operator = (CT(64, job["level"], device) if job["problem"] == "ct" else
                MRI(64, job["level"], parent["operator"]["mask_seed"], device, settings["mri_pattern"]))
    if isinstance(operator, MRI):
        assert torch.equal(operator.mask.cpu(), inputs["mask"])
    assert torch.equal(inputs["clean"], inputs["observed"])
    torch.testing.assert_close(operator(inputs["truth"].to(device)).cpu(), inputs["clean"], rtol=1e-6, atol=1e-7)
    assert settings["flow_steps"] == 100 and settings["dlr"]["trace_probes"] == 0
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tensors = tensor_tree(inputs, device)
    # Ordinary batched graphs bound memory when many distinct ranks are used.
    # Fixed controls use exactly the same backend as the adaptive variants.
    products = GraphedAProducts(model, tensors["initial"], share_linearization=True,
                               full_step_graphs=False, max_jvp_columns=6, max_vjp_columns=4)
    output = args.output / args.case
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"parent": parent, "settings": settings, "ladder": LADDER, "check_every": 5,
                "shrink_hysteresis": .5, "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "backend": "batched graphs, no complete-step graph",
                "rank_policy": "same-source gradient agreement between optimizer steps; rank fixed within flow"}
    # Tuples become lists in JSON, but the PT resume protocol remains typed.
    save_tensor(output / "inputs.pt", inputs)
    if args.validate:
        evaluator = RankEvaluator(model, operator, tensors["observed"], settings, parent["dlr_seed"], products)
        native = evaluator.objective(tensors["initial"])
        _, gx = endpoint_gradient(evaluator.image, operator, tensors["observed"])
        checks = []
        for rank in LADDER:
            graph_gradient = evaluator.rank_gradient(tensors["initial"], rank, commit_basis=False)
            _, factors, alpha, alpha_path = sensitivity_factors(model, tensors["initial"], 100,
                **{**settings["dlr"], "rank": rank}, seed=parent["dlr_seed"], return_alpha_path=True)
            reference = approx_source_gradient(factors, gx, alpha)
            error = float((graph_gradient - reference).norm() / reference.norm().clamp_min(1e-30))
            assert error < .01 and alpha == 0
            assert max(abs(value - (1 - index / 100)) for index, value in enumerate(alpha_path)) < 1e-12
            assert evaluator.previous_v is None
            checks.append({"rank": rank, "graph_eager_relative_gradient_error": error})
        write_json(output / "validation.json", {"phi": native, "checks": checks,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
    interrupted = [False]

    def stop(signum: int, frame: Any) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    results = {}
    for variant in args.variants:
        evaluator = RankEvaluator(model, operator, tensors["observed"], settings, parent["dlr_seed"], products)
        results[variant] = solve(evaluator, tensors, variant, output / f"{variant}.pt", protocol, interrupted)
        if interrupted[0]:
            return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for variant, result in results.items():
        for ax, key, multiplier in zip(axes, ("image_relative_l2", "measurement_relative_l2_clean", "rank"), (100, 100, 1)):
            ax.plot([row["step"] for row in result["history"]],
                    [multiplier * row[key] for row in result["history"]], label=variant)
        panel = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
            {"DLR D-Flow": (result["image"], result["measurement"])}, observation=inputs["observed"],
            histories={"DLR D-Flow": result["history"]}, measurement_log=job["problem"] == "mri",
            residual_vmax=RESIDUAL_VMAX[job["problem"]], title=f"{args.case} | DLR variant: {variant}")
        panel.savefig(output / f"{variant}_comparison.png", dpi=140)
        plt.close(panel)
    for ax, label in zip(axes, ("Image error (%)", "Measurement residual (%)", "Rank")):
        ax.set(xlabel="Optimization attempts", ylabel=label)
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(output / "convergence.png", dpi=150)
    plt.close(fig)
    write_json(output / "complete.json", {"protocol": protocol, "variants": args.variants,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})


if __name__ == "__main__":
    main()

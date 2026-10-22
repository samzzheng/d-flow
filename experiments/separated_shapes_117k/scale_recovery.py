"""Bounded paired endpoint recovery through fallback scale or a frozen sketch seed."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.optimizer import SweepLBFGS
from experiments.ct_mri_sweep.queue import source_hashes
from experiments.separated_shapes_117k.ksl_stability import increment_ksl_step
from experiments.separated_shapes_117k.optimizer_controls import ScaleRecoveryLBFGS
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison
from sampler.ode_samplers import euler_flow


class FrozenSeedEvaluator(run.Evaluator):
    """Replace one initialization column; keep every subsequent KSL update."""

    seed_direction: torch.Tensor

    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        previous = self.previous_v
        self.previous_v = (self.seed_direction if previous is None else
                           torch.cat([self.seed_direction, previous], dim=1))
        try:
            gradient = super().gradient(source, method)
        except Exception:
            self.previous_v = previous
            raise
        if not self.settings.get("dlr_reuse_previous_v", False):
            self.previous_v = previous
        return gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path,
        help="Use a saved inputs bundle instead of the source-pool bundle")
    parser.add_argument("--source-gradient-seed", action="store_true",
        help="Compare cold restarts using one frozen exact-gradient seed, a matched random seed, and no added seed")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.2, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    original = torch.load(args.source, map_location="cpu", weights_only=True)
    inputs = torch.load(args.inputs or args.source.parent / "pool.pt", map_location=device, weights_only=True)
    protocol = original["protocol"]
    parent_recovery = protocol.get("scale_recovery", {})
    if parent_recovery.get("recovery") == "frozen_source_gradient_seed" and not args.source_gradient_seed:
        parser.error("A seeded parent needs the seed-recovery control to preserve its estimator")
    if original["method"] != "DLR D-Flow" or original["status"] not in run.TERMINAL or not original["optimizer"]["consecutive_failures"]:
        parser.error("Recovery needs a completed failed-search source-pool control")
    settings = deepcopy(protocol.get("continuation_settings", protocol.get("settings")))
    if settings is None:
        parser.error("The parent must record its optimizer and factor settings")
    limit = settings["stopping"]["max_updates"]
    if original["optimizer"]["step_attempts"] >= limit:
        parser.error("No original attempt budget remains")
    experiment = protocol.get("experiment", {"budget": limit})
    outside = experiment["budget"] - limit if "continuation_settings" in protocol else 0
    integrator = experiment.get("factor_integrator", "original")
    if integrator not in {"original", "increment"} or experiment.get("goal_steps", 0):
        parser.error("This control supports ordinary fast/increment factors only")
    for key in ("checkpoint_sha256", "operator", "source_seed"):
        assert protocol[key] == inputs["protocol"][key]
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder / "checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
    info = protocol["operator"]
    operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
    if isinstance(operator, MRI):
        assert torch.equal(operator.mask, inputs["mask"])
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    if integrator == "increment":
        enriched._zero_enrichment_step = increment_ksl_step
    products = GraphedAProducts(model, inputs["initial"], share_linearization=True,
        full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
        max_step_jvp_columns=6, max_step_vjp_columns=4)
    evidence = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        optimizer_sha256=hashlib.sha256(Path(ScaleRecoveryLBFGS._direction.__code__.co_filename).read_bytes()).hexdigest(),
        source_hashes=source_hashes(), parent=str(args.source),
        parent_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(original["source"].numpy().tobytes()).hexdigest(),
        attempts_in_other_starts=outside, total_budget=experiment["budget"],
        settings=settings, baseline=original["history"][-1],
        note="Preserve cached gradient, curvature, right basis and all attempt counters; change recovery scale only")
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = {}
    if args.source_gradient_seed:
        seed_path = args.out / "seeds.pt"
        if seed_path.exists():
            cached = torch.load(seed_path, map_location=device, weights_only=True)
            assert cached["source_sha256"] == evidence["source_sha256"]
            seeds = cached["directions"]
        else:
            variable = original["source"].to(device).requires_grad_(True)
            image = euler_flow(model, variable, settings["flow_steps"])
            loss = .5 * (operator(image)-inputs["observed"]).abs().square().sum(dtype=torch.float64)
            exact, = torch.autograd.grad(loss, variable)
            exact = exact.detach().reshape(-1, 1)
            if not torch.isfinite(exact).all() or exact.norm() == 0:
                raise FloatingPointError("The exact gradient cannot supply a seed")
            random = torch.randn(exact.shape, generator=torch.Generator().manual_seed(protocol["dlr_seed"]+80000000)).to(device)
            random *= exact.norm()/random.norm()
            seeds = {"gradient_seed": exact, "random_seed": random}
            if parent_recovery.get("recovery") == "frozen_source_gradient_seed":
                parent_seeds = torch.load(args.source.parents[1] / "seeds.pt", map_location=device, weights_only=True)
                parent_variant = parent_recovery["variant"]
                if parent_variant in parent_seeds["directions"]:
                    previous_seed = parent_seeds["directions"][parent_variant]
                    assert hashlib.sha256(previous_seed.cpu().numpy().tobytes()).hexdigest() == parent_recovery["seed_sha256"][parent_variant]
                    seeds["unchanged"] = previous_seed
            run.save_tensor(seed_path, dict(source_sha256=evidence["source_sha256"], directions=seeds))
            del variable, image, loss
        evidence.update(recovery="frozen_source_gradient_seed",
            exact_gradient_evaluations=parent_recovery.get("exact_gradient_evaluations", 0)+1,
            seed_sha256={name: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for name, value in seeds.items()},
            note="One exact source gradient at the parent, frozen thereafter; matched random seed; same cold optimizer resets and original attempt budget")
    run.write_json(args.out / "config.json", evidence)
    summary = {}
    variants = (("unchanged", SweepLBFGS), ("gradient_seed", SweepLBFGS), ("random_seed", SweepLBFGS)) if args.source_gradient_seed else (("unchanged", SweepLBFGS), ("scaled", ScaleRecoveryLBFGS))
    for name, kind in variants:
        path = args.out / name / "dlr_d_flow.pt"
        path.parent.mkdir(exist_ok=True)
        paired = {**protocol, "scale_recovery": {**evidence, "variant": name}}
        if not path.exists():
            state = original["optimizer"]
            if args.source_gradient_seed:
                state = {**state, "S": [], "Y": [], "f": None, "g": None, "state": {}, "consecutive_failures": 0}
            run.save_tensor(path, {**original, "protocol": paired, "status": "paused", "optimizer": state})
        evaluator_type = FrozenSeedEvaluator if name in seeds else run.Evaluator
        evaluator = evaluator_type(model, operator, inputs["observed"], settings, protocol["dlr_seed"], products)
        if isinstance(evaluator, FrozenSeedEvaluator):
            evaluator.seed_direction = seeds[name]
        with patch.object(run, "SweepLBFGS", kind):
            result = run.solve(evaluator, "DLR D-Flow", original["source"].to(device),
                inputs["truth"], inputs["clean"], path, paired, [False])
        assert result["history"][:len(original["history"])] == original["history"]
        final = result["history"][-1]
        assert final["step"] <= limit
        image_error = (result["image"].to(device).double()-inputs["truth"].double()).norm()/inputs["truth"].double().norm()
        torch.testing.assert_close(image_error, image_error.new_tensor(final["image_relative_l2"]), atol=1e-6, rtol=1e-5)
        summary[name] = dict(status=result["status"], final=final,
            total_attempts=outside+final["step"], additional_attempts=final["step"]-evidence["baseline"]["step"],
            scaled_recoveries=result["optimizer"].get("scaled_recoveries", 0))
        run.write_json(args.out / "summary.json", summary)
        figure = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
            {"DLR D-Flow": (result["image"].to(device), result["measurement"].to(device))},
            observation=inputs["observed"], include_errors=True, measurement_log=isinstance(operator, MRI),
            residual_vmax=RESIDUAL_VMAX[protocol["job"]["problem"]],
            title=f"{resolution} square: {name} {'sketch seed' if args.source_gradient_seed else 'fallback scale'}, original budget {experiment['budget']}")
        figure.savefig(path.parent / "comparison.png", dpi=120)
        run.plt.close(figure)


if __name__ == "__main__":
    main()

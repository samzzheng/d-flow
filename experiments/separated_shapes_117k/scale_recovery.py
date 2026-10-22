"""Paired continuation of a terminal source-pool run with bounded scale recovery."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
    inputs = torch.load(args.source.parent / "pool.pt", map_location=device, weights_only=True)
    protocol = original["protocol"]
    if original["status"] not in run.TERMINAL or not original["optimizer"]["consecutive_failures"]:
        parser.error("Recovery needs a completed failed-search source-pool control")
    settings = deepcopy(protocol["continuation_settings"])
    limit = settings["stopping"]["max_updates"]
    if original["optimizer"]["step_attempts"] >= limit:
        parser.error("No original attempt budget remains")
    experiment = protocol["experiment"]
    outside = experiment["budget"] - limit
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
    run.write_json(args.out / "config.json", evidence)
    summary = {}
    for name, kind in (("unchanged", SweepLBFGS), ("scaled", ScaleRecoveryLBFGS)):
        path = args.out / name / "dlr_d_flow.pt"
        path.parent.mkdir(exist_ok=True)
        paired = {**protocol, "scale_recovery": {**evidence, "variant": name}}
        if not path.exists():
            run.save_tensor(path, {**original, "protocol": paired, "status": "paused"})
        evaluator = run.Evaluator(model, operator, inputs["observed"], settings, protocol["dlr_seed"], products)
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
            title=f"{resolution} square: {name} fallback scale, original budget {experiment['budget']}")
        figure.savefig(path.parent / "comparison.png", dpi=120)
        run.plt.close(figure)


if __name__ == "__main__":
    main()

"""Bounded sketch restarts after failed searches, within the original attempt cap."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.queue import source_hashes
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=3)
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
    inputs = torch.load(args.inputs, map_location=device, weights_only=True)
    protocol = original["protocol"]
    if original["method"] != "DLR D-Flow" or original["status"] not in run.TERMINAL:
        parser.error("Recovery requires a completed DLR control")
    if args.trials < 1 or original["history"][-1]["step"] >= args.budget:
        parser.error("Recovery needs a positive trial count and remaining original budget")
    for key in ("checkpoint_sha256", "operator", "source_seed"):
        assert protocol[key] == inputs["protocol"][key]
    settings = deepcopy(protocol.get("continuation_settings", protocol.get("settings")))
    if settings is None or args.budget > settings["stopping"]["max_updates"]:
        parser.error("Recovery cannot exceed the saved run's remaining attempt cap")
    outside = (protocol["experiment"]["budget"] - settings["stopping"]["max_updates"]
               if "continuation_settings" in protocol else 0)
    offset = 0 if settings["dlr_reuse_previous_v"] else 1
    settings["dlr_reuse_previous_v"] = False
    settings["stopping"]["max_updates"] = args.budget
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
    products = GraphedAProducts(model, inputs["initial"], share_linearization=True,
        full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
        max_step_jvp_columns=6, max_step_vjp_columns=4)
    evidence = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes=source_hashes(), parent_checkpoint=str(args.source),
        parent_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), budget=args.budget,
        attempts_in_other_starts=outside, total_budget=args.budget+outside,
        trials=args.trials, baseline=original["history"][-1],
        note="No source perturbation or extra optimizer budget; reset curvature across sketch switches")
    args.out.mkdir(parents=True, exist_ok=True)
    run.write_json(args.out / "config.json", evidence)
    current = original
    results = []
    for trial in range(args.trials):
        seed = protocol["dlr_seed"] + offset + trial
        stage = args.out / f"trial_{trial:02d}.pt"
        start = current["history"][-1]["step"]
        stage_protocol = {**protocol, "settings": settings, "dlr_seed": seed,
                          "sketch_recovery": {**evidence, "trial": trial}}
        if not stage.exists():
            run.save_tensor(stage, {**current, "protocol": stage_protocol, "status": "paused",
                "previous_v": None, "optimizer": {**current["optimizer"], "S": [], "Y": [],
                    "f": None, "g": None, "state": {}, "consecutive_failures": 0}})
        evaluator = run.Evaluator(model, operator, inputs["observed"], settings, seed, products)
        current = run.solve(evaluator, "DLR D-Flow", current["source"].to(device),
                            inputs["truth"], inputs["clean"], stage, stage_protocol, [False])
        final = current["history"][-1]
        assert final["step"] <= args.budget
        assert current["history"][:len(original["history"])] == original["history"]
        results.append(dict(trial=trial, seed=seed, start_attempts=outside+start,
                            status=current["status"], final=final))
        run.write_json(args.out / "summary.json", dict(trials=results,
            baseline=evidence["baseline"], total_attempts=outside+final["step"],
            winner_attempts=final["step"], attempts_in_other_starts=outside,
            additional_attempts=final["step"]-evidence["baseline"]["step"], budget=args.budget+outside))
        if current["status"] == "step_limit" or final["step"] >= args.budget:
            break
    run.save_tensor(args.out / "result.pt", current)
    figure = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
        {"DLR D-Flow": (current["image"], current["measurement"])}, observation=inputs["observed"],
        include_errors=True, measurement_log=isinstance(operator, MRI),
        residual_vmax=RESIDUAL_VMAX[protocol["job"]["problem"]],
        title=f"{resolution} square: bounded sketch recovery, original budget {args.budget+outside}")
    figure.savefig(args.out / "comparison.png", dpi=120)
    run.plt.close(figure)


if __name__ == "__main__":
    main()

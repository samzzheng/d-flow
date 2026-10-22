"""Paired native-arithmetic continuations of a retained-seed DLR endpoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, residual_startup
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.optimizer import SweepLBFGS
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts
from experiments.separated_shapes_117k.scale_recovery import FrozenSeedEvaluator
from experiments.separated_shapes_117k.sketch_initialization import retained_seed_startup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--additional-attempts", type=int, default=50)
    parser.add_argument("--resume", action="store_true",
        help="Extend the saved paired control within its original budget, preserving curvature")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.3, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    parent = torch.load(args.source, map_location="cpu", weights_only=True)
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    protocol = parent["protocol"]
    recovery = protocol["scale_recovery"]
    if not recovery.get("retain_seed") or recovery["variant"] != "gradient_seed":
        parser.error("Use the completed informed retained-seed control")
    settings = deepcopy(recovery["settings"])
    initial_attempts = parent["optimizer"]["step_attempts"]
    limit = min(initial_attempts + args.additional_attempts, settings["stopping"]["max_updates"])
    if not initial_attempts < limit:
        parser.error("No requested or original attempt budget remains")
    seeds = torch.load(args.source.parents[1]/"seeds.pt", map_location="cpu", weights_only=True)
    seed = seeds["directions"]["gradient_seed"]
    assert hashlib.sha256(seed.numpy().tobytes()).hexdigest() == recovery["seed_sha256"]["gradient_seed"]
    for key in ("checkpoint_sha256", "operator", "source_seed"):
        assert protocol[key] == inputs["protocol"][key]
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    args.out.mkdir(parents=True, exist_ok=True)
    configuration = dict(parent=str(args.source), inputs=str(args.inputs),
        parent_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initial_attempts=initial_attempts, attempt_limit=limit, settings=settings,
        seed_sha256=recovery["seed_sha256"]["gradient_seed"],
        note="Both native autodiff backends; cold curvature; same promoted source, frozen seed, right basis and original attempt count. Each precision recomputes its initial objective.")
    config_path = args.out/"config.json"
    segments = []
    if config_path.exists():
        if not args.resume:
            parser.error("The control exists; use --resume to preserve its states")
        previous_config = json.loads(config_path.read_text())
        for key in ("parent_sha256", "settings", "seed_sha256", "initial_attempts"):
            assert previous_config[key] == configuration[key]
        if limit < previous_config["attempt_limit"]:
            parser.error("Resume cannot lower the recorded attempt limit")
        segments = previous_config.get("segments", []) + [
            {key: previous_config[key] for key in ("attempt_limit", "script_sha256")}]
    elif args.resume:
        parser.error("No saved control to resume")
    configuration["segments"] = segments
    run.write_json(config_path, configuration)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    summary = json.loads((args.out/"summary.json").read_text()) if args.resume else {}
    for dtype in (torch.float32, torch.float64):
        label = str(dtype).split(".")[-1]
        path = args.out/label/"dlr_d_flow.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
        if saved is not None:
            if not args.resume:
                parser.error(f"Refusing to overwrite a completed or partial control: {path}")
            if saved["status"] == "stalled" or saved["optimizer"]["step_attempts"] >= limit:
                continue
        path.parent.mkdir(exist_ok=True)
        model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
        model = model.to(dtype=dtype)
        if dtype == torch.float64:
            model.time_proj.register_forward_pre_hook(lambda module, arguments: (arguments[0].double(),))
        info = protocol["operator"]
        operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                    else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
        if isinstance(operator, CT):
            operator.grids = tuple(grid.to(dtype=dtype) for grid in operator.grids)
        else:
            assert torch.equal(operator.mask.cpu(), inputs["mask"])
        measurement_dtype = (torch.complex128 if dtype == torch.float64 else torch.complex64) if inputs["observed"].is_complex() else dtype
        truth = inputs["truth"].to(device=device, dtype=dtype)
        clean = inputs["clean"].to(device=device, dtype=measurement_dtype)
        observed = inputs["observed"].to(device=device, dtype=measurement_dtype)
        state = parent if saved is None else saved
        source = state["source"].to(device=device, dtype=dtype).clone().requires_grad_(True)
        products = ChunkedAutodiffProducts(model, columns=4)
        evaluator = FrozenSeedEvaluator(model, operator, observed, settings, protocol["dlr_seed"], products)
        evaluator.previous_v = state["previous_v"].to(device=device, dtype=dtype)
        evaluator.seed_direction = seed.to(device=device, dtype=dtype)
        evaluator.method = "DLR D-Flow"
        optimizer = SweepLBFGS(source, evaluator.objective, **{
            **settings["optimizer"], "step_cap": settings["optimizer"]["step_cap"]*source.numel()**.5,
            "project": settings["source_projection"]})
        optimizer.step_attempts = initial_attempts
        if saved is not None:
            for name, value in saved["optimizer"].items():
                setattr(optimizer, name, run.tensor_tree(value, device))
            evaluator.counts = saved["evaluations"].copy()
        evaluator.objective(source.detach())
        history = deepcopy(saved["history"]) if saved is not None else [dict(step=initial_attempts, phi=evaluator.value,
            **run.metrics(evaluator.image, evaluator.measurement, truth, clean, observed))]
        if saved is not None:
            assert math.isclose(evaluator.value, history[-1]["phi"], rel_tol=1e-12, abs_tol=1e-12)
        prior_seconds = 0.0 if saved is None else saved["elapsed_seconds"]
        started = time.monotonic()

        def closure() -> torch.Tensor:
            source.grad = evaluator.gradient(source, "DLR D-Flow")
            return source.new_tensor(evaluator.value, dtype=torch.float64)

        with patch.object(residual_startup, "residual_sketch_startup", retained_seed_startup):
            status = "running"
            while status == "running":
                before = source.detach().clone()
                optimizer.step(closure)
                if not torch.equal(evaluator.source, source.detach()):
                    evaluator.objective(source.detach())
                displacement = float((source-before).detach().norm())
                assert displacement <= optimizer.step_cap*(1+1e-4)
                assert displacement == 0 or evaluator.value < history[-1]["phi"]
                history.append(dict(step=optimizer.step_attempts, phi=evaluator.value,
                    source_step_norm=displacement, line_search_trials=optimizer.last_function_evaluations,
                    **run.metrics(evaluator.image, evaluator.measurement, truth, clean, observed)))
                if optimizer.consecutive_failures >= settings["stopping"]["failed_searches"]:
                    status = "stalled"
                elif optimizer.step_attempts >= limit:
                    status = "step_limit"
                if status != "running" or optimizer.step_attempts % 5 == 0:
                    run.save_tensor(path, dict(protocol={**protocol, "precision_recovery": {**configuration, "dtype": str(dtype)}},
                        method="DLR D-Flow", status=status, source=source.detach(), image=evaluator.image,
                        measurement=evaluator.measurement, previous_v=evaluator.previous_v, history=history,
                        evaluations=evaluator.counts, optimizer={key: value for key, value in vars(optimizer).items()
                            if key not in {"source", "objective", "_direction_graph"}},
                        elapsed_seconds=prior_seconds+time.monotonic()-started))
                    summary[label] = dict(status=status, final=history[-1], additional_attempts=optimizer.step_attempts-initial_attempts)
                    run.write_json(args.out/"summary.json", summary)
                    print(label, status, history[-1], flush=True)
        del evaluator, optimizer, products, model


if __name__ == "__main__":
    main()

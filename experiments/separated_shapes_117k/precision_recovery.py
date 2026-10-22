"""Paired FP32/FP64 continuations of failed DLR endpoints."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, residual_startup, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.optimizer import SweepLBFGS
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts
from experiments.separated_shapes_117k.scale_recovery import FrozenSeedEvaluator
from experiments.separated_shapes_117k.optimizer_controls import AngleSafeguardedLBFGS
from experiments.separated_shapes_117k.sketch_initialization import retained_seed_startup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--additional-attempts", type=int, default=50)
    parser.add_argument("--products", choices=("native", "graphed"), default="native",
        help="Derivative backend; changes on resume are recorded without resetting optimizer state")
    parser.add_argument("--dtypes", choices=("float32", "float64"), nargs="+", default=["float32", "float64"])
    parser.add_argument("--initialization-prefix", type=int,
        help="Cold restart with the existing prefix sketch at the parent's fixed requested rank; leave at least one KSL step")
    parser.add_argument("--resume", action="store_true",
        help="Extend the saved paired control within its original budget, preserving curvature")
    parser.add_argument("--resume-parent-state", action="store_true",
        help="Branch from an existing FP64 precision-control snapshot without resetting optimizer state")
    parser.add_argument("--direction-cosine", type=float, default=0.,
        help="Experimental minimum descent angle; zero keeps the original LBFGS direction")
    args = parser.parse_args()
    if not 0 <= args.direction_cosine < 1 or (args.direction_cosine and not args.resume_parent_state):
        parser.error("Direction safeguarding requires a preserved parent state and a cosine in (0,1)")
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
    recovery = protocol.get("scale_recovery", {})
    retained = recovery.get("retain_seed", False)
    if args.resume_parent_state:
        if retained or "precision_recovery" not in protocol or args.dtypes != ["float64"] or parent["source"].dtype != torch.float64:
            parser.error("Preserved-state branches require an FP64 precision-control parent and --dtypes float64")
    elif args.initialization_prefix is None and not (retained and recovery["variant"] == "gradient_seed") and not recovery.get("rank_floors"):
        parser.error("Use a completed informed retained-seed or adaptive rank-floor control")
    if not args.resume_parent_state and (parent["status"] not in run.TERMINAL or not parent["optimizer"]["consecutive_failures"]):
        parser.error("The parent must have completed with rejected searches")
    settings = deepcopy(protocol["precision_recovery"]["settings"] if args.resume_parent_state else
        recovery.get("settings", protocol.get("continuation_settings", protocol.get("settings"))))
    if args.initialization_prefix is not None:
        experiment = protocol.get("experiment", {})
        if (retained or experiment.get("factor_integrator", "original") != "original"
                or experiment.get("goal_steps", 0) or not 1 <= args.initialization_prefix < settings["flow_steps"]):
            parser.error("Prefix recovery requires an ordinary parent and must leave at least one original KSL step")
        settings["dlr"].update(adaptive_rank=False, residual_sketch_steps=args.initialization_prefix)
    initial_attempts = parent["optimizer"]["step_attempts"]
    limit = min(initial_attempts + args.additional_attempts, settings["stopping"]["max_updates"])
    if not initial_attempts < limit:
        parser.error("No requested or original attempt budget remains")
    seed = None
    if retained:
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
        seed_sha256=recovery["seed_sha256"]["gradient_seed"] if retained else None,
        product_backend=args.products,
        dtypes=args.dtypes, initialization_prefix=args.initialization_prefix,
        resume_parent_state=args.resume_parent_state, direction_cosine=args.direction_cosine,
        attempts_in_other_starts=recovery.get("attempts_in_other_starts",
            protocol.get("experiment", {}).get("budget", settings["stopping"]["max_updates"])-settings["stopping"]["max_updates"]),
        total_budget=recovery.get("total_budget", protocol.get("experiment", {}).get("budget", settings["stopping"]["max_updates"])),
        note="Paired precision with the recorded derivative backend; cold curvature at initial launch; same promoted source, initialization, right basis and original attempt count. Resume preserves optimizer state; each precision recomputes its initial objective.")
    config_path = args.out/"config.json"
    segments = []
    if config_path.exists():
        if not args.resume:
            parser.error("The control exists; use --resume to preserve its states")
        previous_config = json.loads(config_path.read_text())
        for key in ("parent_sha256", "settings", "seed_sha256", "initial_attempts"):
            assert previous_config[key] == configuration[key]
        assert previous_config.get("resume_parent_state", False) == args.resume_parent_state
        assert previous_config.get("direction_cosine", 0.) == args.direction_cosine
        if limit < previous_config["attempt_limit"]:
            parser.error("Resume cannot lower the recorded attempt limit")
        segments = previous_config.get("segments", []) + [
            {**{key: previous_config[key] for key in ("attempt_limit", "script_sha256")},
             "product_backend": previous_config.get("product_backend", "native")}]
    elif args.resume:
        parser.error("No saved control to resume")
    configuration["segments"] = segments
    if args.resume_parent_state:
        configuration["note"] = "Preserve parent source, gradient, curvature, basis, history and attempt count; change only the recorded direction-angle safeguard. Original total budget retained."
    run.write_json(config_path, configuration)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "native"
    tangent_unet.GN_PRIMAL_BACKEND = "native"
    summary = json.loads((args.out/"summary.json").read_text()) if args.resume else {}
    for dtype in map(lambda name: getattr(torch, name), args.dtypes):
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
        products = (GraphedAProducts(model, source, share_linearization=True, full_step_graphs=True,
                    max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)
                    if args.products == "graphed" else ChunkedAutodiffProducts(model, columns=4))
        evaluator_type = FrozenSeedEvaluator if retained else run.Evaluator
        evaluator = evaluator_type(model, operator, observed, settings, protocol["dlr_seed"], products)
        evaluator.previous_v = (None if state.get("previous_v") is None else
                                state["previous_v"].to(device=device, dtype=dtype))
        if retained:
            evaluator.seed_direction = seed.to(device=device, dtype=dtype)
        evaluator.method = "DLR D-Flow"
        optimizer_type = AngleSafeguardedLBFGS if args.direction_cosine else SweepLBFGS
        optimizer = optimizer_type(source, evaluator.objective, **{
            **({"minimum_cosine": args.direction_cosine} if args.direction_cosine else {}),
            **settings["optimizer"], "step_cap": settings["optimizer"]["step_cap"]*source.numel()**.5,
            "project": settings["source_projection"]})
        optimizer.step_attempts = initial_attempts
        optimizer_parent = saved if saved is not None else parent if args.resume_parent_state else None
        if optimizer_parent is not None:
            for name, value in optimizer_parent["optimizer"].items():
                if name == "minimum_cosine":
                    continue
                setattr(optimizer, name, run.tensor_tree(value, device))
            evaluator.counts = optimizer_parent["evaluations"].copy()
        evaluator.objective(source.detach())
        history = deepcopy(optimizer_parent["history"]) if optimizer_parent is not None else [dict(step=initial_attempts, phi=evaluator.value,
            **run.metrics(evaluator.image, evaluator.measurement, truth, clean, observed))]
        if optimizer_parent is not None:
            assert math.isclose(evaluator.value, history[-1]["phi"], rel_tol=1e-12, abs_tol=1e-12)
        prior_seconds = 0.0 if saved is None else saved["elapsed_seconds"]
        started = time.monotonic()

        def closure() -> torch.Tensor:
            source.grad = evaluator.gradient(source, "DLR D-Flow")
            return source.new_tensor(evaluator.value, dtype=torch.float64)

        with (patch.object(residual_startup, "residual_sketch_startup", retained_seed_startup)
              if retained else nullcontext()):
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
                    summary[label] = dict(status=status, final=history[-1], additional_attempts=optimizer.step_attempts-initial_attempts,
                        total_attempts=configuration["attempts_in_other_starts"]+optimizer.step_attempts,
                        terminal_rank=None if evaluator.previous_v is None else evaluator.previous_v.shape[1])
                    run.write_json(args.out/"summary.json", summary)
                    print(label, status, history[-1], flush=True)
        del evaluator, optimizer, products, model


if __name__ == "__main__":
    main()

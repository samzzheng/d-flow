"""Fork a saved CPU sketch32 solve to test memory reset or a late rank increase.

This is a local intervention with preserved optimizer lifetime counters, not
an uninterrupted continuation. No source checkpoint or prior report is edited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import investigate_fast_dlr_accuracy as study

helper, core, enriched = study.helper, study.core, study.enriched
BASE_PARAMETERS = dict(rank=8, trace_probes=4, enrichment_probes=0, initialization_oversampling=32)


def validate_snapshot(saved, threads, checkpoint_hash, model_tag, *, allow_progress_policy_change=False):
    """Require the full accepted state of the existing CPU sketch32 protocol."""
    settings, protocol = saved["settings"], saved["protocol"]
    expected_settings = {"mode": "solve", "profiles": ["sketch32"], "threads": threads,
                         "keep_terminal_alpha": False, "target_tau": 1.0}
    mismatches = [key for key, value in expected_settings.items() if settings.get(key) != value]
    if settings.get("cached_exact_gradient", False):
        mismatches.append("cached_exact_gradient")
    if mismatches:
        raise ValueError(f"Snapshot settings mismatch: {mismatches}")
    for key in ("sample", "seed", "start_step", "history"):
        if not isinstance(settings.get(key), int) or isinstance(settings[key], bool):
            raise ValueError(f"Snapshot requires integer {key}")
    if min(settings["sample"], settings["start_step"]) < 0 or settings["history"] < 1:
        raise ValueError("Invalid snapshot sample, start step, or history size")
    if not isinstance(settings.get("nested_sketch", False), bool):
        raise ValueError("Invalid nested-sketch setting")
    expected_protocol = {"device": "cpu", "resolution": 250, "model": model_tag,
        "checkpoint_sha256": checkpoint_hash, "flow_steps": 100, "integrator": "Euler",
        "noise": helper.NOISE, "base_seed": helper.SEED, "angles": 15,
        "source_prior": "none", "objective": "relative_scaled", "factor_refresh": "every gradient",
        "armijo_initial_step_policy": helper.bench.ArmijoLBFGS.initial_step_policy,
        "history": settings["history"]}
    mismatches = [key for key, value in expected_protocol.items() if protocol.get(key) != value]
    source_progress = protocol.get("armijo_progress_policy", "legacy_rounded_rhs")
    if source_progress != helper.bench.ArmijoLBFGS.progress_policy and not allow_progress_policy_change:
        mismatches.append("armijo_progress_policy (explicit policy-changing fork required)")
    if protocol.get("sample", settings["sample"]) != settings["sample"]:
        mismatches.append("sample")
    if protocol.get("source_state_step", settings["start_step"]) != settings["start_step"]:
        mismatches.append("source_state_step")
    if mismatches:
        raise ValueError(f"Snapshot protocol mismatch: {mismatches}")
    if saved["profile"] != "sketch32" or saved.get("parameters", BASE_PARAMETERS) != BASE_PARAMETERS:
        raise ValueError("Snapshot must use rank8/m4/p0/q32")
    state = saved["optimizer"]
    if set(study.OPTIMIZER_STATE_KEYS) - state.keys():
        raise ValueError("Snapshot optimizer state is incomplete")
    if (not isinstance(saved["step"], int) or saved["step"] < 1
            or state["step_attempts"] != saved["step"]):
        raise ValueError("Snapshot attempt count does not match its saved step")
    source, image, gradient = saved["source"], saved["raw_image"], state["g"]
    if (source.shape != (1, 1, 250, 250) or image.shape != source.shape
            or gradient is None or gradient.shape != (source.numel(),)
            or any(t.device.type != "cpu" or t.dtype != source.dtype or not torch.isfinite(t).all()
                   for t in (source, image, gradient))):
        raise ValueError("Snapshot source, terminal image, or cached gradient is invalid")
    if not state["state"].get("initialised") or state["f"] != saved["metrics"]["objective"]:
        raise ValueError("Cached objective/gradient does not describe the accepted snapshot")
    if not math.isfinite(state["f"]) or not math.isfinite(saved["metrics"]["residual_over_noise"]):
        raise ValueError("Snapshot objective or residual is nonfinite")
    if len(state["S"]) != len(state["Y"]) or len(state["S"]) > settings["history"]:
        raise ValueError("Invalid saved curvature history")
    for vector in state["S"] + state["Y"]:
        if vector.shape != gradient.shape or vector.dtype != source.dtype or not torch.isfinite(vector).all():
            raise ValueError("Invalid saved curvature vector")
    counters = ("capped_steps", "consecutive_failures", "failed_searches",
                "last_function_evaluations", "total_function_evaluations")
    if any(not isinstance(state[key], int) or state[key] < 0 for key in counters):
        raise ValueError("Invalid optimizer counters")
    counts = saved["counts"]
    if (any(not isinstance(counts[key], int) or counts[key] < 0
            for key in ("gradient_evaluations", "true_forward_evaluations"))
            or not math.isfinite(counts["gradient_seconds"]) or counts["gradient_seconds"] < 0
            or counts["true_forward_evaluations"] < state["total_function_evaluations"]):
        raise ValueError("Invalid evaluation counters")


def make_tail_optimizer(source, objective, saved, mode):
    """Discard curvature without repeating the lifetime initial L1 scaling."""
    if mode not in ("reset", "rank12"):
        raise ValueError(f"Unknown intervention: {mode}")
    optimizer = helper.bench.ArmijoLBFGS(source, objective, lr=1., history=saved["settings"]["history"],
                                       max_ls=30, c1=1e-4, step_cap=2.5)
    for key in study.OPTIMIZER_STATE_KEYS:
        setattr(optimizer, key, copy.deepcopy(saved["optimizer"][key]))
    optimizer.S, optimizer.Y = [], []
    if mode == "rank12":
        # The shared class refreshes the gradient through its normal closure
        # when f is None. Keep initialized state and all lifetime counters.
        optimizer.f, optimizer.g = None, None
    return optimizer


def original_elapsed_seconds(saved):
    if "elapsed_seconds" in saved:
        return saved["elapsed_seconds"]
    original_out = Path(saved["settings"]["out"])
    if not original_out.is_absolute():
        original_out = ROOT / original_out
    point_path = original_out / f"sketch32_step{saved['step']:04d}.pt"
    if point_path.exists():
        point = torch.load(point_path, map_location="cpu", weights_only=True)
        if (point["profile"] != "sketch32" or point["step"] != saved["step"]
                or not torch.equal(point["source"], saved["source"])):
            raise ValueError("Original saved step does not match the snapshot")
        return point["metrics"]["elapsed_seconds"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reset", "rank12"), required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-progress-policy-change", action="store_true",
        help="Explicitly fork an older acceptance policy using current production acceptance")
    args = parser.parse_args()
    if not 1 <= args.steps <= 5:
        parser.error("This bounded local study supports 1–5 additional attempts")
    if not 1 <= args.threads <= 24:
        parser.error("Use 1–24 CPU threads")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be a new or empty directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with args.source_file.open("rb") as handle:
        snapshot_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    saved = torch.load(args.source_file, map_location="cpu", weights_only=True)
    checkpoint = ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    model, tag = helper.load(250, torch.device("cpu"))
    if model.mid_attn or any(model.att_channels):
        raise ValueError("This study requires the 250px non-attention model")
    validate_snapshot(saved, args.threads, checkpoint_hash, tag,
                      allow_progress_policy_change=args.allow_progress_policy_change)
    settings = saved["settings"]
    data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
    sample = settings["sample"]
    if sample >= len(data):
        raise ValueError("Snapshot sample is outside the test dataset")
    truth = helper.bench.normalize_images(data[sample:sample + 1])
    angles = torch.linspace(0, math.pi, 16)[:-1]
    clean = helper.bench.sparse_radon(truth, angles)
    _, target = helper.make_problem(250, sample, data, angles, torch.device("cpu"))
    noise_norm = float((target - clean).norm())
    for key, value in (("noise_norm", noise_norm), ("normalized_noise_floor", noise_norm / float(target.norm()))):
        if not math.isclose(saved["protocol"][key], value, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError(f"Snapshot measurement protocol mismatch: {key}")
    parameters = {**BASE_PARAMETERS, "rank": 12 if args.mode == "rank12" else 8}
    core.SHARE_LINEARIZATION = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    source = saved["source"].detach().clone().requires_grad_(True)
    current_x, current_metrics = saved["raw_image"].detach().clone(), dict(saved["metrics"])
    master_sketch = None
    if settings.get("nested_sketch", False):
        master_sketch = enriched._rademacher(source.numel(), 128,
            torch.Generator().manual_seed(settings["seed"] + 1009), source.device, source.dtype)

    def metrics_for_image(image):
        predicted = helper.bench.sparse_radon(image, angles)
        objective, relative_sq = helper.bench.data_objective(predicted, target, 250, "relative_scaled")
        clamped = image.clamp(0, 1)
        predicted_clamped = helper.bench.sparse_radon(clamped, angles)
        return {"objective": float(objective), "residual": math.sqrt(float(relative_sq)),
            "residual_over_noise": float((predicted - target).norm()) / noise_norm,
            "image_relative_l2": float(helper.bench.relative_l2(clamped, truth)),
            "raw_image_relative_l2": float(helper.bench.relative_l2(image, truth)),
            "clamped_noisy_residual": float(helper.bench.relative_l2(predicted_clamped, target)),
            "clamped_clean_residual": float(helper.bench.relative_l2(predicted_clamped, clean))}

    validation_started = time.perf_counter()
    with torch.no_grad():
        recomputed = metrics_for_image(current_x)
    if any(not math.isclose(value, current_metrics[key], rel_tol=1e-6, abs_tol=1e-9)
           for key, value in recomputed.items()):
        raise ValueError("Saved terminal image metrics do not match the reconstructed measurement problem")
    validation_seconds = time.perf_counter() - validation_started
    counts = dict(saved["counts"])
    last_trial = (source.detach().clone(), current_x, current_metrics)

    def objective(candidate):
        nonlocal last_trial
        with torch.no_grad():
            image = helper.bench.solve_flow(model, candidate, 100, "forward_euler")
            metrics = metrics_for_image(image)
        counts["true_forward_evaluations"] += 1
        last_trial = candidate.detach().clone(), image, metrics
        return metrics["objective"]

    optimizer = make_tail_optimizer(source, objective, saved, args.mode)

    def closure():
        nonlocal current_x, current_metrics
        optimizer.zero_grad()
        if not torch.equal(last_trial[0], source.detach()):
            objective(source.detach())
        _, current_x, current_metrics = last_trial
        started_gradient = time.perf_counter()
        sketch_options = {}
        if master_sketch is not None:
            width = parameters["rank"] + parameters["initialization_oversampling"]
            sketch_options["seed_basis"] = master_sketch[:, :width]
        _, factors, _ = core.sensitivity_factors(model, source.detach(), 100,
            **parameters, **sketch_options, seed=settings["seed"], factor_stride=1,
            zero_terminal_baseline=False)
        terminal = helper.terminal_gradient(current_x, target, 250, angles)
        source.grad = core.approx_source_gradient(factors, terminal, 0.).reshape_as(source).detach()
        counts["gradient_evaluations"] += 1
        counts["gradient_seconds"] += time.perf_counter() - started_gradient
        return current_metrics["objective"]

    original_elapsed = original_elapsed_seconds(saved)
    report = {"settings": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
        "protocol": {**saved["protocol"], "sample": sample, "source_state_step": settings["start_step"],
            "armijo_progress_policy": helper.bench.ArmijoLBFGS.progress_policy},
        "intervention": {"kind": "forked local intervention", "mode": args.mode,
            "source_armijo_progress_policy": saved["protocol"].get("armijo_progress_policy", "legacy_rounded_rhs"),
            "armijo_progress_policy_changed": saved["protocol"].get("armijo_progress_policy", "legacy_rounded_rhs") != helper.bench.ArmijoLBFGS.progress_policy,
            "source_snapshot_sha256": snapshot_hash, "original_profile": saved["profile"],
            "original_step": saved["step"], "original_parameters": BASE_PARAMETERS,
            "parameters": parameters, "cleared_curvature_pairs": len(saved["optimizer"]["S"]),
            "initial_cached_gradient_reused": args.mode == "reset", "terminal_alpha": 0.,
            "factor_stride": 1, "seed": settings["seed"],
            "nested_sketch": settings.get("nested_sketch", False),
            "lr": 1., "step_cap": 2.5, "c1": 1e-4, "max_ls": 30,
            "preserved_optimizer_counters": {key: saved["optimizer"][key] for key in (
                "step_attempts", "capped_steps", "consecutive_failures", "failed_searches",
                "last_function_evaluations", "total_function_evaluations")}},
        "original_counts": saved["counts"], "original_elapsed_seconds": original_elapsed,
        "original_settings": settings, "validation_seconds": validation_seconds, "steps": []}
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def emit(event, **details):
        temporary = args.out / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.out / "results.json")
        print(json.dumps({"event": event, **details}), flush=True)

    def record(local_step, accepted):
        added_elapsed = time.perf_counter() - started
        row = {"local_step": local_step, "step": saved["step"] + local_step,
            "accepted": accepted, **current_metrics, **counts,
            "added_gradient_evaluations": counts["gradient_evaluations"] - saved["counts"]["gradient_evaluations"],
            "added_true_forward_evaluations": counts["true_forward_evaluations"] - saved["counts"]["true_forward_evaluations"],
            "added_gradient_seconds": counts["gradient_seconds"] - saved["counts"]["gradient_seconds"],
            "line_search_trials": optimizer.total_function_evaluations,
            "added_line_search_trials": optimizer.total_function_evaluations - saved["optimizer"]["total_function_evaluations"],
            "failed_searches": optimizer.failed_searches,
            "added_failed_searches": optimizer.failed_searches - saved["optimizer"]["failed_searches"],
            "capped_steps": optimizer.capped_steps, "step_attempts": optimizer.step_attempts,
            "history_size": len(optimizer.S), "added_elapsed_seconds": added_elapsed,
            "elapsed_seconds": None if original_elapsed is None else original_elapsed + added_elapsed}
        report["steps"].append(row)
        state = {"protocol": report["protocol"], "intervention": report["intervention"],
            "settings": report["settings"], "parameters": parameters,
            "local_step": local_step, "step": row["step"], "source": source.detach(),
            "raw_image": current_x, "metrics": current_metrics, "counts": dict(counts),
            "optimizer": {key: getattr(optimizer, key) for key in study.OPTIMIZER_STATE_KEYS},
            "report": report}
        torch.save(state, args.out / f"step{local_step:04d}.tmp")
        (args.out / f"step{local_step:04d}.tmp").replace(args.out / f"step{local_step:04d}.pt")
        emit("step", **row)

    record(0, None)
    for local_step in range(1, args.steps + 1):
        if current_metrics["residual_over_noise"] <= 1. or optimizer.stalled:
            break
        previous_failures = optimizer.failed_searches
        optimizer.step(closure)
        record(local_step, optimizer.failed_searches == previous_failures)
    report["stop_reason"] = ("noise_floor" if current_metrics["residual_over_noise"] <= 1.
                             else "line_search_stalled" if optimizer.stalled else "step_budget")
    report["final"] = report["steps"][-1]
    emit("complete", stop_reason=report["stop_reason"], **report["final"])


if __name__ == "__main__":
    main()

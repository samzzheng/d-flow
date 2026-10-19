"""CPU diagnostics for inexpensive improvements to the fast DLR profile.

Point mode measures exact transpose-action errors at one saved source. Solve
mode starts each profile at the same source and uses the shared production
Armijo optimizer. Nonzero start steps are explicitly local continuations with
fresh optimizer history. --resume continues a saved solve with its full
optimizer state and recorded history; only the outer-step budget may increase.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import test_parallel_bug_integrator as helper
from dlr_dflow import dlr_dflow as core, enriched

PROFILES = {
    "fast": dict(rank=8, trace_probes=4, enrichment_probes=0, initialization_oversampling=8),
    "sketch32": dict(rank=8, trace_probes=4, enrichment_probes=0, initialization_oversampling=32),
    "trace8": dict(rank=8, trace_probes=8, enrichment_probes=0, initialization_oversampling=8),
    "rank12": dict(rank=12, trace_probes=4, enrichment_probes=0, initialization_oversampling=8),
    "enrich1": dict(rank=8, trace_probes=4, enrichment_probes=1, initialization_oversampling=8),
    "reference": dict(rank=8, trace_probes=8, enrichment_probes=4, initialization_oversampling=8),
}

OPTIMIZER_STATE_KEYS = (
    "S", "Y", "f", "g", "state", "step_attempts", "capped_steps",
    "consecutive_failures", "failed_searches", "last_function_evaluations",
    "total_function_evaluations")


def validate_resume_settings(saved, requested):
    """Allow only a larger budget and backward-compatible absent false flags."""
    def normalized(settings):
        result = dict(settings)
        result.pop("resume", None)
        result.setdefault("cached_exact_gradient", False)
        result.setdefault("nested_sketch", False)
        for key in ("source_file", "out"):
            result[key] = str(Path(result[key]).resolve())
        return result

    previous, current = normalized(saved), normalized(requested)
    old_budget, new_budget = previous.pop("outer_steps"), current.pop("outer_steps")
    differences = sorted(key for key in previous.keys() | current.keys()
                         if previous.get(key) != current.get(key))
    if differences:
        raise ValueError(f"Resume settings mismatch: {differences}")
    if new_budget < old_budget:
        raise ValueError("Resume may increase --outer-steps, but cannot reduce it")


def load_solve_resume(args, expected_report, initial):
    """Validate and reconcile a checkpoint with its atomic report/step files."""
    profile, = args.profiles
    saved = torch.load(args.out / f"{profile}_resume.pt", map_location="cpu", weights_only=True)
    report_path = args.out / "results.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    elif "report" in saved:
        report = saved["report"]
    else:
        raise ValueError("Legacy resume checkpoint requires the existing results.json history")
    validate_resume_settings(saved["settings"], expected_report["settings"])
    validate_resume_settings(report["settings"], expected_report["settings"])
    if saved["profile"] != profile:
        raise ValueError("Resume profile mismatch")

    def normalized_protocol(protocol, settings):
        result = dict(protocol)
        # These provenance fields were added after the first saved solves.
        # Their source is verified against the original step-zero tensor below.
        result.setdefault("source_state_step", settings["start_step"])
        result.setdefault("sample", settings["sample"])
        result.setdefault("source_generation_optimizer_policy",
                          expected_report["protocol"]["source_generation_optimizer_policy"])
        result.setdefault("source_generation_progress_policy",
                          expected_report["protocol"]["source_generation_progress_policy"])
        return result

    for owner in (saved, report):
        previous = normalized_protocol(owner["protocol"], owner["settings"])
        if previous != expected_report["protocol"]:
            differences = sorted(key for key in previous.keys() | expected_report["protocol"].keys()
                                 if previous.get(key) != expected_report["protocol"].get(key))
            raise ValueError(f"Resume protocol mismatch: {differences}")
    if len(report["runs"]) != 1 or report["runs"][0]["profile"] != profile:
        raise ValueError("Resume requires exactly one matching profile in the report")
    run = report["runs"][0]
    if run["parameters"] != PROFILES[profile] or saved.get("parameters", run["parameters"]) != PROFILES[profile]:
        raise ValueError("Resume profile parameters have changed")
    step = saved["step"]
    if not isinstance(step, int) or not 0 <= step <= args.outer_steps:
        raise ValueError("Invalid saved step or insufficient outer-step budget")
    missing = set(OPTIMIZER_STATE_KEYS) - saved["optimizer"].keys()
    if missing or saved["optimizer"].get("step_attempts") != step:
        raise ValueError(f"Incomplete or inconsistent optimizer state: {sorted(missing)}")
    original = torch.load(args.out / f"{profile}_step0000.pt", map_location="cpu", weights_only=True)
    if (original["source"].shape != initial.shape or original["source"].dtype != initial.dtype
            or not torch.equal(original["source"], initial)):
        raise ValueError("Original source file contents have changed since this solve started")
    rows = run["steps"]
    if [row["step"] for row in rows] != list(range(len(rows))) or len(rows) > step + 1:
        raise ValueError("Report history is duplicated, unordered, or ahead of its optimizer checkpoint")
    # A process may stop after atomically saving the optimizer but before
    # emitting results.json. The already-written step file recovers that row.
    for missing_step in range(len(rows), step + 1):
        point = torch.load(args.out / f"{profile}_step{missing_step:04d}.pt",
                           map_location="cpu", weights_only=True)
        if point["step"] != missing_step or point["profile"] != profile:
            raise ValueError("Saved step record does not match resume history")
        rows.append(point["metrics"])
    expected_last = {**saved["metrics"], **saved["counts"],
                     "line_search_trials": saved["optimizer"]["total_function_evaluations"],
                     "failed_searches": saved["optimizer"]["failed_searches"],
                     "history_size": len(saved["optimizer"]["S"])}
    if any(rows[-1].get(key) != value for key, value in expected_last.items()):
        raise ValueError("Saved metrics/counters do not match the report's final step")
    elapsed = rows[-1]["elapsed_seconds"]
    if not math.isfinite(elapsed) or elapsed < 0 or saved.get("elapsed_seconds", elapsed) != elapsed:
        raise ValueError("Invalid or inconsistent saved elapsed time")
    report.setdefault("resume_history", []).append({
        "step": step, "elapsed_seconds": elapsed, "previous_settings": report["settings"],
        "previous_stop_reason": run.get("stop_reason"), "previous_final": run.get("final")})
    report["settings"] = expected_report["settings"]
    report["protocol"] = expected_report["protocol"]
    run.pop("stop_reason", None)
    run.pop("final", None)
    return saved, report, elapsed


def action_metrics(g, reference):
    g, reference = g.flatten(), reference.flatten()
    norm = float(reference.norm())
    return {"relative_error": float((g - reference).norm()) / norm,
            "cosine": float(torch.nn.functional.cosine_similarity(g[None], reference[None])),
            "norm_ratio": float(g.norm()) / norm}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["points", "solve"], default="points")
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=["fast", "sketch32", "trace8", "reference"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=20)
    parser.add_argument("--source-file", type=Path, default=ROOT / "outputs/accuracy_fast_euler_2026_09_27/exact_from_z0/sources_sample0_exact.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nested-sketch", action="store_true",
        help="Use common columns from an independent initialization sketch for oversampling controls")
    parser.add_argument("--cached-exact-gradient", action="store_true",
        help="Reuse an exact gradient in a matching accepted Euler state (point mode only)")
    parser.add_argument("--outer-steps", type=int, default=10)
    parser.add_argument("--history", type=int, default=100)
    parser.add_argument("--target-tau", type=float, default=1.0)
    parser.add_argument("--keep-terminal-alpha", action="store_true")
    parser.add_argument("--resume", action="store_true",
        help="Resume one solve profile from its checkpoint in --out; only --outer-steps may increase")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/fast_accuracy_recovery_2026_09_27/points")
    args = parser.parse_args()
    if not 1 <= args.threads <= 24:
        parser.error("Use 1–24 CPU threads")
    if args.resume and (args.mode != "solve" or len(args.profiles) != 1):
        parser.error("--resume requires solve mode and exactly one --profiles entry")
    if args.mode == "solve" and args.cached_exact_gradient:
        parser.error("--cached-exact-gradient is only supported in point mode")
    if args.mode == "solve" and not args.resume and ((args.out / "results.json").exists()
            or any((args.out / f"{profile}_resume.pt").exists() for profile in args.profiles)):
        parser.error("Solve output already exists; use --resume or a new --out directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    model, tag = helper.load(250, torch.device("cpu"))
    assert not model.mid_attn and not any(model.att_channels)
    checkpoint = ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    saved = torch.load(args.source_file, map_location="cpu", weights_only=True)
    source_protocol = saved["protocol"]
    source_hash = source_protocol.get("checkpoint_sha256", source_protocol.get("model_checkpoint_sha256"))
    source_sample = saved.get("sample", source_protocol.get("sample"))
    if source_hash != checkpoint_hash or source_sample != args.sample:
        raise ValueError("Source file does not match checkpoint/sample")
    if "sources" in saved:
        initial = saved["sources"][args.start_step].detach().clone()
    else:
        if saved["metrics"]["attempt"] != args.start_step:
            raise ValueError("Accepted-state attempt does not match --start-step")
        initial = saved["source"].detach().clone()
    if args.cached_exact_gradient:
        if args.mode != "points" or "gradient" not in saved:
            parser.error("Cached exact gradients require point mode and an accepted Euler state")
        expected = {"method": "forward_euler", "flow_steps": 100, "device": "cpu",
            "threads": args.threads, "noise_relative_clean": helper.NOISE,
            "base_seed": helper.SEED, "angles": 15, "source_prior": "none",
            "objective": "relative_scaled"}
        mismatched = [key for key, value in expected.items() if source_protocol.get(key) != value]
        if mismatched:
            raise ValueError(f"Cached exact gradient protocol mismatch: {mismatched}")
    data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
    truth = helper.bench.normalize_images(data[args.sample:args.sample + 1])
    angles = torch.linspace(0, math.pi, 16)[:-1]
    clean = helper.bench.sparse_radon(truth, angles)
    _, target = helper.make_problem(250, args.sample, data, angles, torch.device("cpu"))
    noise_norm = float((target - clean).norm())
    core.SHARE_LINEARIZATION = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                           if k != "resume"},
        "protocol": {"device": "cpu", "resolution": 250, "model": tag,
            "checkpoint_sha256": checkpoint_hash, "flow_steps": 100, "integrator": "Euler",
            "noise": helper.NOISE, "base_seed": helper.SEED, "angles": 15,
            "source_prior": "none", "objective": "relative_scaled", "factor_refresh": "every gradient",
            "armijo_initial_step_policy": helper.bench.ArmijoLBFGS.initial_step_policy,
            "armijo_progress_policy": helper.bench.ArmijoLBFGS.progress_policy,
            "source_generation_optimizer_policy": ("initial_random_source" if args.start_step == 0
                else source_protocol.get("armijo_initial_step_policy", "legacy_empty_history")),
            "source_generation_progress_policy": ("initial_random_source" if args.start_step == 0
                else source_protocol.get("armijo_progress_policy", "legacy_rounded_rhs")),
            "source_state_step": args.start_step, "sample": args.sample,
            "optimizer_history_reset_at_start": True,
            "history": args.history,
            "noise_norm": noise_norm, "normalized_noise_floor": noise_norm / float(target.norm()),
            "timing_scope": "CPU diagnostic; not a CUDA speed claim"}, "runs": []}
    resumed, elapsed_before_resume = None, 0.0
    if args.resume:
        resumed, report, elapsed_before_resume = load_solve_resume(args, report, initial)
    master_sketch = None
    if args.nested_sketch:
        # A fixed master width makes the common columns independent of which
        # profiles are selected. The production row-major RNG is not nested
        # when its requested number of columns changes.
        master_sketch = enriched._rademacher(initial.numel(), 128,
            torch.Generator().manual_seed(args.seed + 1009), initial.device, initial.dtype)

    def emit(event, **details):
        temporary = args.out / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.out / "results.json")
        print(json.dumps({"event": event, **details}), flush=True)

    def evaluate(z):
        with torch.no_grad():
            x = helper.bench.solve_flow(model, z, 100, "forward_euler")
            predicted = helper.bench.sparse_radon(x, angles)
            objective, relative_sq = helper.bench.data_objective(predicted, target, 250, "relative_scaled")
            residual_norm = float((predicted - target).norm())
            clamped = x.clamp(0, 1)
            predicted_clamped = helper.bench.sparse_radon(clamped, angles)
            metrics = {"objective": float(objective), "residual": math.sqrt(float(relative_sq)),
                "residual_over_noise": residual_norm / noise_norm,
                "image_relative_l2": float(helper.bench.relative_l2(clamped, truth)),
                "raw_image_relative_l2": float(helper.bench.relative_l2(x, truth)),
                "clamped_noisy_residual": float(helper.bench.relative_l2(predicted_clamped, target)),
                "clamped_clean_residual": float(helper.bench.relative_l2(predicted_clamped, clean))}
        return x, metrics

    def approximate(z, native_x, profile):
        start = time.perf_counter()
        sketch_options = {}
        if master_sketch is not None:
            width = PROFILES[profile]["rank"] + PROFILES[profile]["initialization_oversampling"]
            sketch_options["seed_basis"] = master_sketch[:, :width]
        x, factors, alpha = core.sensitivity_factors(model, z.detach(), 100,
            **PROFILES[profile], **sketch_options, seed=args.seed, factor_stride=1, zero_terminal_baseline=False)
        terminal = helper.terminal_gradient(native_x, target, 250, angles)
        gradient = core.approx_source_gradient(factors, terminal, float(alpha) if args.keep_terminal_alpha else 0.0).reshape_as(z)
        return gradient.detach(), factors, float(alpha), terminal, time.perf_counter() - start, x

    if resumed is None:
        x_initial, initial_metrics = evaluate(initial)
        report["initial"] = initial_metrics
        emit("initial", **initial_metrics)
    else:
        emit("resume", profile=resumed["profile"], step=resumed["step"],
             elapsed_seconds=elapsed_before_resume)
    if args.mode == "points":
        if args.cached_exact_gradient:
            torch.testing.assert_close(x_initial, saved["raw_reconstruction"], rtol=1e-5, atol=1e-6)
            exact = saved["gradient"].detach()
            report["exact_gradient_seconds"] = 0.0
            report["exact_gradient_reused_from"] = str(args.source_file)
        else:
            z = initial.clone().requires_grad_(True)
            start = time.perf_counter()
            x = helper.bench.solve_flow(model, z, 100, "forward_euler", False)
            value, _ = helper.bench.data_objective(helper.bench.sparse_radon(x, angles), target, 250, "relative_scaled")
            exact, = torch.autograd.grad(value, z)
            exact = exact.detach()
            report["exact_gradient_seconds"] = time.perf_counter() - start
            del x, value, z
        tensors = {"source": initial, "exact_gradient": exact, "native_terminal": x_initial}
        gc.collect()
        emit("exact", seconds=report["exact_gradient_seconds"], norm=float(exact.norm()))
        for profile in args.profiles:
            g, factors, alpha, terminal, seconds, primal = approximate(initial, x_initial, profile)
            retained = core.approx_source_gradient(factors, terminal, alpha)
            V = factors[2]
            best_in_subspace = V @ (V.T @ exact.flatten())
            row = {"profile": profile, "parameters": PROFILES[profile], "seconds": seconds,
                "terminal_alpha": alpha, "zero_alpha": action_metrics(core.approx_source_gradient(factors, terminal, 0.0), exact),
                "retained_alpha": action_metrics(retained, exact),
                "best_gradient_in_V_relative_error": float((best_in_subspace - exact.flatten()).norm() / exact.norm()),
                "V_orthogonality_defect": float((V.T @ V - torch.eye(V.shape[1])).norm()),
                "coefficient_component_relative_error": float((core.approx_source_gradient(factors, terminal, 0.0).flatten() - best_in_subspace).norm() / exact.norm()),
                "primal_relative_error": float((primal - x_initial).norm() / x_initial.norm())}
            report["runs"].append(row)
            tensors[profile] = {"gradient_zero_alpha": core.approx_source_gradient(factors, terminal, 0.0),
                "gradient_with_alpha": retained, "factors": factors, "alpha": alpha}
            torch.save(tensors, args.out / "gradients.pt")
            emit("point", **row)
        return

    for profile in args.profiles:
        source = (initial if resumed is None else resumed["source"]).detach().clone().requires_grad_(True)
        if resumed is None:
            current_x, current_metrics = x_initial, dict(initial_metrics)
        else:
            current_x, current_metrics = resumed["raw_image"], dict(resumed["metrics"])
        last_trial = None
        if resumed is None:
            counts = {"gradient_evaluations": 0, "true_forward_evaluations": 0, "gradient_seconds": 0.0}
            run = {"profile": profile, "parameters": PROFILES[profile], "steps": []}
            report["runs"].append(run)
        else:
            counts = dict(resumed["counts"])
            run = report["runs"][0]
        started = time.perf_counter()

        def objective(candidate):
            nonlocal last_trial
            x, metrics = evaluate(candidate)
            counts["true_forward_evaluations"] += 1
            last_trial = (candidate.detach().clone(), x, metrics)
            return metrics["objective"]

        optimizer = helper.bench.ArmijoLBFGS(source, objective, lr=1.0, history=args.history, max_ls=30, c1=1e-4, step_cap=2.5)
        if resumed is not None:
            for key in OPTIMIZER_STATE_KEYS:
                setattr(optimizer, key, resumed["optimizer"][key])

        def closure():
            nonlocal current_x, current_metrics
            optimizer.zero_grad()
            if last_trial is None or not torch.equal(last_trial[0], source.detach()):
                objective(source.detach())
            _, current_x, current_metrics = last_trial
            gradient, _, _, _, seconds, _ = approximate(source, current_x, profile)
            source.grad = gradient
            counts["gradient_evaluations"] += 1
            counts["gradient_seconds"] += seconds
            return current_metrics["objective"]

        def record(step):
            row = {"step": step, **current_metrics, **counts,
                "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
                "line_search_trials": optimizer.total_function_evaluations,
                "failed_searches": optimizer.failed_searches, "history_size": len(optimizer.S)}
            run["steps"].append(row)
            torch.save({"protocol": report["protocol"], "profile": profile, "step": step,
                "source": source.detach(), "raw_image": current_x, "metrics": row},
                args.out / f"{profile}_step{step:04d}.pt")
            resume = {"protocol": report["protocol"], "settings": report["settings"],
                "profile": profile, "parameters": PROFILES[profile], "step": step,
                "source": source.detach(), "raw_image": current_x,
                "metrics": current_metrics, "counts": dict(counts),
                "elapsed_seconds": row["elapsed_seconds"], "report": report,
                "optimizer": {key: getattr(optimizer, key) for key in OPTIMIZER_STATE_KEYS}}
            torch.save(resume, args.out / f"{profile}_resume.tmp")
            (args.out / f"{profile}_resume.tmp").replace(args.out / f"{profile}_resume.pt")
            emit("step", profile=profile, **row)

        start_step = 0 if resumed is None else resumed["step"]
        if resumed is None:
            record(0)
        already_stopped = start_step > 0 and (current_metrics["residual_over_noise"] <= args.target_tau
                                              or optimizer.stalled)
        if not already_stopped:
            for step in range(start_step + 1, args.outer_steps + 1):
                optimizer.step(closure)
                record(step)
                if current_metrics["residual_over_noise"] <= args.target_tau or optimizer.stalled:
                    break
        run["stop_reason"] = ("noise_floor" if current_metrics["residual_over_noise"] <= args.target_tau
            else "line_search_stalled" if optimizer.stalled else "step_budget")
        run["final"] = run["steps"][-1]
        emit("solve_complete", profile=profile, stop_reason=run["stop_reason"])
        del source, optimizer
        gc.collect()


if __name__ == "__main__":
    main()

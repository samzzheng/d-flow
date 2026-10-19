"""CPU exact D-Flow Euler/RK4 pilot with first-crossing noise-floor snapshots.

Both integrators always use 100 flow steps at 250px without attention. Start
from the protocol's original Gaussian source, use the central first-only
Armijo L-BFGS policy, and save every accepted source, gradient and image.
The authoritative resume checkpoint includes the complete optimizer state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench

OPTIMIZER_ATTRIBUTES = ("S", "Y", "f", "g", "state", "step_attempts",
    "capped_steps", "consecutive_failures", "last_function_evaluations",
    "total_function_evaluations", "failed_searches")


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def make_optimizer(source, objective, *, legacy_progress=False, **kwargs):
    if legacy_progress:
        from optimizer_settings_audit import AuditedArmijo
        return AuditedArmijo(source, objective, scaling="once", progress="legacy", **kwargs)
    return bench.ArmijoLBFGS(source, objective, **kwargs)


def validate_resume_protocol(saved, current, *, legacy_progress=False):
    comparable = dict(saved)
    normalized_missing_tag = (
        legacy_progress
        and current.get("armijo_progress_policy") == "legacy_rounded_rhs"
        and "armijo_progress_policy" not in comparable)
    if normalized_missing_tag:
        comparable["armijo_progress_policy"] = "legacy_rounded_rhs"
    if comparable != current:
        raise ValueError("Resume protocol mismatch; optimizer/model/precision/sample must be identical")
    return {"original_progress_policy":saved.get("armijo_progress_policy", "legacy_rounded_rhs"),
            "resumed_progress_policy":current["armijo_progress_policy"],
            "normalized_missing_legacy_tag":normalized_missing_tag}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--method", choices=("forward_euler", "rk4"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-minutes", type=float, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--legacy-progress", action="store_true",
                        help="Explicitly retain historical rounded Armijo acceptance for a compatible continuation")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("Use 1–8 CPU threads on the shared machine")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "accepted").mkdir(exist_ok=True)
    checkpoint_path = args.out / "resume.pt"
    if checkpoint_path.exists() and not args.resume:
        raise ValueError("Output already has a checkpoint; use --resume or a new output directory")
    if args.resume and not checkpoint_path.exists():
        raise ValueError("--resume requires an existing resume.pt")
    checkpoint = ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    with checkpoint.open("rb") as handle:
        model_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    model = bench.load_model(250, 100, torch.device("cpu"),
                            "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
    assert not model.mid_attn and not any(model.att_channels)
    dataset = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu",
                         weights_only=True, mmap=True)["test"]
    truth = bench.normalize_images(dataset[args.sample:args.sample+1])
    angles = torch.linspace(0, math.pi, 16)[:-1]
    clean = bench.sparse_radon(truth, angles)
    observed, _, _ = bench.add_relative_noise(clean, .01,
        torch.Generator().manual_seed(20260807 + 250000 + args.sample))
    noise_norm = float((observed-clean).norm())
    source = torch.randn(truth.shape, generator=torch.Generator().manual_seed(
        20260807 + 10000000 + 250000 + args.sample)).requires_grad_(True)
    protocol = {"resolution":250, "sample":args.sample, "method":args.method,
        "flow_steps":100, "velocity_evaluations_per_flow":100 if args.method=="forward_euler" else 400,
        "model_checkpoint_sha256":model_hash, "attention":False, "device":"cpu",
        "threads":args.threads, "torch":str(torch.__version__), "activation_checkpointing":False,
        "noise_relative_clean":.01, "noise_norm":noise_norm,
        "noise_relative_observed":noise_norm/float(observed.norm()), "angles":15,
        "base_seed":20260807, "source_prior":"none", "objective":"relative_scaled",
        "optimizer":"central ArmijoLBFGS, first-attempt-only L1 scaling",
        "armijo_initial_step_policy":"first_iteration_only",
        "armijo_progress_policy":("legacy_rounded_rhs" if args.legacy_progress
                                  else bench.ArmijoLBFGS.progress_policy),
        "history":100, "lr":1., "step_cap":2.5, "armijo_c1":1e-4, "max_line_search":30,
        "primary_discrepancy_tau":1.0, "secondary_discrepancy_taus":[1.05,1.1],
        "start":"original matched Gaussian source", "metrics_image":"clamped and raw separately",
        "sample_selection":("sample1 selected as pilot because historical local Euler and RK4 both reached strict noise floor"
                            if args.sample == 1 else "additional sample specified by CLI")}
    counts = {"gradients":0, "forward_trials":0}
    accepted = None
    report = {"protocol":protocol, "started_utc":datetime.now(timezone.utc).isoformat(),
              "history":[], "first_crossings":{}, "stop_reason":None}
    elapsed_before = 0.
    started = time.perf_counter()

    def objective(candidate):
        with torch.no_grad():
            x = bench.solve_flow(model, candidate, 100, args.method, False)
            value, _ = bench.data_objective(bench.sparse_radon(x, angles), observed, 250, "relative_scaled")
        counts["forward_trials"] += 1
        return float(value)

    optimizer = make_optimizer(source, objective, legacy_progress=args.legacy_progress,
                               lr=1., history=100, max_ls=30, c1=1e-4, step_cap=2.5)
    if not hasattr(optimizer, "step_attempts"):
        raise RuntimeError("The central first-only optimizer update must be installed before this experiment")
    if optimizer.initial_step_policy != protocol["armijo_initial_step_policy"]:
        raise RuntimeError("Unexpected central Armijo initialization policy")
    if optimizer.progress_policy != protocol["armijo_progress_policy"]:
        raise RuntimeError("Unexpected Armijo progress policy")

    def closure():
        nonlocal accepted
        optimizer.zero_grad()
        x = bench.solve_flow(model, source, 100, args.method, False)
        predicted = bench.sparse_radon(x, angles)
        value, _ = bench.data_objective(predicted, observed, 250, "relative_scaled")
        gradient, = torch.autograd.grad(value, source)
        source.grad = gradient.detach()
        counts["gradients"] += 1
        accepted = {"x":x.detach(), "predicted_raw":predicted.detach(), "objective":float(value.detach())}
        return value.detach()

    def metrics(attempt):
        raw = accepted["x"]
        clamped = raw.clamp(0,1)
        predicted_clamped = bench.sparse_radon(clamped, angles)
        raw_residual = float((accepted["predicted_raw"]-observed).norm())
        mse = float((clamped-truth).square().mean())
        return {"attempt":attempt, "accepted_index":counts["gradients"]-1,
            "objective":accepted["objective"], "residual_relative_observed":raw_residual/float(observed.norm()),
            "residual_over_noise":raw_residual/noise_norm,
            "image_relative_l2":float(bench.relative_l2(clamped,truth)),
            "raw_image_relative_l2":float(bench.relative_l2(raw,truth)),
            "psnr_db":-10*math.log10(max(mse,1e-30)),
            "clamped_noisy_residual":float(bench.relative_l2(predicted_clamped,observed)),
            "clamped_clean_residual":float(bench.relative_l2(predicted_clamped,clean)),
            "source_norm_over_sqrt_d":float(source.detach().norm())/250,
            "elapsed_seconds":elapsed_before+time.perf_counter()-started,
            "failed_searches":optimizer.failed_searches,
            "function_evaluations":optimizer.total_function_evaluations,
            "step_attempts":optimizer.step_attempts, "capped_steps":optimizer.capped_steps,
            "history_length":len(optimizer.S), **counts}

    def save(event, row=None):
        report["elapsed_seconds"] = elapsed_before+time.perf_counter()-started
        payload = {"protocol":protocol, "source":source.detach(), "accepted":accepted,
            "optimizer":{k:getattr(optimizer,k) for k in OPTIMIZER_ATTRIBUTES}, "counts":counts,
            "report":report}
        atomic_torch_save(payload, checkpoint_path)
        temporary = args.out / "results.tmp"
        temporary.write_text(json.dumps(report,indent=2)+"\n")
        temporary.replace(args.out / "results.json")
        print(json.dumps({"event":event,"method":args.method,"sample":args.sample,
                          **({} if row is None else row)},allow_nan=False),flush=True)

    def record(attempt):
        row = metrics(attempt)
        report["history"].append(row)
        state_path = args.out / "accepted" / f"state_{row['accepted_index']:05d}.pt"
        atomic_torch_save({"source":source.detach(), "gradient":optimizer.g,
            "raw_reconstruction":accepted["x"], "metrics":row, "protocol":protocol},state_path)
        for tau in (1.,1.05,1.1):
            if str(tau) not in report["first_crossings"] and row["residual_over_noise"] <= tau:
                report["first_crossings"][str(tau)] = row | {"saved_state":str(state_path)}
        save("accepted",row)

    if args.resume:
        saved = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
        resume_identity = validate_resume_protocol(saved["protocol"], protocol,
                                                    legacy_progress=args.legacy_progress)
        with torch.no_grad(): source.copy_(saved["source"])
        accepted, counts, report = saved["accepted"], saved["counts"], saved["report"]
        for name,value in saved["optimizer"].items(): setattr(optimizer,name,value)
        elapsed_before = report["elapsed_seconds"]
        report["stop_reason"] = None
        started = time.perf_counter()
        provenance_path = args.out / "resume_provenance.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else []
        event = {"utc":datetime.now(timezone.utc).isoformat(), **resume_identity,
                 "original_protocol":saved["protocol"], "resumed_protocol":protocol,
                 "accepted_index":report["history"][-1]["accepted_index"],
                 "step_attempts":optimizer.step_attempts, "elapsed_before_seconds":elapsed_before,
                 "restored_optimizer_attributes":list(OPTIMIZER_ATTRIBUTES),
                 "max_steps":args.max_steps, "max_minutes_cumulative":args.max_minutes,
                 "existing_accepted_states_rewritten":False}
        provenance.append(event)
        provenance_temporary = provenance_path.with_suffix(".tmp")
        provenance_temporary.write_text(json.dumps(provenance,indent=2)+"\n")
        provenance_temporary.replace(provenance_path)
        report.setdefault("resume_events", []).append({key:value for key,value in event.items()
                                                       if key not in ("original_protocol", "resumed_protocol")})
        save("resumed",report["history"][-1])
    else:
        optimizer.f,optimizer.g = optimizer._evaluate_gradient(closure)
        optimizer.state["initialised"] = True
        record(0)

    while optimizer.step_attempts < args.max_steps:
        if "1.0" in report["first_crossings"]:
            report["stop_reason"] = "primary_noise_floor"
            break
        if elapsed_before+time.perf_counter()-started > args.max_minutes*60:
            report["stop_reason"] = "wall_budget_checkpointed"
            break
        before = counts["gradients"]
        optimizer.step(closure)
        if counts["gradients"] > before:
            record(optimizer.step_attempts)
        else:
            save("failed_search",{"attempt":optimizer.step_attempts,"failed_searches":optimizer.failed_searches})
        if optimizer.stalled:
            report["stop_reason"] = "two_failed_searches"
            break
    if report["stop_reason"] is None:
        report["stop_reason"] = "primary_noise_floor" if "1.0" in report["first_crossings"] else "step_budget_checkpointed"
    save("complete",{"stop_reason":report["stop_reason"],"first_crossings":report["first_crossings"]})


if __name__ == "__main__":
    main()

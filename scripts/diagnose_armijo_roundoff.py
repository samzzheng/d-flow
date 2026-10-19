"""Replay one saved line search without constructing a new DLR gradient.

The two acceptance policies share cached, identical native forward trials.
Stop immediately at the accepted-gradient callback: this diagnoses acceptance,
not a resumed optimization trajectory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

import investigate_fast_dlr_tail as tail
from optimizer_settings_audit import AuditedArmijo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output file")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    helper = tail.helper
    saved = torch.load(args.source_file, map_location="cpu", weights_only=True)
    checkpoint = tail.ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    model, tag = helper.load(250, torch.device("cpu"))
    tail.validate_snapshot(saved, args.threads, checkpoint_hash, tag,
                           allow_progress_policy_change=True)
    data = torch.load(tail.ROOT / "data/circles-250-n1.pt", map_location="cpu",
                      weights_only=True, mmap=True)["test"]
    angles = torch.linspace(0, math.pi, 16)[:-1]
    _, target = helper.make_problem(250, saved["settings"]["sample"], data, angles,
                                    torch.device("cpu"))
    cache = []
    unique_forwards = 0
    started = time.perf_counter()

    def native_objective(candidate):
        nonlocal unique_forwards
        for source, value in cache:
            if torch.equal(source, candidate):
                return value, True
        with torch.no_grad():
            image = helper.bench.solve_flow(model, candidate, 100, "forward_euler")
            predicted = helper.bench.sparse_radon(image, angles)
            value, _ = helper.bench.data_objective(predicted, target, 250, "relative_scaled")
        value = float(value)
        cache.append((candidate.detach().clone(), value))
        unique_forwards += 1
        return value, False

    initial, _ = native_objective(saved["source"])
    if initial != saved["optimizer"]["f"]:
        raise ValueError("Native objective does not exactly match the saved state")
    report = {"source_file": str(args.source_file), "protocol": saved["protocol"],
        "source_progress_policy": saved["protocol"].get("armijo_progress_policy", "legacy_rounded_rhs"),
        "threads": args.threads, "device": "cpu", "flow_steps": 100,
        "scope": "Single line-search acceptance replay; no new gradient evaluations. "
                 "Identical native forward candidates are cached across policies.",
        "runs": []}

    class AcceptedCallback(Exception):
        pass

    for mode in ("legacy", "strict"):
        source = saved["source"].detach().clone().requires_grad_(True)
        trials = []

        def objective(candidate):
            value, hit = native_objective(candidate)
            delta = candidate.detach() - saved["source"]
            trials.append({"objective": value, "cache_hit": hit,
                "changed_coordinates": int(torch.count_nonzero(delta)),
                "source_step_norm": float(delta.norm())})
            return value

        optimizer = (AuditedArmijo(source, objective, scaling="once", progress="legacy",
                                  lr=1., history=100, max_ls=30, c1=1e-4, step_cap=2.5)
                     if mode == "legacy" else helper.bench.ArmijoLBFGS(
                         source, objective, lr=1., history=100, max_ls=30, c1=1e-4, step_cap=2.5))
        for key in tail.study.OPTIMIZER_STATE_KEYS:
            setattr(optimizer, key, copy.deepcopy(saved["optimizer"][key]))
        called = False

        def closure():
            nonlocal called
            called = True
            raise AcceptedCallback

        try:
            optimizer.step(closure)
        except AcceptedCallback:
            pass
        delta = source.detach() - saved["source"]
        report["runs"].append({"policy": optimizer.progress_policy,
            "accepted_gradient_callback_reached": called,
            "source_changed": not torch.equal(source.detach(), saved["source"]),
            "source_step_norm": float(delta.norm()), "trials": trials,
            "step_attempts": optimizer.step_attempts,
            "failed_searches": optimizer.failed_searches,
            "curvature_pairs_after": len(optimizer.S)})
    report.update(unique_native_forward_evaluations=unique_forwards,
                  elapsed_seconds=time.perf_counter() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Verify paired proxy-recovery pilots and summarize their saved checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    repeated_trials = []
    for folder in sorted(args.root.glob("proxy_recovery_*/")):
        config = json.loads((folder/"config.json").read_text())
        parent_path = Path(config["parent"])
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        inputs = torch.load(parent_path.with_name("inputs.pt"), map_location="cpu", weights_only=True)
        assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == config["parent_sha256"]
        assert hashlib.sha256(parent["source"].numpy().tobytes()).hexdigest() == config["initial_source_sha256"]
        settings = config["settings"]
        for key in ("optimizer", "stopping", "source_projection", "flow_steps", "dlr"):
            assert settings[key] == parent["protocol"]["settings"][key]
        cap = settings["optimizer"]["step_cap"]*math.sqrt(parent["source"].numel())
        results = {}
        first_events = []
        for arm in ("reset_dlr", "proxy_then_dlr", "proxy_only"):
            path = folder/arm/("p_flow.pt" if arm == "proxy_only" else "dlr_d_flow.pt")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["status"] in ("stalled", "step_limit", "proxy_rejected")
            assert saved["protocol"]["proxy_recovery"] == {**config, "arm": arm}
            for key in ("checkpoint_sha256", "operator", "source_seed", "dlr_seed"):
                assert saved["protocol"][key] == inputs["protocol"][key]
            history = saved["history"]
            assert history[0]["step"] == config["initial_attempt"]
            assert history[0]["phi"] == parent["history"][-1]["phi"]
            assert history[-1]["step"] <= config["attempt_limit"]
            assert all(right["phi"] <= left["phi"] for left, right in zip(history, history[1:]))
            assert all(row.get("source_step_norm", 0) <= cap*(1+1e-4) for row in history)
            truth = inputs["truth"].double()
            dtype = torch.complex128 if inputs["observed"].is_complex() else torch.float64
            observed = inputs["observed"].to(dtype)
            image_error = float((saved["image"].double()-truth).norm()/truth.norm())
            residual = float((saved["measurement"].to(dtype)-observed).norm()/observed.norm())
            objective = float(.5*(saved["measurement"].to(dtype)-observed).abs().square().sum())
            assert abs(image_error-history[-1]["image_relative_l2"]) < 1e-6
            assert abs(residual-history[-1]["measurement_relative_l2_noisy"]) < 1e-6
            assert math.isclose(objective, history[-1]["phi"], rel_tol=2e-6, abs_tol=1e-10)
            events = []
            if arm != "reset_dlr":
                events = list(map(json.loads, (folder/arm/"proxy_trials.jsonl").read_text().splitlines()))
                first_events.append(events[0])
                for event in events:
                    assert len(event["trials"]) <= settings["optimizer"]["max_ls"]
                    assert all(trial["displacement"] <= cap*(1+1e-6) for trial in event["trials"])
                    accepted = [trial for trial in event["trials"] if trial["accepted"]]
                    assert bool(accepted) == event["accepted"]
                    for trial in accepted:
                        bound = min(event["baseline_values"])-event["margin"]
                        assert trial["value"] < bound and trial["repeat_value"] < bound
                if not any(event["accepted"] for event in events):
                    assert torch.equal(saved["source"], parent["source"])
                if saved["status"] == "proxy_rejected":
                    assert saved["optimizer"]["g"] is None
                    assert not saved["optimizer"]["S"] and not saved["optimizer"]["Y"]
            rows.append(dict(case=folder.name, arm=arm, status=saved["status"],
                parent_error_percent=100*history[0]["image_relative_l2"],
                image_error_percent=100*image_error, residual_percent=100*residual,
                initial_phi=history[0]["phi"], final_phi=history[-1]["phi"],
                additional_attempts=history[-1]["step"]-config["initial_attempt"],
                accepted_proxy_steps=sum(event["accepted"] for event in events),
                elapsed_seconds=saved["elapsed_seconds"],
                first_proxy_full_cap_phi=events[0]["trials"][0]["value"] if events else None,
                margin=events[0]["margin"] if events else None))
            results[arm] = saved
        first, second = first_events
        assert first["baseline_values"] == second["baseline_values"]
        assert first["accepted"] == second["accepted"]
        assert len(first["trials"]) == len(second["trials"])
        assert all(abs(left["displacement"]-right["displacement"]) < 1e-7
                   for left, right in zip(first["trials"], second["trials"]))
        repeated_trials.append(dict(case=folder.name,
            maximum_trial_objective_difference=max(abs(left["value"]-right["value"])
                for left, right in zip(first["trials"], second["trials"])),
            note="Same source and proposal rule; CT endpoint gradients recomputed separately can differ numerically. All acceptance decisions agree."))
        figure = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
            {"P-Flow": (results["proxy_only"]["image"], results["proxy_only"]["measurement"]),
             "DLR D-Flow": (results["proxy_then_dlr"]["image"], results["proxy_then_dlr"]["measurement"])},
            observation=inputs["observed"], measurement_log=parent["protocol"]["job"]["problem"] == "mri",
            residual_vmax=RESIDUAL_VMAX[parent["protocol"]["job"]["problem"]], include_errors=True,
            title=f"{folder.name}: proxy-only and one-proxy-then-DLR continuations\nD-Flow continuation unavailable; both proxy searches rejected")
        figure.savefig(folder/"comparison.png", dpi=100, bbox_inches="tight")
        plt.close(figure)
    evidence = dict(rows=rows, repeated_trials=repeated_trials, checks=["parent checkpoint and source hashes", "paired protocols and initial objectives",
        "unchanged optimizer, rank, projection and attempt budget", "monotone saved objectives and displacement caps",
        "complex residual and image metrics from original arrays in FP64", "bounded proxy trials and repeated decrease acceptance",
        "matched first proxy baseline, trial count, step lengths and acceptance decisions", "rejected sources unchanged and empty hybrid optimizer state"],
        limitation="No proxy was accepted in these pilots; the accepted hybrid resume path is not empirically exercised here.")
    (args.root/"proxy_recovery_verification.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()

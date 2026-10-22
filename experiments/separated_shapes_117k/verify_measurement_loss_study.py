"""Independent saved-array checks for completed measurement-loss arms."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.ct_mri_sweep.run import write_json
from experiments.separated_shapes_117k.measurement_losses import measurement_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads((args.root / "config.json").read_text())
    inputs = torch.load(Path("outputs/separated_shapes_117k_noiseless_20261021/production") /
                        config["case"] / "inputs.pt", map_location="cpu", weights_only=True)
    initial_hash = hashlib.sha256(inputs["initial"].numpy().tobytes()).hexdigest()
    assert initial_hash == config["source_sha256"]
    problem = inputs["protocol"]["job"]["problem"]
    observed = inputs["observed"].to(torch.complex128 if problem == "mri" else torch.float64)
    truth = inputs["truth"].double()
    cap = config["settings"]["optimizer"]["step_cap"] * math.sqrt(truth.numel())
    results = []
    for summary_path in sorted(args.root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        source_hash, total, prior_image = initial_hash, 0, None
        stages = sorted(summary_path.parent.glob("stage_*.pt"))
        for stage, path in enumerate(stages):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            experiment = saved["protocol"]["experiment"]
            assert all(saved["protocol"][key] == value for key, value in inputs["protocol"].items())
            assert experiment["source_files"] == config["source_files"]
            assert experiment["starting_source_sha256"] == source_hash
            assert experiment["stage"] == stage
            settings = experiment["settings"]
            assert settings["dlr"]["rank"] == 32 and settings["flow_steps"] == 5
            assert settings["dlr"]["trace_probes"] == settings["dlr"]["enrichment_probes"] == 0
            history = saved["history"]
            assert history[-1]["step"] == saved["optimizer"]["step_attempts"] <= settings["stopping"]["max_updates"]
            assert all(b["phi"] <= a["phi"] for a, b in zip(history, history[1:]))
            assert all(row.get("source_step_norm", 0) <= cap * (1 + 1e-4) for row in history)
            if prior_image is not None:
                assert abs(history[0]["image_relative_l2"] - prior_image) < 1e-6
            image_error = float((saved["image"].double() - truth).norm() / truth.norm())
            residual = saved["measurement"].to(observed.dtype) - observed
            residual_error = float(residual.norm() / observed.norm())
            weighted = float(measurement_loss(residual, problem, **settings["measurement_loss"]))
            assert abs(image_error - history[-1]["image_relative_l2"]) < 1e-6
            assert abs(residual_error - history[-1]["measurement_relative_l2_noisy"]) < 1e-6
            assert math.isclose(weighted, history[-1]["phi"], rel_tol=2e-5, abs_tol=1e-8)
            source_hash = hashlib.sha256(saved["source"].numpy().tobytes()).hexdigest()
            prior_image = history[-1]["image_relative_l2"]
            total += history[-1]["step"]
        assert total == summary["attempts"] <= config["settings"]["stopping"]["max_updates"]
        if experiment["variant"] in ("coarse_to_fine", "l2_resets"):
            assert len(stages) == 4 and settings["measurement_loss"]["sigma"] == 0
        else:
            assert len(stages) == 1
        results.append(dict(arm=str(summary_path.parent.relative_to(args.root)), passed=True,
                            stages=len(stages), attempts=total, image_error_percent=100*image_error,
                            residual_percent=100*residual_error))
    write_json(args.root / "verification.json", dict(completed_arms=results,
        note="Source handoffs, budgets, per-stage descent, caps, weighted loss, complex residuals and image errors verified. No cross-stage monotonicity assumption."))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

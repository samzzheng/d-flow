"""Verify paired terminal-weighting controls and render original-array results."""
import argparse
from copy import deepcopy
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
    queue = json.loads((args.root/"queue.json").read_text())
    case = Path(queue["case"])
    baseline = Path("outputs/separated_shapes_117k_noiseless_20261021/production")
    inputs = torch.load(baseline/case/"inputs.pt", map_location="cpu", weights_only=True)
    source_hash = hashlib.sha256(inputs["initial"].numpy().tobytes()).hexdigest()
    references = {}
    for method, name in (("D-Flow", "d_flow"), ("P-Flow", "p_flow")):
        reference = torch.load(baseline/case/f"{name}.pt", map_location="cpu", weights_only=True)
        assert all(reference["protocol"][key] == value for key, value in inputs["protocol"].items())
        references[method] = (reference["image"], reference["measurement"])
    comparable = None
    rows, states, gradients = [], {}, {}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for variant in ("unchanged", "median_cap", "norm_matched_cap"):
        folder = args.root/variant/case/"rank32"
        if not (folder/"config.json").exists():
            rows.append(dict(variant=variant, status="pending"))
            continue
        config = json.loads((folder/"config.json").read_text())
        assert config["source_sha256"] == source_hash
        assert config["script_sha256"] == queue["script_hashes"][str(Path(__file__).with_name("high_rank_reconstruction.py"))]
        assert config["weighting_script_sha256"] == queue["script_hashes"][str(Path(__file__).with_name("spectral_weighting.py"))]
        reduced = deepcopy(config)
        assert reduced["settings"].pop("terminal_weighting") == variant
        if comparable is None:
            comparable = reduced
        assert comparable == reduced
        if (folder/"first_gradient.pt").exists():
            first = torch.load(folder/"first_gradient.pt", map_location="cpu", weights_only=True)
            assert torch.equal(first["source"], inputs["initial"])
            gradients[variant] = first["gradient"].double().flatten()
        if not (folder/"dlr_d_flow.pt").exists():
            rows.append(dict(variant=variant, status="running_no_checkpoint"))
            continue
        saved = torch.load(folder/"dlr_d_flow.pt", map_location="cpu", weights_only=True)
        assert saved["protocol"]["experiment"] == config
        assert all(saved["protocol"][key] == value for key, value in inputs["protocol"].items())
        history = saved["history"]
        assert history[-1]["step"] <= 1000
        assert all(b["phi"] <= a["phi"] for a, b in zip(history, history[1:]))
        assert all(row.get("source_step_norm", 0) <= 2.5*(1+1e-4) for row in history)
        truth, measurement = inputs["truth"].double(), saved["measurement"].to(torch.complex128)
        residual = measurement-inputs["observed"].to(torch.complex128)
        image_error = float((saved["image"].double()-truth).norm()/truth.norm())
        residual_error = float(residual.norm()/inputs["observed"].to(torch.complex128).norm())
        assert abs(image_error-history[-1]["image_relative_l2"]) < 1e-6
        assert abs(residual_error-history[-1]["measurement_relative_l2_noisy"]) < 1e-6
        assert math.isclose(float(.5*residual.abs().square().sum()), history[-1]["phi"], rel_tol=2e-6)
        rows.append(dict(variant=variant, status=saved["status"], attempts=history[-1]["step"],
            image_error_percent=100*image_error, residual_percent=100*residual_error,
            best_image_error_percent=100*min(row["image_relative_l2"] for row in history)))
        states[variant] = saved
        axes[0].plot([row["step"] for row in history], [100*row["image_relative_l2"] for row in history], label=variant)
        axes[1].plot([row["step"] for row in history], [row["phi"] for row in history], label=variant)
        if saved["status"] in ("stalled", "step_limit", "objective_plateau"):
            weighting = json.loads((folder/"terminal_weighting.json").read_text())
            assert weighting["variant"] == variant
            if variant != "unchanged":
                assert len(weighting["rows"]) == saved["evaluations"]["gradients"]
                for row in weighting["rows"]:
                    assert row["capped_gradient_norm"] <= row["original_gradient_norm"]*(1+1e-5)
                    if variant == "norm_matched_cap" and row["capped_gradient_norm"]:
                        assert math.isclose(row["capped_gradient_norm"]*row["applied_scale"], row["original_gradient_norm"], rel_tol=1e-5)
            panel = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
                {**references, "DLR D-Flow": (saved["image"], saved["measurement"])},
                observation=inputs["observed"], measurement_log=True, include_errors=True,
                residual_vmax=RESIDUAL_VMAX["mri"], title=f"{case}: {variant}\nNoiseless, 5 Euler steps, rank32; D/P are existing paired references")
            panel.savefig(folder/"paired_methods.png", dpi=100, bbox_inches="tight")
            plt.close(panel)
    for axis, label in zip(axes, ("Relative image error (%)", "Measurement objective")):
        axis.set(xlabel="Optimizer attempts", ylabel=label)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.savefig(args.root/"convergence.png", dpi=140)
    plt.close(figure)
    comparisons = {}
    if "unchanged" in gradients:
        ordinary = gradients["unchanged"]
        comparisons = {name: dict(norm_ratio=float(gradient.norm()/ordinary.norm()),
            cosine=float(gradient.dot(ordinary)/(gradient.norm()*ordinary.norm())))
            for name, gradient in gradients.items()}
    old_path = args.root.parent/"deterministic_mri2/reconstruction/deterministic_a"/case/"rank32/dlr_d_flow.pt"
    old_match = None
    if "unchanged" in states and old_path.exists():
        old = torch.load(old_path, map_location="cpu", weights_only=True)
        new = states["unchanged"]
        fields = ("step", "phi", "source_step_norm", "line_search_trials", "image_relative_l2", "measurement_relative_l2_noisy")
        old_match = dict(common_records=min(len(old["history"]), len(new["history"])),
            matching_common_history=all(all(a.get(key) == b.get(key) for key in fields)
                                       for a, b in zip(old["history"], new["history"])),
            same_final_attempt=old["history"][-1]["step"] == new["history"][-1]["step"],
            final_source_bitwise_equal=torch.equal(old["source"], new["source"]))
    report = dict(rows=rows, first_gradient_comparison=comparisons, prior_deterministic_control=old_match,
        checks=["paired source, sketch, model, acquisition and mathematical settings except weighting",
                "recorded implementation hashes", "original attempt budget and caps", "monotone accepted objective",
                "complex residual and image metrics from original arrays", "capped norm and norm-matching algebra"],
        note="Best image error is retrospective. Partial checkpoints do not establish convergence; one selected sample is not population evidence.")
    (args.root/"verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Verify paired execution controls and compare deterministic trajectory repeats."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    case = Path("r250_shapes_mri4/sample_002/rank32")
    inputs = torch.load(Path("outputs/separated_shapes_117k_noiseless_20261021/production")/case.parent/"inputs.pt",
                        map_location="cpu", weights_only=True)
    source_hash = hashlib.sha256(inputs["initial"].numpy().tobytes()).hexdigest()
    truth = inputs["truth"].double()
    observed = inputs["observed"].to(torch.complex128)
    references = {}
    for method, filename in (("D-Flow", "d_flow.pt"), ("P-Flow", "p_flow.pt")):
        saved = torch.load(Path("outputs/separated_shapes_117k_noiseless_20261021/production")/case.parent/filename,
                           map_location="cpu", weights_only=True)
        for key in ("checkpoint_sha256", "operator", "source_seed", "noise_seed"):
            assert saved["protocol"][key] == inputs["protocol"][key]
        references[method] = (saved["image"], saved["measurement"])
    baseline = None
    rows, states = [], {}
    for label in ("default", "deterministic_a", "deterministic_b"):
        folder = args.root/"reconstruction"/label/case
        if not (folder/"config.json").exists():
            rows.append(dict(label=label, status="pending"))
            continue
        config = json.loads((folder/"config.json").read_text())
        assert config["source_sha256"] == source_hash
        comparable = deepcopy(config)
        flag = comparable["settings"]["execution"].pop("cudnn_deterministic")
        assert flag == label.startswith("deterministic")
        if baseline is None:
            baseline = comparable
        assert comparable == baseline
        if not (folder/"dlr_d_flow.pt").exists():
            rows.append(dict(label=label, status="no_optimizer_checkpoint_yet"))
            continue
        saved = torch.load(folder/"dlr_d_flow.pt", map_location="cpu", weights_only=True)
        assert saved["protocol"]["experiment"] == config
        for key in ("checkpoint_sha256", "operator", "source_seed", "noise_seed", "dlr_seed"):
            assert saved["protocol"][key] == inputs["protocol"][key]
        history = saved["history"]
        assert history[-1]["step"] <= 1000
        assert all(right["phi"] <= left["phi"] for left, right in zip(history, history[1:]))
        assert all(row.get("source_step_norm", 0) <= 2.5*(1+1e-4) for row in history)
        image_error = float((saved["image"].double()-truth).norm()/truth.norm())
        residual = float((saved["measurement"].to(torch.complex128)-observed).norm()/observed.norm())
        phi = float(.5*(saved["measurement"].to(torch.complex128)-observed).abs().square().sum())
        assert abs(image_error-history[-1]["image_relative_l2"]) < 1e-6
        assert abs(residual-history[-1]["measurement_relative_l2_noisy"]) < 1e-6
        assert math.isclose(phi, history[-1]["phi"], rel_tol=2e-6)
        first = torch.load(folder/"first_gradient.pt", map_location="cpu", weights_only=True)
        assert torch.equal(first["source"], inputs["initial"])
        rows.append(dict(label=label, status=saved["status"], steps=history[-1]["step"],
            image_error_percent=100*image_error, residual_percent=100*residual,
            first_gradient_sha256=hashlib.sha256(first["gradient"].numpy().tobytes()).hexdigest()))
        states[label] = saved
        if saved["status"] in ("stalled", "step_limit"):
            figure = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
                {**references, "DLR D-Flow": (saved["image"], saved["measurement"])},
                observation=inputs["observed"], measurement_log=True,
                residual_vmax=RESIDUAL_VMAX["mri"], include_errors=True,
                title=f"MRI250 sample2: {label}; rank32, noiseless, 5 Euler steps\nPaired original sources; D/P are the existing reference runs")
            figure.savefig(folder/"paired_methods.png", dpi=100, bbox_inches="tight")
            plt.close(figure)
    repeat = None
    if {"deterministic_a", "deterministic_b"} <= states.keys():
        first, second = states["deterministic_a"], states["deterministic_b"]
        length = min(len(first["history"]), len(second["history"]))
        fields = ("step", "phi", "source_step_norm", "line_search_trials", "image_relative_l2", "measurement_relative_l2_noisy")
        differences = [index for index, (left, right) in enumerate(zip(first["history"], second["history"]))
                       if any(left.get(key) != right.get(key) for key in fields)]
        repeat = dict(common_history_records=length, first_different_record=differences[0] if differences else None,
            matching_non_timing_history=not differences,
            same_final_attempt=first["history"][-1]["step"] == second["history"][-1]["step"],
            source_bitwise_equal=torch.equal(first["source"], second["source"]))
    report = dict(rows=rows, deterministic_repeat=repeat,
        checks=["source/model/operator/seeds and sketch paired", "only cuDNN determinism differs between configurations",
                "original budget, caps and monotone objectives", "image and complex residual metrics from original arrays",
                "common non-timing histories compared when both deterministic checkpoints exist"],
        note="Partial checkpoints are not convergence results. Matching initial gradients does not alone establish matching full trajectories.")
    (args.root/"reconstruction_verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

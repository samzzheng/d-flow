"""Check completed paired high-rank reconstructions from original saved arrays."""
import argparse
from copy import deepcopy
from datetime import datetime
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
    parser.add_argument("--case", default="r250_shapes_mri4/sample_001")
    parser.add_argument("--baseline", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--extra-reconstruction-directory")
    parser.add_argument("--include-live", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    baseline = args.baseline/args.case
    inputs = torch.load(baseline/"inputs.pt", map_location="cpu", weights_only=True)
    source_hash = hashlib.sha256(inputs["initial"].numpy().tobytes()).hexdigest()
    truth = inputs["truth"].double()
    dtype = torch.complex128 if inputs["observed"].is_complex() else torch.float64
    observed = inputs["observed"].to(dtype)
    rows = []
    references = {}
    settings_reference = None
    sketch_reference = None
    paths = [baseline/"d_flow.pt", baseline/"p_flow.pt"]
    paths += sorted((args.root/"reconstruction"/args.case).glob("rank*/dlr_d_flow.pt"),
                    key=lambda path: int(path.parent.name.removeprefix("rank")))
    if args.extra_reconstruction_directory:
        paths += sorted((args.root/args.extra_reconstruction_directory/args.case).glob("rank*/dlr_d_flow.pt"),
                        key=lambda path: int(path.parent.name.removeprefix("rank")))
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        complete = saved["status"] in ("stalled", "step_limit", "objective_plateau")
        if not complete and not args.include_live:
            continue
        protocol = saved["protocol"]
        for key in ("checkpoint_sha256", "operator", "source_seed", "dlr_seed", "noise_seed"):
            assert protocol[key] == inputs["protocol"][key]
        rank = None
        memory_fraction = None
        if path.parent.name.startswith("rank"):
            experiment = protocol["experiment"]
            assert experiment["source_sha256"] == source_hash
            settings = deepcopy(experiment["settings"])
            rank = settings["dlr"].pop("rank")
            memory_fraction = settings["execution"].pop("memory_fraction")
            assert saved["previous_v"].shape == (inputs["initial"].numel(), rank)
            if settings_reference is None:
                settings_reference, sketch_reference = settings, experiment["sketch_sha256"]
            assert settings == settings_reference
            if "original_sketch_sha256" in experiment:
                assert experiment["original_sketch_sha256"] == sketch_reference
                failed_config = json.loads((args.root/"reconstruction"/args.case/f"rank{rank}/config.json").read_text())
                assert experiment["extended_sketch_sha256"] == failed_config["extended_sketch_sha256"]
                assert experiment["original_sketch_sha256"] == failed_config["original_sketch_sha256"]
                assert experiment == json.loads((path.parent/"config.json").read_text())
            else:
                assert experiment["sketch_sha256"] == sketch_reference
            assert settings["flow_steps"] == 5
            assert settings["dlr"]["trace_probes"] == settings["dlr"]["enrichment_probes"] == 0
        history = saved["history"]
        assert all(right["phi"] <= left["phi"] for left, right in zip(history, history[1:]))
        cap = .01*math.sqrt(inputs["initial"].numel())
        assert all(row.get("source_step_norm", 0) <= cap*(1+1e-4) for row in history)
        assert history[-1]["step"] <= 1000
        image_error = float((saved["image"].double()-truth).norm()/truth.norm())
        residual = float((saved["measurement"].to(dtype)-observed).norm()/observed.norm())
        phi = float(.5*(saved["measurement"].to(dtype)-observed).abs().square().sum())
        assert abs(image_error-history[-1]["image_relative_l2"]) < 1e-6
        assert abs(residual-history[-1]["measurement_relative_l2_noisy"]) < 1e-6
        assert math.isclose(phi, history[-1]["phi"], rel_tol=2e-6)
        rows.append(dict(method=saved["method"], rank=rank, status=saved["status"],
            complete=complete, memory_fraction=memory_fraction, checkpoint=str(path),
            steps=history[-1]["step"], image_error_percent=100*image_error,
            residual_percent=100*residual, phi=phi,
            image_norm_over_truth=float(saved["image"].double().norm()/truth.norm()),
            exact_zero_image_objective=float(.5*observed.abs().square().sum()),
            elapsed_seconds=saved["elapsed_seconds"]))
        arrays = (saved["image"], saved["measurement"])
        if rank is None:
            references[saved["method"]] = arrays
        elif complete:
            figure = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
                {**references, "DLR D-Flow": arrays}, observation=inputs["observed"],
                measurement_log=protocol["job"]["problem"] == "mri",
                residual_vmax=RESIDUAL_VMAX[protocol["job"]["problem"]], include_errors=True,
                title=f"{args.case}: fixed rank {rank}; paired original source, noiseless, 5 Euler steps")
            figure.savefig(path.parent/"paired_methods.png", dpi=100, bbox_inches="tight")
            plt.close(figure)
        del saved
    evidence = dict(checked_at=datetime.now().astimezone().isoformat(), rows=rows,
        checks=["same original source hash, model, operator and seeds",
                "identical mathematical DLR settings except rank; memory fraction recorded separately",
                "shared original nested-sketch hash; extended sketch agrees across allocation retry",
                "saved basis dimensions", "monotone objectives, caps and original attempt budget",
                "original-array FP64 image and complex measurement metrics"],
        note="Rows explicitly mark completed endpoints versus live checkpoints. Unchanged stopping rules do not imply equal accepted-step counts. D/P use the same original source and generator but no DLR sketch. No convergence claim follows from a live checkpoint.")
    output = "live_reconstruction_verification.json" if args.include_live else "reconstruction_verification.json"
    (args.root/output).write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()

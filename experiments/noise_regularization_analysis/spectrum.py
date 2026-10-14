"""Bounded native-Euler spectral checks at the saved CT discrepancy sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model, make_terminal_flow_products
from experiments.multicircle_rank_hybrid.ranks import spectrum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/noise_regularization_analysis_20261014/fixed_points/spectra"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model_path = Path("outputs/fm_unet/circles-64-noattn-ch16-b8-e200/n1")
    model, _ = load_flow_model(64, 1, 200, "cpu", model_path)
    directory = Path("outputs/five_step_noise005_n124_20261013/production/r64_n1_ct10/sample_015")
    inputs = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True)
    checkpoint_hash = hashlib.sha256((model_path / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest()
    if inputs["protocol"]["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Model checkpoint mismatch")
    lines = ["# Native Jacobian spectral diagnostic", "",
             "CT10, 5% noise, 64² single-circle sample 15; Jacobian of the actual five-step Euler map. CPU only.", "",
             "Randomized width 32, at most three refinements, 16 independent omitted-energy probes. Energy ranks refer to cumulative squared singular values, not algebraic rank. The minimum singular value and full condition number are not measured.", "",
             "| Source | Largest singular value | 99.9% energy rank estimate | 99.99% energy rank estimate | Omitted-energy fraction estimate | Relevant relative triplet residual |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, source in [
        ("initial", inputs["initial"]),
        ("d_flow_final", torch.load(directory / "d_flow.pt", map_location="cpu", weights_only=True)["source"]),
        ("dlr_final", torch.load(directory / "dlr_d_flow.pt", map_location="cpu", weights_only=True)["source"]),
    ]:
        destination = args.out / f"{label}.json"
        result = None
        if destination.exists():
            result = json.loads(destination.read_text())
            if result["source_sha256"] != hashlib.sha256(source.numpy().tobytes()).hexdigest():
                raise ValueError("Saved spectral source mismatch")
        if result is None or result["status"] == "running":
            native_j, native_jt = make_terminal_flow_products(model, source, 5)
            result = spectrum(native_j, native_jt, source, destination, [32], 3, 16,
                              inputs["protocol"]["dlr_seed"] + 82000000,
                              {"device": "cpu", "torch_threads": 4, "flow_steps": 5,
                               "source_kind": label, "source_directory": str(directory),
                               "checkpoint_sha256": checkpoint_hash,
                               "source_sha256": hashlib.sha256(source.numpy().tobytes()).hexdigest(),
                               "limitation": "Bounded randomized native full-map estimate; unresolved tails or cutoffs are not exact rank/condition-number estimates"})
        latest = result["latest"]
        ranks = latest["effective_ranks"]
        lines.append(f"| {label} | {latest['singular_values'][0]:.3f} | {ranks['0.999'] or 'unresolved'} | {ranks['0.9999'] or 'unresolved'} | {latest['omitted_energy_fraction']:.3g} | {latest['relevant_triplet_residual']:.3g} |")
    lines.extend(["", "In this selected failure case, the largest singular value is much larger at the failed D-Flow source even though its energy-defined effective rank is smaller. This is a cross-source association, not proof that sensitivity growth caused failure. The comparison does not establish the rank of other datasets or a universal conditioning result."])
    (args.out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

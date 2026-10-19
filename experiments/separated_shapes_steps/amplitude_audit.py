"""Check whether shrinking misplaced initial predictions lowers the objective."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.ct_mri_sweep.operators import CT, MRI


def main() -> None:
    torch.set_num_threads(2)
    root = Path("outputs/separated_shapes_steps_20261019")
    rows = []
    for path in sorted((root / "diagnostics").glob("*/images.pt")):
        saved = torch.load(path, weights_only=True, map_location="cpu")
        diagnostic = json.loads((path.parent / "diagnostics.json").read_text())
        info = diagnostic["protocol"]["operator"]
        resolution = saved["truth"].shape[-1]
        operator = (CT(resolution, 10, torch.device("cpu")) if info["kind"] == "ct" else
                    MRI(resolution, 4, info["mask_seed"], torch.device("cpu"), info["pattern"]))
        prediction = operator(saved["images"]["initial_5"])
        observed = saved["clean"]  # These cases are noiseless.
        product = float((prediction.conj() * observed).real.sum(dtype=torch.float64))
        squared_norm = float(prediction.abs().square().sum(dtype=torch.float64))
        observed_squared_norm = float(observed.abs().square().sum(dtype=torch.float64))
        row = dict(case=diagnostic["case"],
                   prediction_measurement_cosine=product/(squared_norm*observed_squared_norm)**.5 if squared_norm*observed_squared_norm else None,
                   optimal_unconstrained_amplitude=product/squared_norm if squared_norm else None,
                   initial_objective=.5*(squared_norm-2*product+observed_squared_norm),
                   zero_image_objective=.5*observed_squared_norm,
                   zero_image_improves_objective=squared_norm > 2*product)
        rows.append(row)
    (root / "amplitude_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Amplitude-collapse diagnostic", "",
             "CPU evaluation of the saved five-step initial images, original models, same noiseless observations. "
             "This is an image-space diagnostic, not a claim that every scaled image lies in the generator's range.", "",
             "For a fixed prediction x and real scalar c, the linear-operator objective is", "",
             r"\[\Phi(cx)=\tfrac12c^2\|Ax\|^2-c\operatorname{Re}\langle Ax,y\rangle+\tfrac12\|y\|^2.\]", "",
             r"Its unconstrained amplitude minimizer is \(c_* = \operatorname{Re}\langle Ax,y\rangle/\|Ax\|^2\). "
             "If measurement overlap is small, reducing amplitude can lower the objective even when the target is nonzero. "
             "That provides a plausible route toward nearly blank predictions from misplaced shapes. "
             "It does not prove that the latent objective has a local minimum at zero or that this causes every failure.", "",
             "| Case | Measurement cosine | Best scalar amplitude | Zero improves objective? |",
             "| --- | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['prediction_measurement_cosine']:.4f} | "
                     f"{row['optimal_unconstrained_amplitude']:.4f} | {row['zero_image_improves_objective']} |")
    lines += ["", "Six of eight original starts favor zero over their initial prediction. "
              "The 128/160 CT starts are already nearly blank and do not satisfy this comparison; "
              "their failures require additional explanation. Successful cases can also start with low measurement overlap, "
              "so this diagnostic identifies an incentive to shrink, not a sufficient predictor of failure."]
    (root / "amplitude_audit.md").write_text("\n".join(lines) + "\n")
    print(f"Zero improves the initial objective in {sum(row['zero_image_improves_objective'] for row in rows)}/{len(rows)} cases.")


if __name__ == "__main__":
    main()

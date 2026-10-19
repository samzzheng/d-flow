"""Audit pairing and summarize the completed separated-shapes controls on CPU."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.separated_shapes_steps.run import CASES, DEFAULT, STEMS


def main() -> None:
    torch.set_num_threads(2)
    baseline = Path("outputs/separated_shapes_noiseless_20261016/production")
    rank32 = Path("outputs/separated_shapes_rank_sweep_20261017/rank_32/production")
    records = []
    audits = []
    for resolution, problem, sample in CASES:
        name = f"r{resolution}_shapes_{problem}/sample_{sample:03d}"
        reference = torch.load(baseline / name / "inputs.pt", weights_only=True, map_location="cpu", mmap=True)
        for steps in (5, 10, 20):
            path = rank32 if steps == 5 else DEFAULT / f"steps_{steps}_rank_32/production"
            inputs = torch.load(path / name / "inputs.pt", weights_only=True, map_location="cpu", mmap=True)
            for key in ("truth", "initial", "clean", "observed", "mask"):
                left, right = reference[key], inputs[key]
                assert (left is None and right is None) or torch.equal(left, right), (name, steps, key)
            for key in ("checkpoint_sha256", "test_subset_sha256", "source_seed", "noise_seed", "dlr_seed", "operator"):
                assert reference["protocol"][key] == inputs["protocol"][key], (name, steps, key)
            assert torch.equal(inputs["clean"], inputs["observed"])
            audits.append(dict(case=name, steps=steps, bitwise_inputs_paired=True, noiseless=True))
            for method in STEMS:
                method_path = baseline if steps == 5 and method != "dlr_d_flow" else path
                result_path = method_path / name / f"{method}.json"
                result = json.loads(result_path.read_text())
                assert result["status"] in {"step_limit", "stalled", "converged"}, result_path
                last = result["history"][-1]
                records.append(dict(case=name, steps=steps, method=result["method"],
                                    error_percent=100*last["image_relative_l2"],
                                    measurement_percent=100*last["measurement_relative_l2_clean"],
                                    attempts=last["step"], path=str(result_path)))
    lines = ["# Separated-shapes collapse investigation", "",
             "Five Euler steps affect the generator substantially, but increasing to 10 or 20 did not consistently resolve collapse. "
             "A 100-step exact D-Flow control also failed. No consistently accurate, fast DLR configuration has yet been established.", "",
             "Eight deliberately selected cases across six resolutions; original epoch-200 models, noiseless CT10/MRI4x, "
             "paired seeds, L-BFGS history 100 and geometric Armijo, 0.01 sqrt(d) displacement cap, up to 1000 attempts. "
             "DLR rank 32 throughout this table, analytical alpha=1-t, zero enrichment, previous-V reuse. "
             "All 24 input bundles were checked bitwise against the five-step baseline, including truths, sources, "
             "observations and MRI masks; checkpoint hashes and acquisition metadata match. "
             "This selected-case pilot is not a population success-rate estimate.", "",
             "| Case | Euler steps | D-Flow error % | P-Flow error % | DLR32 error % |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for resolution, problem, sample in CASES:
        name = f"r{resolution}_shapes_{problem}/sample_{sample:03d}"
        for steps in (5, 10, 20):
            values = [row["error_percent"] for row in records if row["case"] == name and row["steps"] == steps]
            lines.append(f"| {resolution} {problem} sample {sample} | {steps} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "## Additional controls", "",
              "| Control | Case | D-Flow % | P-Flow % | DLR32 % |",
              "| --- | --- | ---: | ---: | ---: |"]
    controls = [
        ("L-BFGS history 5", "optimizer_controls/lbfgs_h5", "r64_shapes_mri4/sample_001"),
        ("L-BFGS history 5", "optimizer_controls/lbfgs_h5", "r128_shapes_ct10/sample_000"),
        ("Tenfold smaller displacement cap", "cap_0001_steps_5_rank_32", "r250_shapes_mri4/sample_000"),
        ("Small model, epoch 1000", "tiny_model_steps_5_rank_32", "r64_shapes_mri4/sample_001"),
        ("Terminal singular values capped at median", "singular_cap_steps_5_rank_32", "r64_shapes_mri4/sample_001"),
        ("100 Euler steps, checkpointed exact gradient", "exact_reference_checkpointed_steps_100", "r250_shapes_mri4/sample_000"),
    ]
    for label, folder, name in controls:
        values = []
        for stem in STEMS:
            path = DEFAULT / folder / "production" / name / f"{stem}.json"
            values.append(f"{100*json.loads(path.read_text())['history'][-1]['image_relative_l2']:.3f}" if path.exists() else "not run")
        lines.append(f"| {label} | {name} | " + " | ".join(values) + " |")
    lines += ["", "### Small-model gradient-origin controls", "",
              "Same 64 MRI sample 1, five Euler steps and rank 32. Exact gradients are computed for diagnostics; "
              "these runs are not speed measurements. Projected-exact is a diagnostic surrogate, not standard DLR.", "",
              "| Gradient | Image error % | Attempts |",
              "| --- | ---: | ---: |"]
    for variant, label in (("baseline", "DLR with previous-V reuse"),
                           ("fresh_basis", "DLR with fresh sketch"),
                           ("projected_exact", "Exact gradient projected onto DLR V")):
        path = DEFAULT / f"gradient_origin_{variant}/production/r64_shapes_mri4/sample_001/dlr_d_flow.json"
        if path.exists():
            last = json.loads(path.read_text())["history"][-1]
            lines.append(f"| {label} | {100*last['image_relative_l2']:.3f} | {last['step']} |")
    lines += ["", "Initially, the gradient component outside DLR's right subspace has norm equal to 66.8% of the full exact gradient norm; "
              "the coefficient error within the subspace is 8.6% of the exact gradient norm. "
              "Replacing those coefficients with exact ones does not rescue this case. "
              "This rules out coefficient error alone as a sufficient explanation under this optimizer; "
              "it does not prove that subspace truncation is the sole cause. "
              "The instrumented baseline also collapses, but its termination iteration differs from the earlier run "
              "(78 versus 93); no bitwise trajectory equivalence is claimed.", "",
              "The small-model result changes both architecture and training duration. It shows that five steps can support "
              "an accurate reconstruction in this case, but does not establish general reliability or resolve DLR failure. "
              "The uncheckpointed 100-step run ran out of allocated memory before optimization; only its completed "
              "checkpointed rerun is reported above.", "",
              "## Evidence and interpretation", "",
              "- Fixed-source replays reproduce all saved five-step final images exactly. Several failed images are nearly blank. "
              "Refining the grid restores image amplitude at those same sources without necessarily recovering the target shapes.",
              "- The exact Euler Jacobian is the ordered product of I+h*A_k. Changing the time grid changes both the generator "
              "and its inverse objective; it is not merely a more accurate gradient of the same objective.",
              "- A successful P-Flow reconstruction on the same five-step generator proves that a good solution is representable "
              "for that sample. D-Flow/DLR failures on it therefore cannot be attributed solely to insufficient generator expressivity.",
              "- Fixed-point gradient errors are source-dependent. At the initial 64 MRI source, DLR32 error falls from 53.3% "
              "to 4.7% when increasing from five to twenty steps. At 128 CT, errors are 12.4%, 63.9%, 43.2% for 5/10/20. "
              "Neither high rank nor more steps guarantees a useful optimization trajectory.",
              "- At one failed 128 CT source, refined FP64 finite differences verify the derivative while revealing a very "
              "narrow descent neighborhood: displacement 1e-4 along normalized descent raises the loss, and 1e-5 lowers it "
              "by only about 3.6e-7. This supports sharp local curvature and precision sensitivity, not a universal derivative bug.",
              "- The sampled rank-pilot truths cover only one- and three-object shapes. Broader validation must include two "
              "and four objects and independent sources before calling any configuration consistent.", "",
              "Detailed evidence: [fixed-source audit](diagnostics/report.md), [gradient audit](gradients/report.md), "
              "[Euler reconstructions](report.md), [short-history controls](optimizer_controls/report.md). "
              "Further diagnostic controls separate coefficient error from subspace loss on the small-model case. "
              "Enrichment and GD are excluded at the user's request. GPU 0 is off limits for subsequent work."]
    (DEFAULT / "investigation_report.md").write_text("\n".join(lines) + "\n")
    (DEFAULT / "paired_audit.json").write_text(json.dumps(dict(audits=audits, records=records), indent=2) + "\n")
    print(f"Verified {len(audits)} paired inputs and summarized {len(records)} terminal results.")


if __name__ == "__main__":
    main()

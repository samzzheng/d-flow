"""Summarize and verify the measurement-informed source controls on CPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    torch.set_num_threads(2)
    root = Path("outputs/separated_shapes_steps_20261019")
    rows = []
    lines = ["# Measurement-informed initialization", "",
             "Five Euler steps, analytical alpha=1-t, zero enrichment, unchanged L-BFGS/Armijo and rank 32 unless stated. "
             "Only the starting source changes. Let q be A*y after subtracting its spatial mean and normalizing its norm to sqrt(d). "
             "Initialize z = sqrt(1-rho^2) z_random + rho q. This uses the observation and acquisition operator, not the ground-truth image. "
             "It changes the source distribution; it is not an unconditional standard-Gaussian draw. "
             "Corrected runs cache one conditioned source for all three methods. The stronger mix was selected using the small-model "
             "64 MRI sample 1 result, so that case is tuning evidence. Timing has not been benchmarked in isolation.", "",
             "| Model | Mix | Rank | Case | D-Flow % | P-Flow % | DLR % |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for manifest_path in sorted(root.glob("measurement_init_*/production/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        settings = manifest["settings"]
        model = "small epoch 1000" if settings["epoch"] == 1000 else "original epoch 200"
        for inputs_path in sorted(manifest_path.parent.glob("*/sample_*/inputs.pt")):
            inputs = torch.load(inputs_path, weights_only=True, map_location="cpu")
            if "initialization_direction" not in inputs:
                continue
            rho = settings["initialization"]["mix"]
            expected = (1-rho**2)**.5 * inputs["unconditioned_initial"] + rho * inputs["initialization_direction"]
            assert torch.equal(expected, inputs["initial"]), inputs_path
            source_hash = hashlib.sha256(inputs["initial"].numpy().tobytes()).hexdigest()
            case = f"{inputs_path.parent.parent.name}/{inputs_path.parent.name}"
            reference_root = (root / "tiny_model_steps_5_rank_32/production" if settings["epoch"] == 1000 else
                              Path("outputs/separated_shapes_noiseless_20261016/production"))
            reference_path = reference_root / case / "inputs.pt"
            if reference_path.exists():
                reference = torch.load(reference_path, weights_only=True, map_location="cpu")
                for key in ("truth", "clean", "observed", "mask"):
                    assert (inputs[key] is None and reference[key] is None) or torch.equal(inputs[key], reference[key]), (case, key)
                assert torch.equal(inputs["unconditioned_initial"], reference["initial"]), case
                for key in ("checkpoint_sha256", "source_seed", "noise_seed", "dlr_seed", "operator"):
                    assert inputs["protocol"][key] == reference["protocol"][key], (case, key)
            values = []
            source_hashes = {json.loads(path.read_text())["protocol"]["initial_source_sha256"]
                             for path in inputs_path.parent.glob("*.json")
                             if path.stem in {"d_flow", "p_flow", "dlr_d_flow"}}
            paired = source_hashes == {source_hash}
            for method in ("d_flow", "p_flow", "dlr_d_flow"):
                path = inputs_path.with_name(f"{method}.json")
                if not path.exists():
                    values.append("not run yet")
                    continue
                result = json.loads(path.read_text())
                last = result["history"][-1]
                error = 100*last["image_relative_l2"]
                terminal = result["status"] in {"step_limit", "stalled", "converged"}
                values.append(("unpaired: " if not paired else "") + f"{error:.3f}" + (" (incomplete)" if not terminal else ""))
                rows.append(dict(model=model, mix=rho, rank=settings["dlr"]["rank"], case=case, method=method,
                                 image_error_percent=error, measurement_error_percent=100*last["measurement_relative_l2_clean"],
                                 status=result["status"], attempts=last["step"],
                                 source_sha256=result["protocol"]["initial_source_sha256"], bitwise_paired=paired,
                                 unchanged_measurements_verified=reference_path.exists(), path=str(path)))
            label = "cached" if "_paired" in manifest_path.parent.parent.name else "original"
            lines.append(f"| {model} ({label}) | {rho:g} | {settings['dlr']['rank']} | {case} | " + " | ".join(values) + " |")
    lines += ["", "A missing result is not a failure. Incomplete values are not final. "
              "Rank-128 runs execute DLR only. Values marked unpaired come from the original CT initializer, "
              "which recomputed the adjoint separately for each method and produced different source hashes; "
              "these are excluded from strictly paired comparisons and are being rerun with one cached source. "
              "The source mixing formula was reconstructed exactly on CPU. When an original reference exists, "
              "truths, observations, masks and random sources were verified bitwise, with matching model and acquisition metadata.", "",
              "The completed 64 MRI sample 1 controls show a large improvement over random initialization: "
              "small-model DLR 99.79% to 2.74% and original-model DLR 99.77% to 2.74%. "
              "The earlier CT runs reached DLR32 34.98% and D-Flow 1.99%, but their source hashes differ; "
              "the cached-source repeat is required before interpreting this comparison as paired. "
              "A consistent accurate configuration across shape counts, resolutions and both operators has not been established.", "",
              "[Amplitude diagnostic](amplitude_audit.md) explains why shrinking poorly placed shapes can decrease the "
              "measurement objective. That mechanism is plausible, not a proof that it causes all observed failures."]
    (root / "initialization_report.md").write_text("\n".join(lines) + "\n")
    (root / "initialization_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Audited {len(rows)} method results.")


if __name__ == "__main__":
    main()

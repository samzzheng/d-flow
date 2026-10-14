"""CPU fixed-source clean/noisy gradient diagnostics for the five-step study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow.api import approx_source_gradient, load_flow_model, make_terminal_flow_products, sensitivity_factors
from dlr_dflow.implementation import enriched
from experiments.ct_mri_sweep.operators import CT, MRI, noisy_measurement
from experiments.ct_mri_sweep.run import write_json
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow

Tensor = torch.Tensor
Record = dict[str, Any]


def compare(estimate: Tensor, reference: Tensor) -> Record:
    a, b = estimate.flatten().double(), reference.flatten().double()
    na, nb = float(a.norm()), float(b.norm())
    error = float((a - b).norm())
    return {"norm": na, "reference_norm": nb, "absolute_error": error,
            "relative_error": error / nb if nb else None,
            "cosine": float(a @ b) / (na * nb) if na and nb else None,
            "norm_ratio": na / nb if nb else None}


def diagnose(model: torch.nn.Module, operator: CT | MRI, inputs: Record,
             source: Tensor, settings: Record, source_kind: str,
             seed_basis: Tensor | None, additional_noises: int, native_control: bool) -> Record:
    started = time.perf_counter()
    variable = source.detach().clone().requires_grad_(True)
    endpoint = euler_flow(model, variable, settings["flow_steps"])
    _, clean_terminal = endpoint_gradient(endpoint, operator, inputs["clean"])

    def transpose(value: Tensor) -> Tensor:
        return torch.autograd.grad(endpoint, variable, value, retain_graph=True)[0].detach()

    exact_clean = transpose(clean_terminal)
    native, factors, alpha, alpha_path = sensitivity_factors(
        model, source, settings["flow_steps"], **settings["dlr"],
        seed=inputs["protocol"]["dlr_seed"], seed_basis=seed_basis,
        return_alpha_path=True)
    endpoint_error = float((native - endpoint.detach()).norm() / endpoint.detach().norm())
    alpha_error = max(abs(a - (1 - i / settings["flow_steps"])) for i, a in enumerate(alpha_path))
    if not torch.isfinite(native).all() or alpha != 0 or alpha_error > 1e-12:
        raise AssertionError(f"Matched endpoint or analytical-alpha check failed: endpoint={endpoint_error}, alpha={alpha}, path={alpha_error}")
    v = factors[2]

    def project(value: Tensor) -> Tensor:
        return (v @ (v.T @ value.reshape(-1, 1))).reshape_as(value)

    dlr_clean = approx_source_gradient(factors, clean_terminal, alpha)
    projected_clean = project(exact_clean)
    generator = torch.Generator().manual_seed(inputs["protocol"]["dlr_seed"] + 72000000)
    direction = torch.randn(source.shape, generator=generator)
    cotangent = torch.randn(source.shape, generator=generator)
    _, tangent = torch.func.jvp(lambda z: euler_flow(model, z, settings["flow_steps"]),
                               (source,), (direction,))
    transpose_cotangent = transpose(cotangent)
    adjoint_error = float(abs((tangent.double() * cotangent.double()).sum() -
                              (direction.double() * transpose_cotangent.double()).sum()) /
                          (tangent.double().norm() * cotangent.double().norm() +
                           direction.double().norm() * transpose_cotangent.double().norm()))
    if adjoint_error > 1e-5:
        raise AssertionError(f"Native JVP/VJP adjoint mismatch: {adjoint_error}")
    jacobian_product, _ = make_terminal_flow_products(model, source, settings["flow_steps"])
    jv = jacobian_product(v)
    us = factors[0] @ factors[1]
    image_operator_difference = (jv - us) @ factors[1].T
    image_operator_proxy = us @ factors[1].T
    result: Record = {
        "source_kind": source_kind,
        "derivative_backend": "native AD, canonical primal control" if native_control else "CPU hand products, shared primal",
        "factor_initialization": "fresh randomized sketch" if seed_basis is None else
            "saved terminal previous_v used for a NEW factor evolution; not a replay of the original last gradient",
        "source_sha256": hashlib.sha256(source.numpy().tobytes()).hexdigest(),
        "endpoint_relative_difference": endpoint_error,
        "endpoint_matches_at_1e_minus_4": endpoint_error < 1e-4,
        "alpha_max_error": alpha_error, "terminal_alpha": alpha,
        "adjoint_scaled_error": adjoint_error,
        "image_relative_l2_percent": 100 * float((endpoint.detach() - inputs["truth"]).norm() / inputs["truth"].norm()),
        "dlr_core_singular_values": torch.linalg.svdvals(factors[1]).tolist(),
        "JV_minus_US_over_US": float((jv.double() - us.double()).norm() / us.double().norm()),
        "JBt_minus_BBt_over_BBt": float(image_operator_difference.double().norm() / image_operator_proxy.double().norm()),
        "dlr_clean_to_exact_clean": compare(dlr_clean, exact_clean),
        "projected_clean_to_exact_clean": compare(projected_clean, exact_clean),
        "noise_trials": [],
    }
    bias2 = float((projected_clean.double() - exact_clean.double()).square().sum())
    for index in range(additional_noises + 1):
        observed = inputs["observed"] if index == 0 else noisy_measurement(
            inputs["clean"], settings["noise_relative"],
            inputs["protocol"]["noise_seed"] + 1000000 + index, inputs["mask"])
        # The operators are linear. Evaluate -A*eta directly to avoid cancellation
        # between two potentially large clean/noisy terminal gradients.
        _, noise_terminal = endpoint_gradient(torch.zeros_like(endpoint), operator,
                                              observed - inputs["clean"])
        exact_noise = transpose(noise_terminal)
        exact_noisy = exact_clean + exact_noise
        dlr_noise = approx_source_gradient(factors, noise_terminal, alpha)
        dlr_noisy = dlr_clean + dlr_noise
        projected_noise = project(exact_noise)
        projected_noisy = projected_clean + projected_noise
        removed_noise2 = float((exact_noise.double() - projected_noise.double()).square().sum())
        noise_norm = float(exact_noise.double().norm())
        dlr_error = float((dlr_noisy.double() - exact_clean.double()).norm())
        projected_error2 = float((projected_noisy.double() - exact_clean.double()).square().sum())
        lhs = projected_error2 - noise_norm ** 2
        rhs = bias2 - removed_noise2
        identity_error = abs(lhs - rhs) / max(bias2 + removed_noise2 + noise_norm ** 2, 1e-30)
        if identity_error > 1e-5:
            raise AssertionError(f"Projection bias/noise identity failed: {identity_error}")
        row = {
            "noise_index": index, "trajectory_noise": index == 0,
            "exact_noise_gradient_norm": noise_norm,
            "DLR_R": dlr_error / noise_norm if noise_norm else None,
            "projected_R": projected_error2 ** .5 / noise_norm if noise_norm else None,
            "dlr_noisy_to_exact_clean": compare(dlr_noisy, exact_clean),
            "dlr_noisy_to_exact_noisy": compare(dlr_noisy, exact_noisy),
            "exact_noisy_to_exact_clean": compare(exact_noisy, exact_clean),
            "dlr_noise_to_exact_noise": compare(dlr_noise, exact_noise),
            "projected_noise_to_exact_noise": compare(projected_noise, exact_noise),
            "discarded_clean_gradient_squared": bias2,
            "discarded_noise_gradient_squared": removed_noise2,
            "projection_identity_scaled_error": identity_error,
        }
        if index == 0:
            _, direct_terminal = endpoint_gradient(endpoint, operator, observed)
            direct_exact = transpose(direct_terminal)
            row["linear_noise_decomposition"] = compare(exact_noisy, direct_exact)
            row["linear_noise_decomposition_scaled_error"] = float(
                (exact_noisy.double() - direct_exact.double()).norm() /
                (exact_clean.double().norm() + exact_noise.double().norm()).clamp_min(1e-30))
            row["p_flow_noisy_to_exact_clean"] = compare(direct_terminal, exact_clean)
            exact_image_noise = jacobian_product(exact_noise.reshape(-1, 1))
            dlr_image_noise = jv @ (v.T @ dlr_noise.reshape(-1, 1))
            row["dlr_image_noise_to_exact_image_noise"] = compare(dlr_image_noise, exact_image_noise)
            if row["linear_noise_decomposition_scaled_error"] > 1e-5:
                raise AssertionError("Canonical noisy gradient differs from clean plus noise")
        result["noise_trials"].append(row)
    additional = result["noise_trials"][1:]
    if additional:
        result["additional_noise_summary"] = {
            "count": len(additional),
            "DLR_R_below_one": sum(row["DLR_R"] < 1 for row in additional),
            "median_DLR_R": statistics.median(row["DLR_R"] for row in additional),
            "median_noise_gradient_norm_ratio": statistics.median(
                row["dlr_noise_to_exact_noise"]["norm_ratio"] for row in additional),
        }
    result["seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/noise_regularization_analysis_20261014/fixed_points"))
    parser.add_argument("--additional-noises", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    model_path = Path("outputs/fm_unet/circles-64-noattn-ch16-b8-e200/n1")
    model, _ = load_flow_model(64, 1, 200, "cpu", model_path)
    checkpoint_hash = hashlib.sha256((model_path / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest()
    cases = [
        (Path("outputs/five_step_noise005_n124_20261013/production"), "r64_n1_ct10", 15),
        (Path("outputs/five_step_noise020_n124_20261013/production"), "r64_n1_mri4", 14),
    ]
    output: Record = {
        "device": "cpu", "torch_threads": 4, "flow_steps": 5, "rank": 8,
        "checkpoint_sha256": checkpoint_hash,
        "selection": "Illustrative discrepancy cases, not a representative sample: CT5% sample15 and MRI20% sample14",
        "limitations": ["CPU recomputation of saved sources; production CUDA kernels are not replayed",
                        "CPU randomized sketches are new draws from the saved seed, not bitwise reproductions of CUDA random draws",
                        "Factor bases are fixed when varying noise at each source",
                        "Ten fresh noise draws are fixed-point diagnostics, not reconstruction trials",
                        "Only saved terminal sources are available; no intermediate source trajectories",
                        "DLR core singular values are not a certified spectrum of the full Jacobian",
                        "All gradients use the canonical Euler endpoint; any internal factor-primal drift is recorded, not substituted into the loss"],
        "cases": [],
    }
    lines = ["# Fixed-source clean/noisy gradient diagnostics", "",
             "Illustrative 64² single-circle cases; five Euler steps, rank 8, analytical alpha = 1-t. CPU only.", "",
             "R = ||approximate noisy gradient - exact clean gradient|| / ||exact noisy gradient - exact clean gradient||.",
             "R < 1 supports the clean-gradient hypothesis at that fixed point. Noise ratio measures only the noise-induced gradient, excluding clean-gradient bias.", "",
             "| Case | Source | Basis seed / backend | Image error (%) | DLR R | Projected R | DLR noise norm / exact noise norm |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for root, job, sample in cases:
        settings = json.loads((root / "manifest.json").read_text())["settings"]
        directory = root / job / f"sample_{sample:03d}"
        inputs = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True)
        if inputs["protocol"]["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Model checkpoint mismatch")
        operator: CT | MRI
        if inputs["protocol"]["job"]["problem"] == "ct":
            operator = CT(64, 10, torch.device("cpu"))
        else:
            operator = MRI(64, 4, inputs["protocol"]["operator"]["mask_seed"], torch.device("cpu"))
            if not torch.equal(operator.mask, inputs["mask"]):
                raise ValueError("MRI mask mismatch")
        if not torch.allclose(operator(inputs["truth"]), inputs["clean"], atol=1e-5, rtol=1e-5):
            raise ValueError("Clean forward measurement mismatch")
        exact_saved = torch.load(directory / "d_flow.pt", map_location="cpu", weights_only=True)
        dlr_saved = torch.load(directory / "dlr_d_flow.pt", map_location="cpu", weights_only=True)
        case: Record = {"directory": str(directory), "noise_percent": settings["noise_relative"] * 100,
                        "protocol": inputs["protocol"], "points": []}
        output["cases"].append(case)
        for label, source, seed_basis, native_control in [
            ("initial", inputs["initial"], None, False),
            ("D-Flow final", exact_saved["source"], None, False),
            ("DLR final", dlr_saved["source"], None, False),
            ("DLR final", dlr_saved["source"], dlr_saved["previous_v"], False),
            ("D-Flow final", exact_saved["source"], None, True),
        ]:
            core.HAND_JVP = core.HAND_VJP = not native_control
            enriched.SHARE_PRIMAL = not native_control
            result = diagnose(model, operator, inputs, source, settings, label, seed_basis, args.additional_noises, native_control)
            case["points"].append(result)
            write_json(args.out / "diagnostics.json", output)
            first = result["noise_trials"][0]
            basis = "fresh" if seed_basis is None else "saved terminal V, new evolution"
            if native_control:
                basis += " / native AD control"
            lines.append(f"| {job} / {case['noise_percent']:g}% / {sample} | {label} | {basis} | {result['image_relative_l2_percent']:.3f} | {first['DLR_R']:.4f} | {first['projected_R']:.4f} | {first['dlr_noise_to_exact_noise']['norm_ratio']:.4f} |")
            print(json.dumps({"job": job, "sample": sample, "source": label, "basis": basis,
                              "R": first["DLR_R"], "projection_R": first["projected_R"],
                              "seconds": result["seconds"]}), flush=True)
    lines.extend(["", "The noise-only gradient is attenuated in these examples, but R is not consistently below one. A smaller noise component does not by itself imply a more accurate total gradient: discarded clean-gradient components contribute bias.", "",
                  "The saved-V calculation evaluates a new factor evolution at the saved source. It does not recover the gradient used by the final production optimizer step.", "",
                  "The following check compares the actual infinitesimal image update J Bᵀ to its symmetric surrogate B Bᵀ, where B = U S Vᵀ. A small discrepancy supports that approximation locally, not throughout the reconstruction.", "",
                  "| Case | Source | Initialization / backend | ||JV-US|| / ||US|| | ||JBᵀ-BBᵀ|| / ||BBᵀ|| | Factor-primal endpoint discrepancy |",
                  "| --- | --- | --- | ---: | ---: | ---: |"])
    for case in output["cases"]:
        for point in case["points"]:
            basis = "fresh" if point["factor_initialization"].startswith("fresh") else "saved terminal V"
            lines.append(f"| {case['protocol']['job']['problem']} {case['noise_percent']:g}% | {point['source_kind']} | {basis} / {point['derivative_backend']} | {point['JV_minus_US_over_US']:.4f} | {point['JBt_minus_BBt_over_BBt']:.4f} | {point['endpoint_relative_difference']:.3g} |")
    lines.extend(["", *[f"- {note}." for note in output["limitations"]]])
    (args.out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

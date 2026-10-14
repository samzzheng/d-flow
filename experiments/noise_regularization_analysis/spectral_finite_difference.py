"""Finite-difference check of the amplified native Jacobian direction on CPU."""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.run import write_json
from sampler.ode_samplers import euler_flow


def main() -> None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model, _ = load_flow_model(64, 1, 200, "cpu", "outputs/fm_unet/circles-64-noattn-ch16-b8-e200/n1")
    directory = Path("outputs/five_step_noise005_n124_20261013/production/r64_n1_ct10/sample_015")
    saved = torch.load(directory / "d_flow.pt", map_location="cpu", weights_only=True)
    result = {"source_directory": str(directory), "flow_steps": 5, "device": "cpu", "checks": []}
    for dtype in (torch.float32, torch.float64):
        local_model = model if dtype == torch.float32 else copy.deepcopy(model).double()
        if dtype == torch.float64:
            # The checkpoint model's sinusoidal time embedding is explicitly
            # float32; cast its output only in this isolated float64 control.
            local_model.time_proj.register_forward_pre_hook(
                lambda module, inputs: (inputs[0].to(torch.float64),))
        source = saved["source"].to(dtype)

        def flow(z: torch.Tensor) -> torch.Tensor:
            return euler_flow(local_model, z, 5)

        generator = torch.Generator().manual_seed(20261014)
        direction = torch.randn(source.shape, dtype=dtype, generator=generator)
        direction /= direction.norm()
        # The selected source has a large leading spectral gap. Four power
        # iterations suffice here; the adjoint residual below checks the result.
        for _ in range(4):
            _, tangent = torch.func.jvp(flow, (source,), (direction,))
            _, transpose = torch.func.vjp(flow, source)
            direction = transpose(tangent)[0]
            direction /= direction.norm()
        _, tangent = torch.func.jvp(flow, (source,), (direction,))
        sigma = float(tangent.norm())
        _, transpose = torch.func.vjp(flow, source)
        residual = float((transpose(tangent / tangent.norm())[0] - tangent.norm() * direction).norm() / tangent.norm())
        check = {"dtype": str(dtype), "jacobian_directional_norm": sigma,
                 "right_triplet_relative_residual": residual, "finite_differences": []}
        for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            with torch.no_grad():
                difference = (flow(source + step * direction) - flow(source - step * direction)) / (2 * step)
            check["finite_differences"].append({
                "step": step, "directional_norm": float(difference.norm()),
                "relative_vector_error": float((difference - tangent).norm() / tangent.norm()),
                "relative_norm_error": abs(float(difference.norm()) / sigma - 1),
            })
        result["checks"].append(check)
    destination = Path("outputs/noise_regularization_analysis_20261014/fixed_points/spectra/finite_difference.json")
    write_json(destination, result)
    print(destination)


if __name__ == "__main__":
    main()

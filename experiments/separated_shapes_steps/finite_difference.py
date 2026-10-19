"""Resolve the failed-source finite-difference anomaly without changing production precision."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import CT
from experiments.ct_mri_sweep.run import write_json
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.15, device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    case = Path("outputs/separated_shapes_noiseless_20261016/production/r128_shapes_ct10/sample_000")
    inputs = torch.load(case / "inputs.pt", map_location=device, weights_only=True)
    saved = torch.load(case / "d_flow.pt", map_location="cpu", weights_only=True, mmap=True)
    source = saved["source"].to(device)
    rows = []
    references: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for dtype in (torch.float32, torch.float64):
        model, _ = load_flow_model(128, 0, 200, str(device), Path("outputs/fm_unet/separated-shapes-128-noattn-ch16-b8-e200"))
        model.to(dtype=dtype)
        if dtype == torch.float64:
            # Keep the model's original FP32 time embedding constants; only cast
            # their fixed values to match the double-precision linear layer.
            def cast_time(module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
                return (values[0].double(),)
            model.time_proj.register_forward_pre_hook(cast_time)
        operator = CT(128, 10, device)
        operator.grids = tuple(grid.to(dtype) for grid in operator.grids)
        observed = inputs["observed"].to(dtype)
        for steps in (5, 20):
            variable = source.to(dtype).detach().requires_grad_(True)
            image = euler_flow(model, variable, steps)
            loss = .5 * (operator(image)-observed).square().sum(dtype=torch.float64)
            gradient, = torch.autograd.grad(loss, variable)
            norm = float(gradient.norm())
            direction = gradient / norm
            row = dict(dtype=str(dtype), steps=steps, gradient_norm=norm, loss=float(loss.detach()), finite_differences=[])
            if dtype == torch.float32:
                references[steps] = (image.detach().double(), gradient.double())
            else:
                ref_image, ref_gradient = references[steps]
                row.update(image_fp32_difference_over_gt=float((image-ref_image).norm()/inputs["truth"].norm()),
                           gradient_fp32_relative_difference=float((gradient-ref_gradient).norm()/gradient.norm()),
                           gradient_fp32_cosine=float(torch.nn.functional.cosine_similarity(gradient.flatten(),ref_gradient.flatten(),dim=0)))
            with torch.no_grad():
                for epsilon in (.1, .03, .01, .003, .001, .0003, .0001, .00003, .00001, .000003, .000001, .0000001):
                    plus = .5 * (operator(euler_flow(model, variable+epsilon*direction, steps))-observed).square().sum(dtype=torch.float64)
                    minus = .5 * (operator(euler_flow(model, variable-epsilon*direction, steps))-observed).square().sum(dtype=torch.float64)
                    numerical = float((plus-minus)/(2*epsilon))
                    row["finite_differences"].append(dict(epsilon=epsilon, derivative=numerical,
                        relative_error=abs(numerical-norm)/norm,
                        plus_loss_change=float(plus-loss),minus_loss_change=float(minus-loss)))
            rows.append(row)
            write_json(Path("outputs/separated_shapes_steps_20261019/gradients/finite_difference.json"), rows)
            print(dtype, steps, "gradient",norm,"best FD error",min(x["relative_error"] for x in row["finite_differences"]),flush=True)
            del image, loss, gradient, variable, direction
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""CPU cross-check of forward and adjoint sensitivity to arithmetic precision."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import CT, MRI
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    parent = torch.load(args.source, map_location="cpu", weights_only=True)
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    protocol = parent["protocol"]
    settings = protocol["settings"]
    for key in ("checkpoint_sha256", "operator", "source_seed"):
        assert protocol[key] == inputs["protocol"][key]
    resolution = protocol["job"]["resolution"]
    steps = settings["flow_steps"]
    step_size = 1.0/steps
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    info = protocol["operator"]
    records, saved = [], []
    for dtype in (torch.float32, torch.float64):
        model, _ = load_flow_model(resolution, 0, 500, "cpu", folder)
        model = model.to(dtype=dtype)
        if dtype == torch.float64:
            model.time_proj.register_forward_pre_hook(lambda module, arguments: (arguments[0].double(),))
        operator = (CT(resolution, protocol["job"]["level"], torch.device("cpu"))
                    if protocol["job"]["problem"] == "ct" else
                    MRI(resolution, 4, info["mask_seed"], torch.device("cpu"), info["pattern"]))
        if isinstance(operator, CT):
            operator.grids = tuple(grid.to(dtype=dtype) for grid in operator.grids)
        else:
            assert torch.equal(operator.mask, inputs["mask"])
        points = [parent["source"].to(dtype=dtype).clone().requires_grad_(True)]
        for index in range(steps):
            points.append(points[-1] + step_size*model(points[-1], points[-1].new_tensor([index*step_size])))
        with torch.no_grad():
            torch.testing.assert_close(points[-1], euler_flow(model, points[0], steps), rtol=0, atol=0)
        measurement_dtype = (torch.complex128 if dtype == torch.float64 else torch.complex64) if inputs["observed"].is_complex() else dtype
        residual = operator(points[-1])-inputs["observed"].to(measurement_dtype)
        loss = .5*residual.abs().square().sum(dtype=torch.float64)
        adjoints = torch.autograd.grad(loss, points)
        saved.append(([value.detach().double() for value in points], [value.detach().double() for value in adjoints]))
        records.append(dict(dtype=str(dtype), objective=float(loss.detach()),
            adjoint_norms=[float(value.norm()) for value in adjoints]))
    comparison = [dict(step=index,
        forward_relative_difference=float((x32-x64).norm()/x64.norm()),
        adjoint_relative_difference=float((g32-g64).norm()/g64.norm()),
        adjoint_cosine=float(torch.nn.functional.cosine_similarity(g32.flatten(), g64.flatten(), dim=0)))
        for index, (x32, x64, g32, g64) in enumerate(zip(saved[0][0], saved[1][0], saved[0][1], saved[1][1]))]
    output = dict(device="cpu", source=str(args.source), checkpoint_sha256=protocol["checkpoint_sha256"],
        source_sha256=hashlib.sha256(parent["source"].numpy().tobytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flow_steps=steps,
        records=records, comparison=comparison,
        note="CPU arithmetic diagnostic. CPU and GPU kernels and FP32 time features need not agree; neither CPU result is a GPU reference.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()

"""Separate CT-adjoint variability from native model-pullback variability."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import CT
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.04, device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    config = json.loads((args.case/"config.json").read_text())
    settings = config["settings"]
    resolution = inputs["protocol"]["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == inputs["protocol"]["checkpoint_sha256"]
    model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
    operator = CT(resolution, inputs["protocol"]["job"]["level"], device)
    cpu_operator = CT(resolution, inputs["protocol"]["job"]["level"], torch.device("cpu"))
    # Preserve the actual CUDA geometry instead of recomputing angles on CPU.
    cpu_operator.grids = tuple(grid.cpu().double() for grid in operator.grids)
    observed = inputs["observed"].to(device)
    rows = []
    for variant in ("initial", "unrestricted256", "gradual256"):
        torch.backends.cudnn.deterministic = False
        saved = None if variant == "initial" else torch.load(args.case/variant/"dlr_d_flow.pt", map_location="cpu", weights_only=True)
        initial = inputs["initial"] if saved is None else saved["source"]
        source = initial.to(device).detach().requires_grad_(True)
        image = euler_flow(model, source, settings["flow_steps"])
        if saved is not None:
            torch.testing.assert_close(image.detach().cpu(), saved["image"], rtol=0, atol=0)
        residual = (operator(image.detach())-observed).detach()
        terminal_gradients = []
        for _ in range(8):
            variable = image.detach().requires_grad_(True)
            terminal_gradients.append(torch.autograd.grad(operator(variable), variable, residual)[0])
        reference_image = image.detach().cpu().double().requires_grad_(True)
        reference_terminal = torch.autograd.grad(cpu_operator(reference_image), reference_image,
            residual.cpu().double())[0].to(device=device, dtype=source.dtype)
        varied = [torch.autograd.grad(image, source, gradient, retain_graph=True)[0]
                  for gradient in terminal_gradients]
        fixed = [torch.autograd.grad(image, source, terminal_gradients[0], retain_graph=True)[0]
                 for _ in range(8)]
        reference_source = torch.autograd.grad(image, source, reference_terminal)[0]
        terminal_norm, source_norm = terminal_gradients[0].double().norm(), varied[0].double().norm()
        terminal_changes = [float((gradient.double()-terminal_gradients[0].double()).norm()/terminal_norm)
                            for gradient in terminal_gradients[1:]]
        source_changes = [float((gradient.double()-varied[0].double()).norm()/source_norm)
                          for gradient in varied[1:]]
        fixed_changes = [float((gradient.double()-fixed[0].double()).norm()/source_norm)
                         for gradient in fixed[1:]]
        torch.backends.cudnn.deterministic = True
        deterministic_image = euler_flow(model, source, settings["flow_steps"])
        deterministic_gradients = [torch.autograd.grad(deterministic_image, source,
            terminal_gradients[0], retain_graph=index < 7)[0] for index in range(8)]
        deterministic_norm = deterministic_gradients[0].double().norm()
        deterministic_changes = [float((gradient.double()-deterministic_gradients[0].double()).norm()/deterministic_norm)
                                 for gradient in deterministic_gradients[1:]]
        rows.append(dict(variant=variant, source_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            terminal_relative_changes=terminal_changes, source_relative_changes=source_changes,
            fixed_terminal_source_relative_changes=fixed_changes,
            cpu_double_adjoint_relative_difference=float((reference_terminal.double()-terminal_gradients[0].double()).norm()/terminal_norm),
            propagated_cpu_double_adjoint_relative_difference=float((reference_source.double()-varied[0].double()).norm()/source_norm),
            deterministic_cudnn_fixed_terminal_relative_changes=deterministic_changes,
            deterministic_cudnn_endpoint_relative_change=float((deterministic_image.detach().double()-image.detach().double()).norm()/image.detach().double().norm()),
            deterministic_cudnn_gradient_relative_change=float((deterministic_gradients[0].double()-fixed[0].double()).norm()/source_norm),
            terminal_norm=float(terminal_norm), source_gradient_norm=float(source_norm)))
        print(variant, max(terminal_changes), max(source_changes), max(fixed_changes), max(deterministic_changes), flush=True)
        del image, varied, fixed, terminal_gradients, source, deterministic_image, deterministic_gradients
    args.out.write_text(json.dumps(dict(rows=rows, checkpoint_sha256=inputs["protocol"]["checkpoint_sha256"],
        protocol="Fixed source, endpoint and residual; repeat native CT adjoint eight times, then propagate each through one fixed native Euler graph. Repeat its pullback separately with identical terminal gradient. CPU FP64 CT adjoint uses copied CUDA grids and the same residual, then casts to FP32 before model pullback.",
        limitation="No optimizer updates. CPU arithmetic is a diagnostic reference, not a promoted operator implementation. Numerical variability alone does not establish the cause of reconstruction failure."), indent=2)+"\n")


if __name__ == "__main__":
    main()

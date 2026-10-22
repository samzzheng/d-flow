"""Audit a saved FP64 L-BFGS direction using native derivatives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.separated_shapes_117k.optimizer_controls import safeguard_direction
from p_flow.armijo import lbfgs_direction
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:1"), default="cpu")
    parser.add_argument("--minimum-direction-cosine", type=float, default=0.)
    parser.add_argument("--line-trials", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.minimum_direction_cosine < 1:
        parser.error("Minimum direction cosine must lie in [0,1)")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    saved = torch.load(args.source, map_location="cpu", weights_only=True)
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if saved["source"].dtype != torch.float64 or saved["optimizer"]["g"] is None:
        parser.error("A saved FP64 source and optimizer gradient are required")
    protocol = saved["protocol"]
    settings = protocol["precision_recovery"]["settings"]
    for key in ("checkpoint_sha256", "operator", "source_seed"):
        assert protocol[key] == inputs["protocol"][key]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.out.with_suffix(".pt")
    if snapshot.exists():
        parser.error("Refusing to replace an existing diagnostic snapshot")
    torch.save(saved, snapshot)
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    model, _ = load_flow_model(resolution, 0, 500, args.device, folder)
    model = model.double()
    model.time_proj.register_forward_pre_hook(lambda module, arguments: (arguments[0].double(),))
    info = protocol["operator"]
    if protocol["job"]["problem"] == "ct":
        # Reproduce the saved CUDA geometry before moving all evaluations to CPU.
        operator = CT(resolution, protocol["job"]["level"], torch.device("cuda:1"))
        operator.grids = tuple(grid.to(device=device, dtype=torch.float64) for grid in operator.grids)
        operator.angles = operator.angles.to(device)
    else:
        operator = MRI(resolution, 4, info["mask_seed"], device, info["pattern"])
        assert torch.equal(operator.mask.cpu(), inputs["mask"])
    observed = inputs["observed"].to(device=device,
        dtype=torch.complex128 if inputs["observed"].is_complex() else torch.float64)
    source = saved["source"].to(device).detach().requires_grad_(True)
    image = euler_flow(model, source, settings["flow_steps"])
    objective = .5*(operator(image)-observed).abs().square().sum()
    exact, = torch.autograd.grad(objective, source)
    exact = exact.flatten()
    state = saved["optimizer"]
    approximate = state["g"].to(device).flatten()
    direction = lbfgs_direction(approximate, [s.to(device) for s in state["S"]],
        [y.to(device) for y in state["Y"]])
    if args.minimum_direction_cosine:
        direction = safeguard_direction(direction, approximate, args.minimum_direction_cosine)
    if settings["source_projection"]:
        parser.error("This audit currently covers the unconstrained source protocol")
    used_fallback = not torch.isfinite(direction).all() or approximate.dot(direction) >= 0
    if used_fallback:
        direction = -approximate.clone()
    norm = direction.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise FloatingPointError("No finite nonzero optimizer direction")
    direction /= norm
    exact_slope = float(exact.dot(direction))
    predicted_slope = float(approximate.dot(direction))
    finite_differences = []
    line_trials = []
    with torch.no_grad():
        for displacement in (1e-5, 1e-6, 1e-7):
            values = []
            for sign in (-1, 1):
                trial = source+sign*displacement*direction.reshape_as(source)
                residual = operator(euler_flow(model, trial, settings["flow_steps"]))-observed
                values.append(float(.5*residual.abs().square().sum()))
            slope = (values[1]-values[0])/(2*displacement)
            finite_differences.append(dict(displacement=displacement, slope=slope,
                relative_slope_error=abs(slope-exact_slope)/max(abs(exact_slope), 1e-30)))
        if args.line_trials:
            displacement = min(float(norm)*settings["optimizer"]["lr"],
                settings["optimizer"]["step_cap"]*source.numel()**.5)
            for _ in range(settings["optimizer"]["max_ls"]):
                trial = source+displacement*direction.reshape_as(source)
                residual = operator(euler_flow(model, trial, settings["flow_steps"]))-observed
                change = float(.5*residual.abs().square().sum()-objective.detach())
                accepted = change < 0 and change <= settings["optimizer"]["c1"]*displacement*predicted_slope
                line_trials.append(dict(displacement=displacement, change=change, accepted=accepted))
                if accepted:
                    break
                displacement *= .5
    endpoint_error = float((image.detach()-saved["image"].to(device)).norm()/saved["image"].norm())
    objective_error = abs(float(objective.detach())-saved["history"][-1]["phi"])
    report = dict(source=str(args.source), snapshot=str(snapshot),
        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        step=state["step_attempts"], curvature_pairs=len(state["S"]),
        device=args.device, endpoint_relative_difference=endpoint_error, objective_absolute_difference=objective_error,
        objective=float(objective.detach()), saved_objective=saved["history"][-1]["phi"],
        gradient_relative_error=float((approximate-exact).norm()/exact.norm()),
        gradient_cosine=float(approximate.dot(exact)/(approximate.norm()*exact.norm())),
        unit_direction_exact_slope=exact_slope, unit_direction_predicted_slope=predicted_slope,
        direction_norm_before_normalization=float(norm), fallback=bool(used_fallback),
        minimum_direction_cosine=args.minimum_direction_cosine, line_trials=line_trials,
        direction_error_slope_fraction=float((exact-approximate).dot(direction)/abs(predicted_slope)),
        exact_direction_cosine=float(-exact.dot(direction)/exact.norm()),
        finite_differences=finite_differences,
        scope="Fixed saved-state diagnostic, native FP64. CPU mode uses CUDA1 only to construct CT geometry. No reconstruction or optimizer changes.")
    args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

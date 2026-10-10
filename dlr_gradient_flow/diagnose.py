"""Fixed-source native gradient diagnostics for a completed DLR Gradient Flow run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from dlr_dflow.api import approx_source_gradient, load_flow_model
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import metrics, tensor_tree, write_json
from sampler.ode_samplers import euler_flow


ROOT = Path(__file__).resolve().parents[1]
Record = dict[str, Any]


def comparison(gradient: torch.Tensor, reference: torch.Tensor) -> Record:
    """Vector metrics, with unavailable ratios for zero references."""
    candidate, exact = gradient.flatten().double(), reference.flatten().double()
    candidate_norm, exact_norm = float(candidate.norm()), float(exact.norm())
    error = float((candidate - exact).norm())
    inner_product = float(candidate @ exact)
    return {
        "gradient_norm": candidate_norm,
        "reference_norm": exact_norm,
        "absolute_error": error,
        "relative_error": error / exact_norm if exact_norm else None,
        "norm_ratio": candidate_norm / exact_norm if exact_norm else None,
        "cosine": inner_product / (candidate_norm * exact_norm)
        if candidate_norm and exact_norm else None,
        "inner_product": inner_product,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Completed state.pt checkpoint or its method directory")
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Original paired sweep inputs.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--line-search-probe", action="store_true",
                        help="Probe eight halving rates along the cached negative gradient")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(
            2**30 / torch.cuda.get_device_properties(device).total_memory, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    state_path = args.run / "state.pt" if args.run.is_dir() else args.run
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    protocol_path = state_path.parent / "protocol.json"
    if not protocol_path.exists():
        protocol_path = state_path.parent.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    settings = protocol["settings"]
    paired = inputs["protocol"]
    if paired != protocol["source_protocol"]:
        raise ValueError("Inputs belong to a different reconstruction protocol")
    input_hash = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    if input_hash != protocol["inputs_sha256"]:
        raise ValueError("Paired input checksum differs from the reconstruction")
    if state["status"] not in {"stalled", "step_limit", "objective_plateau"}:
        raise ValueError("The requested reconstruction is not complete")
    if state["factors"] is None or not torch.equal(state["source"], state["factor_source"]):
        raise ValueError("Saved factors do not correspond to the saved source")

    job = paired["job"]
    model_dir = ROOT / settings["run_template"].format(**job)
    checkpoint_path = model_dir / f"checkpoints/epoch_{settings['epoch']:04d}.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != paired["checkpoint_sha256"]:
        raise ValueError("Model checkpoint differs from the reconstruction")
    model, _ = load_flow_model(job["resolution"], job["circles"],
                              settings["epoch"], str(device), model_dir)
    operator = (CT(job["resolution"], job["level"], device) if job["problem"] == "ct" else
                MRI(job["resolution"], job["level"], paired["operator"]["mask_seed"],
                    device, settings["mri_pattern"]))
    if isinstance(operator, MRI) and not torch.equal(operator.mask.cpu(), inputs["mask"]):
        raise ValueError("Reconstructed MRI mask differs from the saved acquisition")
    state, inputs = tensor_tree(state, device), tensor_tree(inputs, device)
    source = state["source"].detach().clone().requires_grad_(True)
    if device.type == "cuda":
        # Segment checkpointing preserves the native Euler map while bounding
        # activation storage on a GPU shared with reconstruction workers.
        image = source
        step = 1.0 / settings["flow_steps"]
        for start in range(0, settings["flow_steps"], 10):
            def segment(value: torch.Tensor, first: int = start) -> torch.Tensor:
                for index in range(first, min(first + 10, settings["flow_steps"])):
                    model_time = value.new_full((value.shape[0],), index * step)
                    value = value + step * model(value, model_time)
                return value

            image = checkpoint(segment, image, use_reentrant=False)
    else:
        image = euler_flow(model, source, settings["flow_steps"])
    predicted = operator(image)
    objective = 0.5 * (predicted - inputs["observed"]).abs().square().sum(dtype=torch.float64)
    gx, = torch.autograd.grad(objective, image, retain_graph=True)
    exact, = torch.autograd.grad(image, source, grad_outputs=gx)
    approximate = approx_source_gradient(state["factors"], gx.detach(), alpha1=0.0)
    saved_image = state["image"].detach().requires_grad_(True)
    saved_endpoint_loss = 0.5 * (operator(saved_image) - inputs["observed"]).abs().square().sum(dtype=torch.float64)
    saved_endpoint_gx, = torch.autograd.grad(saved_endpoint_loss, saved_image)
    saved_endpoint_approximate = approx_source_gradient(state["factors"], saved_endpoint_gx, alpha1=0.0)
    left, _, right = state["factors"]
    raw_subspace = (right @ (right.T @ exact.flatten())).reshape_as(exact)
    orthogonal_right, _ = torch.linalg.qr(right.double(), mode="reduced")
    subspace = (orthogonal_right @ (orthogonal_right.T @ exact.flatten().double())).reshape_as(exact)
    coefficient_error = (approximate - subspace).double()
    subspace_error = (subspace - exact).double()
    total_squared = float((approximate.double() - exact.double()).square().sum())
    coefficient_squared = float(coefficient_error.square().sum())
    subspace_squared = float(subspace_error.square().sum())
    identity_defect = abs(total_squared - coefficient_squared - subspace_squared)
    saved_gradient = state["optimizer_state"]["g"].reshape_as(exact)
    rank = right.shape[1]
    identity = torch.eye(rank, device=device, dtype=right.dtype)
    report: Record = {
        "run": str(state_path.resolve()), "inputs": str(args.inputs.resolve()),
        "case": state["case"], "mode": state["mode"], "status": state["status"],
        "attempts": state["history"][-1]["step"], "factor_age": state["factor_age"],
        "device": str(device), "threads": 2, "flow_steps": settings["flow_steps"],
        "native_flow_checkpoint_stride": 10 if device.type == "cuda" else None,
        "dtype": str(source.dtype), "torch": str(torch.__version__),
        "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "inputs_sha256": input_hash, "checkpoint_sha256": checkpoint_hash,
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "factor_source_matches": True,
        "native_objective": float(objective.detach()),
        "saved_gpu_objective": state["history"][-1]["phi"],
        "native_metrics": metrics(image.detach(), predicted.detach(), inputs["truth"],
                               inputs["clean"], inputs["observed"]),
        "saved_gpu_metrics": state["history"][-1],
        "endpoint_native_vs_saved_gpu": comparison(image.detach(), state["image"]),
        "saved_factors_vs_exact_native_gradient": comparison(approximate, exact),
        "factor_negative_gradient_native_objective_derivative":
            -float(approximate.flatten().double() @ exact.flatten().double()),
        "right_subspace_vs_exact_native_gradient": comparison(subspace, exact),
        "raw_vvt_subspace_vs_exact_native_gradient": comparison(raw_subspace, exact),
        "coefficient_vs_projected_gradient": comparison(approximate, subspace),
        "error_decomposition": {
            "total_squared": total_squared,
            "coefficient_squared": coefficient_squared,
            "missing_subspace_squared": subspace_squared,
            "coefficient_fraction": coefficient_squared / total_squared if total_squared else None,
            "missing_subspace_fraction": subspace_squared / total_squared if total_squared else None,
            "identity_absolute_defect": identity_defect,
            "identity_relative_defect": identity_defect / total_squared if total_squared else None,
            "left_orthogonality_defect": float((left.T @ left - identity).norm()),
            "right_orthogonality_defect": float((right.T @ right - identity).norm()),
            "projection": "Float64 QR of saved V; same subspace, removes FP32 orthogonality drift",
        },
        "cached_gpu_gradient_vs_exact_native_gradient": comparison(saved_gradient, exact),
        "cached_gpu_gradient_vs_recomputed_factor_gradient": comparison(saved_gradient, approximate),
        "cached_gpu_gradient_vs_factors_at_saved_gpu_endpoint": comparison(saved_gradient, saved_endpoint_approximate),
        "factor_gradient_saved_endpoint_vs_native_endpoint": comparison(saved_endpoint_approximate, approximate),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "Exact and factor gradients use the same newly computed native Euler endpoint "
            "and terminal gradient. The factors were computed on GPU. Cached GPU gradient "
            "comparisons can mix numerical backends; endpoint drift limits GPU-stall attribution. "
            "Positive cosine is local negative-gradient descent, not guaranteed finite-step "
            "progress or descent of the saved L-BFGS direction. No fresh DLR solve was run."
        ),
    }
    if args.line_search_probe:
        direction = -saved_gradient
        source_fixed = source.detach()
        exact_slope = float(exact.flatten().double() @ direction.flatten().double())
        approximate_slope = -float(saved_gradient.double().square().sum())
        optimizer_state = state["optimizer_state"]
        initial_rate = float(optimizer_state["lr"])
        if optimizer_state["step_cap"] > 0:
            initial_rate = min(initial_rate, optimizer_state["step_cap"] / float(direction.norm()))
        base = float(objective.detach())
        trials: list[Record] = []
        with torch.no_grad():
            for halving in range(0, 29, 4):
                rate = initial_rate * 2.0 ** -halving
                candidate = source_fixed + rate * direction
                displacement = candidate - source_fixed
                trial_image = euler_flow(model, candidate, settings["flow_steps"])
                trial_residual = operator(trial_image) - inputs["observed"]
                value = float(0.5 * trial_residual.abs().square().sum(dtype=torch.float64))
                armijo_bound = optimizer_state["c1"] * rate * approximate_slope
                trials.append({
                    "halvings": halving, "rate": rate, "objective": value,
                    "objective_change": value - base,
                    "source_step_norm": float(displacement.norm()),
                    "source_unchanged": torch.equal(candidate, source_fixed),
                    "nominal_exact_first_order_prediction": rate * exact_slope,
                    "rounded_step_exact_first_order_prediction":
                        float(exact.flatten().double() @ displacement.flatten().double()),
                    "armijo_change_bound": armijo_bound,
                    "strict_decrease": value < base,
                    "armijo_accepted": value < base and value - base <= armijo_bound,
                })
        report["line_search_probe"] = {
            "direction": "Negative cached optimizer gradient, without L-BFGS history",
            "initial_capped_rate": initial_rate, "exact_slope": exact_slope,
            "approximate_slope": approximate_slope,
            "saved_history_length": len(optimizer_state["S"]),
            "saved_core_singular_values": torch.linalg.svdvals(state["factors"][1].double()).cpu().tolist(),
            "trials": trials,
        }
        report["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.out, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

"""Audit exact D-Flow source gradients on a real sparse-CT circle case."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from benchmark_dflow_sparse_ct_circles_cpu import (
    ROOT,
    data_objective,
    load_model,
    normalize_images,
    solve_flow,
    sparse_radon,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--method", choices=("forward_euler", "rk4"), required=True)
    parser.add_argument("--flow-steps", type=int, default=100)
    parser.add_argument(
        "--checkpoint-comparison-steps", type=int, default=3,
        help="Short flow used for the checkpointed versus ordinary gradient check.",
    )
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def loss_and_gradient(
    model: torch.nn.Module,
    source: torch.Tensor,
    measurement: torch.Tensor,
    angles: torch.Tensor,
    resolution: int,
    method: str,
    flow_steps: int,
    checkpointing: bool,
) -> tuple[float, torch.Tensor]:
    value = source.detach().clone().requires_grad_(True)
    reconstruction = solve_flow(
        model, value, flow_steps, method, checkpointing,
    )
    predicted = sparse_radon(reconstruction, angles)
    loss, _ = data_objective(
        predicted, measurement, resolution, "measurement_mse",
    )
    gradient, = torch.autograd.grad(loss, value)
    return loss.item(), gradient.detach()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.resolution, args.epoch, device)
    payload = torch.load(
        ROOT / f"data/circles-{args.resolution}-n1.pt",
        map_location="cpu", weights_only=True, mmap=True,
    )
    truth = normalize_images(
        payload["test"][args.sample_index:args.sample_index + 1]
    ).to(device)
    angles = torch.linspace(0, math.pi, 16, device=device)[:-1]
    with torch.no_grad():
        measurement = sparse_radon(truth, angles)
    generator = torch.Generator(device="cpu").manual_seed(
        args.base_seed + 10_000_000 + args.resolution * 1000 + args.sample_index
    )
    source = torch.randn(truth.shape, generator=generator).to(device)

    short_plain_loss, short_plain_gradient = loss_and_gradient(
        model, source, measurement, angles, args.resolution,
        args.method, args.checkpoint_comparison_steps, False,
    )
    short_checkpoint_loss, short_checkpoint_gradient = loss_and_gradient(
        model, source, measurement, angles, args.resolution,
        args.method, args.checkpoint_comparison_steps, True,
    )
    short_gradient_difference = torch.linalg.vector_norm(
        short_checkpoint_gradient - short_plain_gradient
    ) / (torch.linalg.vector_norm(short_plain_gradient) + 1e-30)
    full_loss, full_gradient = loss_and_gradient(
        model, source, measurement, angles, args.resolution,
        args.method, args.flow_steps, True,
    )

    direction_generator = torch.Generator(device="cpu").manual_seed(
        args.base_seed + 20_000_000 + args.resolution * 1000 + args.sample_index
    )
    direction = torch.randn(source.shape, generator=direction_generator).to(device)
    direction = direction / direction.square().mean().sqrt()
    autograd_directional = (full_gradient * direction).sum().item()
    finite_differences = {}
    with torch.no_grad():
        for epsilon in (1e-2, 3e-3, 1e-3):
            losses = []
            for sign in (1.0, -1.0):
                reconstruction = solve_flow(
                    model, source + sign * epsilon * direction,
                    args.flow_steps, args.method, False,
                )
                predicted = sparse_radon(reconstruction, angles)
                loss, _ = data_objective(
                    predicted, measurement, args.resolution, "measurement_mse",
                )
                losses.append(loss.item())
            finite_differences[str(epsilon)] = (losses[0] - losses[1]) / (2 * epsilon)

    result = {
        "resolution": args.resolution,
        "sample_index": args.sample_index,
        "method": args.method,
        "flow_steps": args.flow_steps,
        "objective": "measurement_mse",
        "checkpoint_comparison_steps": args.checkpoint_comparison_steps,
        "short_plain_loss": short_plain_loss,
        "short_checkpoint_loss": short_checkpoint_loss,
        "short_checkpoint_loss_absolute_difference": abs(
            short_checkpoint_loss - short_plain_loss
        ),
        "short_checkpoint_gradient_relative_l2": short_gradient_difference.item(),
        "full_checkpointed_loss": full_loss,
        "full_checkpointed_gradient_norm": torch.linalg.vector_norm(full_gradient).item(),
        "autograd_directional_derivative": autograd_directional,
        "finite_difference_directional_derivatives": finite_differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""P-Flow from the supplied pseudocode: endpoint gradient, then shell projection."""

from __future__ import annotations

import math
from typing import Callable

import torch

from sampler.ode_samplers import euler_flow

ForwardOperator = Callable[[torch.Tensor], torch.Tensor]
GradientEvaluation = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def endpoint_gradient(endpoint: torch.Tensor, operator: ForwardOperator,
                      measurement: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared data misfit and identity-proxy gradient, also for complex data."""
    with torch.enable_grad():
        variable = endpoint.detach().requires_grad_(True)
        loss = 0.5 * (operator(variable) - measurement).abs().square().sum()
        gradient, = torch.autograd.grad(loss, variable)
    return loss.detach(), gradient.detach()


def proxy_gradient(model: torch.nn.Module, source: torch.Tensor,
                   operator: ForwardOperator, measurement: torch.Tensor,
                   n_steps: int = 100) -> GradientEvaluation:
    """Return endpoint, squared measurement loss, and its endpoint gradient.

    No differentiation through the flow. The gradient is placed directly in
    source coordinates, corresponding to the identity sensitivity proxy.
    """
    with torch.no_grad():
        endpoint = euler_flow(model, source, n_steps)
    loss, gradient = endpoint_gradient(endpoint, operator, measurement)
    return endpoint.detach(), loss.detach(), gradient.detach()


def project_source(source: torch.Tensor) -> torch.Tensor:
    """Project each batch member onto its radius-sqrt(d) Gaussian shell."""
    flat = source.flatten(1)
    norms = flat.norm(dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms == 0).any():
        raise FloatingPointError("Cannot project a zero or nonfinite source onto the shell")
    return (math.sqrt(flat.shape[1]) * flat / norms).reshape_as(source)


def step(model: torch.nn.Module, source: torch.Tensor, operator: ForwardOperator,
         measurement: torch.Tensor, learning_rate: float,
         n_steps: int = 100) -> tuple[torch.Tensor, GradientEvaluation]:
    """One literal P-Flow iteration: generate, proxy update, project.

    The caller chooses a fixed learning rate. There is no line search,
    gradient normalization, or L-BFGS-specific first-step scaling here.
    """
    if n_steps < 1 or learning_rate <= 0:
        raise ValueError("n_steps and learning_rate must be positive")
    evaluation = proxy_gradient(model, source, operator, measurement, n_steps)
    with torch.no_grad():
        updated = project_source(source - learning_rate * evaluation[2])
    return updated, evaluation

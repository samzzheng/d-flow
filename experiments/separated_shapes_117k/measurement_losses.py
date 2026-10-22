"""Experimental measurement weighting; no changes to the generator or DLRA."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from dlr_dflow.api import approx_source_gradient, sensitivity_factors
from experiments.ct_mri_sweep.run import Evaluator
from sampler.ode_samplers import euler_flow


def measurement_loss(residual: torch.Tensor, problem: str, kind: str,
                     sigma: float = 0.0, beta: float = 0.1,
                     intensity_weight: float = 9.0) -> torch.Tensor:
    """Sigma is in image/detector pixels; CT sums have unit-norm weights.

    MRI uses the Fourier multiplier of a spatial Gaussian on centered k-space.
    The unacquired residual is already zero. Complex subtraction precedes abs.
    """
    ordinary = 0.5 * residual.abs().square().sum(dtype=torch.float64)
    if kind == "l2" or (kind == "coarse" and sigma == 0):
        return ordinary
    if kind == "intensity":
        dc = (residual.sum(dim=-1) / math.sqrt(residual.shape[-1]) if problem == "ct"
              else residual[..., residual.shape[-2] // 2, residual.shape[-1] // 2])
        return ordinary + 0.5 * intensity_weight * dc.abs().square().sum(dtype=torch.float64)
    if kind != "coarse":
        raise ValueError(f"Unknown loss: {kind}")
    if problem == "ct":
        radius = math.ceil(4 * sigma)
        coordinate = torch.arange(-radius, radius + 1, device=residual.device, dtype=residual.dtype)
        kernel = torch.exp(-0.5 * (coordinate / sigma).square())
        kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
        coarse = functional.conv1d(residual.reshape(-1, 1, residual.shape[-1]),
                                   kernel, padding=radius)
    else:
        fy = torch.fft.fftshift(torch.fft.fftfreq(residual.shape[-2], device=residual.device,
                                                dtype=residual.real.dtype))
        fx = torch.fft.fftshift(torch.fft.fftfreq(residual.shape[-1], device=residual.device,
                                                dtype=residual.real.dtype))
        weight = torch.exp(-2 * math.pi**2 * sigma**2 * (fy[:, None].square() + fx[None, :].square()))
        coarse = weight * residual
    return beta * ordinary + 0.5 * (1 - beta) * coarse.abs().square().sum(dtype=torch.float64)


class LossEvaluator(Evaluator):
    """Use the identical weighted objective in line search and terminal gradients."""

    def loss(self, measurement: torch.Tensor) -> torch.Tensor:
        return measurement_loss(measurement - self.observed, self.operator.metadata["kind"],
                                **self.settings["measurement_loss"])

    def objective(self, source: torch.Tensor) -> float:
        self.objective_loss = self.objective_source = None
        with torch.set_grad_enabled(self.method == "D-Flow"):
            variable = source.detach().requires_grad_(self.method == "D-Flow")
            image = euler_flow(self.model, variable, self.settings["flow_steps"])
            measurement = self.operator(image)
            loss = self.loss(measurement)
            self.image, self.measurement = image.detach(), measurement.detach()
            self.value = float(loss.detach())
            if self.method == "D-Flow":
                self.objective_loss, self.objective_source = loss, variable
        self.source = source.detach().clone()
        self.counts["forward_objectives"] += 1
        return self.value

    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        self.counts["gradients"] += 1
        if method == "D-Flow":
            if self.objective_loss is None:
                self.objective(source.detach())
            gradient, = torch.autograd.grad(self.objective_loss, self.objective_source)
            self.objective_loss = self.objective_source = None
        else:
            with torch.enable_grad():
                endpoint = self.image.detach().requires_grad_(True)
                gradient, = torch.autograd.grad(self.loss(self.operator(endpoint)), endpoint)
            if method == "DLR D-Flow":
                _, factors, alpha = sensitivity_factors(
                    self.model, source.detach(), self.settings["flow_steps"],
                    **self.settings["dlr"], seed=self.seed, products=self.products,
                    seed_basis=self.previous_v)
                if alpha != 0.0:
                    raise FloatingPointError("Terminal analytical alpha must be zero")
                gradient = approx_source_gradient(factors, gradient, alpha)
                if self.settings.get("dlr_reuse_previous_v", False):
                    self.previous_v = factors[2].detach().clone()
        if not torch.isfinite(gradient).all() or not math.isfinite(self.value):
            raise FloatingPointError("Nonfinite weighted objective or gradient")
        return gradient.detach()

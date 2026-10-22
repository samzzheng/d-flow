"""Experimental terminal spectral weighting; factor evolution is unchanged."""
import math

import torch

from dlr_dflow.api import approx_source_gradient, sensitivity_factors
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.run import Evaluator, Settings
from experiments.ct_mri_sweep.operators import CT, MRI
from p_flow import endpoint_gradient


def median_capped_core(core: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap a general square core at its lower median singular value."""
    left, singular, right_t = torch.linalg.svd(core, full_matrices=False)
    return (left*singular.clamp(max=singular.median()))@right_t, singular


class SpectralEvaluator(Evaluator):
    """Compare ordinary DLR, median-cap weighting and its norm-matched control."""

    def __init__(self, model: torch.nn.Module, operator: CT | MRI, observed: torch.Tensor,
                 settings: Settings, seed: int, products: GraphedAProducts | None = None) -> None:
        super().__init__(model, operator, observed, settings, seed, products)
        self.weighting_history: list[Settings] = []

    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        variant = self.settings["terminal_weighting"]
        if variant == "unchanged" or method != "DLR D-Flow":
            return super().gradient(source, method)
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        self.counts["gradients"] += 1
        _, terminal = endpoint_gradient(self.image, self.operator, self.observed)
        _, factors, alpha = sensitivity_factors(
            self.model, source.detach(), self.settings["flow_steps"],
            **self.settings["dlr"], seed=self.seed, products=self.products,
            seed_basis=self.previous_v)
        if alpha != 0.0:
            raise FloatingPointError("Terminal analytical alpha must be exactly zero")
        left, core, right = factors
        capped, singular = median_capped_core(core)
        original = approx_source_gradient(factors, terminal, alpha)
        gradient = approx_source_gradient((left, capped, right), terminal, alpha)
        original_norm, weighted_norm = original.norm(), gradient.norm()
        scale = 1.0
        if variant == "norm_matched_cap" and weighted_norm > 0:
            scale = float(original_norm/weighted_norm)
            gradient = gradient*scale
        if not torch.isfinite(gradient).all() or not math.isfinite(self.value):
            raise FloatingPointError("Nonfinite objective or weighted gradient")
        self.weighting_history.append(dict(
            gradient_evaluation=self.counts["gradients"],
            largest_singular_value=float(singular.max()),
            median_singular_value=float(singular.median()),
            original_gradient_norm=float(original_norm), capped_gradient_norm=float(weighted_norm),
            applied_scale=scale))
        if self.settings.get("dlr_reuse_previous_v", False):
            self.previous_v = right.detach().clone()
        return gradient.detach()

"""Stable, small public API for DLR-D-Flow.

Implementation modules remain importable for research diagnostics, but new
code can use this module without knowing where the production solver and
optional graph backend are implemented.
"""

from __future__ import annotations

from .dlr_dflow import (
    approx_source_gradient,
    make_A_products,
    sensitivity_factors,
    terminal_gradient,
    velocity,
)
from .implementation.enriched import (
    direct_terminal_rank_factors,
    evolve_trace_centered_enriched,
    make_terminal_flow_products,
)
from .implementation.model_io import load_flow_model
from .implementation.optimizer_adapters import solve_inverse, solve_inverse_lbfgs
from .implementation.types import Factors
from .optimization import OptimizerFactory, objective, optimize

__all__ = [
    "Factors",
    "OptimizerFactory",
    "approx_source_gradient",
    "direct_terminal_rank_factors",
    "evolve_trace_centered_enriched",
    "load_flow_model",
    "make_A_products",
    "make_terminal_flow_products",
    "objective",
    "optimize",
    "sensitivity_factors",
    "solve_inverse",
    "solve_inverse_lbfgs",
    "terminal_gradient",
    "velocity",
]

"""DLR-D-Flow package.

Use :mod:`dlr_dflow.api` for the stable solver interface. The implementation
modules remain available for experiments and diagnostics, including optional
CUDA-graph and hand-written tangent backends.
"""

from __future__ import annotations

from importlib import import_module
from sys import modules
from typing import Any


_API_NAMES = (
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
)

_COMPAT_MODULES = {
    "dlr_dflow_diagnostics": "dlr_dflow_diagnostics",
    "enriched": "enriched",
    "fast_groupnorm": "fused_norm",
    "fused_norm": "fused_norm",
    "fused_primal": "fused_primal",
    "graphed_evolution": "graphed_evolution",
    "graphed_products": "graphed_products",
    "tangent_unet": "tangent_unet",
    "types": "types",
}

# Keep the historical ``dlr_dflow.<module>`` import paths working while the
# implementations live in one explicit subpackage. Registering the aliases
# once avoids duplicate shim files at the package root.
for _module_name, _implementation_name in _COMPAT_MODULES.items():
    _module = import_module(f".implementation.{_implementation_name}", __name__)
    globals()[_module_name] = _module
    modules[f"{__name__}.{_module_name}"] = _module


def __getattr__(name: str) -> Any:
    """Load public solver symbols only when a caller asks for them."""
    if name in _API_NAMES:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["api", *_API_NAMES]

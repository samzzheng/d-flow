"""Shared type aliases for the DLR-D-Flow implementation.

The solver and trace-centered integrator use the same factor contract:
``(U, S, V)`` with ``U`` and ``V`` shaped ``(dimension, rank)`` and ``S``
shaped ``(rank, rank)``.
"""

from __future__ import annotations

from typing import Callable, Union

import torch


# ``tuple[...]`` is available on the minimum supported Python 3.9 runtime;
# avoiding ``typing.TypeAlias`` keeps import-time compatibility there.
Factors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""An orthonormal low-rank factorization ``Y = U @ S @ V.T``."""

TensorOperator = Callable[[torch.Tensor], torch.Tensor]
"""A forward operator or a Jacobian action on tensors."""

OptimizationResult = tuple[torch.Tensor, list[dict[str, Union[float, int, bool, str]]]]
"""Selected source and per-iteration optimization metrics."""

SensitivityState = tuple[torch.Tensor, Factors, float]
EvolutionResult = Union[
    SensitivityState,
    tuple[torch.Tensor, Factors, float, list[float]],
    tuple[torch.Tensor, Factors, float, dict[str, object]],
]
"""Terminal state, optionally followed by an alpha path or diagnostics."""

"""Experiment-only rank retention; reuse the existing initialization and KSL algebra."""
from contextlib import contextmanager
from collections.abc import Iterator
from math import ceil
from typing import Any
from unittest.mock import patch

import torch

from dlr_dflow.implementation import enriched
from dlr_dflow.implementation.types import EvolutionResult, Factors


@contextmanager
def progressive_rank(floor: int, retain_fraction: float = .5) -> Iterator[None]:
    """Keep rank_max at initialization, then truncate by at most 1-fraction.

    Scoped to a single experiment process, never used concurrently by threads.
    With zero enrichment ranks cannot grow within a flow pass. Each new
    gradient evaluation starts again at rank_max. The original singular-value
    cutoff still decides whether to shrink; this policy only bounds the loss.
    """
    if floor < 1 or not 0 <= retain_fraction <= 1:
        raise ValueError("Require a positive floor and a retention fraction in [0,1]")
    original_evolve = enriched.evolve_trace_centered_enriched
    original_step = enriched._fused_enriched_step

    def evolve(*args: Any, **kwargs: Any) -> EvolutionResult:
        if (not kwargs.get("adaptive_rank") or kwargs.get("rank_min") != floor
                or kwargs.get("enrichment_probes") != 0 or kwargs["rank_max"] < floor):
            raise ValueError("Progressive ranks require matching adaptive bounds and zero enrichment")
        # The existing initializer applies rank_min after its spectral cutoff.
        # Raise that floor only here; the step wrapper restores the real floor.
        kwargs["rank_min"] = kwargs["rank_max"]
        return original_evolve(*args, **kwargs)

    def step(*args: Any, **kwargs: Any) -> tuple[Factors, torch.Tensor, float | torch.Tensor]:
        rank = args[2][1].shape[0]
        kwargs["rank_min"] = max(floor, ceil(retain_fraction * rank))
        return original_step(*args, **kwargs)

    with patch.object(enriched, "evolve_trace_centered_enriched", evolve), \
            patch.object(enriched, "_fused_enriched_step", step):
        yield

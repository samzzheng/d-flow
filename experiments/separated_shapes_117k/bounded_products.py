"""Bound graph memory for experimental, continuously changing adaptive ranks."""
from typing import Any

import torch

from dlr_dflow.implementation.graphed_products import GraphedAProducts


class BoundedGraphedProducts(GraphedAProducts):
    """Evict least-recently-used product graphs without changing product widths."""

    def __init__(self, model: torch.nn.Module, x_like: torch.Tensor,
                 cache_size: int = 8, **kwargs: Any) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        super().__init__(model, x_like, **kwargs)
        self.cache_size = cache_size

    def _capture(self, kind: str, width: int) -> None:
        if len(self._graphs) >= self.cache_size:
            # Returned actions own their buffers. Finish prior replays before
            # releasing the old graph and its private memory pool.
            torch.cuda.synchronize(self.x_buf.device)
            del self._graphs[next(iter(self._graphs))]
        super()._capture(kind, width)

    def _run_block(self, kind: str, values: torch.Tensor) -> torch.Tensor:
        result = super()._run_block(kind, values)
        key = (kind, values.shape[1])
        self._graphs[key] = self._graphs.pop(key)
        return result

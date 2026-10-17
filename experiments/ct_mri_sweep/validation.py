"""Memory-bounded eager products for high-rank backend validation only."""
from __future__ import annotations

import torch

from dlr_dflow.api import make_A_products


class ChunkedEagerProducts:
    """Keep the eager factor evolution while bounding derivative batch width."""

    def __init__(self, model: torch.nn.Module, columns: int = 8) -> None:
        self.model, self.columns = model, columns

    def set_state(self, time: float, image: torch.Tensor) -> None:
        self.forward, self.transpose = make_A_products(self.model, time, image)

    def A(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(map(self.forward, values.split(self.columns, dim=1))), dim=1)

    def AT(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(map(self.transpose, values.split(self.columns, dim=1))), dim=1)

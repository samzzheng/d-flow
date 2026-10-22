"""Memory-bounded eager products for high-rank backend validation only."""
from __future__ import annotations

import torch

from dlr_dflow.api import make_A_products


class ChunkedAutodiffProducts:
    """Independent PyTorch derivatives of the native model, without hand tapes."""

    def __init__(self, model: torch.nn.Module, columns: int = 4) -> None:
        self.model, self.columns = model, columns

    def set_state(self, time: float, image: torch.Tensor) -> None:
        self.image = image.detach()
        self.time = image.new_tensor([time])
        _, self.pullback = torch.func.vjp(self.velocity, self.image)

    def velocity(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image, self.time)

    def A(self, values: torch.Tensor) -> torch.Tensor:
        actions = []
        for block in values.split(self.columns, dim=1):
            directions = block.T.reshape(-1, *self.image.shape)
            _, derivative = torch.func.vmap(
                lambda direction: torch.func.jvp(self.velocity, (self.image,), (direction,)),
                out_dims=(None, 0))(directions)
            actions.append(derivative.reshape(block.shape[1], -1).T)
        return torch.cat(actions, dim=1)

    def AT(self, values: torch.Tensor) -> torch.Tensor:
        actions = []
        for block in values.split(self.columns, dim=1):
            cotangents = block.T.reshape(-1, *self.image.shape)
            derivative = torch.func.vmap(lambda cotangent: self.pullback(cotangent)[0])(cotangents)
            actions.append(derivative.reshape(block.shape[1], -1).T)
        return torch.cat(actions, dim=1)


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

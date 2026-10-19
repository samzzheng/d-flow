"""Execution-only acceleration of the sweep's unchanged LBFGS recursion."""
from __future__ import annotations

import torch

from p_flow.armijo import ArmijoLBFGS, lbfgs_direction


class GraphedDirection:
    """Replay the same operations, with fresh history values on every call."""

    def __init__(self, gradient: torch.Tensor, displacements: list[torch.Tensor],
                 changes: list[torch.Tensor]) -> None:
        self.gradient = gradient.clone()
        self.displacements, self.changes = torch.stack(displacements), torch.stack(changes)
        s, y = list(self.displacements.unbind()), list(self.changes.unbind())
        stream = torch.cuda.Stream(device=gradient.device)
        stream.wait_stream(torch.cuda.current_stream(gradient.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                lbfgs_direction(self.gradient, s, y)
        torch.cuda.current_stream(gradient.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.result = lbfgs_direction(self.gradient, s, y)

    def __call__(self, gradient: torch.Tensor, displacements: list[torch.Tensor],
                 changes: list[torch.Tensor]) -> torch.Tensor:
        self.gradient.copy_(gradient)
        torch.stack(displacements, out=self.displacements)
        torch.stack(changes, out=self.changes)
        self.graph.replay()
        return self.result


class SweepLBFGS(ArmijoLBFGS):
    """Capture only a full history; growing or reset histories remain eager.

    This avoids capturing a different graph for every growing history length.
    The graph is disposable runtime state and must not be checkpointed.
    """

    def _direction(self) -> torch.Tensor:
        if not self.g.is_cuda or len(self.S) != self.history_size or self.history_size < 8:
            return super()._direction()
        if getattr(self, "_direction_graph", None) is None:
            self._direction_graph = GraphedDirection(self.g, self.S, self.Y)
        return self._direction_graph(self.g, self.S, self.Y)

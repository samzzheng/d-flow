"""Replay native FM microbatch forward/backward without changing the optimizer."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class GraphedMicrobatches:
    """Replay native BF16 kernels; the caller must retain gradient storage.

    Use ``zero_grad(set_to_none=False)`` and keep optimizer/EMA updates outside
    capture. Graphs are rebuilt lazily after checkpoint resume.
    """

    def __init__(self, model: nn.Module,
                 loss_fn: Callable[[nn.Module, Tensor, str, float], Tensor],
                 device: str, t_alpha: float = 1.0) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.t_alpha = t_alpha
        self.graphs: dict[tuple[int, ...], tuple[torch.cuda.CUDAGraph, Tensor, Tensor, Tensor]] = {}
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)

    def __call__(self, images: Tensor, micro_batch: int) -> Tensor:
        total = len(images)
        result = torch.zeros((), device=images.device, dtype=torch.float64)
        for chunk in images.split(micro_batch):
            shape = tuple(chunk.shape)
            if shape not in self.graphs:
                # Capture warmup must not consume training draws or gradients,
                # including when a partial chunk introduces a new shape.
                rng = torch.cuda.get_rng_state(self.device)
                saved_gradients = [p.grad.clone() for p in self.model.parameters()]
                static = torch.empty_like(chunk).copy_(chunk)
                weight = torch.tensor(len(chunk) / total, device=chunk.device)
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                            loss = self.loss_fn(self.model, static, self.device, self.t_alpha)
                            (loss * weight).backward()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                del loss
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        loss = self.loss_fn(self.model, static, self.device, self.t_alpha)
                        (loss * weight).backward()
                for parameter, saved in zip(self.model.parameters(), saved_gradients):
                    parameter.grad.copy_(saved)
                torch.cuda.set_rng_state(rng, self.device)
                self.graphs[shape] = graph, static, weight, loss.detach()
            graph, static, weight, loss = self.graphs[shape]
            static.copy_(chunk)
            weight.fill_(len(chunk) / total)
            graph.replay()
            result.add_(loss.detach().to(torch.float64) * (len(chunk) / total))
        return result

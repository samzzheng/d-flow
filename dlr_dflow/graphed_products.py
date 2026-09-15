"""CUDA-graph replay for the velocity Jacobian products A_t Q and A_t^T Q.

Profiling (2026-09-05, 64px, rank 8, 100 steps) showed the enriched evolution
is CPU-dispatch-bound: ~9,400 kernel launches per flow step from the
torch.func jvp/vjp decompositions, with the GPU busy only ~15% of the wall
time.  All product shapes are static across steps, so each (kind, width)
call is captured once as a CUDA graph and replayed; the flow state x_t and
time t live in persistent buffers that are copied into place per step.
Numerics are identical to make_A_products (same kernels, same order).
"""
from __future__ import annotations

import torch


class GraphedAProducts:
    """Drop-in (A, AT) provider with CUDA-graph replay per input width."""

    def __init__(self, model: torch.nn.Module, x_like: torch.Tensor):
        H, W = x_like.shape[-2:]
        self.h, self.w, self.d = H, W, H * W
        self.x_buf = torch.zeros(1, 1, H, W, device=x_like.device,
                                 dtype=x_like.dtype)
        self.t_buf = torch.zeros(1, device=x_like.device, dtype=x_like.dtype)
        self.model = model
        self._graphs = {}

        def f(X):
            return model(X, self.t_buf)

        def lanes(Q):
            return Q.T.reshape(-1, 1, 1, H, W)

        def A_eager(Q):
            out = torch.func.vmap(
                lambda tangent: torch.func.jvp(f, (self.x_buf,), (tangent,))[1]
            )(lanes(Q))
            return out.reshape(Q.shape[1], self.d).T

        def AT_eager(Q):
            _, vjp_fn = torch.func.vjp(f, self.x_buf)
            out = torch.func.vmap(lambda cot: vjp_fn(cot)[0])(lanes(Q))
            return out.reshape(Q.shape[1], self.d).T

        self._eager = {"A": A_eager, "AT": AT_eager}

    def set_state(self, t: float, x_img: torch.Tensor):
        self.t_buf.fill_(float(t))
        self.x_buf.copy_(x_img.reshape(self.x_buf.shape))

    def _capture(self, kind: str, width: int):
        # flush pending CUDAGraph destructions (e.g. a previous solve's
        # products being garbage-collected) before starting a capture:
        # graph teardown during an active capture invalidates it
        import gc
        gc.collect()
        torch.cuda.synchronize(self.x_buf.device)
        fn = self._eager[kind]
        # capture must happen with the buffers' device current: a capture
        # begun on another device's stream is invalidated by the first
        # cross-device kernel (cudaErrorStreamCaptureUnsupported)
        with torch.cuda.device(self.x_buf.device):
            q_in = torch.zeros(self.d, width, device=self.x_buf.device,
                               dtype=self.x_buf.dtype)
            side = torch.cuda.Stream(self.x_buf.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    fn(q_in)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                q_out = fn(q_in)
        self._graphs[(kind, width)] = (graph, q_in, q_out)

    def _run(self, kind: str, Q: torch.Tensor) -> torch.Tensor:
        key = (kind, Q.shape[1])
        if key not in self._graphs:
            self._capture(kind, Q.shape[1])
        graph, q_in, q_out = self._graphs[key]
        with torch.cuda.device(self.x_buf.device):
            q_in.copy_(Q)
            graph.replay()
            return q_out.clone()

    def A(self, Q: torch.Tensor) -> torch.Tensor:
        return self._run("A", Q)

    def AT(self, Q: torch.Tensor) -> torch.Tensor:
        return self._run("AT", Q)

"""CUDA-graph replay for the velocity Jacobian products A_t Q and A_t^T Q.

Profiling (2026-09-05, 64px, rank 8, 100 steps) showed the enriched evolution
is CPU-dispatch-bound: ~9,400 kernel launches per flow step from the
torch.func jvp/vjp decompositions, with the GPU busy only ~15% of the wall
time.  All product shapes are static across steps, so each (kind, width)
call is captured once as a CUDA graph and replayed; the flow state x_t and
time t live in persistent buffers that are copied into place per step.
The products represent the same operators as make_A_products; shared primal
intermediates and reordered contractions can introduce floating-point changes.
"""
from __future__ import annotations

from numbers import Integral

import torch


class GraphedAProducts:
    """Drop-in (A, AT) provider with CUDA-graph replay per input width.

    Use a fixed evaluation-mode model, image shape, device, dtype and derivative
    configuration for the provider's lifetime. Before launching products on
    multiple streams, call ``A.prepare_linearization()`` on their producer
    stream and make the derivative streams wait for it. The parallel integrator
    handles this dependency automatically.

    Optional ``max_jvp_columns`` and ``max_vjp_columns`` bound the widths of
    ordinary A/AT product graphs by evaluating columns in successive blocks.
    They leave the complete-step graph backend unchanged. Unset limits retain
    the original single-graph call for each input width.
    ``max_step_jvp_columns`` and ``max_step_vjp_columns`` independently bound
    derivative batches inside a complete-step graph, reusing its primal tape.

    ``checkpoint_blocks=True`` recomputes each UNet block's local tape during
    JVP/VJP actions. It applies to ordinary product and complete-step graphs;
    choose the setting before constructing this provider and keep it fixed.
    """

    def __init__(self, model: torch.nn.Module, x_like: torch.Tensor,
                 share_linearization: bool | None = None,
                 full_step_graphs: bool = True,
                 max_jvp_columns: int | None = None,
                 max_vjp_columns: int | None = None,
                 checkpoint_blocks: bool = False,
                 max_step_jvp_columns: int | None = None,
                 max_step_vjp_columns: int | None = None):
        from .. import dlr_dflow as core
        from . import tangent_unet
        self._gn_action_backend = getattr(tangent_unet, "GN_ACTION_BACKEND", "native")
        self._gn_primal_backend = tangent_unet.GN_PRIMAL_BACKEND
        for name, value in (("max_jvp_columns", max_jvp_columns),
                            ("max_vjp_columns", max_vjp_columns),
                            ("max_step_jvp_columns", max_step_jvp_columns),
                            ("max_step_vjp_columns", max_step_vjp_columns)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
            setattr(self, name, None if value is None else int(value))
        self._step_column_limits = (self.max_step_jvp_columns, self.max_step_vjp_columns)
        H, W = x_like.shape[-2:]
        self.h, self.w, self.d = H, W, H * W
        self.x_buf = torch.zeros(1, 1, H, W, device=x_like.device,
                                 dtype=x_like.dtype)
        self.t_buf = torch.zeros(1, device=x_like.device, dtype=x_like.dtype)
        self.model = model
        self._graphs = {}
        self.share_linearization = (
            core.SHARE_LINEARIZATION if share_linearization is None else share_linearization
        ) and core.HAND_JVP and core.HAND_VJP
        if checkpoint_blocks and not self.share_linearization:
            raise ValueError("checkpoint_blocks requires shared hand-written JVP/VJP products")
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self._checkpoint_blocks = self.checkpoint_blocks
        self._linearization = None
        self._primal_graph = None
        self._state_dirty = True
        self.full_step_graphs = full_step_graphs and self.share_linearization
        self._step_graphs = {}

        def f(X):
            return model(X, self.t_buf)

        def lanes(Q):
            return Q.T.reshape(-1, 1, 1, H, W)

        def A_eager(Q):
            from .. import dlr_dflow as core
            if self.share_linearization:
                out = self._linearization.jvp(Q.T.reshape(-1, 1, H, W))
                return out.reshape(Q.shape[1], self.d).T, self._linearization.primal
            if core.HAND_JVP:
                from .tangent_unet import unet_jvp
                primal, out = unet_jvp(model, self.t_buf, self.x_buf,
                                       Q.T.reshape(-1, 1, H, W))
                return out.reshape(Q.shape[1], self.d).T, primal
            primal, out = torch.func.vmap(
                lambda tangent: torch.func.jvp(f, (self.x_buf,), (tangent,)),
                out_dims=(None, 0))(lanes(Q))
            return out.reshape(Q.shape[1], self.d).T, primal

        def AT_eager(Q):
            from .. import dlr_dflow as core
            if self.share_linearization:
                out = self._linearization.vjp(Q.T.reshape(-1, 1, H, W))
                return out.reshape(Q.shape[1], self.d).T
            if core.HAND_VJP:
                from .tangent_unet import unet_vjp
                _, out = unet_vjp(model, self.t_buf, self.x_buf,
                                  Q.T.reshape(-1, 1, H, W))
                return out.reshape(Q.shape[1], self.d).T
            _, vjp_fn = torch.func.vjp(f, self.x_buf)
            out = torch.func.vmap(lambda cot: vjp_fn(cot)[0])(lanes(Q))
            return out.reshape(Q.shape[1], self.d).T

        self._eager = {"A": A_eager, "AT": AT_eager}
        # plain closures (not bound methods) so A.last_primal can be set,
        # matching make_A_products' interface for the shared state advance
        self.A = lambda Q: self._run("A", Q)
        self.AT = lambda Q: self._run("AT", Q)
        self.A.prepare_linearization = self.prepare_linearization

    def prepare_linearization(self):
        """Produce the shared primal before derivative streams fork."""
        self._check_configuration()
        if self.share_linearization:
            self._ensure_linearization()

    def _check_configuration(self):
        from . import tangent_unet
        if self.checkpoint_blocks != self._checkpoint_blocks:
            raise ValueError("checkpoint_blocks changed; construct a new product provider")
        if getattr(tangent_unet, "GN_ACTION_BACKEND", "native") != self._gn_action_backend:
            raise ValueError("GN_ACTION_BACKEND changed; construct a new product provider")
        if tangent_unet.GN_PRIMAL_BACKEND != self._gn_primal_backend:
            raise ValueError("GN_PRIMAL_BACKEND changed; construct a new product provider")
        if (self.max_step_jvp_columns, self.max_step_vjp_columns) != self._step_column_limits:
            raise ValueError("step column limits changed; construct a new product provider")

    def set_state(self, t: float, x_img: torch.Tensor):
        self._check_configuration()
        self.t_buf.fill_(float(t))
        self.x_buf.copy_(x_img.reshape(self.x_buf.shape))
        self._state_dirty = True

    def prepare_step_graph(self, x, factors, sensitivities, probes, alpha, step_size):
        """Reuse a complete p=0 step graph across source-gradient evaluations."""
        from . import enriched
        from .graphed_evolution import GraphedZeroEnrichmentStep
        self._check_configuration()
        flags = tuple(getattr(enriched, name) for name in GraphedZeroEnrichmentStep._FLAG_NAMES)
        key = (factors[0].shape[1], probes.shape[1], float(step_size), flags,
               self.checkpoint_blocks, self._gn_action_backend, self._gn_primal_backend,
               self._step_column_limits)
        if key not in self._step_graphs:
            self._step_graphs[key] = GraphedZeroEnrichmentStep(
                self.model, x, factors, sensitivities, probes, alpha, step_size,
                checkpoint_blocks=self.checkpoint_blocks,
                max_step_jvp_columns=self.max_step_jvp_columns,
                max_step_vjp_columns=self.max_step_vjp_columns)
        else:
            self._step_graphs[key].reset(x, factors, sensitivities, alpha, probes)
        return self._step_graphs[key]

    def _capture_linearization(self):
        """Capture one primal sweep whose buffers feed every product width."""
        import gc
        from .tangent_unet import unet_linearize
        gc.collect()
        torch.cuda.synchronize(self.x_buf.device)
        with torch.cuda.device(self.x_buf.device):
            side = torch.cuda.Stream(self.x_buf.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    unet_linearize(self.model, self.t_buf, self.x_buf,
                                   checkpoint_blocks=self.checkpoint_blocks)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tape = unet_linearize(self.model, self.t_buf, self.x_buf,
                                      checkpoint_blocks=self.checkpoint_blocks)
        self._linearization = tape
        self._primal_graph = graph

    def _ensure_linearization(self):
        if self._primal_graph is None:
            self._capture_linearization()
        if self._state_dirty:
            with torch.cuda.device(self.x_buf.device):
                self._primal_graph.replay()
            self._state_dirty = False

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
        self._check_configuration()
        limit = self.max_jvp_columns if kind == "A" else self.max_vjp_columns
        if limit is None or Q.shape[1] <= limit:
            return self._run_block(kind, Q)
        # Each block is an owned clone, so replaying a reusable graph for a
        # later block cannot overwrite an earlier result before concatenation.
        return torch.cat([self._run_block(kind, block)
                          for block in Q.split(limit, dim=1)], dim=1)

    def _run_block(self, kind: str, Q: torch.Tensor) -> torch.Tensor:
        if self.share_linearization:
            self._ensure_linearization()
        key = (kind, Q.shape[1])
        if key not in self._graphs:
            self._capture(kind, Q.shape[1])
        graph, q_in, q_out = self._graphs[key]
        with torch.cuda.device(self.x_buf.device):
            q_in.copy_(Q)
            graph.replay()
            if kind == "A":
                out, primal = q_out
                self.A.last_primal = primal.clone()
                return out.clone()
            return q_out.clone()

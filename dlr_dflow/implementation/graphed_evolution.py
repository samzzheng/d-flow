"""Optional capture of a complete fixed-rank, zero-enrichment DLRA step.

Initialization remains the enriched solver's first-step randomized sketch.
This backend starts from the resulting state and advances one Euler step per
``run(t)`` call. The graphed product provider selects this backend only for
fixed-rank, zero-enrichment, stride-one evolution with the default scalar rule.
"""

from __future__ import annotations

import gc
from numbers import Integral

import torch

from . import enriched, tangent_unet
from .tangent_unet import unet_linearize


class GraphedZeroEnrichmentStep:
    """Replay one complete sensitivity/state step using one primal tape.

    All factor and probe widths are fixed for the object's lifetime. The model
    must remain in evaluation mode with unchanged weights and tangent-kernel
    settings. Construct a new provider when changing derivative settings.
    ``run(t)`` mutates owned buffers; use ``snapshot()`` to obtain independent output tensors.
    ``reset`` starts a new trajectory while reusing a previously captured graph.

    ``capture=False`` executes exactly the same operations eagerly, including
    on CPU, for verification. No custom alpha rate, adaptive rank, factor stride
    or enrichment is supported by this specialized backend.
    ``checkpoint_blocks`` selects local UNet recomputation for the entire
    backend lifetime; it does not checkpoint the sensitivity trajectory.
    Optional ``max_step_jvp_columns``/``max_step_vjp_columns`` partition the
    derivative columns while retaining a single shared primal tape per step.
    """

    _FLAG_NAMES = ("SKIP_UNTRUNCATED_SVD", "FUSE_K_IMAGE",
                   "M_FROM_TRANSPOSE", "RECOMPRESS_FROM_TRANSPOSE",
                   "SPECIALIZE_ZERO_ENRICHMENT")

    @torch.no_grad()
    def __init__(self, model, x, factors, probe_sensitivities, probes,
                 alpha, step_size: float, *, capture: bool = True,
                 checkpoint_blocks: bool = False,
                 max_step_jvp_columns: int | None = None,
                 max_step_vjp_columns: int | None = None):
        for name, value in (("max_step_jvp_columns", max_step_jvp_columns),
                            ("max_step_vjp_columns", max_step_vjp_columns)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
            setattr(self, name, None if value is None else int(value))
        self._step_column_limits = (self.max_step_jvp_columns, self.max_step_vjp_columns)
        if model.training:
            raise ValueError("graphed evolution requires model.eval()")
        if capture and x.device.type != "cuda":
            raise ValueError("CUDA capture requires CUDA tensors")
        if not enriched.SKIP_UNTRUNCATED_SVD:
            raise ValueError("graphed evolution requires SKIP_UNTRUNCATED_SVD=True")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if x.ndim != 4 or x.shape[:2] != (1, 1):
            raise ValueError("expected a single-channel image of shape (1, 1, H, W)")
        self.model = model
        self.step_size = float(step_size)
        self.capture = capture
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self._checkpoint_blocks = self.checkpoint_blocks
        self._gn_action_backend = getattr(tangent_unet, "GN_ACTION_BACKEND", "native")
        self._gn_primal_backend = tangent_unet.GN_PRIMAL_BACKEND
        self._flags = tuple(getattr(enriched, name) for name in self._FLAG_NAMES)
        self.x = x.detach().clone()
        self.factors = tuple(value.detach().clone() for value in factors)
        self.sensitivities = probe_sensitivities.detach().clone()
        self.probes = probes.detach().clone()
        self.alpha = torch.as_tensor(alpha, device=x.device, dtype=x.dtype).reshape(()).clone()
        self.time = x.new_zeros(1)
        self.graph = None
        self._validate_state()

    def _validate_state(self):
        u, s, v = self.factors
        dimension = self.x.numel()
        if u.ndim != 2 or u.shape[0] != dimension or u.shape[1] < 1:
            raise ValueError("left factor must have shape (image dimension, positive rank)")
        rank = u.shape[1]
        if v.shape != u.shape or s.shape != (rank, rank):
            raise ValueError("inconsistent low-rank factor shapes")
        if (self.probes.ndim != 2 or self.probes.shape[0] != dimension
                or self.sensitivities.shape != self.probes.shape):
            raise ValueError("probes and sensitivities must have matching (dimension, count) shapes")
        for value in (*self.factors, self.probes, self.sensitivities):
            if value.device != self.x.device or value.dtype != self.x.dtype:
                raise ValueError("all state tensors must share a device and dtype")

    def _check_configuration(self):
        if (self.max_step_jvp_columns, self.max_step_vjp_columns) != self._step_column_limits:
            raise ValueError("step column limits changed; construct a new graph backend")
        if self.checkpoint_blocks != self._checkpoint_blocks:
            raise ValueError("checkpoint_blocks changed; construct a new graph backend")
        if getattr(tangent_unet, "GN_ACTION_BACKEND", "native") != self._gn_action_backend:
            raise ValueError("GN_ACTION_BACKEND changed; construct a new graph backend")
        if tangent_unet.GN_PRIMAL_BACKEND != self._gn_primal_backend:
            raise ValueError("GN_PRIMAL_BACKEND changed; construct a new graph backend")
        if self.model.training:
            raise ValueError("graphed evolution requires model.eval()")
        if tuple(getattr(enriched, name) for name in self._FLAG_NAMES) != self._flags:
            raise ValueError("enriched algebra flags changed; construct a new graph backend")

    @torch.no_grad()
    def reset(self, x, factors, probe_sensitivities, alpha, probes=None):
        """Replace initial conditions without invalidating a captured graph."""
        if len(factors) != 3:
            raise ValueError("factors must contain exactly three tensors (U, S, V)")
        values = (x, *factors, probe_sensitivities)
        buffers = (self.x, *self.factors, self.sensitivities)
        if probes is not None:
            values += (probes,)
            buffers += (self.probes,)
        # Validate everything before mutating a buffer.
        for value, buffer in zip(values, buffers):
            if (value.shape != buffer.shape or value.device != buffer.device
                    or value.dtype != buffer.dtype):
                raise ValueError("reset must preserve every buffer's shape, device and dtype")
        alpha_value = torch.as_tensor(alpha, device=self.x.device, dtype=self.x.dtype)
        if alpha_value.numel() != 1:
            raise ValueError("alpha must be scalar")
        for value, buffer in zip(values, buffers):
            buffer.copy_(value)
        self.alpha.copy_(alpha_value.reshape(()))

    @torch.no_grad()
    def snapshot(self, *, canonicalize: bool = True):
        """Return independent (x, factors, probe sensitivities, alpha) tensors."""
        factors = self.factors
        if canonicalize:
            factors = enriched._canonicalize_factors(factors)
        return (self.x.clone(), tuple(value.clone() for value in factors),
                self.sensitivities.clone(), self.alpha.clone())

    @torch.no_grad()
    def _advance(self):
        height, width = self.x.shape[-2:]
        dimension = self.x.numel()
        tape = unet_linearize(self.model, self.time, self.x,
                              checkpoint_blocks=self.checkpoint_blocks)

        def apply_product(action, q, limit):
            def apply_block(block):
                return action(block.T.reshape(-1, 1, height, width)).reshape(
                    block.shape[1], dimension).T
            if limit is None or q.shape[1] <= limit:
                return apply_block(q)
            # Each eager action owns its result; the enclosing CUDA graph
            # captures every chunk once. All chunks read this step's same tape.
            return torch.cat([apply_block(block) for block in q.split(limit, dim=1)], dim=1)

        def a(q):
            return apply_product(tape.jvp, q, self.max_step_jvp_columns)

        def at(q):
            return apply_product(tape.vjp, q, self.max_step_vjp_columns)

        factors, derivative, _ = enriched._fused_enriched_step(
            a, at, self.factors, self.sensitivities, self.probes,
            self.step_size, self.alpha, 0, None)
        sensitivities = self.sensitivities + self.step_size * derivative
        if self.probes.shape[1]:
            alpha = (self.probes * sensitivities).sum() / (self.probes.shape[1] * dimension)
        else:
            alpha = self.alpha - self.step_size   # analytic alpha_t = 1 - t
        x = self.x + self.step_size * tape.primal
        # Read all old-state quantities before writing any persistent buffer.
        for buffer, value in zip(self.factors, factors):
            buffer.copy_(value)
        self.sensitivities.copy_(sensitivities)
        self.alpha.copy_(alpha)
        self.x.copy_(x)

    def _capture(self):
        saved = self.snapshot(canonicalize=False)
        gc.collect()
        torch.cuda.synchronize(self.x.device)
        with torch.cuda.device(self.x.device):
            side = torch.cuda.Stream(self.x.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._advance()
                    self.reset(*saved)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._advance()
            # Capture executes the step once. Restore its inputs before the
            # first replay so warmup/capture cannot advance the trajectory.
            self.reset(*saved)
        self.graph = graph

    @torch.no_grad()
    def run(self, t: float):
        """Advance once at exactly the caller's time, retaining owned buffers."""
        self._check_configuration()
        self.time.fill_(float(t))
        if not self.capture:
            self._advance()
            return
        if self.graph is None:
            self._capture()
        with torch.cuda.device(self.x.device):
            self.graph.replay()

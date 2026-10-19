"""Experimental compiler backend retaining native GroupNorm arithmetic."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any
import torch
from torch._inductor.compile_fx import compile_fx
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.lowering import make_fallback, require_contiguous


def compile_native_norm(graph: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Callable[..., Any]:
    decompositions = dict(select_decomp_table())
    for op in (torch.ops.aten.native_group_norm.default, torch.ops.aten.native_group_norm_backward.default):
        decompositions.pop(op, None)
        make_fallback(op, layout_constraint=require_contiguous, override_decomp=True)
    return compile_fx(graph, inputs, decompositions=decompositions,
                      config_patches={'emulate_precision_casts': True})

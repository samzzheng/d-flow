"""Exact tangent/adjoint checks against PyTorch on small, nontrivial networks.

Run with: python -m unittest discover -s tests -p 'test_tangent_products.py'
"""
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import dlr_dflow.tangent_unet as tangent
from models.unet.default import Unet
from models.unet.unet import AttentionBlock


class _CastEmbedding(nn.Module):
    """The model's sinusoidal embedding always returns float32."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, value):
        return value.to(self.dtype)


class TangentProductsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(1701)

    def assert_close(self, actual, expected):
        if actual.dtype == torch.float64:
            torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=4e-6)

    @staticmethod
    def reference_products(fn, x, directions):
        with sdpa_kernel(SDPBackend.MATH):
            primal, pullback = torch.func.vjp(fn, x)
            jvp = torch.cat([
                torch.func.jvp(fn, (x,), (lane[None],))[1]
                for lane in directions
            ])
            vjp = torch.cat([pullback(lane[None])[0] for lane in directions])
        return primal, jvp, vjp

    def test_group_norm_with_zero_and_nonuniform_weights(self):
        for dtype in (torch.float32, torch.float64):
            m = nn.GroupNorm(2, 8).to(dtype).eval()
            with torch.no_grad():
                m.weight.copy_(torch.linspace(-0.8, 1.3, 8, dtype=dtype))
                m.weight[2] = 0
                m.bias.normal_(0, 0.2)
            x = torch.randn(1, 8, 3, 5, dtype=dtype)
            directions = torch.randn(3, 8, 3, 5, dtype=dtype)
            primal, jvp, vjp = self.reference_products(m, x, directions)
            configurations = ((False, "manual"), (True, "manual"), (True, "native"))
            for fused, stats_mode in configurations:
                with self.subTest(dtype=dtype, fused=fused, stats_mode=stats_mode), \
                        patch.object(tangent, "FUSED_GN", fused), \
                        patch.object(tangent, "GN_STATS_MODE", stats_mode):
                    p, d = tangent._gn(m, x, directions)
                    self.assert_close(p, primal)
                    self.assert_close(d, jvp)
                    self.assert_close(tangent._gn_T(m, x, directions), vjp)

    def test_attention_multiple_query_chunks(self):
        for dtype in (torch.float32, torch.float64):
            m = AttentionBlock(8, num_heads=2, groups=2).to(dtype).eval()
            x = torch.randn(1, 8, 3, 5, dtype=dtype)
            directions = torch.randn(3, 8, 3, 5, dtype=dtype)
            primal, jvp, vjp = self.reference_products(m, x, directions)
            # Force eight query chunks, including a short final chunk.
            score_bytes = directions.shape[0] * m.num_heads * 15 * x.element_size() * 2
            for mode in ("moments", "chunked"):
                with self.subTest(dtype=dtype, mode=mode), patch.object(tangent, "ATTENTION_MODE", mode), patch.object(tangent, "ATTENTION_SCORE_BYTES", score_bytes):
                    p, d = tangent._attn(m, x, directions)
                    pv, tape = tangent._attn_fwd(m, x)
                    self.assert_close(p, primal)
                    self.assert_close(pv, primal)
                    self.assert_close(d, jvp)
                    self.assert_close(tangent._attn_T(m, tape, directions), vjp)

    def test_unet_odd_resolution_and_both_conditioning_modes(self):
        for dtype in (torch.float32, torch.float64):
            for scale_shift in (False, True):
                model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, 1],
                             groups=2, dropout=0.1, scale_shift=scale_shift).to(dtype).eval()
                model.time_proj = nn.Sequential(_CastEmbedding(dtype), model.time_proj)
                # Include learned, nonuniform normalization parameters and a
                # zero channel weight so a divide-by-weight implementation fails.
                with torch.no_grad():
                    for module in model.modules():
                        if isinstance(module, nn.GroupNorm):
                            module.weight.uniform_(0.4, 1.4)
                            module.weight[0] = 0
                            module.bias.normal_(0, 0.1)
                x = torch.randn(1, 1, 5, 7, dtype=dtype)
                directions = torch.randn(2, 1, 5, 7, dtype=dtype)
                time = torch.tensor([0.37], dtype=dtype)
                primal, jvp, vjp = self.reference_products(lambda value: model(value, time), x, directions)
                for mode in ("moments", "chunked"):
                    with self.subTest(dtype=dtype, scale_shift=scale_shift, mode=mode), patch.object(tangent, "ATTENTION_MODE", mode):
                        p, d = tangent.unet_jvp(model, time, x, directions)
                        pv, adjoint = tangent.unet_vjp(model, time, x, directions)
                        self.assert_close(p, primal)
                        self.assert_close(pv, primal)
                        self.assert_close(d, jvp)
                        self.assert_close(adjoint, vjp)


if __name__ == "__main__":
    unittest.main()

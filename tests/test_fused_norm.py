"""Exact GN derivative formula; optional CUDA tests never launch on CPU-only runs."""

import unittest
from itertools import product
from unittest.mock import patch

import torch
from torch import nn

from dlr_dflow.fused_norm import group_norm_action
from dlr_dflow import tangent_unet


class FusedNormCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_formula_matches_native_derivatives_and_adjoint(self):
        torch.manual_seed(279)
        for dtype in (torch.float32, torch.float64):
            for shape in ((1, 8, 5, 7), (1, 8, 17, 19)):
                for channels_last in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, channels_last=channels_last):
                        p = torch.randn(shape, dtype=dtype)
                        y = torch.randn((3, *shape[1:]), dtype=dtype)
                        cotangent = torch.randn_like(y)
                        coefficient = torch.randn_like(p)
                        coefficient[:, 0] = 0
                        if channels_last:
                            p = p.contiguous(memory_format=torch.channels_last)
                            y = y.contiguous(memory_format=torch.channels_last)
                        module = nn.GroupNorm(2, 8, affine=False).to(dtype)
                        _, mean, rstd = torch.ops.aten.native_group_norm(
                            p, None, None, 1, 8, shape[-1] * shape[-2], 2, module.eps)
                        jvp = group_norm_action(p, y, mean, rstd, 2, coefficient)
                        vjp = group_norm_action(p, cotangent, mean, rstd, 2, coefficient, True)
                        # The installed PyTorch GN forward-AD decomposition
                        # requires contiguous input even when the primal
                        # operator supports channels-last. Compare the same
                        # mathematical tensors in its supported layout.
                        reference_p = p.contiguous()
                        expected_jvp = torch.cat([torch.func.jvp(module, (reference_p,), (lane[None].contiguous(),))[1]
                                                  for lane in y]) * coefficient
                        _, pullback = torch.func.vjp(module, reference_p)
                        expected_vjp = torch.cat([pullback(lane[None] * coefficient)[0]
                                                  for lane in cotangent])
                        tolerance = dict(rtol=4e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=3e-11, atol=3e-12)
                        torch.testing.assert_close(jvp, expected_jvp, **tolerance)
                        torch.testing.assert_close(vjp, expected_vjp, **tolerance)
                        torch.testing.assert_close((jvp * cotangent).sum(), (y * vjp).sum(), **tolerance)

    def test_cpu_fallback_does_not_import_cuda_backend(self):
        p = torch.randn(1, 4, 3, 5, dtype=torch.float64)
        y = torch.randn(2, 4, 3, 5, dtype=torch.float64)
        mean, rstd, _ = tangent_unet._gn_stats(nn.GroupNorm(2, 4).double(), p)
        with patch("dlr_dflow.fused_norm._cuda_kernels", side_effect=AssertionError("loaded CUDA")):
            actual = group_norm_action(p, y, mean, rstd, 2)
            expected = group_norm_action(p, y, mean, rstd, 2, backend="torch")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "CUDA float32"):
            group_norm_action(p, y, mean, rstd, 2, backend="triton")

    def test_validation_and_broadcast_coefficient(self):
        p, y = torch.randn(1, 4, 3, 5), torch.randn(2, 4, 3, 5)
        mean, rstd, _ = tangent_unet._gn_stats(nn.GroupNorm(2, 4), p)
        coefficient = torch.randn(1, 4, 1, 1)
        torch.testing.assert_close(group_norm_action(p, y, mean, rstd, 2, coefficient),
                                   group_norm_action(p, y, mean, rstd, 2) * coefficient)
        for kwargs in (dict(groups=3), dict(groups=2, block_size=3), dict(groups=2, backend="bad")):
            with self.assertRaises(ValueError):
                group_norm_action(p, y, mean, rstd, **kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton GN kernel")
class FusedNormCUDA(unittest.TestCase):
    def test_independent_input_layouts_preserve_lane_layout(self):
        for spatial in ((17, 19), (33, 35)):
            p_base = torch.randn(1, 8, *spatial, device="cuda")
            y_base = torch.randn(2, 8, *spatial, device="cuda")
            coefficient_base = torch.randn_like(p_base)
            mean, rstd, _ = tangent_unet._gn_stats(nn.GroupNorm(2, 8), p_base)
            for p_last, y_last, coeff_last in product((False, True), repeat=3):
                p = p_base.contiguous(memory_format=torch.channels_last if p_last else torch.contiguous_format)
                y = y_base.contiguous(memory_format=torch.channels_last if y_last else torch.contiguous_format)
                coefficient = coefficient_base.contiguous(
                    memory_format=torch.channels_last if coeff_last else torch.contiguous_format)
                for transpose in (False, True):
                    with self.subTest(spatial=spatial, layouts=(p_last, y_last, coeff_last), transpose=transpose):
                        actual = group_norm_action(p, y, mean, rstd, 2, coefficient,
                                                   transpose, backend="triton")
                        expected = group_norm_action(p, y, mean, rstd, 2, coefficient,
                                                     transpose, backend="torch")
                        torch.testing.assert_close(actual, expected, rtol=7e-5, atol=4e-6)
                        self.assertTrue(actual.is_contiguous(memory_format=(
                            torch.channels_last if y_last else torch.contiguous_format)))

    def test_arbitrary_strides_fall_back_to_contiguous_storage(self):
        p = torch.randn(1, 8, 17, 38, device="cuda")[..., ::2]
        y = torch.randn(2, 8, 17, 38, device="cuda")[..., ::2]
        coefficient = torch.randn(1, 8, 1, 1, device="cuda")
        mean, rstd, _ = tangent_unet._gn_stats(nn.GroupNorm(2, 8), p)
        for transpose in (False, True):
            actual = group_norm_action(p, y, mean, rstd, 2, coefficient, transpose, backend="triton")
            expected = group_norm_action(p, y, mean, rstd, 2, coefficient, transpose, backend="torch")
            torch.testing.assert_close(actual, expected, rtol=7e-5, atol=4e-6)
            self.assertTrue(actual.is_contiguous())

    def test_masked_tiles_and_mixed_widths(self):
        torch.manual_seed(279)
        for spatial in ((5, 7), (33, 35), (63, 65)):
            for lanes in (1, 3):
                p = torch.randn(1, 8, *spatial, device="cuda")
                y = torch.randn(lanes, 8, *spatial, device="cuda")
                coefficient = torch.randn_like(p)
                coefficient[:, 0] = 0
                mean, rstd, _ = tangent_unet._gn_stats(nn.GroupNorm(2, 8), p)
                for transpose in (False, True):
                    for coeff in (None, coefficient):
                        expected = group_norm_action(p, y, mean, rstd, 2, coeff, transpose, backend="torch")
                        actual = group_norm_action(p, y, mean, rstd, 2, coeff, transpose, backend="triton")
                        torch.testing.assert_close(actual, expected, rtol=7e-5, atol=4e-6)

    def test_capture_replays_new_inputs(self):
        p = torch.randn(1, 8, 17, 19, device="cuda")
        y = torch.randn(3, 8, 17, 19, device="cuda")
        coeff = torch.randn_like(p)
        mean, rstd, _ = tangent_unet._gn_stats(nn.GroupNorm(2, 8), p)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                group_norm_action(p, y, mean, rstd, 2, coeff, True, backend="triton")
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = group_norm_action(p, y, mean, rstd, 2, coeff, True, backend="triton")
        for scale in (1.3, -.5):
            y.mul_(scale)
            coeff.add_(.1)
            graph.replay()
            expected = group_norm_action(p, y, mean, rstd, 2, coeff, True, backend="torch")
            torch.testing.assert_close(out, expected, rtol=7e-5, atol=4e-6)


if __name__ == "__main__":
    unittest.main()

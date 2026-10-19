"""Verify optional pointwise primal fusion without changing GN statistics."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dlr_dflow.fused_primal import normalization_activation_record


class FusedPrimalCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    @staticmethod
    def reference(x, weight, bias, scale, shift):
        value = x * weight + bias
        if scale is not None:
            value = value * (1 + scale) + shift
        return F.silu(value)

    def test_output_and_coefficient_match_autodiff(self):
        torch.manual_seed(921)
        for dtype in (torch.float32, torch.float64):
            for conditioned in (False, True):
                x = torch.randn(1, 8, 5, 7, dtype=dtype)
                weight, bias = torch.randn(1, 8, 1, 1, dtype=dtype), torch.randn(1, 8, 1, 1, dtype=dtype)
                weight[:, 0] = 0
                scale = torch.randn_like(weight) if conditioned else None
                shift = torch.randn_like(weight) if conditioned else None
                expected, coefficient = torch.func.jvp(
                    lambda q: self.reference(q, weight, bias, scale, shift), (x,), (torch.ones_like(x),))
                for fused in (False, True):
                    with self.subTest(dtype=dtype, conditioned=conditioned, fused=fused):
                        actual, actual_coefficient = normalization_activation_record(
                            x, weight, bias, scale, shift, fuse_silu=fused)
                        tolerance = dict(rtol=3e-5, atol=5e-7) if dtype == torch.float32 else dict(rtol=3e-12, atol=3e-13)
                        torch.testing.assert_close(actual, expected, **tolerance)
                        torch.testing.assert_close(actual_coefficient, coefficient, **tolerance)
                        if not fused:
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cpu_fallback_is_lazy_and_rejects_incompatible_requests(self):
        x, weight, bias = torch.randn(1, 4, 5, 7), torch.ones(4), torch.zeros(4)
        with patch("dlr_dflow.fused_primal._cuda_kernel", side_effect=AssertionError("loaded CUDA")):
            normalization_activation_record(x, weight, bias)
        with self.assertRaisesRegex(ValueError, "CUDA float32"):
            normalization_activation_record(x, weight, bias, backend="triton")
        with self.assertRaisesRegex(ValueError, "both"):
            normalization_activation_record(x, weight, bias, scale=weight)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for primal fusion")
class FusedPrimalCUDA(unittest.TestCase):
    def test_cuda_parity_and_conservative_bitwise_primal(self):
        for conditioned in (False, True):
            x = torch.randn(1, 8, 33, 35, device="cuda")
            weight, bias = torch.randn(8, device="cuda"), torch.randn(8, device="cuda")
            weight[0] = 0
            scale = torch.randn_like(weight) if conditioned else None
            shift = torch.randn_like(weight) if conditioned else None
            expected = normalization_activation_record(x, weight, bias, scale, shift, backend="torch")
            for fused in (False, True):
                actual = normalization_activation_record(x, weight, bias, scale, shift,
                                                          backend="triton", fuse_silu=fused)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, rtol=3e-5, atol=1e-6)
                    if not fused:
                        torch.testing.assert_close(value, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

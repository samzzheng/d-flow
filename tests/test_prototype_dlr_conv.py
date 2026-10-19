"""Indexing checks for a standalone implicit-GEMM convolution experiment."""

import unittest

import torch
import torch.nn.functional as F

from scripts.prototype_dlr_conv import conv3x3_nchw, gather_conv3x3_rows


class ImplicitConvIndexing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_padding_channel_order_and_batch_boundaries(self):
        torch.manual_seed(292)
        for channels in (16, 48):
            for height, width in ((1, 1), (3, 5), (7, 11)):
                x = torch.randn(2, channels, height, width, dtype=torch.float64)
                weight = torch.randn(16, channels, 3, 3, dtype=torch.float64)
                rows = torch.arange(2 * height * width)
                selected = gather_conv3x3_rows(x, rows) @ weight.reshape(16, -1).T
                expected = F.conv2d(x, weight, padding=1).permute(0, 2, 3, 1).reshape(-1, 16)
                torch.testing.assert_close(selected, expected, rtol=2e-12, atol=2e-12)

    def test_padding_is_zero_not_wrapped_neighbors(self):
        x = torch.ones(2, 16, 3, 5, dtype=torch.float64)
        rows = torch.tensor([0, 4, 5, 7, 14, 15, 29])
        nonzero_per_channel = gather_conv3x3_rows(x, rows).reshape(-1, 16, 9).sum(-1)
        expected = torch.tensor([4, 4, 6, 9, 4, 4, 4], dtype=torch.float64)
        torch.testing.assert_close(nonzero_per_channel, expected[:, None].expand_as(nonzero_per_channel))

    def test_unsupported_execution_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "CUDA float32"):
            conv3x3_nchw(torch.randn(1, 16, 3, 5), torch.randn(16, 16, 3, 3))
        with self.assertRaisesRegex(ValueError, "Cin16/48"):
            conv3x3_nchw(torch.randn(1, 32, 3, 5), torch.randn(16, 32, 3, 3))


if __name__ == "__main__":
    unittest.main()

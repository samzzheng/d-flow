"""CPU tests for column partitioning and reusable graph-buffer ownership."""
from contextlib import nullcontext, redirect_stderr
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from dlr_dflow.graphed_products import GraphedAProducts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


class LinearVelocity(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        matrix = torch.arange(dimension ** 2, dtype=torch.float64).reshape(dimension, dimension)
        self.register_buffer("matrix", (matrix - dimension) / dimension ** 2)

    def operator(self, time):
        return self.matrix + time * torch.eye(self.matrix.shape[0], dtype=self.matrix.dtype)

    def forward(self, x, time):
        return (self.operator(time[0]) @ x.flatten()).reshape_as(x)


class Replay:
    def __init__(self, operation):
        self.operation = operation

    def replay(self):
        self.operation()


class CPUReusableProducts(GraphedAProducts):
    """Exercise real _run/_run_block using simulated reusable graph buffers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captures = []
        self.replays = []
        self.primal_builds = 0

    def _ensure_linearization(self):
        if self._state_dirty:
            self.primal_builds += 1
            self._state_dirty = False

    def _capture(self, kind, width):
        self.captures.append((kind, width))
        q_in = torch.empty(self.d, width, dtype=self.x_buf.dtype)
        out = torch.empty_like(q_in)
        primal = torch.empty_like(self.x_buf)

        def operation():
            self.replays.append((kind, width))
            matrix = self.model.operator(self.t_buf[0])
            out.copy_((matrix if kind == "A" else matrix.T) @ q_in)
            primal.copy_(self.model(self.x_buf, self.t_buf))

        self._graphs[(kind, width)] = (Replay(operation), q_in,
                                      (out, primal) if kind == "A" else out)


class ProductPartitionTests(unittest.TestCase):
    def setUp(self):
        self.model = LinearVelocity(16)
        self.x = torch.linspace(-1., 1., 16, dtype=torch.float64).reshape(1, 1, 4, 4)
        self.cuda_context = patch("torch.cuda.device", side_effect=lambda *_: nullcontext())
        self.cuda_context.start()
        self.addCleanup(self.cuda_context.stop)

    def provider(self, **kwargs):
        kwargs.setdefault("share_linearization", False)
        return CPUReusableProducts(self.model, self.x, **kwargs)

    @staticmethod
    def widths(width, limit):
        if limit is None or width <= limit:
            return [width]
        return [min(limit, width - start) for start in range(0, width, limit)]

    def test_linear_products_partition_independently_and_keep_column_order(self):
        for width, jvp_limit, vjp_limit in (
            (13, 5, 4), (8, 3, None), (8, None, 2),
            (8, 4, 4), (4, 8, 9), (1, 1, 1), (7, None, None),
        ):
            with self.subTest(width=width, jvp=jvp_limit, vjp=vjp_limit):
                products = self.provider(max_jvp_columns=jvp_limit, max_vjp_columns=vjp_limit)
                products.set_state(.25, self.x)
                q = torch.arange(16 * width * 2, dtype=torch.float64).reshape(16, 2 * width)[:, ::2]
                self.assertFalse(q.is_contiguous())
                expected_input = q.clone()
                matrix = self.model.operator(.25)
                torch.testing.assert_close(products.A(q), matrix @ q, rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(products.AT(q), matrix.T @ q, rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(products.A.last_primal, self.model(self.x, torch.tensor([.25])))
                expected = ([('A', n) for n in self.widths(width, jvp_limit)]
                            + [('AT', n) for n in self.widths(width, vjp_limit)])
                self.assertEqual(products.replays, expected)
                self.assertEqual(set(products.captures), set(expected))
                self.assertEqual(len(products.captures), len(set(expected)))
                torch.testing.assert_close(q, expected_input, rtol=0, atol=0)

    def test_results_and_primal_remain_owned_across_repeated_blocks_and_states(self):
        products = self.provider(max_jvp_columns=5, max_vjp_columns=4)
        products.set_state(.1, self.x)
        q = torch.arange(16 * 13, dtype=torch.float64).reshape(16, 13)
        saved = [products.A(q), products.A.last_primal, products.AT(q),
                 products.A(q[:, :5]), products.A.last_primal]
        snapshots = [value.clone() for value in saved]
        q.fill_(-99.)
        products.set_state(.9, self.x + 2.)
        products.A(q)
        products.AT(q)
        products.A(q[:, :5])
        for value, expected in zip(saved, snapshots):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        torch.testing.assert_close(products.A.last_primal, self.model(self.x + 2., torch.tensor([.9], dtype=torch.float64)))

    def test_shared_primal_is_prepared_once_per_state_across_all_blocks(self):
        products = self.provider(share_linearization=True, max_jvp_columns=3, max_vjp_columns=2)
        q = torch.ones(16, 8, dtype=torch.float64)
        products.set_state(.1, self.x)
        products.A(q)
        products.AT(q)
        self.assertEqual(products.primal_builds, 1)
        products.set_state(.2, self.x)
        products.AT(q)
        self.assertEqual(products.primal_builds, 2)

    def test_invalid_limits_are_rejected(self):
        for name in ("max_jvp_columns", "max_vjp_columns"):
            for value in (0, -1, 1.5, True, False, "2"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.provider(**{name: value})


class ProductPartitionCLITests(unittest.TestCase):
    def test_cli_defaults_are_unlimited_and_explicit_limits_are_positive(self):
        with patch.object(sys, "argv", ["benchmark"]):
            defaults = bench.parse_args()
        self.assertIsNone(defaults.dlr_max_jvp_columns)
        self.assertIsNone(defaults.dlr_max_vjp_columns)
        with patch.object(sys, "argv", ["benchmark", "--dlr-max-jvp-columns", "20", "--dlr-max-vjp-columns", "16"]):
            args = bench.parse_args()
        self.assertEqual((args.dlr_max_jvp_columns, args.dlr_max_vjp_columns), (20, 16))
        for name in ("--dlr-max-jvp-columns", "--dlr-max-vjp-columns"):
            with patch.object(sys, "argv", ["benchmark", name, "0"]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args()

    def test_resume_cannot_mix_partition_settings(self):
        bench.validate_resume_settings({}, {"dlr_max_jvp_columns": None, "dlr_max_vjp_columns": None})
        for name in ("dlr_max_jvp_columns", "dlr_max_vjp_columns"):
            with self.assertRaisesRegex(RuntimeError, name):
                bench.validate_resume_settings({}, {name: 20})


if __name__ == "__main__":
    unittest.main()

"""CUDA graph replay must preserve the eager velocity Jacobian products.

Run with ``python -m unittest discover -s tests -p 'test_graphed_products.py'``.
The tests skip cleanly when CUDA is unavailable.
"""

import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from dlr_dflow import dlr_dflow as core
from dlr_dflow import tangent_unet
from dlr_dflow.graphed_products import GraphedAProducts
from models.unet.default import Unet


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph replay requires CUDA")
class GraphedAProductsTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.device = torch.device("cuda")
        self.context.enter_context(torch.random.fork_rng(devices=[torch.cuda.current_device()]))
        torch.manual_seed(812)
        # Forward AD needs the differentiable math attention backend.
        self.context.enter_context(sdpa_kernel(SDPBackend.MATH))
        self.model = Unet(
            ch=8, ch_mul=[1, 2], att_channels=[0, 1], groups=4, dropout=0.0,
        ).to(self.device).eval().requires_grad_(False)
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, nn.GroupNorm):
                    module.weight.uniform_(0.5, 1.5)
                    module.bias.normal_(std=0.1)
        self.x = torch.randn(1, 1, 8, 8, device=self.device)

    def assertProductClose(self, actual, expected):
        # Graph replay may select a different CUDA kernel than eager execution,
        # so individual values near zero need a loose elementwise allowance.
        # The relative norm below is the meaningful whole-product check.
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-4)
        relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        # This matches the normal fp32/TF32 replay spread measured for the
        # full sensitivity evolution while still catching stale graph inputs.
        self.assertLess(relative_error.item(), 1.2e-4)

    def autograd_products(self, t, x, q):
        """Independent reverse-over-reverse JVPs and ordinary autograd VJPs."""
        tvec = torch.full((1,), t, device=x.device, dtype=x.dtype)

        def velocity(value):
            return self.model(value, tvec)

        jvp_columns = [
            torch.autograd.functional.jvp(
                velocity, x, column.reshape_as(x), strict=True,
            )[1].flatten()
            for column in q.T
        ]
        variable = x.detach().requires_grad_(True)
        primal = velocity(variable)
        vjp_columns = [
            torch.autograd.grad(
                primal, variable, column.reshape_as(primal), retain_graph=True,
            )[0].flatten()
            for column in q.T
        ]
        return torch.stack(jvp_columns, dim=1), torch.stack(vjp_columns, dim=1), primal.detach()

    def check_replay(self, hand_jvp, hand_vjp, shared=False):
        self.context.enter_context(mock.patch.object(core, "HAND_JVP", hand_jvp))
        self.context.enter_context(mock.patch.object(core, "HAND_VJP", hand_vjp))
        self.context.enter_context(mock.patch.object(core, "SHARE_LINEARIZATION", shared))
        # The optimized selector uses native GroupNorm statistics at this
        # small resolution; the manual large-image path is covered separately
        # by test_tangent_products in both float32 and float64.
        self.context.enter_context(mock.patch.object(
            tangent_unet, "GN_STATS_MODE", "native",
        ))
        jvp_spy = self.context.enter_context(mock.patch.object(
            tangent_unet, "unet_jvp", wraps=tangent_unet.unet_jvp,
        ))
        vjp_spy = self.context.enter_context(mock.patch.object(
            tangent_unet, "unet_vjp", wraps=tangent_unet.unet_vjp,
        ))
        linearize_spy = self.context.enter_context(mock.patch.object(
            tangent_unet, "unet_linearize", wraps=tangent_unet.unet_linearize,
        )) if shared else None
        products = GraphedAProducts(self.model, self.x)
        saved_outputs = []
        captured_graphs = {}
        # Change time and image separately so stale copies of either buffer
        # cannot accidentally agree with the reference.
        states = (
            (0.13, self.x),
            (0.71, self.x),
            (0.71, self.x + 0.25 * torch.randn_like(self.x)),
        )
        for t, x in states:
            products.set_state(t, x)
            eager_a, eager_at = core.make_A_products(self.model, t, x)
            for width in (1, 2, 1):
                with self.subTest(time=t, width=width, hand_jvp=hand_jvp, hand_vjp=hand_vjp):
                    q = torch.randn(x.numel(), width, device=self.device)
                    # Read the dispatch spies before calling the eager reference:
                    # those reference calls must not make a missing graph branch pass.
                    jvp_calls, vjp_calls = jvp_spy.call_count, vjp_spy.call_count
                    primal_calls = linearize_spy.call_count if shared else 0
                    first_primal = shared and products._primal_graph is None
                    first_width = ("A", width) not in captured_graphs
                    # Either product must work first after set_state, including
                    # when its graph is captured for a previously unseen width.
                    if shared and width == 2:
                        actual_at = products.AT(q)
                    actual_a = products.A(q)
                    actual_primal = products.A.last_primal
                    if not (shared and width == 2):
                        actual_at = products.AT(q)
                    self.assertEqual(jvp_spy.call_count > jvp_calls, first_width and hand_jvp and not shared)
                    self.assertEqual(vjp_spy.call_count > vjp_calls, first_width and hand_vjp and not shared)
                    if shared:
                        self.assertEqual(linearize_spy.call_count > primal_calls, first_primal)

                    for kind in ("A", "AT"):
                        graph = products._graphs[(kind, width)][0]
                        if (kind, width) in captured_graphs:
                            self.assertIs(graph, captured_graphs[(kind, width)])
                        else:
                            captured_graphs[(kind, width)] = graph

                    expected_a, expected_at, expected_primal = self.autograd_products(t, x, q)
                    self.assertProductClose(actual_a, eager_a(q))
                    self.assertProductClose(actual_at, eager_at(q))
                    self.assertProductClose(actual_a, expected_a)
                    self.assertProductClose(actual_at, expected_at)
                    self.assertProductClose(actual_primal, expected_primal)

                    # Replaying another state or width must not mutate products
                    # already retained by the low-rank integrator.
                    for retained, snapshot in saved_outputs:
                        torch.testing.assert_close(retained, snapshot, rtol=0, atol=0)
                    saved_outputs.extend(
                        (value, value.clone())
                        for value in (actual_a, actual_at, actual_primal)
                    )
        self.assertEqual(set(products._graphs), {("A", 1), ("A", 2), ("AT", 1), ("AT", 2)})

    def test_hand_products_match_eager_and_autograd(self):
        self.check_replay(hand_jvp=True, hand_vjp=True)

    def test_torch_func_products_match_eager_and_autograd(self):
        self.check_replay(hand_jvp=False, hand_vjp=False)

    def test_shared_primal_products_match_eager_and_autograd(self):
        self.check_replay(hand_jvp=True, hand_vjp=True, shared=True)


if __name__ == "__main__":
    unittest.main()

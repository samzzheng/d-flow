"""Small optional CUDA replay regressions for checkpointed non-attention DLR.

Run only this file with a selected CUDA device; each model uses four base
channels and an 8x8 image. No trained checkpoint or reconstruction is loaded.
"""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core, enriched, tangent_unet
from dlr_dflow.graphed_evolution import GraphedZeroEnrichmentStep
from dlr_dflow.graphed_products import GraphedAProducts
from models.unet.default import Unet


def residual_matrix(factors):
    u, s, v = factors
    return u @ s @ v.T


@unittest.skipUnless(torch.cuda.is_available(), "checkpointed replay requires CUDA")
class CheckpointedCUDAReplayTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.context.enter_context(torch.random.fork_rng(devices=[self.device.index]))
        self.context.enter_context(patch.multiple(
            core, HAND_JVP=True, HAND_VJP=True, SHARE_LINEARIZATION=True))
        self.context.enter_context(patch.object(tangent_unet, "GN_STATS_MODE", "native"))
        self.context.enter_context(patch.multiple(
            enriched, SKIP_UNTRUNCATED_SVD=True, FUSE_K_IMAGE=True,
            M_FROM_TRANSPOSE=True, RECOMPRESS_FROM_TRANSPOSE=True))
        torch.manual_seed(9827)
        self.model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, 0],
                          groups=2, dropout=0., mid_attn=False
                          ).to(self.device).eval().requires_grad_(False)
        self.x = torch.randn(1, 1, 8, 8, device=self.device)

    def assert_action_close(self, actual, expected, tolerance=2e-4):
        relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        self.assertLess(relative.item(), tolerance)

    def test_product_replays_update_time_state_width_and_preserve_owned_outputs(self):
        products = GraphedAProducts(self.model, self.x, checkpoint_blocks=True,
                                    full_step_graphs=False)
        retained = []
        graphs = {}
        states = ((.13, self.x), (.71, self.x), (.71, self.x + .3))
        for state_index, (time, state) in enumerate(states):
            products.set_state(time, state)
            # The default full-tape implementation is an uncaptured reference.
            eager_a, eager_at = core.make_A_products(self.model, time, state)
            for width in (1, 2, 1):
                with self.subTest(state=state_index, width=width):
                    q = torch.randn(state.numel(), width, device=self.device)
                    # Exercise transpose-first replay after changing the state.
                    actual_at = products.AT(q)
                    actual_a = products.A(q)
                    actual_primal = products.A.last_primal
                    self.assert_action_close(actual_a, eager_a(q))
                    self.assert_action_close(actual_at, eager_at(q))
                    self.assert_action_close(actual_primal, eager_a.last_primal)
                    for kind in ("A", "AT"):
                        graph = products._graphs[(kind, width)][0]
                        if (kind, width) in graphs:
                            self.assertIs(graph, graphs[(kind, width)])
                        graphs[(kind, width)] = graph
                    for value, saved in retained:
                        torch.testing.assert_close(value, saved, atol=0, rtol=0)
                    retained.extend((value, value.clone())
                                    for value in (actual_a, actual_at, actual_primal))

        # Independently check the last one-column product against native AD.
        time_tensor = state.new_full((1,), time)
        velocity = lambda value: self.model(value, time_tensor)
        _, expected_jvp = torch.autograd.functional.jvp(
            velocity, state, q.T.reshape_as(state), strict=True)
        variable = state.detach().requires_grad_(True)
        primal = velocity(variable)
        expected_vjp, = torch.autograd.grad(primal, variable, q.T.reshape_as(primal))
        self.assert_action_close(actual_a, expected_jvp.reshape(-1, 1))
        self.assert_action_close(actual_at, expected_vjp.reshape(-1, 1))
        self.assertEqual(set(graphs), {("A", 1), ("A", 2), ("AT", 1), ("AT", 2)})

    def test_complete_step_graph_matches_default_across_reset(self):
        dimension, rank = self.x.numel(), 2
        u = torch.linalg.qr(torch.randn(dimension, rank, device=self.device))[0]
        v = torch.linalg.qr(torch.randn(dimension, rank, device=self.device))[0]
        s = torch.diag(self.x.new_tensor([.1, .04]))
        factors = (u, s, v)
        probes = torch.randn(dimension, 2, device=self.device)
        sensitivities = .7 * probes + u @ (s @ (v.T @ probes))
        backends = [GraphedZeroEnrichmentStep(
            self.model, self.x, factors, sensitivities, probes, .7, .03,
            checkpoint_blocks=checkpoint_blocks) for checkpoint_blocks in (False, True)]
        captured = None
        saved_output = None
        for restart in (False, True):
            if restart:
                for backend in backends:
                    backend.reset(.3 * self.x, factors, -.4 * sensitivities,
                                  .4, probes=-probes)
            for time in (.2, .6):
                for backend in backends:
                    backend.run(time)
                expected, actual = [backend.snapshot() for backend in backends]
                for value, reference in ((actual[0], expected[0]),
                                         (residual_matrix(actual[1]), residual_matrix(expected[1])),
                                         (actual[2], expected[2]), (actual[3], expected[3])):
                    self.assert_action_close(value, reference, tolerance=5e-4)
                if captured is None:
                    captured = [backend.graph for backend in backends]
                    values = (actual[0], *actual[1], actual[2], actual[3])
                    saved_output = [(value, value.clone()) for value in values]
                else:
                    for backend, graph in zip(backends, captured):
                        self.assertIs(backend.graph, graph)
                    for value, saved in saved_output:
                        torch.testing.assert_close(value, saved, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

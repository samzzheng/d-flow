"""Optional fused derivatives preserve the whole non-attention UNet map."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from dlr_dflow import tangent_unet as tangent
from dlr_dflow import enriched
from dlr_dflow.graphed_evolution import GraphedZeroEnrichmentStep
from dlr_dflow.graphed_products import GraphedAProducts
from models.unet.default import Unet


def make_model(device, channels_last=False):
    torch.manual_seed(728)
    model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, 0], groups=2,
                 dropout=0., mid_attn=False).to(device).eval().requires_grad_(False)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return model


class FusedUnetCPU(unittest.TestCase):
    def test_native_fallback_and_invalid_backend(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        model = make_model("cpu")
        x, q = torch.randn(1, 1, 7, 9), torch.randn(2, 1, 7, 9)
        with patch.object(tangent, "GN_ACTION_BACKEND", "native"):
            native = tangent.unet_linearize(model, .4, x)
            expected = (native.primal, native.jvp(q), native.vjp(q))
        with patch.object(tangent, "GN_ACTION_BACKEND", "triton"):
            fallback = tangent.unet_linearize(model, .4, x)
            for actual, reference in zip((fallback.primal, fallback.jvp(q), fallback.vjp(q)), expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        with patch.object(tangent, "GN_ACTION_BACKEND", "invalid"):
            with self.assertRaisesRegex(ValueError, "GN_ACTION_BACKEND"):
                tangent.unet_linearize(model, .4, x)


@unittest.skipUnless(torch.cuda.is_available(), "fused UNet checks require CUDA")
class FusedUnetCUDA(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(tangent, "GN_ACTION_BACKEND", "triton"))
        self.stack.enter_context(patch.object(tangent, "GN_STATS_MODE", "manual"))

    def assert_relative(self, actual, reference, tolerance=3e-4):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        error = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
        self.assertLess(float(error.detach()), tolerance)

    def test_products_match_native_autodiff_in_both_layouts(self):
        for channels_last in (False, True):
            with self.subTest(channels_last=channels_last):
                model = make_model("cuda", channels_last)
                x = torch.randn(1, 1, 7, 9, device="cuda")
                q = torch.randn(2, 1, 7, 9, device="cuda")
                time = x.new_tensor([.4])
                # Reverse-over-reverse JVP also supports channels-last models.
                velocity = lambda value: model(value, time)
                expected_j = torch.cat([torch.autograd.functional.jvp(
                    velocity, x, lane[None], strict=True)[1] for lane in q])
                variable = x.detach().requires_grad_(True)
                primal = velocity(variable)
                expected_t = torch.cat([torch.autograd.grad(
                    primal, variable, lane[None], retain_graph=True)[0] for lane in q])
                for checkpoint in (False, True):
                    tape = tangent.unet_linearize(model, time, x, checkpoint_blocks=checkpoint)
                    self.assert_relative(tape.primal, primal)
                    self.assert_relative(tape.jvp(q), expected_j)
                    self.assert_relative(tape.vjp(q), expected_t)

    def test_graphed_products_preserve_changed_states_and_outputs(self):
        for channels_last in (False, True):
            model = make_model("cuda", channels_last)
            x = torch.randn(1, 1, 7, 9, device="cuda")
            products = GraphedAProducts(model, x, full_step_graphs=False)
            retained = []
            for time, state in ((.2, x), (.6, x), (.6, .4 * x)):
                products.set_state(time, state)
                with patch.object(tangent, "GN_ACTION_BACKEND", "native"):
                    tape = tangent.unet_linearize(model, time, state)
                    q = torch.randn(2, 1, 7, 9, device="cuda")
                    expected_j, expected_t = tape.jvp(q), tape.vjp(q)
                matrix = q.reshape(2, -1).T
                actual_t, actual_j = products.AT(matrix), products.A(matrix)
                self.assert_relative(actual_j.T.reshape_as(q), expected_j)
                self.assert_relative(actual_t.T.reshape_as(q), expected_t)
                for value, reference in retained:
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                retained.extend((value, value.clone()) for value in (actual_j, actual_t))

    def test_complete_step_chunking_matches_reference_across_reset(self):
        model = make_model("cuda")
        x = torch.randn(1, 1, 7, 9, device="cuda")
        rank, dimension = 4, x.numel()
        u = torch.linalg.qr(torch.randn(dimension, rank, device="cuda"))[0]
        v = torch.linalg.qr(torch.randn(dimension, rank, device="cuda"))[0]
        factors = (u, torch.diag(x.new_tensor([.4, .2, .03, .001])), v)
        probes = torch.randn(dimension, 3, device="cuda")
        with patch.object(enriched, "SPECIALIZE_ZERO_ENRICHMENT", True):
            backends = [GraphedZeroEnrichmentStep(
                model, x, factors, .6 * probes, probes, .6, .03,
                max_step_jvp_columns=5 if chunked else None,
                max_step_vjp_columns=2 if chunked else None)
                for chunked in (False, True)]
            for reset in (False, True):
                if reset:
                    for backend in backends:
                        backend.reset(.3 * x, factors, -.4 * probes, .2, probes=-probes)
                for time in (.3, .6):
                    for backend in backends:
                        backend.run(time)
                    expected, actual = [backend.snapshot() for backend in backends]
                    self.assert_relative(actual[0], expected[0])
                    self.assert_relative(actual[2], expected[2])
                    self.assert_relative(actual[3], expected[3])
                    action = lambda fs: fs[2] @ (fs[1].T @ (fs[0].T @ probes))
                    self.assert_relative(action(actual[1]), action(expected[1]))


if __name__ == "__main__":
    unittest.main()

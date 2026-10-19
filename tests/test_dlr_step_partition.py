"""Partition complete-step derivatives without rebuilding the shared primal."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dlr_dflow import enriched, graphed_evolution, tangent_unet
from dlr_dflow.graphed_evolution import GraphedZeroEnrichmentStep
from dlr_dflow.graphed_products import GraphedAProducts
from models.unet.default import Unet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


def represented(factors):
    u, s, v = factors
    return u @ s @ v.T


class StepPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(523)
        self.model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, 0], mid_attn=False,
                          groups=2, dropout=0.).eval().requires_grad_(False)
        self.x = torch.randn(1, 1, 8, 8)
        u = torch.linalg.qr(torch.randn(64, 8))[0]
        v = torch.linalg.qr(torch.randn(64, 8))[0]
        s = torch.diag(torch.linspace(.2, .01, 8))
        self.factors = (u, s, v)
        self.probes = torch.randn(64, 4)
        self.sensitivities = .7 * self.probes + u @ (s @ (v.T @ self.probes))

    def backend(self, **kwargs):
        return GraphedZeroEnrichmentStep(
            self.model, self.x, self.factors, self.sensitivities,
            self.probes, .7, .01, capture=False, **kwargs)

    @staticmethod
    def widths(width, limit):
        if limit is None or width <= limit:
            return [width]
        return [min(limit, width - offset) for offset in range(0, width, limit)]

    def test_partitioned_steps_match_reference_with_one_primal_per_step(self):
        configurations = [(None, None, False, False), (5, 3, False, False),
                          (5, None, False, False), (None, 3, False, False),
                          (20, 20, False, False), (5, 3, True, False),
                          (5, 3, False, True)]
        original_jvp, original_vjp = tangent_unet.UnetLinearization.jvp, tangent_unet.UnetLinearization.vjp
        original_linearize = graphed_evolution.unet_linearize
        for jvp_limit, vjp_limit, checkpoint, specialized in configurations:
            with self.subTest(jvp=jvp_limit, vjp=vjp_limit, checkpoint=checkpoint, specialized=specialized), \
                    patch.object(enriched, "SPECIALIZE_ZERO_ENRICHMENT", specialized):
                reference = self.backend(checkpoint_blocks=checkpoint)
                actual = self.backend(checkpoint_blocks=checkpoint,
                                      max_step_jvp_columns=jvp_limit,
                                      max_step_vjp_columns=vjp_limit)
                retained = None
                for restart, time in ((False, .2), (False, .4), (True, .6)):
                    if restart:
                        for backend in (reference, actual):
                            backend.reset(.3 * self.x, self.factors, -.4 * self.sensitivities,
                                          .4, probes=-self.probes)
                    reference.run(time)
                    calls = []
                    def jvp(tape, lanes):
                        calls.append(("A", lanes.shape[0]))
                        return original_jvp(tape, lanes)
                    def vjp(tape, lanes):
                        calls.append(("AT", lanes.shape[0]))
                        return original_vjp(tape, lanes)
                    with patch.object(tangent_unet.UnetLinearization, "jvp", jvp), \
                            patch.object(tangent_unet.UnetLinearization, "vjp", vjp), \
                            patch.object(graphed_evolution, "unet_linearize", wraps=original_linearize) as linearize:
                        actual.run(time)
                    self.assertEqual(linearize.call_count, 1)
                    self.assertEqual(calls, [("A", width) for width in self.widths(12, jvp_limit)]
                                     + [("AT", width) for width in self.widths(8, vjp_limit)])
                    expected_state, actual_state = reference.snapshot(), actual.snapshot()
                    for value, expected in ((actual_state[0], expected_state[0]),
                                            (represented(actual_state[1]), represented(expected_state[1])),
                                            (actual_state[2], expected_state[2]),
                                            (actual_state[3], expected_state[3])):
                        torch.testing.assert_close(value, expected, rtol=5e-5, atol=3e-6)
                    if retained is None:
                        retained = [(value, value.clone()) for value in
                                    (actual_state[0], *actual_state[1], actual_state[2], actual_state[3])]
                    else:
                        for value, saved in retained:
                            torch.testing.assert_close(value, saved, rtol=0, atol=0)

    def test_provider_propagates_independent_step_limits_and_reuses_backend(self):
        provider = GraphedAProducts(self.model, self.x, max_jvp_columns=20, max_vjp_columns=16,
                                    max_step_jvp_columns=5, max_step_vjp_columns=3)
        args = (self.x, self.factors, self.sensitivities, self.probes, .7, .01)
        with patch.object(graphed_evolution, "GraphedZeroEnrichmentStep", wraps=GraphedZeroEnrichmentStep) as backend:
            backend.side_effect = lambda *a, **kw: SimpleNamespace(reset=lambda *args: None)
            provider.prepare_step_graph(*args)
            provider.prepare_step_graph(*args)
        self.assertEqual(backend.call_count, 1)
        self.assertEqual(backend.call_args.kwargs["max_step_jvp_columns"], 5)
        self.assertEqual(backend.call_args.kwargs["max_step_vjp_columns"], 3)
        self.assertEqual(next(iter(provider._step_graphs))[-1], (5, 3))
        self.assertEqual((provider.max_jvp_columns, provider.max_vjp_columns), (20, 16))

    def test_invalid_limits_and_mutation_are_rejected(self):
        for name in ("max_step_jvp_columns", "max_step_vjp_columns"):
            for value in (0, -1, 1.5, True, False, "2"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.backend(**{name: value})
                    with self.assertRaisesRegex(ValueError, name):
                        GraphedAProducts(self.model, self.x, **{name: value})
            backend = self.backend(**{name: 3})
            setattr(backend, name, 4)
            with self.assertRaisesRegex(ValueError, "step column limits changed"):
                backend.run(.2)
            provider = GraphedAProducts(self.model, self.x, **{name: 3})
            setattr(provider, name, 4)
            with self.assertRaisesRegex(ValueError, "step column limits changed"):
                provider.set_state(.2, self.x)


class StepPartitionCLITests(unittest.TestCase):
    def test_defaults_explicit_limits_and_resume_identity(self):
        with patch.object(sys, "argv", ["benchmark"]):
            defaults = bench.parse_args()
        self.assertIsNone(defaults.dlr_step_jvp_columns)
        self.assertIsNone(defaults.dlr_step_vjp_columns)
        with patch.object(sys, "argv", ["benchmark", "--dlr-step-jvp-columns", "5",
                                       "--dlr-step-vjp-columns", "3"]):
            args = bench.parse_args()
        self.assertEqual((args.dlr_step_jvp_columns, args.dlr_step_vjp_columns), (5, 3))
        for name in ("dlr_step_jvp_columns", "dlr_step_vjp_columns"):
            bench.validate_resume_settings({}, {name: None})
            with self.assertRaisesRegex(RuntimeError, name):
                bench.validate_resume_settings({}, {name: 3})
            with self.assertRaisesRegex(RuntimeError, name):
                bench.validate_resume_settings({name: 3}, {name: 5})
            with patch.object(sys, "argv", ["benchmark", "--" + name.replace("_", "-"), "0"]), \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args()


if __name__ == "__main__":
    unittest.main()

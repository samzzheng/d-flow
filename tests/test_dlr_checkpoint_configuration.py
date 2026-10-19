"""Checkpoint configuration reaches every supported DLR execution path."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow import enriched, tangent_unet
from dlr_dflow.graphed_evolution import GraphedZeroEnrichmentStep
from dlr_dflow.graphed_products import GraphedAProducts
from models.unet.default import Unet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


def represented_matrix(factors):
    u, s, v = factors
    return u @ s @ v.T


class CheckpointConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(922)
        self.model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, 0],
                          groups=2, dropout=0., mid_attn=False).eval().requires_grad_(False)
        self.z = torch.randn(1, 1, 7, 9)

    def test_short_evolution_preserves_state_and_source_gradient(self):
        terminal_gradient = torch.randn_like(self.z)
        results = []
        for checkpoint_blocks in (False, True):
            kwargs = dict(rank=2, trace_probes=2, initialization_oversampling=2,
                          enrichment_probes=0, zero_terminal_baseline=True,
                          checkpoint_blocks=checkpoint_blocks)
            results.append(core.sensitivity_factors(self.model, self.z, 3, **kwargs))
        expected, actual = results
        torch.testing.assert_close(actual[0], expected[0], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(represented_matrix(actual[1]),
                                   represented_matrix(expected[1]), atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(core.approx_source_gradient(actual[1], terminal_gradient, actual[2]),
                                   core.approx_source_gradient(expected[1], terminal_gradient, expected[2]),
                                   atol=3e-6, rtol=3e-5)
        self.assertEqual(actual[2], expected[2])

    def test_complete_step_backend_preserves_replay_and_reset(self):
        _, factors, _ = core.sensitivity_factors(
            self.model, self.z, 2, rank=2, trace_probes=2,
            initialization_oversampling=2, enrichment_probes=0)
        probes = torch.randn(self.z.numel(), 2)
        backends = [GraphedZeroEnrichmentStep(
            self.model, self.z, factors, .5 * probes, probes, .5, .1,
            capture=False, checkpoint_blocks=checkpoint_blocks)
            for checkpoint_blocks in (False, True)]
        for restart in (False, True):
            if restart:
                for backend in backends:
                    backend.reset(.4 * self.z, factors, .3 * probes, .3, probes=-probes)
            for time in (.2, .3):
                for backend in backends:
                    backend.run(time)
                expected, actual = [backend.snapshot() for backend in backends]
                for value, reference in ((actual[0], expected[0]),
                                         (represented_matrix(actual[1]), represented_matrix(expected[1])),
                                         (actual[2], expected[2]), (actual[3], expected[3])):
                    torch.testing.assert_close(value, reference, atol=3e-6, rtol=3e-5)
        backends[1].checkpoint_blocks = False
        with self.assertRaisesRegex(ValueError, "checkpoint_blocks changed"):
            backends[1].run(.4)

    def test_graph_provider_propagates_mode_and_scopes_graph_lifetime(self):
        products = GraphedAProducts(self.model, self.z, checkpoint_blocks=True)
        args = (self.z, (torch.empty(63, 2), torch.empty(2, 2), torch.empty(63, 2)),
                torch.empty(63, 2), torch.empty(63, 2), .5, .1)
        with patch("dlr_dflow.graphed_evolution.GraphedZeroEnrichmentStep",
                   wraps=GraphedZeroEnrichmentStep) as backend_class:
            # Avoid any CUDA allocation while verifying the real provider cache.
            backend_class.side_effect = lambda *a, **kw: SimpleNamespace(reset=lambda *args: None)
            products.prepare_step_graph(*args)
            products.prepare_step_graph(*args)
            self.assertEqual(backend_class.call_count, 1)
            self.assertTrue(backend_class.call_args.kwargs["checkpoint_blocks"])
        products.checkpoint_blocks = False
        for operation in (lambda: products.set_state(.2, self.z),
                          lambda: products.A(torch.ones(63, 2)),
                          lambda: products.prepare_step_graph(*args)):
            with self.assertRaisesRegex(ValueError, "checkpoint_blocks changed"):
                operation()

    def test_incompatible_configurations_fail_explicitly(self):
        for name in ("SHARE_LINEARIZATION", "HAND_JVP", "HAND_VJP"):
            with self.subTest(flag=name), patch.object(core, name, False):
                with self.assertRaisesRegex(ValueError, "shared hand-written"):
                    core.make_A_products(self.model, .2, self.z, checkpoint_blocks=True)
                with self.assertRaisesRegex(ValueError, "shared hand-written"):
                    GraphedAProducts(self.model, self.z, checkpoint_blocks=True)
        for method in ("legacy", "direct"):
            with self.assertRaisesRegex(ValueError, "enriched"):
                core.sensitivity_factors(self.model, self.z, 2, method=method, checkpoint_blocks=True)
            with self.assertRaisesRegex(ValueError, "enriched"):
                core.solve_inverse_lbfgs(self.model, self.z, lambda x: x, self.z,
                                         method=method, dlr_activation_checkpointing=True)
        for requested in (False, True):
            provider = SimpleNamespace(checkpoint_blocks=not requested)
            with self.assertRaisesRegex(ValueError, "must match"):
                enriched.evolve_trace_centered_enriched(
                    self.model, self.z, 2, products=provider,
                    checkpoint_blocks=requested)

    def test_core_optimizer_forwards_checkpointing_to_refresh_and_provider(self):
        dimension = self.z.numel()
        factors = (torch.ones(dimension, 1), torch.ones(1, 1), torch.ones(dimension, 1))
        with patch.object(core, "sensitivity_factors", return_value=(self.z, factors, 0.)) as sensitivity, \
                patch("dlr_dflow.graphed_products.GraphedAProducts") as products:
            core.solve_inverse_lbfgs(
                self.model, self.z, lambda x: x, torch.zeros_like(self.z),
                n_steps=1, rank=1, outer_steps=1,
                dlr_activation_checkpointing=True, use_cuda_graphs=True)
        self.assertTrue(sensitivity.call_args.kwargs["checkpoint_blocks"])
        self.assertTrue(products.call_args.kwargs["checkpoint_blocks"])
        self.assertIs(sensitivity.call_args.kwargs["products"], products.return_value)

    def test_graph_configuration_rejects_changed_normalization_backend(self):
        with patch.object(tangent_unet, "GN_ACTION_BACKEND", "native", create=True):
            products = GraphedAProducts(self.model, self.z)
            basis = torch.linalg.qr(torch.randn(self.z.numel(), 2))[0]
            probes = torch.randn(self.z.numel(), 2)
            backend = GraphedZeroEnrichmentStep(
                self.model, self.z, (basis, torch.eye(2), basis), probes, probes,
                1., .01, capture=False)
        with patch.object(tangent_unet, "GN_ACTION_BACKEND", "triton", create=True):
            for operation in (lambda: products.set_state(.2, self.z),
                              lambda: products.A(probes),
                              lambda: products.prepare_linearization(),
                              lambda: backend.run(.2)):
                with self.assertRaisesRegex(ValueError, "GN_ACTION_BACKEND changed"):
                    operation()


class CheckpointBenchmarkTests(unittest.TestCase):
    def test_specialized_algebra_and_channels_last_cli_resume(self):
        names = ("dlr_specialized_algebra", "channels_last")
        with patch.object(sys, "argv", ["benchmark"]):
            defaults = bench.parse_args()
        for name in names:
            self.assertFalse(getattr(defaults, name))
            flag = name.replace("_", "-")
            with patch.object(sys, "argv", ["benchmark", "--" + flag]):
                self.assertTrue(getattr(bench.parse_args(), name))
            with patch.object(sys, "argv", ["benchmark", "--no-" + flag]):
                self.assertFalse(getattr(bench.parse_args(), name))
            bench.validate_resume_settings({}, {name: False})
            bench.validate_resume_settings({name: False}, {})
            for existing in ({}, {name: False}):
                with self.assertRaisesRegex(RuntimeError, name):
                    bench.validate_resume_settings(existing, {name: True})

    def test_execution_configuration_applies_all_flags_before_capture(self):
        original_model = torch.nn.Conv2d(4, 8, 3)
        with patch.object(tangent_unet, "GN_ACTION_BACKEND", "native", create=True), \
                patch.object(tangent_unet, "GN_PRIMAL_BACKEND", "native"), \
                patch.object(enriched, "SPECIALIZE_ZERO_ENRICHMENT", False), \
                patch.object(bench, "configure_dlr_kernels") as kernels:
            args = SimpleNamespace(dlr_fused_norm=True, dlr_fused_primal="full", dlr_specialized_algebra=True,
                                   channels_last=True)
            actual = bench.configure_execution(original_model, 250, args)
            self.assertIs(actual, original_model)
            kernels.assert_called_once_with(250)
            self.assertEqual(tangent_unet.GN_ACTION_BACKEND, "triton")
            self.assertEqual(tangent_unet.GN_PRIMAL_BACKEND, "full")
            self.assertTrue(enriched.SPECIALIZE_ZERO_ENRICHMENT)
            self.assertTrue(actual.weight.is_contiguous(memory_format=torch.channels_last))
            default_model = torch.nn.Conv2d(4, 8, 3)
            args = SimpleNamespace(dlr_fused_norm=False, dlr_fused_primal="native", dlr_specialized_algebra=False,
                                   channels_last=False)
            actual = bench.configure_execution(default_model, 250, args)
            self.assertEqual(tangent_unet.GN_ACTION_BACKEND, "native")
            self.assertFalse(enriched.SPECIALIZE_ZERO_ENRICHMENT)
            self.assertTrue(actual.weight.is_contiguous())

    def test_fused_norm_cli_and_legacy_resume_identity(self):
        with patch.object(sys, "argv", ["benchmark"]):
            default = bench.parse_args()
        self.assertFalse(default.dlr_fused_norm)
        with patch.object(sys, "argv", ["benchmark", "--dlr-fused-norm"]):
            self.assertTrue(bench.parse_args().dlr_fused_norm)
        with patch.object(sys, "argv", ["benchmark", "--no-dlr-fused-norm"]):
            self.assertFalse(bench.parse_args().dlr_fused_norm)
        bench.validate_resume_settings({}, {"dlr_fused_norm": False})
        bench.validate_resume_settings({"dlr_fused_norm": False}, {})
        for existing in ({}, {"dlr_fused_norm": False}):
            with self.assertRaisesRegex(RuntimeError, "dlr_fused_norm"):
                bench.validate_resume_settings(existing, {"dlr_fused_norm": True})

    def test_cli_modes_are_independent_and_immutable_on_resume(self):
        with patch.object(sys, "argv", ["benchmark"]):
            default = bench.parse_args()
        self.assertFalse(default.dlr_activation_checkpointing)
        self.assertIsNone(default.activation_checkpointing)
        with patch.object(sys, "argv", ["benchmark", "--dlr-activation-checkpointing",
                                       "--no-activation-checkpointing"]):
            enabled = bench.parse_args()
        self.assertTrue(enabled.dlr_activation_checkpointing)
        self.assertFalse(enabled.activation_checkpointing)
        bench.validate_resume_settings(vars(enabled), vars(enabled).copy())
        bench.validate_resume_settings({}, {"dlr_activation_checkpointing": False})
        bench.validate_resume_settings({"dlr_activation_checkpointing": False}, {})
        for existing in ({}, {"dlr_activation_checkpointing": False}):
            with self.assertRaisesRegex(RuntimeError, "dlr_activation_checkpointing"):
                bench.validate_resume_settings(existing, {"dlr_activation_checkpointing": True})
        with patch.object(sys, "argv", ["benchmark", "--no-dlr-activation-checkpointing",
                                       "--activation-checkpointing"]):
            disabled = bench.parse_args()
        self.assertFalse(disabled.dlr_activation_checkpointing)
        self.assertTrue(disabled.activation_checkpointing)

    def test_benchmark_forwards_checkpoint_mode_for_ksl(self):
        with patch.object(sys, "argv", [
            "benchmark", "--dlr-activation-checkpointing",
            "--outer-steps", "1", "--lbfgs-max-iter", "1", "--euler-flow-steps", "1",
        ]):
            args = bench.parse_args()
        state = torch.ones(1, 1, 2, 2)
        factors = (torch.eye(4), torch.eye(4), torch.eye(4))
        provider = SimpleNamespace(checkpoint_blocks=True)
        with patch.object(core, "sensitivity_factors", return_value=(2 * state, factors, 0.)) as evolve, \
                patch.object(bench, "solve_flow", side_effect=lambda model, x, *a: x), \
                patch.object(bench, "sparse_radon", side_effect=lambda x, *a: x), \
                patch.object(bench, "save_diagnostic_figure"), \
                patch.object(bench.np, "savez_compressed"), redirect_stdout(io.StringIO()):
            bench.run_inverse(
                torch.nn.Identity(), state, state, 1.01 * state,
                torch.ones(1), 2, 0, "dlr", 2 * state, "test", "CPU", args, products=provider)
        self.assertTrue(evolve.call_args.kwargs["checkpoint_blocks"])
        self.assertIs(evolve.call_args.kwargs["products"], provider)


if __name__ == "__main__":
    unittest.main()

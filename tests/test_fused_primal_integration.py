"""Optional primal fusion reaches shared tapes without changing CPU defaults."""

from contextlib import ExitStack, redirect_stderr
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from dlr_dflow import fused_primal, tangent_unet
from dlr_dflow.graphed_evolution import GraphedZeroEnrichmentStep
from dlr_dflow.graphed_products import GraphedAProducts
from models.unet.default import Unet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


def model_problem(device="cpu"):
    torch.manual_seed(853)
    model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, 0], mid_attn=False,
                 groups=2, dropout=0., scale_shift=True).to(device).eval().requires_grad_(False)
    return model, torch.randn(1, 1, 7, 9, device=device)


class FusedPrimalIntegrationCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_cpu_tapes_remain_bitwise_native_and_do_not_call_cuda_helper(self):
        model, x = model_problem()
        tangents, cotangents = torch.randn(2, 1, 7, 9), torch.randn(3, 1, 7, 9)
        for checkpoint_blocks in (False, True):
            with patch.object(tangent_unet, "GN_PRIMAL_BACKEND", "native"):
                reference = tangent_unet.unet_linearize(model, .37, x, checkpoint_blocks=checkpoint_blocks)
                expected = (reference.primal, reference.jvp(tangents), reference.vjp(cotangents))
            for mode in ("affine", "full"):
                with self.subTest(mode=mode, checkpoint=checkpoint_blocks), \
                        patch.object(tangent_unet, "GN_PRIMAL_BACKEND", mode), \
                        patch.object(fused_primal, "normalization_activation_record",
                                     side_effect=AssertionError("CPU imported a CUDA primal path")):
                    tape = tangent_unet.unet_linearize(model, .37, x, checkpoint_blocks=checkpoint_blocks)
                    actual = (tape.primal, tape.jvp(tangents), tape.vjp(cotangents))
                for value, reference_value in zip(actual, expected):
                    torch.testing.assert_close(value, reference_value, rtol=0, atol=0)

    def test_invalid_backend_and_graph_policy_mutation_are_rejected(self):
        model, x = model_problem()
        with patch.object(tangent_unet, "GN_PRIMAL_BACKEND", "bad"):
            with self.assertRaisesRegex(ValueError, "GN_PRIMAL_BACKEND"):
                tangent_unet.unet_linearize(model, .3, x)
        with patch.object(tangent_unet, "GN_PRIMAL_BACKEND", "native"):
            provider = GraphedAProducts(model, x)
            basis = torch.linalg.qr(torch.randn(x.numel(), 2))[0]
            probes = torch.randn(x.numel(), 2)
            backend = GraphedZeroEnrichmentStep(model, x, (basis, torch.eye(2), basis),
                                                probes, probes, 1., .01, capture=False)
        with patch.object(tangent_unet, "GN_PRIMAL_BACKEND", "full"):
            for operation in (lambda: provider.set_state(.3, x),
                              lambda: provider.A(probes), lambda: provider.prepare_linearization(),
                              lambda: backend.run(.3)):
                with self.assertRaisesRegex(ValueError, "GN_PRIMAL_BACKEND changed"):
                    operation()

    def test_cli_and_legacy_resume_identity(self):
        with patch.object(sys, "argv", ["benchmark"]):
            self.assertEqual(bench.parse_args().dlr_fused_primal, "native")
        for mode in ("native", "affine", "full"):
            with patch.object(sys, "argv", ["benchmark", "--dlr-fused-primal", mode]):
                self.assertEqual(bench.parse_args().dlr_fused_primal, mode)
            bench.validate_resume_settings({"dlr_fused_primal": mode}, {"dlr_fused_primal": mode})
        bench.validate_resume_settings({}, {"dlr_fused_primal": "native"})
        bench.validate_resume_settings({"dlr_fused_primal": "native"}, {})
        for mode in ("affine", "full"):
            with self.assertRaisesRegex(RuntimeError, "dlr_fused_primal"):
                bench.validate_resume_settings({}, {"dlr_fused_primal": mode})
        with self.assertRaisesRegex(RuntimeError, "dlr_fused_primal"):
            bench.validate_resume_settings({"dlr_fused_primal": "affine"}, {"dlr_fused_primal": "full"})
        with patch.object(sys, "argv", ["benchmark", "--dlr-fused-primal", "bad"]), \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for fused primal tape")
class FusedPrimalIntegrationCUDA(unittest.TestCase):
    def test_small_nonattention_tape_matches_native_autodiff(self):
        device = torch.device("cuda", torch.cuda.current_device())
        with ExitStack() as contexts:
            contexts.enter_context(torch.random.fork_rng(devices=[device.index]))
            contexts.enter_context(patch.multiple(tangent_unet, GN_STATS_MODE="manual",
                                                 FUSED_GN=True, FUSE_POINTWISE_DERIVATIVES=True,
                                                 GN_ACTION_BACKEND="native"))
            model, x = model_problem(device)
            directions = torch.randn(2, 1, 7, 9, device=device)
            cotangents = torch.randn(2, 1, 7, 9, device=device)
            time = x.new_tensor([.37])
            velocity = lambda value: model(value, time)
            expected_jvp = torch.cat([torch.autograd.functional.jvp(
                velocity, x, direction[None], strict=True)[1] for direction in directions])
            variable = x.detach().requires_grad_(True)
            native_primal = velocity(variable)
            expected_vjp = torch.cat([torch.autograd.grad(native_primal, variable,
                cotangent[None], retain_graph=True)[0] for cotangent in cotangents])
            for mode in ("affine", "full"):
                for checkpoint_blocks in (False, True):
                    with self.subTest(mode=mode, checkpoint=checkpoint_blocks), \
                            patch.object(tangent_unet, "GN_PRIMAL_BACKEND", mode), \
                            patch.object(fused_primal, "normalization_activation_record",
                                         wraps=fused_primal.normalization_activation_record) as fused:
                        tape = tangent_unet.unet_linearize(model, time, x, checkpoint_blocks=checkpoint_blocks)
                        actual_jvp, actual_vjp = tape.jvp(directions), tape.vjp(cotangents)
                        self.assertGreater(fused.call_count, 0)
                        for value, reference in ((tape.primal, native_primal.detach()),
                                                 (actual_jvp, expected_jvp), (actual_vjp, expected_vjp)):
                            relative = (value - reference).norm() / reference.norm().clamp_min(1e-12)
                            self.assertLess(relative.item(), 3e-4)


if __name__ == "__main__":
    unittest.main()

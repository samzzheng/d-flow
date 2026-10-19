"""State ownership and replay checks for the optional full-step backend."""

import unittest
from unittest.mock import patch

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow import enriched
from dlr_dflow.graphed_evolution import GraphedZeroEnrichmentStep
from models.unet.default import Unet


def reconstruct(factors):
    u, s, v = factors
    return u @ s @ v.T


def problem(device="cpu"):
    torch.manual_seed(1307)
    model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, 1], groups=2,
                 dropout=0.).to(device).eval().requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(807)
    x = torch.randn(1, 1, 8, 8, device=device, generator=generator)
    u = torch.linalg.qr(torch.randn(64, 3, device=device, generator=generator))[0]
    v = torch.linalg.qr(torch.randn(64, 3, device=device, generator=generator))[0]
    s = torch.diag(torch.tensor([.1, .05, .02], device=device))
    probes = torch.randn(64, 2, device=device, generator=generator)
    sensitivities = .7 * probes + u @ (s @ (v.T @ probes))
    return model, x, (u, s, v), sensitivities, probes, .7


class EagerWholeStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_eager_step_matches_reference_and_owns_state(self):
        model, x, factors, sensitivities, probes, alpha = problem()
        saved_x = x.clone()
        saved_factors = tuple(value.clone() for value in factors)
        backend = GraphedZeroEnrichmentStep(model, x, factors, sensitivities,
                                           probes, alpha, .05, capture=False)
        for t in (.1, .15, .2):
            with patch.multiple(core, SHARE_LINEARIZATION=True, HAND_JVP=True, HAND_VJP=True):
                a, at = core.make_A_products(model, t, x)
                factors, derivative, _ = enriched._fused_enriched_step(
                    a, at, factors, sensitivities, probes, .05, alpha, 0, None)
                sensitivities = sensitivities + .05 * derivative
                alpha = (probes * sensitivities).sum() / probes.numel()
                x = x + .05 * a.last_primal
            backend.run(t)
            state, actual, q, scalar = backend.snapshot()
            torch.testing.assert_close(state, x, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(reconstruct(actual), reconstruct(factors),
                                       atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(q, sensitivities, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(scalar, alpha)
            torch.testing.assert_close(actual[1], torch.diag(torch.diag(actual[1])), atol=0, rtol=0)
        backend.reset(saved_x, saved_factors, probes, .8)
        snapshot = backend.snapshot(canonicalize=False)
        snapshot[0].zero_()
        snapshot[1][1].zero_()
        torch.testing.assert_close(backend.x, saved_x, atol=0, rtol=0)
        torch.testing.assert_close(backend.factors[1], saved_factors[1], atol=0, rtol=0)

    def test_rejects_invalid_reset_before_mutation(self):
        model, x, factors, sensitivities, probes, alpha = problem()
        # Match the probe width to the rank so a missing V would otherwise
        # let zip pair sensitivities with the V buffer without a shape error.
        probes = torch.cat((probes, probes[:, :1]), dim=1)
        sensitivities = torch.cat((sensitivities, sensitivities[:, :1]), dim=1)
        backend = GraphedZeroEnrichmentStep(model, x, factors, sensitivities,
                                           probes, alpha, .05, capture=False)
        with self.assertRaisesRegex(ValueError, "reset must preserve"):
            backend.reset(torch.zeros_like(x), factors, sensitivities[:, :1], alpha)
        torch.testing.assert_close(backend.x, x, atol=0, rtol=0)
        saved = backend.snapshot(canonicalize=False)
        for malformed in (factors[:2], (*factors, factors[2])):
            with self.subTest(factor_count=len(malformed)):
                with self.assertRaisesRegex(ValueError, "exactly three tensors"):
                    backend.reset(torch.zeros_like(x), malformed,
                                  torch.zeros_like(sensitivities), .2)
                actual = backend.snapshot(canonicalize=False)
                for value, reference in zip((actual[0], *actual[1], actual[2], actual[3]),
                                            (saved[0], *saved[1], saved[2], saved[3])):
                    torch.testing.assert_close(value, reference, atol=0, rtol=0)
        with patch.object(enriched, "FUSE_K_IMAGE", not enriched.FUSE_K_IMAGE):
            with self.assertRaisesRegex(ValueError, "algebra flags changed"):
                backend.run(.1)

    def test_enriched_provider_integration_and_return_ownership(self):
        model, z, _, _, _, _ = problem()

        class Provider:
            full_step_graphs = True

            def __init__(self):
                self.backend = None
                self.preparations = 0

                def a(q):
                    result = self._a(q)
                    self.A.last_primal = self._a.last_primal
                    return result

                self.A = a
                self.AT = lambda q: self._at(q)

            def set_state(self, t, x):
                self._a, self._at = core.make_A_products(model, t, x)

            def prepare_step_graph(self, x, factors, sensitivities, probes, alpha, step_size):
                self.preparations += 1
                if self.backend is None:
                    self.backend = GraphedZeroEnrichmentStep(
                        model, x, factors, sensitivities, probes, alpha,
                        step_size, capture=False)
                else:
                    self.backend.reset(x, factors, sensitivities, alpha, probes)
                return self.backend

        def run(source, provider=None):
            with patch.multiple(core, SHARE_LINEARIZATION=True, HAND_JVP=True, HAND_VJP=True):
                return enriched.evolve_trace_centered_enriched(
                    model, source, n_steps=5, rank=3, trace_probes=2,
                    enrichment_probes=0, initialization_oversampling=2,
                    return_alpha_path=True, products=provider)

        provider = Provider()
        reference = run(z)
        actual = run(z, provider)
        original = (actual[0].clone(), tuple(value.clone() for value in actual[1]))
        self.assertEqual(provider.preparations, 1)
        torch.testing.assert_close(actual[0], reference[0], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(reconstruct(actual[1]), reconstruct(reference[1]),
                                   atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(torch.tensor(actual[3]), torch.tensor(reference[3]))
        run(.3 * z, provider)
        self.assertEqual(provider.preparations, 2)
        torch.testing.assert_close(actual[0], original[0], atol=0, rtol=0)
        for value, saved in zip(actual[1], original[1]):
            torch.testing.assert_close(value, saved, atol=0, rtol=0)

    def test_disabled_shared_primal_uses_velocity_override_and_product_fallback(self):
        model, z, _, _, _, _ = problem()

        class Provider:
            full_step_graphs = True

            def __init__(self):
                def a(q):
                    result = self._a(q)
                    self.A.last_primal = self._a.last_primal
                    return result

                self.A = a
                self.AT = lambda q: self._at(q)

            def set_state(self, t, x):
                self._a, self._at = core.make_A_products(model, t, x)

            def prepare_step_graph(self, *args):
                raise AssertionError("SHARE_PRIMAL=False must disable whole-step replay")

        original_velocity = core.velocity

        def override(model, t, x):
            return original_velocity(model, t, x) + .07

        results = []
        for provider in (None, Provider()):
            with patch.multiple(core, SHARE_LINEARIZATION=True, HAND_JVP=True, HAND_VJP=True), \
                    patch.object(enriched, "SHARE_PRIMAL", False), \
                    patch.object(core, "velocity", side_effect=override) as velocity:
                results.append(enriched.evolve_trace_centered_enriched(
                    model, z, n_steps=5, rank=3, trace_probes=2,
                    enrichment_probes=0, initialization_oversampling=2,
                    return_alpha_path=True, products=provider))
                self.assertEqual(velocity.call_count, 5)
        reference, actual = results
        torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
        torch.testing.assert_close(reconstruct(actual[1]), reconstruct(reference[1]), atol=0, rtol=0)
        self.assertEqual(actual[3], reference[3])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph replay requires CUDA")
class CapturedWholeStepTests(unittest.TestCase):
    def test_replay_and_reset_match_eager_backend(self):
        model, x, factors, sensitivities, probes, alpha = problem("cuda")
        eager = GraphedZeroEnrichmentStep(model, x, factors, sensitivities,
                                         probes, alpha, .05, capture=False)
        captured = GraphedZeroEnrichmentStep(model, x, factors, sensitivities,
                                            probes, alpha, .05)
        for restart in (False, True):
            if restart:
                for backend in (eager, captured):
                    backend.reset(.3 * x, factors, .5 * sensitivities, .35,
                                  probes=-probes)
            for t in (.1, .15, .2):
                eager.run(t)
                captured.run(t)
                expected = eager.snapshot()
                actual = captured.snapshot()
                for value, reference in ((actual[0], expected[0]),
                                         (reconstruct(actual[1]), reconstruct(expected[1])),
                                         (actual[2], expected[2]), (actual[3], expected[3])):
                    # QR and convolution kernel selection may differ during
                    # capture; norm error catches stale state more directly.
                    relative = (value - reference).norm() / reference.norm().clamp_min(1e-12)
                    self.assertLess(relative.item(), 2e-4)


if __name__ == "__main__":
    unittest.main()

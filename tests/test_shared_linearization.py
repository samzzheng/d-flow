"""One hand primal tape must supply both exact Jacobian actions repeatedly."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from dlr_dflow import tangent_unet as tangent
from models.unet.default import Unet


class CastEmbedding(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, value):
        return value.to(self.dtype)


class SharedLinearizationTests(unittest.TestCase):
    def test_wide_time_embedding_matches_autodiff(self) -> None:
        torch.manual_seed(420)
        model = Unet(ch=6, ch_mul=[1, 2, 3, 3], att_channels=[0]*4,
                     groups=2, dropout=0, mid_attn=False,
                     time_embed_dim=32).eval().requires_grad_(False)
        x = torch.randn(1, 1, 16, 16)
        direction = torch.randn_like(x)
        time = torch.tensor([.37])

        def velocity(value: torch.Tensor) -> torch.Tensor:
            return model(value, time)

        primal, pullback = torch.func.vjp(velocity, x)
        expected_jvp = torch.func.jvp(velocity, (x,), (direction,))[1]
        expected_vjp = pullback(direction)[0]
        for checkpoint in (False, True):
            tape = tangent.unet_linearize(model, time, x, checkpoint_blocks=checkpoint)
            self.assert_close(tape.primal, primal)
            self.assert_close(tape.jvp(direction), expected_jvp)
            self.assert_close(tape.vjp(direction), expected_vjp)
        jvp_primal, actual_jvp = tangent.unet_jvp(model, time, x, direction)
        vjp_primal, actual_vjp = tangent.unet_vjp(model, time, x, direction)
        self.assert_close(jvp_primal, primal)
        self.assert_close(vjp_primal, primal)
        self.assert_close(actual_jvp, expected_jvp)
        self.assert_close(actual_vjp, expected_vjp)

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_model(self, dtype, scale_shift, attention):
        torch.manual_seed(420)
        model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, int(attention)],
                     groups=2, dropout=.1, scale_shift=scale_shift,
                     mid_attn=attention).to(dtype).eval().requires_grad_(False)
        model.time_proj = nn.Sequential(CastEmbedding(dtype), model.time_proj)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.GroupNorm):
                    module.weight.uniform_(-.4, 1.4)
                    module.weight[0] = 0
                    module.bias.normal_(0, .1)
        return model

    def assert_close(self, actual, expected):
        if actual.dtype == torch.float64:
            torch.testing.assert_close(actual, expected, rtol=3e-9, atol=3e-10)
        else:
            torch.testing.assert_close(actual, expected, rtol=4e-4, atol=6e-6)

    def test_primal_and_products_match_autodiff(self):
        for dtype in (torch.float32, torch.float64):
            for scale_shift in (False, True):
                for attention in (False, True):
                    model = self.make_model(dtype, scale_shift, attention)
                    x = torch.randn(1, 1, 5, 7, dtype=dtype)
                    directions = torch.randn(2, 1, 5, 7, dtype=dtype)
                    cotangents = torch.randn(3, 1, 5, 7, dtype=dtype)
                    time = torch.tensor([.37], dtype=dtype)
                    with sdpa_kernel(SDPBackend.MATH):
                        velocity = lambda value: model(value, time)
                        primal, pullback = torch.func.vjp(velocity, x)
                        jvp = torch.cat([torch.func.jvp(
                            velocity, (x,), (lane[None],))[1] for lane in directions])
                        vjp = torch.cat([pullback(lane[None])[0] for lane in cotangents])
                    modes = (("moments", "moments"), ("chunked", "chunked"),
                             ("moments", "chunked"), ("chunked", "moments")) if attention else (("moments", "moments"),)
                    for jvp_mode, vjp_mode in modes:
                        for stats_mode in ("manual", "native"):
                            with self.subTest(dtype=dtype, scale_shift=scale_shift,
                                              attention=attention, jvp=jvp_mode,
                                              vjp=vjp_mode, stats=stats_mode), \
                                    patch.object(tangent, "ATTENTION_JVP_MODE", jvp_mode), \
                                    patch.object(tangent, "ATTENTION_VJP_MODE", vjp_mode), \
                                    patch.object(tangent, "ATTENTION_SCORE_BYTES", 256), \
                                    patch.object(tangent, "GN_STATS_MODE", stats_mode):
                                tape = tangent.unet_linearize(model, time, x)
                                self.assert_close(tape.primal, primal)
                                self.assert_close(tape.jvp(directions), jvp)
                                self.assert_close(tape.vjp(cotangents), vjp)
                                self.assertFalse(tape.primal.requires_grad)

    def test_repeated_mixed_width_actions_preserve_tape_and_outputs(self):
        model = self.make_model(torch.float64, True, True)
        x = torch.randn(1, 1, 7, 9, dtype=torch.float64)
        directions = torch.randn(3, 1, 7, 9, dtype=torch.float64)
        cotangents = torch.randn(3, 1, 7, 9, dtype=torch.float64)
        tape = tangent.unet_linearize(model, .29, x)

        def tensors(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, dict):
                for child in value.values():
                    yield from tensors(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from tensors(child)

        saved = [(value, value.clone()) for value in tensors([
            tape.primal, tape.tape_down, tape.tape_up, tape.mid0, tape.mid1,
            tape.final, x, directions, cotangents])]
        # Neither action should rebuild primal normalization statistics,
        # attention probabilities, or the time conditioning.
        with patch.object(tangent, "_gn_stats", side_effect=AssertionError("recomputed GN")), \
                patch.object(tangent, "_attn_fwd", side_effect=AssertionError("recomputed attention")), \
                patch.object(model.time_proj, "forward", side_effect=AssertionError("recomputed time")):
            jvp = tape.jvp(directions)
            vjp = tape.vjp(cotangents)
            saved.extend([(jvp, jvp.clone()), (vjp, vjp.clone())])
            self.assert_close(tape.jvp(directions[:1]), jvp[:1])
            self.assert_close(tape.vjp(cotangents[1:]), vjp[1:])
            self.assert_close(tape.jvp(directions[1:]), jvp[1:])
            self.assert_close(tape.vjp(cotangents[:1]), vjp[:1])
            self.assert_close((jvp * cotangents).sum(), (directions * vjp).sum())
        for actual, snapshot in saved:
            torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)

    def test_fused_pointwise_matches_separate_operations(self):
        """Cached factors must include GN weights exactly once, including zeros."""
        for dtype in (torch.float32, torch.float64):
            for scale_shift in (False, True):
                for fused_gn in (False, True):
                    with self.subTest(dtype=dtype, scale_shift=scale_shift,
                                      fused_gn=fused_gn), \
                            patch.object(tangent, "FUSED_GN", fused_gn):
                        model = self.make_model(dtype, scale_shift, True)
                        x = torch.randn(1, 1, 5, 7, dtype=dtype)
                        directions = torch.randn(3, 1, 5, 7, dtype=dtype)
                        cotangents = torch.randn(2, 1, 5, 7, dtype=dtype)
                        with patch.object(tangent, "FUSE_POINTWISE_DERIVATIVES", False):
                            separate = tangent.unet_linearize(model, .43, x)
                        with patch.object(tangent, "FUSE_POINTWISE_DERIVATIVES", True):
                            fused = tangent.unet_linearize(model, .43, x)
                        torch.testing.assert_close(fused.primal, separate.primal,
                                                   rtol=0, atol=0)
                        self.assertNotIn("pointwise_derivative", separate.final)
                        self.assertIn("pointwise_derivative", fused.final)
                        for actual, expected in ((fused.jvp(directions), separate.jvp(directions)),
                                                 (fused.vjp(cotangents), separate.vjp(cotangents))):
                            self.assert_close(actual, expected)
                            relative = (actual - expected).norm() / expected.norm()
                            self.assertLess(relative.item(),
                                            1e-5 if dtype == torch.float32 else 1e-12)

    def test_training_model_is_rejected(self):
        model = self.make_model(torch.float32, True, False).train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            tangent.unet_linearize(model, .2, torch.randn(1, 1, 5, 7))


if __name__ == "__main__":
    unittest.main()

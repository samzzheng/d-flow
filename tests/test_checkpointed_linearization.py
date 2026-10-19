"""Block recomputation preserves exact hand derivatives and reduces tape storage."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from dlr_dflow import tangent_unet as tangent
from models.unet.default import Unet


class _CastEmbedding(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, value):
        return value.to(self.dtype)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensors(child)


def _tape_tensors(tape):
    # Deliberately omit the model: parameters are shared, not activation tape.
    return list(_tensors([tape.primal, tape.tape_down, tape.tape_up,
                          tape.mid0, tape.mid1, tape.final]))


def _owned_bytes(tape):
    external = {parameter.untyped_storage().data_ptr()
                for parameter in tape.model.parameters()}
    external.update(buffer.untyped_storage().data_ptr()
                    for buffer in tape.model.buffers())
    storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                for value in _tape_tensors(tape)
                if value.untyped_storage().data_ptr() not in external}
    return sum(storages.values())


class CheckpointedLinearizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    @staticmethod
    def make_model(dtype=torch.float64, scale_shift=True, attention=False):
        torch.manual_seed(20260928)
        model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, int(attention)],
                     groups=2, dropout=.2, scale_shift=scale_shift,
                     mid_attn=attention).to(dtype).eval().requires_grad_(False)
        model.time_proj = nn.Sequential(_CastEmbedding(dtype), model.time_proj)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.GroupNorm):
                    module.weight.uniform_(-.5, 1.5)
                    module.weight[0] = 0
                    module.bias.normal_(0, .1)
        return model

    @staticmethod
    def assert_close(actual, expected):
        if actual.dtype == torch.float64:
            torch.testing.assert_close(actual, expected, rtol=4e-9, atol=4e-10)
        else:
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=8e-6)

    def test_primal_jvp_vjp_match_autodiff_on_odd_shapes(self):
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
                        expected_jvp = torch.cat([torch.func.jvp(
                            velocity, (x,), (lane[None],))[1] for lane in directions])
                        expected_vjp = torch.cat([pullback(lane[None])[0]
                                                  for lane in cotangents])
                    modes = ("moments", "chunked") if attention else ("moments",)
                    for mode in modes:
                        for stats in ("manual", "native"):
                            with self.subTest(dtype=dtype, scale_shift=scale_shift,
                                              attention=attention, mode=mode, stats=stats), \
                                    patch.object(tangent, "GN_STATS_MODE", stats), \
                                    patch.object(tangent, "ATTENTION_MODE", mode), \
                                    patch.object(tangent, "ATTENTION_SCORE_BYTES", 256):
                                tape = tangent.unet_linearize(
                                    model, time, x, checkpoint_blocks=True)
                                self.assert_close(tape.primal, primal)
                                self.assert_close(tape.jvp(directions), expected_jvp)
                                self.assert_close(tape.vjp(cotangents), expected_vjp)
                                self.assertFalse(tape.primal.requires_grad)

    def test_repeated_actions_adjoint_identity_and_input_ownership(self):
        model = self.make_model(attention=True)
        x = torch.randn(1, 1, 7, 9, dtype=torch.float64)
        time = torch.tensor([.29], dtype=torch.float64)
        directions = torch.randn(3, 1, 7, 9, dtype=torch.float64)
        cotangents = torch.randn(3, 1, 7, 9, dtype=torch.float64)
        tape = tangent.unet_linearize(model, time, x, checkpoint_blocks=True)
        cached = tangent.unet_linearize(model, time, x)
        tensors_before = _tape_tensors(tape)
        saved = [(value, value.clone()) for value in
                 tensors_before + [directions, cotangents]]
        storage_before = _owned_bytes(tape)
        # The source/time belong to the caller. Mutating them must not alter
        # any checkpoint or its later relinearization.
        x.add_(100)
        time.fill_(.91)
        n_blocks = 2 * (len(model.down) + len(model.up)) + 2
        with patch.object(tangent, "_resblock_fwd", wraps=tangent._resblock_fwd) as rebuilt, \
                patch.object(model.time_proj, "forward", side_effect=AssertionError("recomputed time")):
            jvp = tape.jvp(directions)
            vjp = tape.vjp(cotangents)
            self.assertEqual(rebuilt.call_count, 2 * n_blocks)
            self.assert_close(jvp, cached.jvp(directions))
            self.assert_close(vjp, cached.vjp(cotangents))
            saved.extend([(jvp, jvp.clone()), (vjp, vjp.clone())])
            self.assert_close(tape.jvp(directions[:1]), jvp[:1])
            self.assert_close(tape.vjp(cotangents[1:]), vjp[1:])
            self.assert_close(tape.jvp(directions[1:]), jvp[1:])
            self.assert_close(tape.vjp(cotangents[:1]), vjp[:1])
            self.assert_close((jvp * cotangents).sum(), (directions * vjp).sum())
        for actual, snapshot in saved:
            torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)
        self.assertEqual([id(value) for value in tensors_before],
                         [id(value) for value in _tape_tensors(tape)])
        self.assertEqual(_owned_bytes(tape), storage_before)

    def test_default_unchanged_and_alternate_derivative_rules(self):
        for fused_gn in (False, True):
            for fused_pointwise in (False, True):
                with self.subTest(fused_gn=fused_gn, fused_pointwise=fused_pointwise), \
                        patch.object(tangent, "FUSED_GN", fused_gn), \
                        patch.object(tangent, "FUSE_POINTWISE_DERIVATIVES", fused_pointwise):
                    model = self.make_model()
                    x = torch.randn(1, 1, 7, 9, dtype=torch.float64)
                    lanes = torch.randn(2, 1, 7, 9, dtype=torch.float64)
                    default = tangent.unet_linearize(model, .43, x)
                    explicit = tangent.unet_linearize(model, .43, x, checkpoint_blocks=False)
                    checkpointed = tangent.UnetLinearization(model, .43, x, checkpoint_blocks=True)
                    for actual in (explicit, checkpointed):
                        torch.testing.assert_close(actual.primal, default.primal, rtol=0, atol=0)
                        torch.testing.assert_close(actual.jvp(lanes), default.jvp(lanes), rtol=0, atol=0)
                        torch.testing.assert_close(actual.vjp(lanes), default.vjp(lanes), rtol=0, atol=0)

    def test_reduced_unique_activation_storage(self):
        for attention in (False, True):
            with self.subTest(attention=attention):
                model = self.make_model(attention=attention)
                x = torch.randn(1, 1, 17, 19, dtype=torch.float64)
                cached = tangent.unet_linearize(model, .3, x)
                checkpointed = tangent.unet_linearize(model, .3, x, checkpoint_blocks=True)
                self.assertLess(_owned_bytes(checkpointed), _owned_bytes(cached) * .6)
                self.assertEqual(set(checkpointed.final), {"checkpoint_input"})
                self.assertEqual(set(checkpointed.mid0), {"checkpoint_input", "checkpoint_emb"})

    def test_training_model_is_rejected(self):
        model = self.make_model(torch.float32).train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            tangent.unet_linearize(model, .2, torch.randn(1, 1, 5, 7), checkpoint_blocks=True)


if __name__ == "__main__":
    unittest.main()

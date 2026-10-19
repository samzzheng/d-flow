"""Local intervention tests without flow-model evaluations."""
import copy
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import investigate_fast_dlr_tail as tail


class TailOptimizerTests(unittest.TestCase):
    def snapshot(self):
        source = torch.full((64,), 10., dtype=torch.float64, requires_grad=True)

        def objective(candidate):
            return float(-.5 * candidate.square().sum())

        optimizer = tail.helper.bench.ArmijoLBFGS(source, objective, step_cap=2.5)

        def closure():
            optimizer.zero_grad()
            source.grad = -source.detach().clone()
            return objective(source.detach())

        for _ in range(2):
            optimizer.step(closure)
        # Represent curvature which the intervention must remove, independently
        # of the negative-curvature problem used to expose initial L1 scaling.
        optimizer.S = [torch.ones_like(source)]
        optimizer.Y = [2 * torch.ones_like(source)]
        return {"source": source.detach().clone(), "settings": {"history": 100},
                "optimizer": {key: copy.deepcopy(getattr(optimizer, key))
                              for key in tail.study.OPTIMIZER_STATE_KEYS}}

    def test_reset_reuses_gradient_without_restarting_initial_step_scaling(self):
        self.check_intervention("reset", expected_gradient_calls=1)

    def test_rank_change_refreshes_gradient_but_preserves_lifetime_step_scaling(self):
        self.check_intervention("rank12", expected_gradient_calls=2)

    def check_intervention(self, mode, expected_gradient_calls):
        saved = self.snapshot()
        source = saved["source"].clone().requires_grad_(True)

        def objective(candidate):
            return float(-.5 * candidate.square().sum())

        optimizer = tail.make_tail_optimizer(source, objective, saved, mode)
        self.assertEqual(optimizer.S, [])
        self.assertEqual(optimizer.Y, [])
        for key in tail.study.OPTIMIZER_STATE_KEYS:
            if key not in ("S", "Y", "f", "g"):
                self.assertEqual(getattr(optimizer, key), saved["optimizer"][key])
        if mode == "rank12":
            self.assertIsNone(optimizer.f)
            self.assertIsNone(optimizer.g)
        else:
            self.assertEqual(optimizer.f, saved["optimizer"]["f"])
            torch.testing.assert_close(optimizer.g, saved["optimizer"]["g"], rtol=0, atol=0)
        evaluations = 0

        def closure():
            nonlocal evaluations
            evaluations += 1
            optimizer.zero_grad()
            source.grad = -(2 if mode == "rank12" else 1) * source.detach().clone()
            return objective(source.detach())

        optimizer.step(closure)
        self.assertEqual(evaluations, expected_gradient_calls)
        self.assertAlmostEqual(float((source.detach() - saved["source"]).norm()), 2.5)
        self.assertEqual(optimizer.step_attempts, saved["optimizer"]["step_attempts"] + 1)
        self.assertEqual(optimizer.total_function_evaluations,
                         saved["optimizer"]["total_function_evaluations"] + 1)
        self.assertLess(optimizer.f, saved["optimizer"]["f"])
        # Fork construction and future optimizer mutation must not mutate the input.
        self.assertEqual(len(saved["optimizer"]["S"]), 1)
        self.assertEqual(len(saved["optimizer"]["Y"]), 1)
        self.assertEqual(saved["optimizer"]["step_attempts"], 2)
        torch.testing.assert_close(saved["optimizer"]["g"], -saved["source"], rtol=0, atol=0)

    def test_snapshot_validation_rejects_settings_model_and_lifetime_changes(self):
        source = torch.ones((1, 1, 250, 250))
        saved = {"settings": {"mode": "solve", "profiles": ["sketch32"], "threads": 8,
                    "keep_terminal_alpha": False, "target_tau": 1., "sample": 0,
                    "seed": 0, "start_step": 0, "history": 100},
            "protocol": {"device": "cpu", "resolution": 250, "model": "mock model",
                "checkpoint_sha256": "mock hash", "flow_steps": 100, "integrator": "Euler",
                "noise": tail.helper.NOISE, "base_seed": tail.helper.SEED, "angles": 15,
                "source_prior": "none", "objective": "relative_scaled", "factor_refresh": "every gradient",
                "armijo_initial_step_policy": tail.helper.bench.ArmijoLBFGS.initial_step_policy,
                "armijo_progress_policy": tail.helper.bench.ArmijoLBFGS.progress_policy,
                "history": 100}, "profile": "sketch32", "step": 2,
            "source": source, "raw_image": source.clone(),
            "metrics": {"objective": 1., "residual_over_noise": 2.},
            "counts": {"gradient_evaluations": 3, "true_forward_evaluations": 3, "gradient_seconds": 2.},
            "optimizer": {"S": [], "Y": [], "f": 1., "g": source.flatten().clone(),
                "state": {"initialised": True}, "step_attempts": 2, "capped_steps": 1,
                "consecutive_failures": 0, "failed_searches": 0,
                "last_function_evaluations": 1, "total_function_evaluations": 2}}
        tail.validate_snapshot(saved, 8, "mock hash", "mock model")
        old_progress = copy.deepcopy(saved)
        old_progress["protocol"].pop("armijo_progress_policy")
        with self.assertRaisesRegex(ValueError, "armijo_progress_policy"):
            tail.validate_snapshot(old_progress, 8, "mock hash", "mock model")
        tail.validate_snapshot(old_progress, 8, "mock hash", "mock model",
                               allow_progress_policy_change=True)
        for threads, digest, tag in ((4, "mock hash", "mock model"), (8, "new hash", "mock model"),
                                     (8, "mock hash", "different model")):
            with self.subTest(threads=threads, digest=digest, tag=tag), self.assertRaises(ValueError):
                tail.validate_snapshot(saved, threads, digest, tag)
        for owner, key, value in (("settings", "keep_terminal_alpha", True),
                                   ("settings", "profiles", ["fast"]),
                                   ("protocol", "integrator", "RK4"),
                                   ("optimizer", "step_attempts", 0),
                                   ("optimizer", "f", 2.)):
            changed = copy.deepcopy(saved)
            changed[owner][key] = value
            with self.subTest(owner=owner, key=key), self.assertRaises(ValueError):
                tail.validate_snapshot(changed, 8, "mock hash", "mock model")

    def test_main_saves_fork_and_separates_added_costs_without_model_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            checkpoint = root / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"mock checkpoint")
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            source = torch.ones((1, 1, 250, 250), dtype=torch.float64)
            truth, target = .5 * source, .51 * source
            noise_norm = float((target - truth).norm())
            (root / "data").mkdir()
            torch.save({"test": truth}, root / "data/circles-250-n1.pt")
            relative_l2 = tail.helper.bench.relative_l2
            value = float(.5 * (source - target).square().sum())
            saved = {"settings": {"mode": "solve", "profiles": ["sketch32"], "threads": 8,
                        "keep_terminal_alpha": False, "target_tau": 1., "sample": 0,
                        "seed": 7, "start_step": 0, "history": 100},
                "protocol": {"device": "cpu", "resolution": 250, "model": "mock model",
                    "checkpoint_sha256": checkpoint_hash, "flow_steps": 100, "integrator": "Euler",
                    "noise": tail.helper.NOISE, "base_seed": tail.helper.SEED, "angles": 15,
                    "source_prior": "none", "objective": "relative_scaled", "factor_refresh": "every gradient",
                    "armijo_initial_step_policy": tail.helper.bench.ArmijoLBFGS.initial_step_policy,
                    "armijo_progress_policy": tail.helper.bench.ArmijoLBFGS.progress_policy,
                    "history": 100, "noise_norm": noise_norm,
                    "normalized_noise_floor": noise_norm / float(target.norm())},
                "profile": "sketch32", "step": 2, "source": source, "raw_image": source.clone(),
                "metrics": {"objective": value, "residual": float(relative_l2(source, target)),
                    "residual_over_noise": float((source - target).norm()) / noise_norm,
                    "image_relative_l2": float(relative_l2(source, truth)),
                    "raw_image_relative_l2": float(relative_l2(source, truth)),
                    "clamped_noisy_residual": float(relative_l2(source, target)),
                    "clamped_clean_residual": float(relative_l2(source, truth))},
                "counts": {"gradient_evaluations": 3, "true_forward_evaluations": 3, "gradient_seconds": 2.},
                "elapsed_seconds": 10.,
                "optimizer": {"S": [], "Y": [], "f": value, "g": (source - target).flatten(),
                    "state": {"initialised": True}, "step_attempts": 2, "capped_steps": 1,
                    "consecutive_failures": 0, "failed_searches": 0,
                    "last_function_evaluations": 1, "total_function_evaluations": 2}}
            # An explicitly permitted fork from historical acceptance records
            # both policies; it must never be mistaken for exact continuation.
            saved["protocol"].pop("armijo_progress_policy")
            snapshot = root / "optimizer_snapshot.pt"
            torch.save(saved, snapshot)
            original_bytes = snapshot.read_bytes()
            stack.enter_context(patch.object(tail, "ROOT", root))
            stack.enter_context(patch.object(torch, "set_num_threads"))
            stack.enter_context(patch.object(torch, "set_num_interop_threads"))
            stack.enter_context(patch.object(tail.helper, "load", return_value=(
                SimpleNamespace(mid_attn=False, att_channels=[]), "mock model")))
            stack.enter_context(patch.object(tail.helper.bench, "normalize_images", side_effect=lambda x: x))
            stack.enter_context(patch.object(tail.helper.bench, "sparse_radon", side_effect=lambda x, angles: x))
            flow = stack.enter_context(patch.object(tail.helper.bench, "solve_flow",
                side_effect=lambda model, z, *args: z.detach().clone()))
            stack.enter_context(patch.object(tail.helper, "make_problem", return_value=(None, target)))
            stack.enter_context(patch.object(tail.helper.bench, "data_objective",
                side_effect=lambda image, target, *args: (.5 * (image - target).square().sum(),
                    (image - target).square().sum() / target.square().sum())))
            stack.enter_context(patch.object(tail.helper, "terminal_gradient",
                side_effect=lambda image, target, *args: (image - target).flatten()))
            factors = stack.enter_context(patch.object(tail.core, "sensitivity_factors",
                side_effect=lambda model, z, *args, **kwargs: (z.detach(), None, 0.)))
            stack.enter_context(patch.object(tail.core, "approx_source_gradient",
                side_effect=lambda factors, terminal, alpha: terminal))
            for mode, gradient_count in (("reset", 1), ("rank12", 2)):
                with self.subTest(mode=mode):
                    out = root / mode
                    factors.reset_mock()
                    flow.reset_mock()
                    arguments = ["investigate_fast_dlr_tail.py", "--mode", mode, "--source-file", str(snapshot),
                                 "--steps", "1", "--threads", "8", "--out", str(out),
                                 "--allow-progress-policy-change"]
                    with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                        tail.main()
                    report = json.loads((out / "results.json").read_text())
                    self.assertEqual(report["intervention"]["kind"], "forked local intervention")
                    self.assertEqual(report["intervention"]["source_armijo_progress_policy"], "legacy_rounded_rhs")
                    self.assertTrue(report["intervention"]["armijo_progress_policy_changed"])
                    self.assertEqual(report["protocol"]["armijo_progress_policy"],
                                     tail.helper.bench.ArmijoLBFGS.progress_policy)
                    self.assertEqual(report["final"]["added_gradient_evaluations"], gradient_count)
                    self.assertEqual(report["final"]["added_true_forward_evaluations"], 1)
                    self.assertEqual(report["final"]["added_line_search_trials"], 1)
                    self.assertEqual(report["final"]["step_attempts"], 3)
                    self.assertEqual(factors.call_count, gradient_count)
                    self.assertEqual(flow.call_count, 1)
                    for call in factors.call_args_list:
                        self.assertEqual(call.args[2], 100)
                        self.assertEqual(call.kwargs["rank"], 8 if mode == "reset" else 12)
                        self.assertEqual(call.kwargs["initialization_oversampling"], 32)
                        self.assertEqual(call.kwargs["seed"], 7)
                        self.assertEqual(call.kwargs["factor_stride"], 1)
                    point = torch.load(out / "step0001.pt", map_location="cpu", weights_only=True)
                    self.assertEqual(point["step"], 3)
                    self.assertAlmostEqual(float((point["source"] - source).norm()), 2.5)
                    self.assertEqual(snapshot.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()

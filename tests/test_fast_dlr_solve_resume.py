"""Cheap end-to-end checkpoint tests with the production Armijo optimizer."""
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import investigate_fast_dlr_accuracy as study


class FastSolveResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        checkpoint = self.root / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"mock quadratic, no model checkpoint")
        self.checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.initial = torch.tensor([4., -3., 2., 5.], dtype=torch.float64).reshape(1, 1, 2, 2)
        self.truth = torch.ones_like(self.initial)
        self.target = self.truth + .01
        self.weights = torch.tensor([1., 2., 3., 4.], dtype=torch.float64).reshape_as(self.initial)
        (self.root / "data").mkdir()
        torch.save({"test": self.truth}, self.root / "data/circles-250-n1.pt")
        self.source_file = self.root / "source.pt"
        torch.save({"protocol": {"checkpoint_sha256": self.checkpoint_hash}, "sample": 0,
                    "sources": [self.initial]}, self.source_file)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(study, "ROOT", self.root))
        self.stack.enter_context(patch.object(torch, "set_num_threads"))
        self.stack.enter_context(patch.object(torch, "set_num_interop_threads"))
        self.stack.enter_context(patch.object(study.time, "perf_counter", side_effect=itertools.count()))
        self.stack.enter_context(patch.object(study.helper, "load", return_value=(
            SimpleNamespace(mid_attn=False, att_channels=[]), "mock quadratic")))
        self.stack.enter_context(patch.object(study.helper.bench, "normalize_images", side_effect=lambda x: x))
        self.stack.enter_context(patch.object(study.helper.bench, "sparse_radon", side_effect=lambda x, angles: x))
        self.forward = self.stack.enter_context(patch.object(study.helper.bench, "solve_flow",
            side_effect=lambda model, source, *args: source.detach().clone()))
        self.stack.enter_context(patch.object(study.helper, "make_problem", return_value=(None, self.target)))
        self.stack.enter_context(patch.object(study.helper.bench, "data_objective",
            side_effect=lambda prediction, target, *args: (
                .5 * (self.weights * (prediction - target).square()).sum(),
                (prediction - target).square().sum() / target.square().sum())))
        self.stack.enter_context(patch.object(study.helper, "terminal_gradient",
            side_effect=lambda image, target, *args: self.weights * (image - target)))
        identity = torch.eye(4, dtype=self.initial.dtype)
        self.factors = self.stack.enter_context(patch.object(study.core, "sensitivity_factors",
            side_effect=lambda model, source, *args, **kwargs: (
                source.detach().clone(), (identity, identity, identity), 0.)))

    def run_study(self, output, steps, resume=False):
        arguments = ["investigate_fast_dlr_accuracy.py", "--mode", "solve", "--profiles", "fast",
            "--threads", "1", "--start-step", "0", "--source-file", str(self.source_file),
            "--seed", "3", "--outer-steps", str(steps), "--target-tau", "0", "--out", str(output)]
        if resume:
            arguments.append("--resume")
        with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            study.main()
        saved = torch.load(output / "fast_resume.pt", map_location="cpu", weights_only=True)
        report = json.loads((output / "results.json").read_text())
        return saved, report

    def assert_same_optimizer(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            if isinstance(actual[key], torch.Tensor):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            elif key in ("S", "Y"):
                self.assertEqual(len(actual[key]), len(expected[key]))
                for left, right in zip(actual[key], expected[key]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            else:
                self.assertEqual(actual[key], expected[key])

    def test_resume_matches_uninterrupted_trajectory_and_accepts_legacy_false_defaults(self):
        uninterrupted, continuous_report = self.run_study(self.root / "continuous", 5)
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                output = self.root / f"resumed_{legacy}"
                saved, report = self.run_study(output, 2)
                original_rows = copy.deepcopy(report["runs"][0]["steps"])
                if legacy:
                    for owner in (saved, report):
                        owner["settings"].pop("cached_exact_gradient")
                        owner["settings"].pop("nested_sketch")
                        for key in ("source_state_step", "sample", "source_generation_optimizer_policy"):
                            owner["protocol"].pop(key)
                    for key in ("parameters", "report", "elapsed_seconds"):
                        saved.pop(key)
                    torch.save(saved, output / "fast_resume.pt")
                    (output / "results.json").write_text(json.dumps(report))
                before_gradients = self.factors.call_count
                resumed, new_report = self.run_study(output, 5, resume=True)
                self.assertEqual(self.factors.call_count - before_gradients, 3)
                for key in ("source", "raw_image"):
                    torch.testing.assert_close(resumed[key], uninterrupted[key], rtol=0, atol=0)
                for key in ("metrics", "counts", "elapsed_seconds"):
                    self.assertEqual(resumed[key], uninterrupted[key])
                self.assert_same_optimizer(resumed["optimizer"], uninterrupted["optimizer"])
                self.assertEqual(new_report["initial"], report["initial"])
                self.assertEqual(new_report["runs"][0]["steps"][:3], original_rows)
                self.assertEqual(new_report["runs"][0]["steps"], continuous_report["runs"][0]["steps"])
                self.assertEqual(new_report["resume_history"][0]["step"], 2)
                self.assertEqual(new_report["resume_history"][0]["previous_settings"]["outer_steps"], 2)
                self.assertEqual(new_report["resume_history"][0]["previous_stop_reason"], "step_budget")

    def test_recovers_row_when_report_write_lags_checkpoint(self):
        output = self.root / "lagged"
        _, report = self.run_study(output, 2)
        expected_rows = copy.deepcopy(report["runs"][0]["steps"])
        report["runs"][0]["steps"].pop()
        (output / "results.json").write_text(json.dumps(report))
        _, restored = self.run_study(output, 3, resume=True)
        self.assertEqual(restored["runs"][0]["steps"][:3], expected_rows)
        self.assertEqual([row["step"] for row in restored["runs"][0]["steps"]], [0, 1, 2, 3])

    def test_checkpoint_embeds_history_if_report_is_missing(self):
        output = self.root / "missing_report"
        _, report = self.run_study(output, 2)
        (output / "results.json").unlink()
        _, restored = self.run_study(output, 3, resume=True)
        self.assertEqual(restored["runs"][0]["steps"][:3], report["runs"][0]["steps"])

    def test_rejects_settings_protocol_profile_and_original_source_changes(self):
        output = self.root / "validation"
        saved, report = self.run_study(output, 2)
        args = SimpleNamespace(profiles=["fast"], out=output, outer_steps=3)
        expected = copy.deepcopy(report)
        expected["settings"]["outer_steps"] = 3
        for key, value in (("seed", 42), ("threads", 2), ("nested_sketch", True),
                           ("history", 10), ("keep_terminal_alpha", True), ("sample", 1),
                           ("source_file", "different.pt"), ("target_tau", 1.), ("cached_exact_gradient", True)):
            with self.subTest(setting=key):
                changed = copy.deepcopy(expected)
                changed["settings"][key] = value
                with self.assertRaisesRegex(ValueError, "settings mismatch"):
                    study.load_solve_resume(args, changed, self.initial)
        changed = copy.deepcopy(expected)
        changed["settings"]["outer_steps"] = 1
        with self.assertRaisesRegex(ValueError, "cannot reduce"):
            study.load_solve_resume(args, changed, self.initial)
        changed = copy.deepcopy(expected)
        changed["protocol"]["checkpoint_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            study.load_solve_resume(args, changed, self.initial)
        with self.assertRaisesRegex(ValueError, "source file contents"):
            study.load_solve_resume(args, expected, self.initial + 1)
        corrupted = copy.deepcopy(saved)
        corrupted["optimizer"].pop("step_attempts")
        torch.save(corrupted, output / "fast_resume.pt")
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            study.load_solve_resume(args, expected, self.initial)
        torch.save(saved, output / "fast_resume.pt")
        report["runs"][0]["parameters"]["rank"] = 9
        (output / "results.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "profile parameters"):
            study.load_solve_resume(args, expected, self.initial)

    def test_same_budget_resume_does_not_repeat_evaluations_or_rows(self):
        output = self.root / "already_complete"
        saved, report = self.run_study(output, 2)
        evaluations, gradients = self.forward.call_count, self.factors.call_count
        _, resumed = self.run_study(output, 2, resume=True)
        self.assertEqual(self.forward.call_count, evaluations)
        self.assertEqual(self.factors.call_count, gradients)
        self.assertEqual(resumed["runs"][0]["steps"], report["runs"][0]["steps"])
        self.assertEqual(resumed["runs"][0]["final"], report["runs"][0]["final"])
        unchanged = torch.load(output / "fast_resume.pt", map_location="cpu", weights_only=True)
        self.assert_same_optimizer(unchanged["optimizer"], saved["optimizer"])

    def test_old_progress_policy_cannot_resume_as_current_before_any_evaluations(self):
        output = self.root / "old_progress"
        saved, report = self.run_study(output, 2)
        for policy in (None, "legacy_rounded_rhs"):
            with self.subTest(policy=policy):
                for owner in (saved, report):
                    if policy is None:
                        owner["protocol"].pop("armijo_progress_policy", None)
                    else:
                        owner["protocol"]["armijo_progress_policy"] = policy
                torch.save(saved, output / "fast_resume.pt")
                (output / "results.json").write_text(json.dumps(report))
                evaluations, gradients = self.forward.call_count, self.factors.call_count
                with self.assertRaisesRegex(ValueError, "armijo_progress_policy"):
                    self.run_study(output, 3, resume=True)
                self.assertEqual(self.forward.call_count, evaluations)
                self.assertEqual(self.factors.call_count, gradients)


if __name__ == "__main__":
    unittest.main()

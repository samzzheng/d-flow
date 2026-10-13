"""CPU checks for paired operators, complex gradients, and exact resumption."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from dlr_dflow import dlr_dflow as core
from p_flow import proxy_gradient
from .operators import CT, MRI, noisy_measurement
from .run import Evaluator, Job, jobs, model_ready, reviewed_source_hashes, solve
from . import queue
from scripts.reconstruction_figures import METHODS, RESIDUAL_VMAX, plot_reconstruction_comparison
import matplotlib.pyplot as plt
import numpy as np


class LinearVelocity(torch.nn.Module):
    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return 0.2 * image


class SweepTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.settings = json.loads(Path(__file__).with_name("config.json").read_text())

    def test_complete_grid(self) -> None:
        grid = jobs(self.settings)
        self.assertEqual(len(grid), 168)
        self.assertEqual(len({job.name for job in grid}), 168)
        self.assertEqual(len(grid) * self.settings["samples"] * 3, 15120)

    def test_training_gate_requires_summary_and_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.ct_mri_sweep.run.ROOT", Path(tmp)):
            job = Job(64, 1, "ct", 10)
            run = Path(tmp) / self.settings["run_template"].format(resolution=64, circles=1)
            run.mkdir(parents=True)
            (run / "run_summary.json").write_text(json.dumps({"status": "complete", "epochs_completed": 200}))
            self.assertFalse(model_ready(job, self.settings))
            (run / "checkpoints").mkdir()
            (run / "checkpoints/epoch_0200.pt").touch()
            self.assertTrue(model_ready(job, self.settings))

    def test_source_review_preserves_manifest_and_unreviewed_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest = {"source_sha256": {"plot.py": "old", "algorithm.py": "unchanged"}}
            original = json.dumps(manifest).encode()
            path.write_bytes(original)
            self.assertEqual(reviewed_source_hashes(path, manifest), manifest["source_sha256"])
            review = {"manifest_sha256": hashlib.sha256(original).hexdigest(),
                      "changes": {"plot.py": {"before": "old", "after": "new"}}}
            review_path = path.with_name("source_review.json")
            review_path.write_text(json.dumps(review))
            self.assertEqual(reviewed_source_hashes(path, manifest),
                             {"plot.py": "new", "algorithm.py": "unchanged"})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(manifest["source_sha256"]["plot.py"], "old")
            review["changes"]["plot.py"]["before"] = "incorrect"
            review_path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, "baseline mismatch"):
                reviewed_source_hashes(path, manifest)
            review["manifest_sha256"] = "different"
            review_path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, "different manifest"):
                reviewed_source_hashes(path, manifest)

    def test_scheduler_waits_for_training_and_external_gpu_jobs(self) -> None:
        for ready, occupancy, expected_status, launches in (
            (False, {0: set(), 1: set()}, "waiting_for_training", 0),
            (True, {0: {111}, 1: {222}}, "waiting_for_gpus", 0),
            (True, {0: {111}, 1: set()}, "running", 1),
        ):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                out = Path(tmp)
                (out / "manifest.json").write_text(json.dumps({"settings": self.settings,
                    "source_sha256": {}, "jobs": [{}] * 168}))
                stack.enter_context(patch("sys.argv", ["queue", "run", "--out", tmp]))
                stack.enter_context(patch.object(queue, "source_hashes", return_value={}))
                stack.enter_context(patch.object(queue, "model_ready", return_value=ready))
                stack.enter_context(patch.object(queue, "gpu_processes", return_value=occupancy))
                stack.enter_context(patch.object(queue.subprocess, "check_output", return_value="0, 49140, 2\n1, 49140, 2\n"))
                stack.enter_context(patch.object(queue.time, "sleep", side_effect=KeyboardInterrupt))
                stack.enter_context(patch.object(queue.signal, "signal"))
                process = MagicMock(pid=777)
                process.poll.return_value = None
                launch = stack.enter_context(patch.object(queue.subprocess, "Popen", return_value=process))
                with self.assertRaises(KeyboardInterrupt):
                    queue.main()
                self.assertEqual(json.loads((out / "queue_status.json").read_text())["status"], expected_status)
                self.assertEqual(launch.call_count, launches)
                if launches:
                    self.assertIn("cuda:1", launch.call_args.args[0])
                    process.send_signal.assert_called()

    def test_shards_and_memory_packing(self) -> None:
        self.settings.update(resolutions=[64, 128, 250], circles=[1, 2, 3], ct_angles=[10], mri_accelerations=[4])
        self.settings["scheduling"] = {"samples_per_worker": 5, "gpu_headroom_mib": 2048,
            "max_workers_per_gpu": 20, "worker_memory_mib": {"64": 3000, "128": 8000, "250": 30000}}
        tasks = queue.tasks(self.settings)
        self.assertEqual(len(tasks), 108)
        self.assertEqual(len({(task.job.name, sample) for task in tasks for sample in range(task.start, task.stop)}), 540)
        running = [tasks[0]]
        remaining = tasks[1:]
        while (task := queue.fitting_task(remaining, running, self.settings, 49140, 2)) is not None:
            running.append(task)
            remaining.remove(task)
        self.assertEqual([task.job.resolution for task in running], [250, 128, 128])
        self.assertLessEqual(sum(queue.memory_reservation(task, self.settings, 49140) for task in running), 49140-2048)
        self.assertIsNone(queue.fitting_task(tasks, [], self.settings, 49140, 47000))

    def test_scheduler_adds_gpu_zero_and_recognizes_all_own_workers(self) -> None:
        self.settings["scheduling"] = {"samples_per_worker": 5, "gpu_headroom_mib": 2048,
            "max_workers_per_gpu": 2, "worker_memory_mib": {str(r): 8000 for r in self.settings["resolutions"]}}
        processes = [MagicMock(pid=pid) for pid in (777, 778, 779, 780)]
        for process in processes:
            process.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out = Path(tmp)
            (out / "manifest.json").write_text(json.dumps({"settings": self.settings,
                "source_sha256": {}, "jobs": [{}] * 168}))
            stack.enter_context(patch("sys.argv", ["queue", "run", "--out", tmp]))
            stack.enter_context(patch.object(queue, "source_hashes", return_value={}))
            stack.enter_context(patch.object(queue, "model_ready", return_value=True))
            stack.enter_context(patch.object(queue, "gpu_processes", side_effect=[
                {0: {111}, 1: set()}, {0: set(), 1: {777, 778}},
                {0: {779, 780}, 1: {777, 778, 222}}]))
            stack.enter_context(patch.object(queue.subprocess, "check_output", return_value="0, 20000, 2\n1, 20000, 2\n"))
            stack.enter_context(patch.object(queue.time, "sleep", side_effect=[None, None, KeyboardInterrupt]))
            stack.enter_context(patch.object(queue.signal, "signal"))
            launch = stack.enter_context(patch.object(queue.subprocess, "Popen", side_effect=processes))
            with self.assertRaises(KeyboardInterrupt):
                queue.main()
            self.assertEqual(launch.call_count, 4)
            self.assertEqual([call.args[0][call.args[0].index("--device") + 1]
                              for call in launch.call_args_list], ["cuda:1", "cuda:1", "cuda:0", "cuda:0"])
            records = [json.loads(line) for line in (out / "queue_history.jsonl").read_text().splitlines()]
            self.assertEqual(len(records[0]["active"]), 2)
            self.assertEqual(len(records[1]["active"]), 4)
            self.assertEqual(records[1]["external_gpu_processes"], {"1": [], "0": []})
            self.assertEqual(records[2]["external_gpu_processes"]["1"], [222])
            self.assertEqual([p.send_signal.call_count for p in processes], [2, 2, 1, 1])

    def test_ct_linearity_adjoint_and_all_angles(self) -> None:
        generator = torch.Generator().manual_seed(3)
        image, direction = [torch.randn(1, 1, 16, 16, generator=generator) for _ in range(2)]
        for angles in (10, 20, 23, 180):
            operator = CT(16, angles, torch.device("cpu"))
            torch.testing.assert_close(operator(2 * image - direction), 2 * operator(image) - operator(direction), atol=5e-7, rtol=1e-5)
            variable = image.clone().requires_grad_(True)
            measurement = torch.randn(1, 1, angles, 16, generator=generator)
            product = (operator(variable) * measurement).sum()
            transpose, = torch.autograd.grad(product, variable)
            torch.testing.assert_close((operator(direction) * measurement).sum(), (direction * transpose).sum(), atol=3e-6, rtol=1e-5)

    def test_mri_sampling_noise_and_complex_gradient(self) -> None:
        image = torch.randn(1, 1, 16, 16)
        for pattern in ("cartesian", "points"):
            for acceleration in (4, 8, 24):
                operator = MRI(16, acceleration, 11, torch.device("cpu"), pattern)
                duplicate = MRI(16, acceleration, 11, torch.device("cpu"), pattern)
                self.assertTrue(torch.equal(operator.mask, duplicate.mask))
                self.assertTrue(bool(operator.mask[0, 0, 8, 8]))
                count = max(1, round((16 if pattern == "cartesian" else 256) / acceleration))
                self.assertEqual(int(operator.mask.sum()), count * (16 if pattern == "cartesian" else 1))
                clean = operator(image)
                observed = noisy_measurement(clean, .01, 12, operator.mask)
                self.assertEqual(int(observed[~operator.mask].count_nonzero()), 0)
                self.assertAlmostEqual(float((observed-clean).norm() / clean.norm()), .01, places=6)
                endpoint, loss, proxy = proxy_gradient(LinearVelocity(), image, operator, observed, 3)
                residual = operator(endpoint) - observed
                expected = torch.fft.ifft2(torch.fft.ifftshift(operator.mask * residual, dim=(-2,-1)), norm="ortho").real
                torch.testing.assert_close(proxy, expected)
                torch.testing.assert_close(loss, .5 * residual.abs().square().sum())

    def test_noiseless_measurements_bypass_sampling_and_have_zero_gt_residual(self) -> None:
        truth = torch.randn(1, 1, 16, 16)
        for operator in (CT(16, 10, torch.device("cpu")), MRI(16, 4, 5, torch.device("cpu"))):
            clean = operator(truth)
            with patch("torch.randn", side_effect=AssertionError("Zero noise must not sample")):
                observed = noisy_measurement(clean, 0, 23, getattr(operator, "mask", None))
            self.assertTrue(torch.equal(clean, observed))
            fig = plot_reconstruction_comparison(truth, clean, {}, observation=observed)
            try:
                self.assertEqual(np.count_nonzero(fig.axes[15].images[0].get_array()), 0)
                self.assertIn("0.00%", fig.axes[15].get_title())
            finally:
                plt.close(fig)

    def test_aligned_panels_amp_and_complex_error_histories(self) -> None:
        truth = torch.ones(1, 1, 16, 16)
        for shape in ((1, 1, 10, 16), (1, 1, 16, 16)):
            clean = torch.ones(shape, dtype=torch.complex64)
            reconstruction = 1j * clean
            history = [{"step": 0, "image_relative_l2": 1., "measurement_relative_l2_noisy": 2.},
                       {"step": 7, "image_relative_l2": 0., "measurement_relative_l2_noisy": 2**.5}]
            fig = plot_reconstruction_comparison(truth, clean, {name: (truth, reconstruction) for name in METHODS},
                observation=clean, histories={name: history for name in METHODS})
            try:
                fig.canvas.draw()
                self.assertEqual(len(fig.axes), 24)
                axes = np.asarray([*fig.axes[:20], None, *fig.axes[20:]], dtype=object).reshape(5, 5)
                boxes = [[axis.get_window_extent() for axis in row[:4]] for row in axes[:4]]
                np.testing.assert_allclose([box.width for row in boxes for box in row], boxes[0][0].width, atol=.1)
                np.testing.assert_allclose([box.height for row in boxes for box in row], boxes[0][0].height, atol=.1)
                for row in boxes[1:]:
                    np.testing.assert_allclose([box.x0 for box in row], [box.x0 for box in boxes[0]], atol=.1)
                for axis in axes[2:4, :4].flat:
                    self.assertEqual(axis.images[0].get_cmap().name, "amp")
                self.assertIn("141.42%", axes[1, 1].get_title())
                np.testing.assert_allclose(axes[4, 1].lines[0].get_xdata(), [0, 7])
                np.testing.assert_allclose(axes[4, 1].lines[1].get_ydata(), [200, 100 * 2**.5])
                self.assertEqual(axes[4, 1].get_xlim(), axes[4, 3].get_xlim())
                self.assertEqual(axes[4, 1].get_ylim(), axes[4, 3].get_ylim())
                self.assertEqual(axes[4, 1].get_xlabel(), "Optimizer attempt")
            finally:
                plt.close(fig)

    def test_image_training_scale_saturates_display_without_clipping_metrics(self) -> None:
        truth = torch.ones(1, 1, 2, 2)
        image = torch.tensor([[[[-.2, 0.], [.5, 1.4]]]])
        fig = plot_reconstruction_comparison(truth, truth, {"DLR D-Flow": (image, image)})
        try:
            expected = 100 * float((image - truth).double().norm() / truth.double().norm())
            axis = fig.axes[3]
            artist = axis.images[0]
            self.assertEqual(artist.get_clim(), (0., 1.))
            np.testing.assert_allclose(artist.get_array(), image.squeeze().numpy())
            self.assertIn(f"{expected:.2f}%", axis.get_title())
            self.assertIn("Range [-0.2, 1.4]", axis.get_title())
            np.testing.assert_array_equal(artist.get_cmap()(-1.), artist.get_cmap()(0.))
            np.testing.assert_array_equal(artist.get_cmap()(2.), artist.get_cmap()(1.))
            self.assertEqual(fig.axes[4]._colorbar.extend, "both")
            self.assertEqual(fig.axes[0].images[0].get_clim(), (0., 1.))
        finally:
            plt.close(fig)

    def test_fixed_residual_scales_preserve_complex_errors_and_mark_overflow(self) -> None:
        truth = torch.ones(1, 1, 8, 8)
        clean = torch.ones_like(truth, dtype=torch.complex64)
        for problem, limit in RESIDUAL_VMAX.items():
            for amplitude in (.01, 2 * limit):
                predicted = clean + 1j * amplitude
                fig = plot_reconstruction_comparison(truth, clean, {"DLR D-Flow": (truth, predicted)},
                    measurement_log=problem == "mri", residual_vmax=limit)
                try:
                    residual_axis = fig.axes[18]
                    artist = residual_axis.images[0]
                    self.assertEqual(artist.get_clim(), (0, limit))
                    np.testing.assert_allclose(artist.get_array(), amplitude)
                    self.assertIn(f"{100 * amplitude:.2f}%", residual_axis.get_title())
                    np.testing.assert_allclose(fig.axes[19].get_yticks(), np.linspace(0, limit, 6))
                    self.assertEqual(artist.colorbar.extend, "max" if amplitude > limit else "neither")
                finally:
                    plt.close(fig)

    def test_full_rank_dlr_and_exact_gradient_match_euler_map(self) -> None:
        settings = self.settings
        settings["flow_steps"] = 3
        settings["dlr"].update(rank=4, initialization_oversampling=0)
        source = torch.randn(1, 1, 2, 2, dtype=torch.float64).requires_grad_(True)
        operator = MRI(2, 1, 1, torch.device("cpu"))
        observed = operator(torch.ones_like(source))
        evaluator = Evaluator(LinearVelocity(), operator, observed, settings, 13)
        exact = evaluator.gradient(source, "D-Flow")
        with ExitStack() as stack:
            for flag in ("HAND_JVP", "HAND_VJP", "SHARE_LINEARIZATION"):
                stack.enter_context(patch.object(core, flag, False))
            approximate = evaluator.gradient(source, "DLR D-Flow")
        torch.testing.assert_close(approximate, exact, atol=1e-10, rtol=1e-10)
        direction = torch.randn_like(source)
        epsilon = 1e-5
        finite_difference = (evaluator.objective(source.detach() + epsilon*direction) - evaluator.objective(source.detach() - epsilon*direction)) / (2*epsilon)
        self.assertAlmostEqual(finite_difference, float((exact*direction).sum()), places=8)

    def test_resume_preserves_lbfgs_schedule_and_trajectory(self) -> None:
        settings = self.settings
        settings["flow_steps"] = 3
        settings["dlr"].update(rank=2, initialization_oversampling=0)
        settings["execution"]["checkpoint_every"] = 1
        settings["stopping"].update(min_updates=4, max_updates=4, window=2)
        initial = torch.tensor([3., 4., 2., 1.], dtype=torch.float64).reshape(1,1,2,2)
        truth = torch.ones_like(initial)
        operator = MRI(2, 1, 1, torch.device("cpu"))
        clean = operator(truth)
        for method in METHODS:
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                for flag in ("HAND_JVP", "HAND_VJP", "SHARE_LINEARIZATION"):
                    stack.enter_context(patch.object(core, flag, False))
                root = Path(tmp)
                evaluator = Evaluator(LinearVelocity(), operator, clean, settings, 0)
                whole = solve(evaluator, method, initial, truth, clean, root / "whole.pt", {}, [False])
                evaluator = Evaluator(LinearVelocity(), operator, clean, settings, 0)
                interrupted = [False]
                gradient = evaluator.gradient

                def interrupt_after_two(source: torch.Tensor, name: str) -> torch.Tensor:
                    result = gradient(source, name)
                    interrupted[0] = evaluator.counts["gradients"] == 3
                    return result

                with patch.object(evaluator, "gradient", interrupt_after_two):
                    paused = solve(evaluator, method, initial, truth, clean, root / "split.pt", {}, interrupted)
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(paused["optimizer"]["step_attempts"], 2)
                evaluator = Evaluator(LinearVelocity(), operator, clean, settings, 0)
                resumed = solve(evaluator, method, initial, truth, clean, root / "split.pt", {}, [False])
                torch.testing.assert_close(resumed["source"], whole["source"], rtol=0, atol=0)
                self.assertEqual(resumed["optimizer"]["step_attempts"], 4)
                self.assertEqual([row["phi"] for row in resumed["history"]], [row["phi"] for row in whole["history"]])
                self.assertEqual(resumed["optimizer"]["backtracking"], "geometric")
                if method == "DLR D-Flow":
                    torch.testing.assert_close(resumed["previous_v"], whole["previous_v"], rtol=0, atol=0)
                    self.assertEqual(resumed["evaluations"]["gradients"], whole["evaluations"]["gradients"])
                (root / "split.json").unlink()
                evaluator = Evaluator(LinearVelocity(), operator, clean, settings, 0)
                solve(evaluator, method, initial, truth, clean, root / "split.pt", {}, [False])
                self.assertEqual(json.loads((root / "split.json").read_text())["status"], "step_limit")
                self.assertEqual(evaluator.counts["gradients"], 0)

    def test_previous_basis_seeds_only_the_next_gradient(self) -> None:
        settings = self.settings
        settings["flow_steps"] = 3
        settings["dlr"].update(rank=2, initialization_oversampling=0)
        source = torch.randn(1, 1, 2, 2, dtype=torch.float64)
        operator = MRI(2, 1, 1, torch.device("cpu"))
        evaluator = Evaluator(LinearVelocity(), operator, operator(torch.ones_like(source)), settings, 13)
        with ExitStack() as stack:
            for flag in ("HAND_JVP", "HAND_VJP", "SHARE_LINEARIZATION"):
                stack.enter_context(patch.object(core, flag, False))
            evolve = stack.enter_context(patch("experiments.ct_mri_sweep.run.sensitivity_factors", wraps=core.sensitivity_factors))
            evaluator.gradient(source, "DLR D-Flow")
            self.assertIsNone(evolve.call_args.kwargs["seed_basis"])
            previous = evaluator.previous_v.clone()
            evaluator.objective(source + .01)
            torch.testing.assert_close(evaluator.previous_v, previous, rtol=0, atol=0)
            evaluator.gradient(source + .01, "DLR D-Flow")
            torch.testing.assert_close(evolve.call_args.kwargs["seed_basis"], previous, rtol=0, atol=0)
            self.assertEqual(evolve.call_count, 2)


if __name__ == "__main__":
    unittest.main()

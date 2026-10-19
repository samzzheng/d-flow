"""Cover multi-sample admission and the smoke barrier without GPU work."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from experiments.ct_mri_sweep.queue import tasks
from .run import main


class QueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name)
        with patch("sys.argv", ["pilot", "init", "--out", str(self.out), "--resolutions", "64", "160", "250",
                                "--samples", "30", "--noise-relative", ".01", "--max-workers", "4"]):
            main()

    def test_grid_and_smoke_scope(self) -> None:
        settings = json.loads((self.out / "production/manifest.json").read_text())["settings"]
        grid = tasks(settings)
        self.assertEqual(len(grid), 180)
        self.assertEqual(len({task.name for task in grid}), 180)
        for resolution in (64, 160, 250):
            for problem in ("ct", "mri"):
                self.assertEqual({task.start for task in grid if task.job.resolution == resolution and
                                  task.job.problem == problem}, set(range(30)))
        smoke = json.loads((self.out / "smoke/manifest.json").read_text())["settings"]
        self.assertEqual(smoke["samples"], 1)
        self.assertEqual(smoke["stopping"]["max_updates"], 2)
        self.assertEqual(settings["stopping"]["max_updates"], 1000)
        self.assertEqual(settings["noise_relative"], .01)
        self.assertEqual(json.loads((self.out / "summary.json").read_text())["total_runs"], 540)

    def test_separated_shapes_use_mixed_models_and_full_paired_grid(self) -> None:
        out = self.out / "shapes"
        with patch("sys.argv", ["pilot", "init", "--out", str(out),
                                "--dataset", "separated_shapes", "--samples", "30", "--max-workers", "4"]):
            main()
        manifest = json.loads((out / "production/manifest.json").read_text())
        settings = manifest["settings"]
        grid = tasks(settings)
        self.assertEqual(len(grid), 360)
        self.assertEqual(len({task.job.name for task in grid}), 12)
        self.assertTrue(all("_shapes_" in task.job.name for task in grid))
        self.assertEqual(settings["dataset_template"].format(resolution=96), "data/separated-shapes-96.pt")
        self.assertEqual(settings["flow_steps"], 5)
        self.assertEqual(settings["noise_relative"], 0)
        self.assertEqual(settings["stopping"]["max_updates"], 1000)
        baseline = json.loads((self.out / "production/manifest.json").read_text())["settings"]
        self.assertEqual(settings["optimizer"], baseline["optimizer"])
        self.assertEqual(settings["dlr"], baseline["dlr"])
        self.assertEqual(json.loads((out / "summary.json").read_text())["total_runs"], 1080)
        smoke = json.loads((out / "smoke/manifest.json").read_text())["settings"]
        self.assertEqual(smoke["resolutions"], [64, 250])
        self.assertEqual(len(tasks(smoke)), 4)

    def test_smoke_barrier_and_nonzero_sample_arguments(self) -> None:
        for smoke_complete in (False, True):
            with self.subTest(smoke_complete=smoke_complete):
                for stage in ("smoke", "production"):
                    root = self.out / stage
                    settings = json.loads((root / "manifest.json").read_text())["settings"]
                    for task in tasks(settings):
                        if stage == "smoke" and not smoke_complete:
                            continue
                        if stage == "production" and task.job.name == "r250_n1_mri4" and task.start == 29:
                            continue
                        marker = task.marker(root)
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.write_text("{}")
                process = MagicMock(pid=12345)
                process.poll.return_value = None
                with ExitStack() as stack:
                    stack.enter_context(patch("sys.argv", ["pilot", "run", "--out", str(self.out)]))
                    stack.enter_context(patch("experiments.five_step_pilot.run.gpu_processes", return_value={1: set()}))
                    stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.check_output", return_value="49140, 2"))
                    launch = stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.Popen", return_value=process))
                    stack.enter_context(patch("experiments.five_step_pilot.run.signal.signal"))
                    stack.enter_context(patch("experiments.five_step_pilot.run.time.sleep", side_effect=KeyboardInterrupt))
                    with self.assertRaises(KeyboardInterrupt):
                        main()
                commands = [call.args[0] for call in launch.call_args_list]
                if smoke_complete:
                    self.assertEqual(len(commands), 1)
                    command = commands[0]
                    self.assertEqual(command[command.index("--sample-start") + 1], "29")
                    self.assertEqual(command[command.index("--sample-stop") + 1], "30")
                    self.assertEqual(command[command.index("--device") + 1], "cuda:1")
                else:
                    self.assertEqual(len(commands), 2)
                    self.assertTrue(all("/smoke/" in command[command.index("--manifest") + 1] for command in commands))

    def test_rank16_only_preserves_baseline_and_selects_dlr_worker(self) -> None:
        out = self.out / "rank16"
        with patch("sys.argv", ["pilot", "init", "--out", str(out), "--resolutions", "64", "160", "250",
                                "--samples", "30", "--noise-relative", ".01", "--max-workers", "4",
                                "--rank", "16", "--dlr-only", "--baseline-out", str(self.out)]):
            main()
        baseline = json.loads((self.out / "production/manifest.json").read_text())["settings"]
        manifest = json.loads((out / "production/manifest.json").read_text())
        settings = manifest["settings"]
        expected = {**settings, "dlr": {**settings["dlr"], "rank": 8}}
        del expected["baseline_out"]
        self.assertEqual(expected, baseline)
        self.assertEqual(manifest["protocol"]["methods"], ["DLR D-Flow"])
        self.assertEqual(json.loads((out / "summary.json").read_text())["total_runs"], 180)
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["pilot", "run", "--out", str(out)]))
            stack.enter_context(patch("experiments.five_step_pilot.run.gpu_processes", return_value={1: set()}))
            stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.check_output", return_value="49140, 2"))
            launch = stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.Popen", return_value=process))
            stack.enter_context(patch("experiments.five_step_pilot.run.signal.signal"))
            stack.enter_context(patch("experiments.five_step_pilot.run.time.sleep", side_effect=KeyboardInterrupt))
            with self.assertRaises(KeyboardInterrupt):
                main()
        self.assertEqual(launch.call_count, 2)
        self.assertTrue(all(call.args[0][2] == "experiments.noisy_ct_mri.rank_worker" for call in launch.call_args_list))

    def test_higher_rank_subset_preserves_baseline_fingerprint_population(self) -> None:
        out = self.out / "rank64"
        with patch("sys.argv", ["pilot", "init", "--out", str(out), "--resolutions", "64", "160", "250",
                                "--samples", "30", "--sample-limit", "5", "--noise-relative", ".01",
                                "--max-workers", "4", "--rank", "64", "--dlr-only", "--baseline-out", str(self.out),
                                "--worker-memory-mib", "16384"]):
            main()
        manifest = json.loads((out / "production/manifest.json").read_text())
        settings = manifest["settings"]
        self.assertEqual(settings["samples"], 30)
        self.assertEqual(settings["dlr"]["rank"], 64)
        self.assertEqual(set(settings["scheduling"]["worker_memory_mib"].values()), {16384})
        self.assertEqual(manifest["protocol"]["samples"], list(range(5)))
        self.assertEqual(len(tasks(settings)), 30)
        self.assertEqual({task.start for task in tasks(settings)}, set(range(5)))
        self.assertEqual(json.loads((out / "summary.json").read_text())["total_runs"], 30)
        smoke = json.loads((out / "smoke/manifest.json").read_text())
        self.assertEqual(smoke["protocol"]["samples"], [0])
        self.assertEqual(len(tasks(smoke["settings"])), 2)
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["pilot", "run", "--out", str(out)]))
            stack.enter_context(patch("experiments.five_step_pilot.run.gpu_processes", return_value={1: set()}))
            stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.check_output", return_value="49140, 2"))
            launch = stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.Popen", return_value=process))
            stack.enter_context(patch("experiments.five_step_pilot.run.signal.signal"))
            stack.enter_context(patch("experiments.five_step_pilot.run.time.sleep", side_effect=KeyboardInterrupt))
            with self.assertRaises(KeyboardInterrupt):
                main()
        for call in launch.call_args_list:
            command = call.args[0]
            self.assertAlmostEqual(float(command[command.index("--memory-fraction") + 1]), (16384 - 768) / 49140)

    def test_multicircle_grid_keeps_untrained_models_pending(self) -> None:
        out = self.out / "multicircle"
        additional = self.out / "additional"
        for root, noise in ((out, ".2"), (additional, ".01")):
            with patch("sys.argv", ["pilot", "init", "--out", str(root), "--circles", "1", "2", "4",
                                    "--samples", "30", "--noise-relative", noise, "--max-workers", "4"]):
                main()
        settings = json.loads((out / "production/manifest.json").read_text())["settings"]
        self.assertEqual(len(tasks(settings)), 1080)
        self.assertEqual(json.loads((out / "summary.json").read_text())["total_runs"], 3240)
        smoke = json.loads((out / "smoke/manifest.json").read_text())["settings"]
        self.assertEqual(len(tasks(smoke)), 6)
        for root in (out, additional):
            for task in tasks(smoke):
                marker = task.marker(root / "smoke")
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}")
        for selection in ([], ["--run-circles", "1", "--priority-out", str(additional)]):
            if selection:
                for root in (out, additional):
                    for task in tasks(smoke):
                        if task.job.circles != 1:
                            task.marker(root / "smoke").unlink()
            process = MagicMock(pid=12345)
            process.poll.return_value = None
            with ExitStack() as stack:
                stack.enter_context(patch("sys.argv", ["pilot", "run", "--out", str(out),
                                                       "--additional-out", str(additional), *selection]))
                stack.enter_context(patch("experiments.five_step_pilot.run.model_ready",
                                          side_effect=lambda job, settings: (job.resolution, job.circles) != (64, 1)))
                stack.enter_context(patch("experiments.five_step_pilot.run.gpu_processes", return_value={1: set()}))
                stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.check_output", return_value="49140, 2"))
                launch = stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.Popen", return_value=process))
                stack.enter_context(patch("experiments.five_step_pilot.run.signal.signal"))
                stack.enter_context(patch("experiments.five_step_pilot.run.time.sleep", side_effect=KeyboardInterrupt))
                with self.assertRaises(KeyboardInterrupt):
                    main()
            self.assertEqual(launch.call_count, 4)
            self.assertEqual({Path(call.args[0][call.args[0].index("--manifest") + 1]).parents[1]
                              for call in launch.call_args_list}, {out, additional})
            self.assertTrue(all(not call.args[0][call.args[0].index("--job") + 1].startswith("r64_n1_")
                                for call in launch.call_args_list))
            status = json.loads((out / "queue_status.json").read_text())
            self.assertEqual(status["training_pending"], ["r64_n1"])
            self.assertEqual(sum("/r64_n1_" in name for name in status["pending"]), 120)
            if selection:
                commands = [call.args[0] for call in launch.call_args_list]
                self.assertTrue(all("_n1_" in command[command.index("--job") + 1] for command in commands))
                self.assertEqual(sum(Path(command[command.index("--manifest") + 1]).parents[1] == additional
                                     for command in commands), 1)
                self.assertTrue(all("_n1_" in name for name in status["pending"]))
                self.assertTrue(status["preflight_passed"])

    def test_replacement_admitted_before_reporting_and_readiness_deduplicated(self) -> None:
        for stage in ("smoke", "production"):
            manifest_path = self.out / stage / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["settings"]["scheduling"]["max_workers_per_gpu"] = 1
            manifest_path.write_text(json.dumps(manifest))
            for task in tasks(manifest["settings"]):
                if stage == "production" and task.job.resolution == 64 and task.job.problem == "ct":
                    continue
                marker = task.marker(manifest_path.parent)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}")
        replacement_started = threading.Event()
        report_finished = threading.Event()
        first, second = MagicMock(pid=12345), MagicMock(pid=12346)
        first.poll.return_value = 0
        first.returncode = 0
        second.poll.return_value = None
        launches = []

        def launch(command: list[str], **kwargs: object) -> MagicMock:
            launches.append(command)
            if len(launches) == 1:
                marker = self.out / "production/r64_n1_ct10/shard_000_001.json"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}")
                return first
            replacement_started.set()
            return second

        def report(out: Path) -> None:
            self.assertTrue(replacement_started.is_set(), "Report blocked replacement admission")
            report_finished.set()

        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["pilot", "run", "--out", str(self.out)]))
            stack.enter_context(patch("experiments.five_step_pilot.run.gpu_processes", return_value={1: set()}))
            stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.check_output", return_value="49140, 2"))
            stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.Popen", side_effect=launch))
            stack.enter_context(patch("experiments.five_step_pilot.run.signal.signal"))
            stack.enter_context(patch("experiments.five_step_pilot.run.time.sleep", side_effect=[None, KeyboardInterrupt]))
            ready = stack.enter_context(patch("experiments.five_step_pilot.run.model_ready", return_value=True))
            stack.enter_context(patch("experiments.five_step_pilot.run.report", side_effect=report))
            with self.assertRaises(KeyboardInterrupt):
                main()
        self.assertEqual(len(launches), 2)
        self.assertEqual(ready.call_count, 2)  # Once per model per tick, not once per sample.
        self.assertTrue(report_finished.is_set())

    def test_sample_groups_are_disjoint_and_recover_only_unfinished_members(self) -> None:
        for exit_code, workers in ((0, 2), (1, 2), (None, 4)):
            out = self.out / f"group_{exit_code}"
            for stage in ("smoke", "production"):
                root = out / stage
                root.mkdir(parents=True)
                manifest = json.loads((self.out / stage / "manifest.json").read_text())
                manifest["settings"]["scheduling"]["max_workers_per_gpu"] = workers
                (root / "manifest.json").write_text(json.dumps(manifest))
                for task in tasks(manifest["settings"]):
                    if stage == "production" and task.job.name == "r64_n1_ct10" and task.start < 6:
                        continue
                    marker = task.marker(root)
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("{}")
            first, second, third, fourth = [MagicMock(pid=pid) for pid in (12345, 12346, 12347, 12348)]
            first.poll.return_value = exit_code
            first.returncode = exit_code
            second.poll.return_value = third.poll.return_value = fourth.poll.return_value = None
            commands = []

            def launch(command: list[str], **kwargs: object) -> MagicMock:
                commands.append(command)
                if len(commands) == 1:
                    # Sample 0 finished; sample 1 paused or failed; sample 2 was not attempted.
                    marker = out / "production/r64_n1_ct10/shard_000_001.json"
                    marker.write_text("{}")
                return (first, second, third, fourth)[len(commands) - 1]

            with ExitStack() as stack:
                stack.enter_context(patch("sys.argv", ["pilot", "run", "--out", str(out), "--samples-per-process", "3"]))
                stack.enter_context(patch("experiments.five_step_pilot.run.gpu_processes", return_value={1: set()}))
                stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.check_output", return_value="49140, 2"))
                stack.enter_context(patch("experiments.five_step_pilot.run.subprocess.Popen", side_effect=launch))
                stack.enter_context(patch("experiments.five_step_pilot.run.signal.signal"))
                stack.enter_context(patch("experiments.five_step_pilot.run.time.sleep",
                                          side_effect=[KeyboardInterrupt] if exit_code is None else [None, KeyboardInterrupt]))
                stack.enter_context(patch("experiments.five_step_pilot.run.model_ready", return_value=True))
                stack.enter_context(patch("experiments.five_step_pilot.run.report"))
                with self.assertRaises(KeyboardInterrupt):
                    main()
            ranges = [(int(command[command.index("--sample-start") + 1]),
                       int(command[command.index("--sample-stop") + 1])) for command in commands]
            self.assertEqual(ranges, [(0, 1), (1, 2), (2, 4), (4, 6)] if exit_code is None else
                             [(0, 3), (3, 6), (2 if exit_code else 1, 3)])
            status = json.loads((out / "queue_status.json").read_text())
            self.assertEqual(len(status["failures"]), exit_code or 0)
            if exit_code:
                self.assertTrue(next(iter(status["failures"])).endswith("r64_n1_ct10_001_002"))
            self.assertEqual(status["pending"], [])


if __name__ == "__main__":
    unittest.main()

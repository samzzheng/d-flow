"""Cover multi-sample admission and the smoke barrier without GPU work."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
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


if __name__ == "__main__":
    unittest.main()

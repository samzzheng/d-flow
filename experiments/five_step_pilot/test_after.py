"""The dependency launcher must not admit GPU work prematurely."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.ct_mri_sweep.queue import tasks
from .after import main


class DependencyTests(unittest.TestCase):
    def test_requires_markers_and_successful_finished_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior, out = root / "prior", root / "next"
            (prior / "production").mkdir(parents=True)
            out.mkdir()
            settings = json.loads(Path("experiments/ct_mri_sweep/config_fp32.json").read_text())
            settings.update(resolutions=[64], circles=[1], samples=1)
            (prior / "production/manifest.json").write_text(json.dumps({"settings": settings}))
            for complete, status, failures, ready in ((False, "finished", {}, False),
                (True, "running", {}, False), (True, "finished", {"job": 1}, False), (True, "finished", {}, True)):
                with self.subTest(complete=complete, status=status, failures=failures):
                    if complete:
                        for task in tasks(settings):
                            marker = task.marker(prior / "production")
                            marker.parent.mkdir(parents=True, exist_ok=True)
                            marker.write_text("{}")
                    (prior / "queue_status.json").write_text(json.dumps({"status": status, "failures": failures}))
                    with ExitStack() as stack:
                        stack.enter_context(patch("sys.argv", ["after", "--out", str(out), "--after", str(prior)]))
                        stack.enter_context(patch("experiments.five_step_pilot.after.signal.signal"))
                        stack.enter_context(patch("experiments.five_step_pilot.after.time.sleep", side_effect=KeyboardInterrupt))
                        launch = stack.enter_context(patch("experiments.five_step_pilot.after.os.execv", side_effect=KeyboardInterrupt))
                        with self.assertRaises(KeyboardInterrupt):
                            main()
                    self.assertEqual(launch.call_count, int(ready))


if __name__ == "__main__":
    unittest.main()

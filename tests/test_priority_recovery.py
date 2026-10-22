"""The experiment scheduler must resume its suspended baseline on exit."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PriorityRecoveryTests(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux process-state control")
    def test_baseline_resumes_after_success_and_failed_control(self) -> None:
        script = Path(__file__).resolve().parents[1]/"experiments/separated_shapes_117k/priority_recovery.py"
        for child_exit in (0, 7):
            with self.subTest(child_exit=child_exit), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                baseline = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
                try:
                    process = Path(f"/proc/{baseline.pid}/stat")
                    queue = directory/"queue.json"
                    queue.write_text(json.dumps(dict(status="queued", file_hashes={},
                        command=[sys.executable, "-c", f"raise SystemExit({child_exit})"],
                        log=str(directory/"control.log"))))
                    plan = directory/"plan.json"
                    plan.write_text(json.dumps(dict(baseline_pid=baseline.pid,
                        baseline_start_time=process.read_text().rsplit(")", 1)[1].split()[19],
                        queues=[str(queue)])))
                    result = subprocess.run([sys.executable, str(script), str(plan)],
                        capture_output=True, timeout=10)
                    self.assertEqual(result.returncode == 0, child_exit == 0, result.stderr.decode())
                    self.assertIsNone(baseline.poll())
                    self.assertNotEqual(process.read_text().rsplit(")", 1)[1].split()[0], "T")
                    self.assertIn("baseline_resumed_at", json.loads(plan.read_text()))
                finally:
                    baseline.kill()
                    baseline.wait()


if __name__ == "__main__":
    unittest.main()

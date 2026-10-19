"""Exercise completion, restart skipping and failure handoff without GPU work."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import queue_small_circle_training as queue


class TrainingQueueTest(unittest.TestCase):
    def test_success_skips_completed_and_restores_reconstruction(self) -> None:
        self.check_handoff(fail=False)

    def test_failure_stops_training_and_restores_reconstruction(self) -> None:
        self.check_handoff(fail=True)

    def check_handoff(self, fail: bool) -> None:
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            jobs = [dict(resolution=64, circles=i, micro_batch=512,
                         run=str(out / f'n{i}'), command=['fake-trainer', str(i)]) for i in (1, 2, 3)]
            (out / 'manifest.json').write_text(json.dumps(dict(epochs=1000, jobs=jobs, source_sha256={})))
            (out / 'reconstruction_queue.json').write_text(json.dumps(dict(pid=999999999, argv=['fake-controller'])))
            (out / 'n1').mkdir()
            (out / 'n1/run_summary.json').write_text(json.dumps(dict(status='complete', epochs_completed=1000)))
            calls: list[list[str]] = []

            def start(command: list[str], **kwargs: object) -> SimpleNamespace:
                calls.append(command)
                code = 0
                if command[0] == 'fake-trainer':
                    code = 1 if fail else 0
                    if not fail:
                        (out / f'n{command[1]}/run_summary.json').write_text(json.dumps(dict(status='complete', epochs_completed=1000)))
                return SimpleNamespace(pid=7777, returncode=code, poll=lambda: code)

            with patch.object(queue.sys, 'argv', ['queue', '--out', str(out)]), \
                 patch.object(queue, 'find_controller', return_value=None), \
                 patch.object(queue.signal, 'signal'), \
                 patch.object(queue.subprocess, 'Popen', side_effect=start):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'Training failed'):
                        queue.main()
                else:
                    queue.main()
            state = json.loads((out / 'status.json').read_text())
            self.assertEqual(state['status'], 'failed' if fail else 'complete')
            self.assertEqual(calls[0], ['fake-trainer', '2'])
            self.assertEqual(calls[-1], ['fake-controller'])
            self.assertEqual(len(calls), 2 if fail else 3)
            self.assertEqual(state['jobs'][0]['status'], 'complete')
            self.assertEqual(state['jobs'][2]['status'], 'pending' if fail else 'complete')


if __name__ == '__main__':
    unittest.main()

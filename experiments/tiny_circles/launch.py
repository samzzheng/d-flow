"""Train the small-model pilot, then resume the checkpointed reconstruction queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / 'outputs/tiny_circles_20261019'
    queue = json.loads((out / 'paused_queue.json').read_text())
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'}
    model_out = root / 'outputs/fm_unet/circles-64-noattn-ch4-e1000/n1'
    command = [sys.executable, '-u', 'scripts/train_small_circle_models.py',
               '--out-dir', str(model_out), '--epochs', '1000', '--device', 'cuda:1']
    with (model_out / 'train.log').open('ab') as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)

    def forward_signal(signum: int, frame: Any) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    status = dict(pid=os.getpid(), training_pid=child.pid, command=command, status='training')
    (out / 'launch.json').write_text(json.dumps(status, indent=2))
    code = child.wait()
    # Another session may have already restarted the same queue during training.
    queue_pid = None
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().strip('\0').split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if argv == queue['argv']:
            queue_pid = int(path.parent.name)
            break
    if queue_pid is None:
        with (out / 'reconstruction_resume.log').open('ab') as log:
            resumed = subprocess.Popen(queue['argv'], cwd=root,
                env={**env, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        queue_pid = resumed.pid
    status.update(status='training_complete' if code == 0 else 'training_exited',
                  training_returncode=code, reconstruction_controller_pid=queue_pid)
    (out / 'launch.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()

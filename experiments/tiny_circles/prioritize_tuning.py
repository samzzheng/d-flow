"""Checkpoint-pause our reconstruction workers while GPU 1 is calibrated."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import FrameType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=int, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/microbatch_tuning_20261019'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proc = Path(f'/proc/{args.controller}')
    command = proc.joinpath('cmdline').read_bytes().split(b'\0')
    if b'experiments.five_step_pilot.run' not in command:
        raise RuntimeError('Controller identity mismatch')
    if proc.joinpath('stat').read_text().split()[2] == 'T':
        raise RuntimeError('Controller already paused; do not take ownership of its resumption')
    child: subprocess.Popen | None = None

    def stop(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    record = dict(controller=args.controller, controller_command=[c.decode() for c in command if c], workers=[])
    os.kill(args.controller, signal.SIGSTOP)
    try:
        for pid in map(int, proc.joinpath(f'task/{args.controller}/children').read_text().split()):
            worker = Path(f'/proc/{pid}')
            worker_command = worker.joinpath('cmdline').read_bytes().split(b'\0')
            if not worker_command or worker_command == [b'']:
                continue
            if b'experiments.ct_mri_sweep.run' not in worker_command:
                raise RuntimeError(f'Unexpected worker {pid}; refusing to signal it')
            record['workers'].append(dict(pid=pid, command=[c.decode() for c in worker_command if c]))
            os.kill(pid, signal.SIGTERM)
        (args.out/'scheduling.json').write_text(json.dumps({**record, 'status':'checkpointing'}, indent=2))
        deadline = time.monotonic() + 180
        while any((p := Path(f'/proc/{w["pid"]}/stat')).exists() and p.read_text().split()[2] != 'Z'
                  for w in record['workers']):
            if time.monotonic() > deadline:
                raise TimeoutError('Workers have not checkpointed; controller will resume')
            time.sleep(1)
        (args.out/'scheduling.json').write_text(json.dumps({**record, 'status':'tuning'}, indent=2))
        print('Reconstruction workers checkpointed; starting tuning.', flush=True)
        child = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.tiny_circles.tune_microbatches',
                                  '--device', 'cuda:1', '--out', str(args.out)])
        if child.wait():
            raise RuntimeError('Microbatch tuning failed')
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        if proc.exists():
            os.kill(args.controller, signal.SIGCONT)
        (args.out/'scheduling.json').write_text(json.dumps({**record, 'status':'controller_resumed'}, indent=2))
        print('Reconstruction controller resumed.', flush=True)


if __name__ == '__main__':
    main()

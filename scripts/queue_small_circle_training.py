"""Retrain the 24 small circle models on GPU 1 using calibrated microbatches."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import FrameType
from typing import Any


def save_status(path: Path, state: dict[str, Any]) -> None:
    state['updated_at'] = datetime.now().astimezone().isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def find_controller(command: list[str]) -> int | None:
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().strip('\0').split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if argv == command:
            return int(path.parent.name)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/small_circle_retraining_20261019'))
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['epochs'] != args.epochs:
            parser.error('Epoch budget differs from the saved queue')
    else:
        profile = json.loads((root / 'experiments/tiny_circles/microbatch_profile.json').read_text())
        jobs = []
        for resolution in (64, 96, 128, 160, 200, 250):
            micro_batch = profile['micro_batches'][str(resolution)]
            for circles in (1, 2, 3, 4):
                dataset = root / f'data/circles-{resolution}-n{circles}.pt'
                if not dataset.exists():
                    raise FileNotFoundError(dataset)
                run = root / f'outputs/fm_unet/circles-{resolution}-noattn-ch4-e{args.epochs}-mbtuned-20261019/n{circles}'
                if run.exists():
                    raise FileExistsError(f'Fresh retraining requires a new output: {run}')
                jobs.append(dict(resolution=resolution, circles=circles, micro_batch=micro_batch, run=str(run),
                    command=[sys.executable, '-u', 'scripts/train_small_circle_models.py',
                             '--resolution', str(resolution), '--n-circles', str(circles),
                             '--micro-batch', str(micro_batch), '--epochs', str(args.epochs),
                             '--device', 'cuda:1', '--out-dir', str(run)]))
        paths = ['train_fm_unet.py', 'scripts/train_small_circle_models.py',
                 'models/unet/default.py', 'models/unet/unet.py', 'sampler/ode_samplers.py']
        manifest = dict(epochs=args.epochs, device='cuda:1', parameters=22037, effective_batch=512,
                        microbatch_profile=profile, jobs=jobs,
                        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths})
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.prepare_only:
        print(f'Prepared {len(manifest["jobs"])} fresh training jobs.', flush=True)
        lock.close()
        return
    reconstruction = json.loads((args.out / 'reconstruction_queue.json').read_text())
    state: dict[str, Any] = dict(pid=os.getpid(), status='waiting_for_checkpoints', device='cuda:1',
                                jobs=[{**job, 'status': 'pending'} for job in manifest['jobs']])
    status_path = args.out / 'status.json'
    save_status(status_path, state)
    child: subprocess.Popen[bytes] | None = None
    stopping = False

    def stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'OPENBLAS_NUM_THREADS': '1', 'TQDM_MININTERVAL': '30', 'MPLBACKEND': 'Agg'}
    try:
        # Also handles resuming this training queue after reconstruction restarted.
        existing = find_controller(reconstruction['argv'])
        if existing is not None:
            os.kill(existing, signal.SIGTERM)
        old = Path(f'/proc/{existing or reconstruction["pid"]}/stat')
        deadline = time.monotonic() + 180
        while old.exists() and old.read_text().split()[2] != 'Z' and not stopping:
            if time.monotonic() > deadline:
                raise TimeoutError('Reconstruction controller has not finished checkpointing')
            time.sleep(1)
        for job in state['jobs']:
            if stopping:
                break
            run = Path(job['run'])
            summary = run / 'run_summary.json'
            if summary.exists():
                result = json.loads(summary.read_text())
                if result['status'] == 'complete' and result['epochs_completed'] == args.epochs:
                    job['status'] = 'complete'
                    continue
            for name, expected in manifest['source_sha256'].items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Training source changed since queue preparation: {name}')
            run.mkdir(parents=True, exist_ok=True)
            with (run / 'train.log').open('ab') as log:
                child = subprocess.Popen(job['command'], cwd=root, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
            if stopping:
                child.terminate()
            job.update(status='running', pid=child.pid, started_at=datetime.now().astimezone().isoformat())
            state['status'] = 'running'
            print(f'Started {job["resolution"]}² n={job["circles"]}, microbatch {job["micro_batch"]}, PID {child.pid}', flush=True)
            save_status(status_path, state)
            while child.poll() is None:
                time.sleep(5)
            job['returncode'] = child.returncode
            job['finished_at'] = datetime.now().astimezone().isoformat()
            if stopping:
                job['status'] = 'paused'
                break
            result = json.loads(summary.read_text()) if summary.exists() else {}
            job['status'] = 'complete' if child.returncode == 0 and result.get('epochs_completed') == args.epochs and result.get('status') == 'complete' else 'failed'
            save_status(status_path, state)
            if job['status'] == 'failed':
                raise RuntimeError(f'Training failed; see {run / "train.log"}')
        state['status'] = 'paused' if stopping else 'complete'
    except Exception as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        # Avoid duplicating a controller that another session has already restarted.
        controller_pid = find_controller(reconstruction['argv'])
        if controller_pid is None:
            with (args.out / 'reconstruction_resume.log').open('ab') as log:
                controller = subprocess.Popen(reconstruction['argv'], cwd=root,
                    env={**env, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            controller_pid = controller.pid
        state['reconstruction_controller_pid'] = controller_pid
        save_status(status_path, state)
        lock.close()


if __name__ == '__main__':
    main()

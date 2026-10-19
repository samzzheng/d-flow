"""Paired three-sample inverse tests and isolated per-update timings of small FM models."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import torch

from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.queue import source_hashes


class TimedEvaluator(run.Evaluator):
    def objective(self, source: torch.Tensor) -> float:
        torch.cuda.synchronize(source.device)
        started = time.perf_counter()
        value = super().objective(source)
        torch.cuda.synchronize(source.device)
        if not hasattr(self, 'objective_seconds'):
            self.objective_seconds: list[float] = []
        self.objective_seconds.append(time.perf_counter() - started)
        return value


class TimedLBFGS(run.SweepLBFGS):
    def step(self, closure: Callable[[], torch.Tensor]) -> float:
        evaluator = self.objective.__self__
        start_index = len(evaluator.objective_seconds)
        torch.cuda.synchronize(self.source.device)
        started = time.perf_counter()
        value = super().step(closure)
        torch.cuda.synchronize(self.source.device)
        total = time.perf_counter() - started
        trials = evaluator.objective_seconds[start_index:]
        accepted = self.consecutive_failures == 0
        if not hasattr(self, 'timings'):
            self.timings: list[dict[str, Any]] = []
        self.timings.append(dict(step=self.step_attempts, accepted=accepted,
            total_seconds=total, objective_seconds=sum(trials), objective_calls=len(trials),
            no_extra_backtracking_seconds=total - sum(trials[:-1]) if accepted else None,
            history_length=len(self.S)))
        return value


def summarize(out: Path) -> None:
    rows = []
    for size in ('small', 'large'):
        for problem in ('ct10', 'mri4'):
            for sample in range(3):
                case = out / size / f'r64_n1_{problem}' / f'sample_{sample:03d}'
                for method in run.METHODS:
                    path = case / (method.lower().replace('-', '_').replace(' ', '_') + '.pt')
                    if not path.exists():
                        continue
                    result = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                    if result['status'] not in run.TERMINAL:
                        continue
                    timings = [item['no_extra_backtracking_seconds'] for item in result['optimizer']['timings']
                               if item['accepted'] and item['step'] > 5]
                    last = result['history'][-1]
                    rows.append(dict(size=size, problem=problem, sample=sample, method=method,
                        image_error_percent=100*last['image_relative_l2'],
                        measurement_error_percent=100*last['measurement_relative_l2_clean'],
                        attempts=last['step'], status=result['status'],
                        accepted_updates=sum(item['source_step_norm'] > 0 for item in result['history'][1:]),
                        iteration_ms=1000*statistics.median(timings) if timings else None,
                        timed_updates=len(timings)))
    aggregated = []
    for problem in ('ct10', 'mri4'):
        for method in run.METHODS:
            for size in ('small', 'large'):
                group = [row for row in rows if (row['problem'], row['method'], row['size']) == (problem, method, size)]
                if group:
                    times = [row['iteration_ms'] for row in group if row['iteration_ms'] is not None]
                    aggregated.append(dict(problem=problem, method=method, size=size, completed=len(group),
                        median_image_error_percent=statistics.median(row['image_error_percent'] for row in group),
                        median_iteration_ms=statistics.median(times) if times else None))
    run.write_json(out / 'results.json', dict(samples=rows, aggregate=aggregated,
        timing_definition='Wall-clock optimizer update including one accepted forward objective, gradient and LBFGS; extra rejected Armijo forward evaluations removed. Failed attempts and first five attempts excluded. Plot/checkpoint I/O outside timing.',
        limitation='Actual trajectories and LBFGS history lengths differ; these are measured solver-update costs, not a fixed-history microbenchmark. Three paired samples do not establish general performance.'))
    for problem in ('ct10', 'mri4'):
        for sample in range(3):
            paths = [out / size / f'r64_n1_{problem}' / f'sample_{sample:03d}/inputs.pt' for size in ('small', 'large')]
            if all(path.exists() for path in paths):
                a, b = [torch.load(path, weights_only=True, map_location='cpu', mmap=True) for path in paths]
                for key in ('truth', 'clean', 'observed', 'initial', 'mask'):
                    assert a[key] is None and b[key] is None or torch.equal(a[key], b[key]), f'Unpaired {key}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.worker:
        run.Evaluator, run.SweepLBFGS = TimedEvaluator, TimedLBFGS
        sys.argv = ['inverse_worker', '--manifest', str(args.manifest), '--job', args.job, '--device', 'cuda:1']
        run.main()
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((run.ROOT / 'outputs/five_step_noise000_n124_20261014/production/manifest.json').read_text())['settings']
    sources = source_hashes()
    sources['experiments/tiny_circles/inverse_compare.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for size, directory, epoch in [('small', 'circles-64-noattn-ch4-e1000/n1', 1000),
                                    ('large', 'circles-64-noattn-ch16-b8-e200/n1', 200)]:
        settings = deepcopy(base)
        settings.update(resolutions=[64], circles=[1], samples=3, epoch=epoch,
                        run_template='outputs/fm_unet/' + directory)
        manifest = dict(settings=settings, source_sha256=sources,
                        protocol=dict(smoke_only=False, note='Paired small/large model inverse pilot; no noise; five Euler steps; original 1000-attempt safeguarded optimizer.'))
        path = out / size / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Existing manifest differs; use a new output directory')
        run.write_json(path, manifest)
    if not args.execute:
        print('Prepared', out)
        return
    # Pause only our known running shapes trainer. Do not touch GPU 0 or other users.
    paused: list[int] = []
    child: subprocess.Popen | None = None
    trainer = run.ROOT / 'outputs/fm_unet/separated-shapes-64-noattn-ch4-e1000'
    allowed = {os.getpid()}
    if json.loads((trainer / 'status.json').read_text())['status'] == 'running':
        pid = json.loads((trainer / 'launch.json').read_text())['pid']
        proc = Path(f'/proc/{pid}')
        if proc.exists():
            command = (proc / 'cmdline').read_bytes().decode().replace('\0', ' ')
            if proc.stat().st_uid != os.getuid() or 'scripts/train_separated_shapes.py' not in command or '--small-model' not in command:
                raise RuntimeError('Trainer identity mismatch')
            if (proc / 'stat').read_text().split()[2] == 'T':
                raise RuntimeError('Trainer was already paused; do not change its state')
            allowed.add(pid)
    active = {int(line) for line in subprocess.check_output(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines() if line.strip()}
    if active - allowed:
        raise RuntimeError(f'GPU 1 has other active processes: {active - allowed}; timing needs an isolated device')
    def stop(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for pid in allowed - {os.getpid()}:
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
        run.write_json(out / 'execution.json', dict(pid=os.getpid(), paused_training_pids=paused,
            started_at=datetime.now().astimezone().isoformat(), status='running'))
        # Alternate model order across acquisitions to reduce order bias.
        for size, job in [('large', 'r64_n1_ct10'), ('small', 'r64_n1_ct10'),
                          ('small', 'r64_n1_mri4'), ('large', 'r64_n1_mri4')]:
            command = [sys.executable, '-u', '-m', 'experiments.tiny_circles.inverse_compare',
                       '--worker', '--out', str(out), '--manifest', str(out / size / 'manifest.json'), '--job', job]
            with (out / f'{size}_{job}.log').open('ab') as log:
                child = subprocess.Popen(command, cwd=run.ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'})
            print('Started', size, job, child.pid, flush=True)
            code = child.wait()
            child = None
            summarize(out)
            if code != 0:
                raise RuntimeError(f'{size} {job} exited {code}; see log')
        run.write_json(out / 'complete.json', dict(completed_at=datetime.now().astimezone().isoformat()))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        for pid in paused:
            if Path(f'/proc/{pid}').exists():
                os.kill(pid, signal.SIGCONT)
        run.write_json(out / 'execution.json', dict(pid=os.getpid(), resumed_training_pids=paused,
            finished_at=datetime.now().astimezone().isoformat(), status='complete' if (out / 'complete.json').exists() else 'interrupted_or_failed'))


if __name__ == '__main__':
    main()

"""Concurrent accuracy solves followed by isolated no-line-search iteration timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any

import torch
from experiments.ct_mri_sweep import run


def alive(pid: int) -> bool:
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2] != 'Z'


def report(out: Path) -> None:
    rows = []
    timing_path = out / 'isolated_timings.json'
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else []
    for size in ('small', 'large'):
        for problem in ('ct10', 'mri4'):
            for sample in range(3):
                case = out / size / f'r64_n1_{problem}' / f'sample_{sample:03d}'
                for method in run.METHODS:
                    stem = method.lower().replace('-', '_').replace(' ', '_')
                    path = case / f'{stem}.json'
                    if not path.exists():
                        continue
                    result = json.loads(path.read_text())
                    if result['status'] not in run.TERMINAL:
                        continue
                    last = result['history'][-1]
                    timing = next((v for v in timings if (v['size'], v['problem'], v['sample'], v['method']) == (size, problem, sample, method)), None)
                    rows.append(dict(size=size, problem=problem, sample=sample, method=method,
                        image_error_percent=100*last['image_relative_l2'],
                        measurement_error_percent=100*last['measurement_relative_l2_clean'], attempts=last['step'],
                        accepted_updates=sum(v['source_step_norm'] > 0 for v in result['history'][1:]),
                        status=result['status'], iteration_ms=timing['median_ms'] if timing else None))
    aggregate = []
    for problem in ('ct10', 'mri4'):
        for method in run.METHODS:
            for size in ('small', 'large'):
                group = [v for v in rows if (v['problem'],v['method'],v['size']) == (problem,method,size)]
                if group:
                    times = [v['iteration_ms'] for v in group if v['iteration_ms'] is not None]
                    aggregate.append(dict(problem=problem,method=method,size=size,completed=len(group),
                        median_image_error_percent=statistics.median(v['image_error_percent'] for v in group),
                        median_iteration_ms=statistics.median(times) if times else None))
    run.write_json(out/'results.json',dict(samples=rows,aggregate=aggregate,
        timing_definition='Isolated replay at each final source: one forward objective, one gradient, saved-history LBFGS direction and capped update proposal; no Armijo search, checkpoint or plotting. Five warmups then 20 repeats. Actual terminal history lengths recorded.',
        limitation='Three paired samples; different training epochs and final optimizer histories. Earlier contended/serial solver timings are not used for the comparison.'))


def timing(out: Path) -> None:
    torch.set_num_threads(2)
    torch.cuda.set_device(1)
    records = []
    for size in ('large', 'small'):
        settings = json.loads((out/size/'manifest.json').read_text())['settings']
        execution = settings['execution']
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model, _ = run.load_flow_model(64, 1, settings['epoch'], 'cuda:1', run.ROOT/settings['run_template'])
        run.tangent_unet.GN_STATS_MODE = 'native'
        run.tangent_unet.GN_ACTION_BACKEND = execution['dlr_fused_norm']
        run.tangent_unet.GN_PRIMAL_BACKEND = execution['dlr_fused_primal']
        run.enriched.SPECIALIZE_ZERO_ENRICHMENT = execution['dlr_specialized_algebra']
        run.enriched.SKIP_UNTRUNCATED_SVD = True
        zero = torch.zeros(1,1,64,64,device='cuda:1')
        products = run.GraphedAProducts(model, zero, share_linearization=True,
            full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
            max_step_jvp_columns=execution['dlr_jvp_columns'], max_step_vjp_columns=execution['dlr_vjp_columns'])
        for problem in ('ct10','mri4'):
            for sample in range(3):
                case = out/size/f'r64_n1_{problem}'/f'sample_{sample:03d}'
                inputs = torch.load(case/'inputs.pt',map_location='cuda:1',weights_only=True)
                seed = settings['seed'] + 640000 + 100 + sample
                operator = run.CT(64,10,torch.device('cuda:1')) if problem=='ct10' else run.MRI(64,4,seed+10000000,torch.device('cuda:1'),'cartesian')
                for method in run.METHODS:
                    stem = method.lower().replace('-','_').replace(' ','_')
                    saved = torch.load(case/f'{stem}.pt',map_location='cuda:1',weights_only=True)
                    source = saved['source'].detach().requires_grad_(True)
                    evaluator = run.Evaluator(model,operator,inputs['observed'],settings,seed+30000000,products)
                    evaluator.method = method
                    optimizer = run.SweepLBFGS(source,evaluator.objective,**{**settings['optimizer'],'step_cap':.64,'project':False})
                    for key in ('S','Y','step_attempts'):
                        setattr(optimizer,key,saved['optimizer'][key])
                    durations = []
                    for repeat in range(25):
                        evaluator.previous_v = saved.get('previous_v')
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        evaluator.objective(source)
                        optimizer.g = evaluator.gradient(source,method).flatten()
                        direction = optimizer._direction()
                        slope = float(optimizer.g.dot(direction))
                        if not math.isfinite(slope) or slope >= 0:
                            direction = -optimizer.g.clone()
                        rate = optimizer.lr
                        if optimizer.step_attempts == 0:
                            rate *= min(1.,1./float(optimizer.g.abs().sum()))
                        norm = float(direction.norm())
                        if rate*norm > optimizer.step_cap:
                            rate = optimizer.step_cap/norm
                        proposal = source.detach() + rate*direction.reshape_as(source)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-start
                        if repeat >= 5:
                            durations.append(1000*elapsed)
                    assert torch.isfinite(proposal).all()
                    records.append(dict(size=size,problem=problem,sample=sample,method=method,
                        median_ms=statistics.median(durations),min_ms=min(durations),max_ms=max(durations),
                        repeats=len(durations),history_length=len(optimizer.S)))
                    run.write_json(out/'isolated_timings.json',records)
                other = out/('small' if size=='large' else 'large')/f'r64_n1_{problem}'/f'sample_{sample:03d}/inputs.pt'
                paired = torch.load(other,map_location='cuda:1',weights_only=True)
                for key in ('truth','clean','observed','initial','mask'):
                    assert inputs[key] is None and paired[key] is None or torch.equal(inputs[key],paired[key]),key
                report(out)
        del products, model, evaluator, optimizer
        torch.cuda.empty_cache()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--size')
    parser.add_argument('--problem')
    parser.add_argument('--sample',type=int)
    args=parser.parse_args()
    out=args.out.resolve()
    if args.worker:
        sys.argv=['worker','--manifest',str(out/args.size/'manifest.json'),'--job',f'r64_n1_{args.problem}',
                  '--device','cuda:1','--sample-start',str(args.sample),'--sample-stop',str(args.sample+1),'--memory-fraction','0.2']
        run.main()
        return
    previous=json.loads((out/'execution.json').read_text())
    if previous['status']=='running' and alive(previous['pid']):
        command=Path(f'/proc/{previous["pid"]}/cmdline').read_bytes()
        if b'experiments.tiny_circles.inverse_compare' not in command:
            raise RuntimeError('Previous runner identity mismatch')
        os.kill(previous['pid'],signal.SIGTERM)
        while alive(previous['pid']):
            time.sleep(.5)
    trainer=run.ROOT/'outputs/fm_unet/separated-shapes-64-noattn-ch4-e1000'
    pid=json.loads((trainer/'launch.json').read_text())['pid']
    paused=[]
    children: list[subprocess.Popen] = []
    def stop(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    try:
        if alive(pid):
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'train_separated_shapes.py' not in command or b'--small-model' not in command:
                raise RuntimeError('Trainer identity mismatch')
            os.kill(pid,signal.SIGSTOP);paused.append(pid)
        run.write_json(out/'parallel_execution.json',dict(status='running',pid=os.getpid(),paused_training=paused,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),timing='Separate isolated final-source replay; all trajectory timings ignored'))
        pending=[(size,problem,sample) for sample in range(3) for problem in ('ct10','mri4') for size in ('small','large')]
        while pending or children:
            for child in list(children):
                if child.poll() is not None:
                    children.remove(child)
                    if child.returncode:
                        raise RuntimeError(f'Worker {child.pid} failed: {child.returncode}')
                    report(out)
            while pending and len(children)<4:
                size,problem,sample=pending.pop(0)
                command=[sys.executable,'-u','-m','experiments.tiny_circles.inverse_parallel','--worker','--out',str(out),'--size',size,'--problem',problem,'--sample',str(sample)]
                with (out/f'{size}_{problem}_{sample}_parallel.log').open('ab') as log:
                    child=subprocess.Popen(command,cwd=run.ROOT,stdout=log,stderr=subprocess.STDOUT,
                        env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
                children.append(child)
                print('Accuracy',size,problem,sample,child.pid,flush=True)
            time.sleep(1)
        print('Isolated timing',flush=True)
        active={int(line) for line in subprocess.check_output(['nvidia-smi','--id=1',
            '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if line.strip()}
        if active-set(paused)-{os.getpid()}:
            raise RuntimeError('GPU 1 acquired another process; refusing contended timing')
        timing(out)
        report(out)
        run.write_json(out/'complete.json',dict(status='complete',paired_cases=6,method_model_solves=36))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for pid in paused:
            if alive(pid):
                os.kill(pid,signal.SIGCONT)
        run.write_json(out/'parallel_execution.json',dict(status='complete' if (out/'complete.json').exists() else 'interrupted_or_failed',resumed_training=paused))


if __name__=='__main__':
    main()

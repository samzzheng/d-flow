"""Re-evaluate fixed initial and optimized sources on refined Euler grids."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import metrics, write_json
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=Path('outputs/separated_shapes_steps_20261019/diagnostics'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.75, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    base = Path('outputs/separated_shapes_noiseless_20261016/production')
    dlr_base = Path('outputs/separated_shapes_rank_sweep_20261017/rank_128/production')
    cases = [(64,'ct10',0), (64,'mri4',1), (96,'mri4',1), (128,'ct10',0),
             (128,'mri4',2), (160,'ct10',0), (200,'ct10',1), (250,'mri4',0)]
    records = []
    for resolution, problem, sample in cases:
        name = f'r{resolution}_shapes_{problem}/sample_{sample:03d}'
        case = base/name
        inputs = torch.load(case/'inputs.pt',map_location=device,weights_only=True)
        protocol = inputs['protocol']
        folder = Path(f'outputs/fm_unet/separated-shapes-{resolution}-noattn-ch16-b8-e200')
        checkpoint = folder/'checkpoints/epoch_0200.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==protocol['checkpoint_sha256']
        model, _ = load_flow_model(resolution,0,200,str(device),folder)
        info = protocol['operator']
        operator = CT(resolution,10,device) if problem=='ct10' else MRI(resolution,4,info['mask_seed'],device,info['pattern'])
        if problem=='mri4':
            assert torch.equal(operator.mask,inputs['mask'])
        assert torch.equal(inputs['clean'],inputs['observed'])
        saved = {method:torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                 for method,path in [('D-Flow',case/'d_flow.pt'),('P-Flow',case/'p_flow.pt'),
                                      ('DLR128',dlr_base/name/'dlr_d_flow.pt')]}
        sources = {'initial':inputs['initial'],**{method:s['source'].to(device) for method,s in saved.items()}}
        results = dict(case=name,protocol=protocol,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       sources=[], note='Fixed sources, not reoptimized reconstructions. Models are the original larger networks.')
        snapshots = dict(truth=inputs['truth'].cpu(),clean=inputs['clean'].cpu(),images={})
        for label, source in sources.items():
            for steps in (5,10,20,50,100):
                start = time.perf_counter()
                variable = source.detach().requires_grad_(True)
                image = euler_flow(model,variable,steps)
                prediction = operator(image)
                loss = .5*(prediction-inputs['observed']).abs().square().sum(dtype=torch.float64)
                exact, = torch.autograd.grad(loss,variable)
                image = image.detach()
                _, gx = endpoint_gradient(image,operator,inputs['observed'])
                gradient_norm, gx_norm = float(exact.norm()),float(gx.norm())
                row = dict(source=label,steps=steps,objective=float(loss.detach()),
                           **metrics(image,prediction.detach(),inputs['truth'],inputs['clean'],inputs['observed']),
                           source_rms=float(source.norm())/resolution,
                           image_norm_over_gt=float(image.norm()/inputs['truth'].norm()),
                           image_min=float(image.min()),image_max=float(image.max()),
                           foreground_pixels=int((image>.5).sum()),truth_foreground_pixels=int((inputs['truth']>.5).sum()),
                           exact_gradient_norm=gradient_norm,endpoint_gradient_norm=gx_norm,
                           exact_over_endpoint_gradient_norm=gradient_norm/gx_norm if gx_norm else None,
                           exact_proxy_cosine=float((exact*gx).sum())/(gradient_norm*gx_norm) if gradient_norm*gx_norm else None)
                if steps==5 and label!='initial':
                    row['replay_image_error_over_gt']=float((image-saved[label]['image'].to(device)).norm()/inputs['truth'].norm())
                if steps in (5,20):
                    trials=[]
                    for direction_name,gradient in [('exact',exact),('proxy',gx)]:
                        norm=float(gradient.norm())
                        if not norm:
                            continue
                        direction=-gradient/norm
                        slope=float((exact*direction).sum())
                        for fraction in (1.,.125,.015625):
                            displacement=.01*resolution*fraction
                            with torch.no_grad():
                                trial_image=euler_flow(model,source+displacement*direction,steps)
                                value=float(.5*(operator(trial_image)-inputs['observed']).abs().square().sum(dtype=torch.float64))
                            trials.append(dict(direction=direction_name,displacement=displacement,true_slope=slope,
                                               objective_change=value-row['objective']))
                    row['direction_trials']=trials
                row['seconds']=time.perf_counter()-start
                results['sources'].append(row)
                snapshots['images'][f'{label}_{steps}']=image.cpu()
                write_json(args.out/name.replace('/','_')/'diagnostics.json',results)
                print(name,label,steps,'image%',round(100*row['image_relative_l2'],3),
                      'norm_ratio',round(row['image_norm_over_gt'],3),
                      'gradient',round(gradient_norm,6),flush=True)
                del image,prediction,loss,exact,gx,variable
            torch.cuda.empty_cache()
        torch.save(snapshots,args.out/name.replace('/','_')/'images.pt')
        records.append(results)
        write_json(args.out/'summary.json',dict(status='running',cases=records))
        del model,saved,sources,inputs
        torch.cuda.empty_cache()
    write_json(args.out/'summary.json',dict(status='complete',cases=records))


if __name__=='__main__':
    main()

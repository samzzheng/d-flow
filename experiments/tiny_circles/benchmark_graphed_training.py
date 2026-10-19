"""Alternating native/graph update timings with shared cuDNN choices and real data."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from models.unet.default import Unet
from train_fm_unet import autocast_context, batch_fm_loss, fm_loss
from training.fm_acceleration import GraphedMicrobatches


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolution',type=int,required=True)
    parser.add_argument('--micro-batch',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.cuda.set_device(1);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(191)
    base=Unet(ch=4,ch_mul=[1,2,2],att_channels=[0,0,0],groups=2,mid_attn=False,dropout=0.).to('cuda:1',memory_format=torch.channels_last)
    models=[base,deepcopy(base)]
    opts=[torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4,fused=True) for m in models]
    emas=[AveragedModel(m,multi_avg_fn=get_ema_multi_avg_fn(.999)) for m in models]
    replay=GraphedMicrobatches(models[1],fm_loss,'cuda:1')
    dataset=torch.load(f'data/circles-{args.resolution}-n1.pt',mmap=True,weights_only=True)
    resident={key:dataset[key].to('cuda:1') for key in ('train','val','test')}
    indices=torch.arange(512,device='cuda:1')
    timings=[[],[]]
    for round_index in range(5):
        for variant in ((0,1) if round_index%2==0 else (1,0)):
            torch.cuda.synchronize();start=time.perf_counter()
            losses=[]
            for _ in range(8):
                images=resident['train'][indices].float().div_(255).contiguous(memory_format=torch.channels_last)
                opts[variant].zero_grad(set_to_none=variant==0)
                with autocast_context('cuda:1'):
                    loss=batch_fm_loss(models[variant],images,'cuda:1',args.micro_batch,backward=True) if variant==0 else replay(images,args.micro_batch)
                opts[variant].step();emas[variant].update_parameters(models[variant]);losses.append(loss)
            float(torch.stack(losses).mean());torch.cuda.synchronize()
            seconds=(time.perf_counter()-start)/8
            if round_index:timings[variant].append(seconds)
    result=dict(resolution=args.resolution,micro_batch=args.micro_batch,effective_batch=512,
                native_seconds=statistics.median(timings[0]),graph_seconds=statistics.median(timings[1]),
                rounds=timings,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    result['reduction_percent']=100*(1-result['graph_seconds']/result['native_seconds'])
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'timing.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()

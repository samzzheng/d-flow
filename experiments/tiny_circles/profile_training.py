"""Profile and benchmark the current effective-batch-512 training step on GPU 1."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from models.unet.default import Unet
from experiments.tiny_circles.compile_backend import compile_native_norm
from train_fm_unet import autocast_context, batch_fm_loss, fm_loss
from training.fm_acceleration import GraphedMicrobatches


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolution',type=int,default=250)
    parser.add_argument('--micro-batch',type=int,default=32)
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--graph',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--native-norm',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_device(1)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(912)
    model=Unet(ch=4,ch_mul=[1,2,2],att_channels=[0,0,0],groups=2,mid_attn=False,dropout=0.).to('cuda:1',memory_format=torch.channels_last)
    start=time.monotonic()
    if args.compile:
        model=torch.compile(model,backend=compile_native_norm) if args.native_norm else torch.compile(model,options={'emulate_precision_casts': True})
    ema=AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.999))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,fused=True)
    images=torch.randint(0,256,(512,1,args.resolution,args.resolution),device='cuda:1',dtype=torch.uint8)
    accelerated=GraphedMicrobatches(model,fm_loss,'cuda:1') if args.graph else None
    losses=[]
    for _ in range(5):
        optimizer.zero_grad(set_to_none=not args.graph)
        with autocast_context('cuda:1'):
            loss=accelerated(images.float().div(255).contiguous(memory_format=torch.channels_last),args.micro_batch) if accelerated else batch_fm_loss(model,images.float().div(255).contiguous(memory_format=torch.channels_last),'cuda:1',args.micro_batch,backward=True)
        optimizer.step();ema.update_parameters(model)
    torch.cuda.synchronize();startup=time.monotonic()-start
    rounds=[]
    for _ in range(3):
        start=time.perf_counter()
        for _ in range(8):
            optimizer.zero_grad(set_to_none=not args.graph)
            with autocast_context('cuda:1'):
                loss=accelerated(images.float().div(255).contiguous(memory_format=torch.channels_last),args.micro_batch) if accelerated else batch_fm_loss(model,images.float().div(255).contiguous(memory_format=torch.channels_last),'cuda:1',args.micro_batch,backward=True)
            optimizer.step();ema.update_parameters(model);losses.append(loss)
        float(torch.stack(losses).mean());torch.cuda.synchronize()
        rounds.append((time.perf_counter()-start)/8)
    result=dict(resolution=args.resolution,micro_batch=args.micro_batch,compiled=args.compile,graphed=args.graph,
                startup_seconds=startup,round_seconds=rounds,median_seconds=statistics.median(rounds),
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,loss=float(loss))
    (args.out/'timing.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            for _ in range(2):
                optimizer.zero_grad(set_to_none=not args.graph)
                with autocast_context('cuda:1'):
                    loss=accelerated(images.float().div(255).contiguous(memory_format=torch.channels_last),args.micro_batch) if accelerated else batch_fm_loss(model,images.float().div(255).contiguous(memory_format=torch.channels_last),'cuda:1',args.micro_batch,backward=True)
                optimizer.step();ema.update_parameters(model)
            torch.cuda.synchronize()
        table=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25)
        (args.out/'profile.txt').write_text(table);print(table,flush=True)


if __name__=='__main__':main()

"""Native-operation graph replay equivalence through partial batches and resume."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
from models.unet.default import Unet
from train_fm_unet import autocast_context,batch_fm_loss,fm_loss
from training.fm_acceleration import GraphedMicrobatches


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolution',type=int,default=64)
    parser.add_argument('--micro-batch',type=int,default=32)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.cuda.set_device(1);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(781)
    base=Unet(ch=4,ch_mul=[1,2,2],att_channels=[0,0,0],groups=2,mid_attn=False,dropout=0.).to('cuda:1',memory_format=torch.channels_last)
    models=[base,deepcopy(base)]
    optimizers=[torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,fused=True) for model in models]
    emas=[AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.999)) for model in models]
    replay=GraphedMicrobatches(models[1],fm_loss,'cuda:1')
    rows=[]
    for index,count in enumerate((64,40,64,13,64,40,64,64)):
        images=torch.rand(count,1,args.resolution,args.resolution,device='cuda:1').contiguous(memory_format=torch.channels_last)
        losses=[];gradients=[]
        for variant,(model,optimizer,ema) in enumerate(zip(models,optimizers,emas)):
            torch.manual_seed(20+index)
            optimizer.zero_grad(set_to_none=variant==0)
            optimizer.param_groups[0]['lr']=2e-4 if index<4 else 1e-4
            with autocast_context('cuda:1'):
                loss=batch_fm_loss(model,images,'cuda:1',args.micro_batch,backward=True) if variant==0 else replay(images,args.micro_batch)
            losses.append(float(loss))
            gradients.append(torch.cat([p.grad.detach().flatten() for p in model.parameters()]))
            optimizer.step();ema.update_parameters(model)
        row=dict(update=index+1,count=count,loss_difference=abs(losses[0]-losses[1]),
                 gradient_relative_difference=float((gradients[0]-gradients[1]).norm()/gradients[0].norm()))
        rows.append(row);print(json.dumps(row),flush=True)
        torch.testing.assert_close(gradients[0],gradients[1],rtol=0,atol=0)
        assert losses[0]==losses[1]
        for a,b in zip(models[0].parameters(),models[1].parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(emas[0].state_dict().values(),emas[1].state_dict().values()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(optimizers[0].state.values(),optimizers[1].state.values()):
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        if index==3:
            # Rebuild runtime graphs around loaded parameters/moments, as on full checkpoint resume.
            state=deepcopy(dict(model=models[1].state_dict(),optimizer=optimizers[1].state_dict(),ema=emas[1].state_dict()))
            models[1]=deepcopy(models[0]);models[1].load_state_dict(state['model'])
            optimizers[1]=torch.optim.AdamW(models[1].parameters(),lr=2e-4,fused=True)
            optimizers[1].load_state_dict(state['optimizer'])
            emas[1]=AveragedModel(models[1],multi_avg_fn=get_ema_multi_avg_fn(.999));emas[1].load_state_dict(state['ema'])
            replay=GraphedMicrobatches(models[1],fm_loss,'cuda:1')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'verification.json').write_text(json.dumps(dict(status='passed',resolution=args.resolution,rows=rows),indent=2))


if __name__=='__main__':main()

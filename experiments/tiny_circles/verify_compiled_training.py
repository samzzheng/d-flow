"""Compare native and compiled FM forward/gradient/update behavior with paired draws."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from unittest.mock import patch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from models.unet.default import Unet
from experiments.tiny_circles.compile_backend import compile_native_norm
from train_fm_unet import autocast_context, batch_fm_loss


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolution',type=int,default=64)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--native-norm',action='store_true')
    args=parser.parse_args()
    torch.cuda.set_device(1);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(879)
    base=Unet(ch=4,ch_mul=[1,2,2],att_channels=[0,0,0],groups=2,mid_attn=False,dropout=0.).to('cuda:1',memory_format=torch.channels_last)
    if args.checkpoint:
        state=torch.load(args.checkpoint,weights_only=True,map_location='cuda:1')
        base.load_state_dict(state['model'])
    dataset=torch.load(f'data/circles-{args.resolution}-n1.pt',weights_only=True,mmap=True)['train']
    compiled=torch.compile(deepcopy(base),backend=compile_native_norm) if args.native_norm else torch.compile(deepcopy(base),options={'emulate_precision_casts': True})
    models=(base,compiled)
    optimizers=[torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,fused=True) for model in models]
    emas=[AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.999)) for model in models]
    rows=[]
    for index,count in enumerate((64,64,40,64,40,64,64,40)):
        images=dataset[index*64:index*64+count].to('cuda:1').float().div(255).contiguous(memory_format=torch.channels_last)
        torch.manual_seed(100+index)
        times=torch.rand(count,device=images.device)
        noise=torch.randn_like(images)
        gradients=[];losses=[]
        for model,optimizer,ema in zip(models,optimizers,emas):
            torch.manual_seed(100+index)
            optimizer.zero_grad(set_to_none=True)
            offset=0
            def fixed_loss(m: torch.nn.Module, x: torch.Tensor, device: str, t_alpha: float=1.) -> torch.Tensor:
                nonlocal offset
                t=times[offset:offset+len(x)]
                z=noise[offset:offset+len(x)]
                offset+=len(x)
                xt=(1-t[:,None,None,None])*z+t[:,None,None,None]*x
                return (m(xt,t)-(x-z)).square().mean()
            with patch('train_fm_unet.fm_loss',fixed_loss), autocast_context('cuda:1'):
                loss=batch_fm_loss(model,images,'cuda:1',32,backward=True)
            gradients.append(torch.cat([p.grad.detach().flatten() for p in model.parameters()]))
            losses.append(float(loss))
            optimizer.step();ema.update_parameters(model)
        gradient_relative=float((gradients[0]-gradients[1]).norm()/gradients[0].norm())
        weights=[torch.cat([p.detach().flatten() for p in model.parameters()]) for model in models]
        row=dict(update=index+1,count=count,native_loss=losses[0],compiled_loss=losses[1],
                 relative_loss_difference=abs(losses[0]-losses[1])/abs(losses[0]),
                 relative_gradient_difference=gradient_relative,
                 relative_parameter_difference=float((weights[0]-weights[1]).norm()/weights[0].norm()))
        rows.append(row);print(json.dumps(row),flush=True)
        if index == 0:
            offset = 0
            details = []
            for name, parameter in base.named_parameters():
                left = gradients[0][offset:offset+parameter.numel()]
                right = gradients[1][offset:offset+parameter.numel()]
                details.append(dict(name=name, native_norm=float(left.norm()), compiled_norm=float(right.norm()), difference=float((left-right).norm())))
                offset += parameter.numel()
            args.out.mkdir(parents=True,exist_ok=True)
            (args.out/'gradient_details.json').write_text(json.dumps(sorted(details,key=lambda item:item['difference'],reverse=True),indent=2))
        assert row['relative_loss_difference']<.01,row
        assert gradient_relative<.05,row
        assert row['relative_parameter_difference']<.005,row
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'verification.json').write_text(json.dumps(dict(resolution=args.resolution,rows=rows,status='passed'),indent=2))


if __name__=='__main__':main()

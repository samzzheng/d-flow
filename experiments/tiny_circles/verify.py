"""GPU 1 check of capture, partial batches, LR changes and state restoration."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from experiments.tiny_circles.train import TrainingStep
from models.unet.default import Unet


def main() -> None:
    torch.set_num_threads(4)
    torch.cuda.set_device(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(712)
    model = Unet(ch=4, ch_mul=[1,2,2], att_channels=[0,0,0], groups=2,
                 mid_attn=False, dropout=0.).to('cuda:1', memory_format=torch.channels_last)
    eager = TrainingStep(deepcopy(model), 16, 2e-4, False)
    graph = TrainingStep(deepcopy(model), 16, 2e-4, True)
    restored = TrainingStep(deepcopy(model), 16, 2e-4, True)
    worst = 0.0
    for index in range(10):
        count = 7 if index in (2, 6) else 16
        x1 = torch.rand(count, 1, 64, 64, device='cuda:1').contiguous(memory_format=torch.channels_last)
        x0, t = torch.randn_like(x1), torch.rand(count, device='cuda:1')
        for engine in (eager, graph, restored):
            engine.lr.fill_(2e-4 if index < 5 else 1e-4)
            if engine.graph is None or count != 16:
                engine.eager(x1, x0, t)
            else:
                engine.x1.copy_(x1)
                engine.x0.copy_(x0)
                engine.t.copy_(t)
                engine.graph.replay()
        if index == 4:
            restored.restore(deepcopy(dict(model=graph.model.state_dict(), ema=graph.ema.state_dict(),
                                          optimizer=graph.optimizer.state_dict(), updates=graph.updates)))
        for other in (graph, restored):
            for left, right in zip(eager.model.parameters(), other.model.parameters()):
                worst = max(worst, float((left-right).detach().abs().max()))
                torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-7)
            for left, right in zip(eager.ema.parameters(), other.ema.parameters()):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-7)
            for left, right in zip(eager.optimizer.state.values(), other.optimizer.state.values()):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=1e-5, atol=2e-7)
    result = dict(status='passed', updates=10, max_parameter_difference=worst,
                  checked=['full/partial/full batches', 'EMA', 'Adam moments', 'LR change', 'checkpoint restoration'])
    out = Path('outputs/tiny_circles_20261019')
    out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()

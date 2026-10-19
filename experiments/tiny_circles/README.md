# Small circles-64 flow-matching pilot

Start with the single-circle dataset and a 22,037-parameter non-attention U-Net
(channels 4/8/8, GroupNorm with 2 groups), compared with the existing 472,017-parameter
4-level model. This changes width and depth; it is a capacity pilot, not a
controlled width-only ablation or a demonstrated minimum model size.

- Existing paired 48,000/1,000/1,000 training/validation/test images at 64².
- 1,000 full epochs, batch 512 including the final partial batch; AdamW LR 2e-4,
  weight decay 1e-4, cosine decay to 1e-5; EMA 0.999.
- Same uniform-time conditional flow-matching objective as `train_fm_unet.py`.
- BF16 autocast training and FP32 evaluation, TF32 disabled. GPU 1 only.
- GPU-resident compact uint8 dataset; channels-last tensors; fused AdamW;
  captured forward/backward/optimizer/EMA step. Graph capture warmup is undone.
  No per-minibatch scalar transfer or data-loader workers.
- Full optimizer, EMA, RNG, epoch and history checkpoint every epoch. SIGTERM
  saves at the end of the current epoch. `--resume` restores the Adam buffers
  in place so CUDA graph pointers remain valid.
- Fixed validation noise/time draws shared with the existing epoch-200 baseline.
  Select the small model's best checkpoint using validation only. Evaluate it
  once on the held-out test split after training. Compare both models' raw
  5- and 100-Euler-step samples with identical source seeds; plots display [0,1].
- Different training budgets are intentional: the small model gets 5× the epochs.
  FM loss and unconditional samples do not establish inverse-reconstruction
  accuracy; that requires a subsequent paired CT/MRI test.

Run from the repository root:

```sh
.venv/bin/python -m experiments.tiny_circles.verify
.venv/bin/python -m experiments.tiny_circles.train \
  --out outputs/tiny_circles_20261019/n1_22037_e1000 --epochs 1000
```

The active run uses `experiments.tiny_circles.launch` to resume the previously
checkpoint-paused reconstruction queue after training exits. Runtime records,
benchmark evidence, logs, checkpoints and sample grids are under
`outputs/tiny_circles_20261019/`. Do not launch a duplicate trainer.

# Small circles-64 flow-matching pilot

The active 22,037-parameter model now uses the standard `train_fm_unet.py` trainer
and the normal FM output layout:

```
outputs/fm_unet/circles-64-noattn-ch4-e1000/n1/
  config.json
  launch.json
  train.log
  loss_history.json
  loss_curves.png
  checkpoints/epoch_XXXX.pt
  checkpoints/training_state.pt
  eval/epoch_XXXX.png
  status.json
  run_summary.json          # created on completion
```

Run/resume with:

```sh
.venv/bin/python scripts/train_small_circle_models.py --device cuda:1
```

Do not launch a duplicate trainer. The active supervisor is
`experiments.tiny_circles.launch`; it restores the previous reconstruction queue
when training exits.

- Single-circle 64² dataset: 48,000/1,000/1,000 train/validation/test images.
- Channels 4/8/8, no attention, GroupNorm groups 2; 22,037 parameters versus
  the existing model's 472,017. This is a width-and-depth capacity pilot.
- 1,000 epochs, batch/micro-batch 512, AdamW LR 2e-4, weight decay 1e-4,
  cosine LR decay to 1e-5, EMA .999, uniform-time FM objective, BF16 training.
- Same standard stochastic validation, loss-curve formatting, and 100-step
  paired Euler/RK4 evaluation grids as the other FM models.
- GPU-resident compact data, channels-last and fused AdamW retain efficient
  training. The standard eager loop replaces the pilot's CUDA-graph loop.
- Full model/EMA/Adam/RNG checkpoint every epoch; SIGTERM checkpoints at the
  epoch boundary. Loss curves update every 10 epochs and at sampling checkpoints.

The earlier isolated trainer (`train.py`) and its verification script remain as
archived pilot/benchmark code. Its first 135 epochs, model, EMA and 12,690 Adam
updates were migrated without resetting. Epochs 1–135 used fixed-draw FP32
validation and sample-weighted training means; subsequent epochs use the
standard trainer's stochastic BF16 validation and batch-averaged training means.
These estimate the same objective but are not identical measurement protocols;
the transition is recorded in `config.json`. The original history is preserved.

Migration checks and the two-epoch standard-trainer smoke are under
`outputs/tiny_circles_20261019/`. Smoke state is separate from production.
Matching FM loss or unconditional samples does not establish matching inverse
reconstruction accuracy; a later paired CT/MRI comparison is still needed.

The same architecture is also tested on separated-shapes-64 using
`python scripts/train_separated_shapes.py --small-model --device cuda:1`.
Its default 1,000-epoch output is
`outputs/fm_unet/separated-shapes-64-noattn-ch4-e1000/`.
After training, `scripts/compare_small_fm_models.py` produces paired held-out
FM loss comparisons and raw/generated sample panels against the existing large model.

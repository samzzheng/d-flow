# Starting the DLR search at the displacement cap

Two completed 250², noiseless, five-Euler-step continuations compare a cold L-BFGS reset against the same reset with subsequent searches beginning at the existing displacement cap. The approximate gradient, rank settings, previous-V seed, projection, Armijo coefficient, halving budget and original total1000-attempt limit are unchanged. The first-attempt-only scaling remains intact; these continuations start at the parent attempt counter.

| Parent endpoint | Parent image error | Reset DLR | Cap-started DLR | Additional attempts per arm |
|---|---:|---:|---:|---:|
| MRI sample2, adaptive32–128 | 99.7683% | 99.7683% | 99.7683% | 2 |
| CT sample0, conditioned, rank32 | 37.1184% | 37.1184% | 37.1184% | 2 |

Both alternatives reject both searches and preserve their parent sources. There is no recovery. The reset direction norms were about2.085 and0.00497, versus the full displacement cap2.5. The earlier rank1024 MRI1 endpoint is a different source where a full-cap proposal did decrease the objective; this negative transfer result does not contradict that fixed-source observation or establish that the cap policy always fails. The preserved-state FP64 CT200 control remains separately queued.

A small CPU squared-objective check verifies that the cap policy preserves the first-attempt behavior and can find a useful later step outside the original halving interval. That mechanism check is not reconstruction evidence. Original-array verification checks pairing, unchanged mathematical configuration except the selected policy, monotone objectives, caps and complex residual metrics. Each arm took less than ten seconds on shared GPU1. Production defaults are unchanged.

[Verification](cap_started_verification.json) · [MRI panel](cap_started_mri2/comparison.png) · [CT panel](cap_started_conditioned_ct0/comparison.png).

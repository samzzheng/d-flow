# Amplitude-collapse diagnostic

CPU evaluation of the saved five-step initial images, original models, same noiseless observations. This is an image-space diagnostic, not a claim that every scaled image lies in the generator's range.

For a fixed prediction x and real scalar c, the linear-operator objective is

\[\Phi(cx)=\tfrac12c^2\|Ax\|^2-c\operatorname{Re}\langle Ax,y\rangle+\tfrac12\|y\|^2.\]

Its unconstrained amplitude minimizer is \(c_* = \operatorname{Re}\langle Ax,y\rangle/\|Ax\|^2\). If measurement overlap is small, reducing amplitude can lower the objective even when the target is nonzero. That provides a plausible route toward nearly blank predictions from misplaced shapes. It does not prove that the latent objective has a local minimum at zero or that this causes every failure.

| Case | Measurement cosine | Best scalar amplitude | Zero improves objective? |
| --- | ---: | ---: | --- |
| r128_shapes_ct10/sample_000 | 0.4967 | 6.7692 | False |
| r128_shapes_mri4/sample_002 | 0.0060 | 0.0074 | True |
| r160_shapes_ct10/sample_000 | 0.5022 | 8.8316 | False |
| r200_shapes_ct10/sample_001 | 0.1387 | 0.2620 | True |
| r250_shapes_mri4/sample_000 | 0.0391 | 0.0887 | True |
| r64_shapes_ct10/sample_000 | 0.2592 | 0.3421 | True |
| r64_shapes_mri4/sample_001 | -0.0037 | -0.0086 | True |
| r96_shapes_mri4/sample_001 | 0.0043 | 0.0036 | True |

Six of eight original starts favor zero over their initial prediction. The 128/160 CT starts are already nearly blank and do not satisfy this comparison; their failures require additional explanation. Successful cases can also start with low measurement overlap, so this diagnostic identifies an incentive to shrink, not a sufficient predictor of failure.

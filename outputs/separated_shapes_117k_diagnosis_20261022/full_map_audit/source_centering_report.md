# Source-mean counterfactual diagnostic

Several failed reconstructions had a large negative mean in the optimized source. This check removes 0%,25%,50%,100% of that mean at fixed saved sources, then reevaluates the unchanged five-step generator. No optimizer, projection constraint or regularizer is introduced. All inputs are noiseless.

| Saved reconstruction | Method | Original source mean | Original image error | Fully centered image error |
| --- | --- | ---: | ---: | ---: |
| initialization_controls/measurement/r250_shapes_ct10/sample_000 | D-Flow | -0.06773 | 10.192% | 34.972% |
| initialization_controls/measurement/r250_shapes_ct10/sample_000 | P-Flow | -0.16095 | 78.841% | 158.247% |
| initialization_controls/measurement/r250_shapes_ct10/sample_000 | DLR D-Flow | -0.20011 | 37.118% | 159.631% |
| mri_initialization_controls/measurement/r250_shapes_mri4/sample_001 | D-Flow | -0.00376 | 1.454% | 1.535% |
| mri_initialization_controls/measurement/r250_shapes_mri4/sample_001 | P-Flow | -0.02203 | 2.639% | 3.363% |
| mri_initialization_controls/measurement/r250_shapes_mri4/sample_001 | DLR D-Flow | -0.14193 | 12.306% | 15.418% |
| task4_ct0/measurement/r250_shapes_ct10/sample_000 | DLR D-Flow | -0.10177 | 26.097% | 83.178% |
| task_terminal_mri1/measurement/r250_shapes_mri4/sample_001 | DLR D-Flow | -0.00471 | 1.607% | 1.715% |

Every partial and full centering change worsens both image error and measurement fit in these eight selected reconstructions. This rejects direct post-hoc centering as a correction here. It does not prove that constraining an entire optimization trajectory could never help; that different experiment is not warranted by this check alone.

[Raw metrics and source hashes](source_centering.json)

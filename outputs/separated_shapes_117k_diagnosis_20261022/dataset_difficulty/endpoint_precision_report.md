# Precision sensitivity at failed and successful sources

Compare native gradients at identical source values using the same model weights, observations and MRI masks. All scalar objective reductions use FP64 and TF32 is disabled. FP64 model evaluation is the reference; its slopes at the six failed sources were independently checked by central differences. No DLR truncation is involved in these gradient comparisons.

| Endpoint | Sample | FP32 gradient difference | FP32 cosine | FP64 normalization only | FP64 Euler state only | Both |
|---|---:|---:|---:|---:|---:|---:|
| D-Flow failed | 2 | 5.708% | 1.000000 | 6.090% | 10.349% | 2.311% |
| P-Flow successful | 2 | 0.071% | 1.000000 | 0.009% | 0.025% | 0.011% |
| D-Flow failed | 20 | 71.335% | 0.984664 | 41.379% | 130.452% | 95.345% |
| P-Flow successful | 20 | 0.019% | 1.000000 | 0.003% | 0.025% | 0.003% |
| D-Flow failed | 0 | 309.760% | -0.266768 | 199.372% | 117.387% | 139.986% |
| P-Flow successful | 0 | 0.120% | 0.999999 | 0.143% | 0.129% | 0.198% |
| D-Flow failed | 14 | 58.022% | 0.997726 | 73.153% | 7.367% | 10.675% |
| P-Flow successful | 14 | 0.047% | 1.000000 | 0.045% | 0.027% | 0.026% |
| DLR failed | 0 | 74.542% | 0.998772 | 102.741% | 104.924% | 46.676% |
| DLR failed | 1 | 56.578% | 0.891201 | 102.074% | 132.261% | 64.015% |
| P-Flow successful | 1 | 0.032% | 1.000000 | 0.016% | 0.019% | 0.009% |

All difference columns are gradient differences relative to the full-FP64 gradient norm. Normalization-only computes each GroupNorm in FP64 then casts its output back to FP32. Euler-state variants retain FP64 accumulated states while evaluating the model in FP32, and also use FP64 MRI FFTs. These mixed modes change rounding throughout the map; they are not isolated backward-only changes.

At the six selected failed endpoints, FP32 gradient differences span 5.708%–309.760%; at five distinct successful endpoints they span 0.019%–0.120%. D-Flow sample 0 even has negative FP32/FP64 gradient cosine. These are selected endpoint diagnostics, not a failure predictor calibrated on an independent population.

The cheaper mixed-precision options are inconsistent. For corrected DLR sample 0, normalization-only and Euler-state-only each reverse the gradient alignment relative to FP64; combining them improves the magnitude error but leaves about 47%. For corrected DLR sample 1, all mixed variants are worse than ordinary FP32. Do not promote these partial-precision changes.

Small objective or image differences do not guarantee accurate derivatives. The observed numerical sensitivity is separate from DLR projection error: even the exact native backpropagated gradient can be unreliable relative to the higher-precision reference in these regions. Higher rank cannot repair this arithmetic effect.

The next paired control starts both FP32 and FP64 DLR from the original cached Gaussian source, retaining rank 32, five Euler steps, analytic alpha, previous-V reuse, and the original LBFGS/geometric Armijo budget. This tests whether higher precision avoids entering the failed region, rather than only attempting to recover after collapse. Initial native/backend agreement is checked before each run. No default is changed.

[Full measurements](endpoint_mixed_precision.json) · [Local geometry and derivative checks](source_bridge_report.md)

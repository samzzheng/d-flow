# Five-step Euler pilot

20% relative measurement noise; CT10 and Cartesian MRI4; 30 single-circle test samples at each resolution. Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed line searches stop a method. A step limit does not certify convergence. DLR uses rank8, analytical alpha=1-t, no trace probes/enrichment and previous V.

Final saved checkpoints, not retrospective best-image selection. Five-step Euler changes the generator being optimized; more optimizer steps do not remove its time-discretization error.

| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |
| --- | --- | --- | --- | --- | --- | --- |

Terminal method runs: 0/1080. Running rows are provisional.

Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, including image/residual panels and error versus optimizer attempts.

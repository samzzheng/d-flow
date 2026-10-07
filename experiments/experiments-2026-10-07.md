# Experiments — 2026-10-07

Times are local America/New_York; durations in HH:MM.

| Time | Experiment | Motivation | What Tried / Evidence | Correctness | Timing | Merge decision |
| --- | --- | --- | --- | --- | --- | --- |
| 00:50 | Shared D-Flow / P-Flow / DLR D-Flow figure validation | Comparing multiple measurement operators requires consistent panels, explicit clean/noisy references, and complex-valued error calculations for MRI. | Added `scripts/reconstruction_figures.py` with GT / D-Flow / P-Flow / DLR D-Flow columns and four rows: images, clean-reference measurements, absolute image error, and observed measurement residual. Rendered identical-GT synthetic inputs to `outputs/reconstruction_figure_validation_2026_10_07/layout.{png,pdf}`; these are layout checks, not reconstruction results. | CPU assertions passed for common row color limits, zero image/clean-measurement error, nonzero GT residual against noise, missing-method labels, undefined zero-reference ratios, and mismatched shapes. A 90-degree complex phase change correctly reports 141.42% error despite identical magnitudes. PNG visually inspected; Python compilation passed. | 00:00 (under one minute); one CPU thread; no GPU. | Adopt the shared plotter for new comparisons. No algorithm, training, or historical benchmark changes. |

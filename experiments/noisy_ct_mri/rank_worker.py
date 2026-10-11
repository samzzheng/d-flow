"""Run the existing reconstruction worker with only its DLR method enabled.

The imported plotter retains the standard four columns and marks missing
D/P controls unavailable. No objective, optimizer, or algorithm is replaced.
"""
from experiments.ct_mri_sweep import run

if __name__ == "__main__":
    run.METHODS = ("DLR D-Flow",)
    run.main()

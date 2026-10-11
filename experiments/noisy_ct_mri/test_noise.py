"""Pairing, noise semantics, rank selection and memory admission invariants."""
from __future__ import annotations

from pathlib import Path
import unittest

import torch
import matplotlib.pyplot as plt

from experiments.ct_mri_sweep.operators import CT, MRI, noisy_measurement
from experiments.ct_mri_sweep.queue import Task
from experiments.ct_mri_sweep.run import Job
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison
from .queue import Work, choose
from .setup import settings_for


class NoiseStudyTests(unittest.TestCase):
    def test_pairing_and_one_percent_noise(self) -> None:
        torch.set_num_threads(2)
        for problem in ("ct", "mri"):
            cases = []
            for rank in (2, 4, 8, 12, 16, None):
                settings = settings_for(rank)
                seed = settings["seed"] + 10000 * 64 + 100 * 3
                operator = CT(64, 10, torch.device("cpu")) if problem == "ct" else MRI(64, 4, seed + 10000000, torch.device("cpu"))
                truth = torch.ones(1, 1, 64, 64)
                initial = torch.randn(truth.shape, generator=torch.Generator().manual_seed(seed))
                clean = operator(truth)
                mask = getattr(operator, "mask", None)
                observed = noisy_measurement(clean, settings["noise_relative"], seed + 20000000, mask)
                self.assertAlmostEqual(float((observed - clean).norm() / clean.norm()), .01, places=6)
                if mask is not None:
                    self.assertEqual(int(observed[~mask].count_nonzero()), 0)
                cases.append((initial, clean, observed))
            for case in cases[1:]:
                for left, right in zip(cases[0], case):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            fig = plot_reconstruction_comparison(truth, clean, {}, observation=observed, residual_vmax=RESIDUAL_VMAX[problem])
            try:
                expected = 100 * float((clean - observed).norm() / observed.norm())
                self.assertIn(f"{expected:.2f}%", fig.axes[15].get_title())
                self.assertEqual(fig.axes[15].images[0].get_clim(), (0, RESIDUAL_VMAX[problem]))
            finally:
                plt.close(fig)

    def test_reservations_include_legacy_and_future_growth(self) -> None:
        settings = settings_for()
        old = Work(Path("old/manifest.json"), Task(Job(250, 3, "ct", 10), 25, 30), settings, False)
        large = Work(Path("new/manifest.json"), Task(Job(250, 3, "mri", 4), 0, 5), settings, False)
        small = Work(Path("rank/manifest.json"), Task(Job(64, 4, "ct", 10), 0, 5), settings_for(16), True)
        self.assertIs(choose([large, small], [old], 49140, 1000), small)
        self.assertIsNone(choose([large], [old], 49140, 1000))
        self.assertIsNone(choose([small], [], 49140, 47000))


if __name__ == "__main__":
    unittest.main()

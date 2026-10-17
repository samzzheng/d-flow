"""Check that unfinished ranks cannot change the comparison population."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.ct_mri_sweep.run import write_json
from .sweep_report import report


class ReportTests(unittest.TestCase):
    def test_only_common_completed_samples_enter_medians(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = {str(rank): str(root / str(rank)) for rank in (8, 16, 32, 64, 128)}
            write_json(root / "plan.json", {"rank_roots": roots, "resolutions": [64, 96, 128, 160, 200, 250], "samples": list(range(5))})
            for rank, path in roots.items():
                for sample, error in ((0, .02), (1, 9.)):
                    status = "step_limit" if sample == 0 or rank == "8" else "running"
                    write_json(Path(path) / "production/r64_shapes_ct10" / f"sample_{sample:03d}/dlr_d_flow.json",
                               {"status": status, "history": [{"step": 20, "image_relative_l2": error,
                                                               "measurement_relative_l2_noisy": .01}]})
            report(root)
            rows = json.loads((root / "summary.json").read_text())["rows"]
            self.assertEqual(rows[0]["common_samples"], [0])
            self.assertEqual(set(rows[0]["median_image_percent"].values()), {2.})
            self.assertEqual(rows[0]["completed_by_rank"]["8"], 2)
            self.assertEqual(rows[1]["common_samples"], [])


if __name__ == "__main__":
    unittest.main()

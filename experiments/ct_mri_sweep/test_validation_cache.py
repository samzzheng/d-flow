"""Exercise the worker's calibration cache using CPU linear dynamics."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from dlr_dflow import dlr_dflow as core
from . import run
from .test_sweep import LinearVelocity


class ValidationCacheTests(unittest.TestCase):
    def test_frozen_configuration_reuses_calibration_and_invalidates_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            settings = json.loads(Path(__file__).with_name("config_fp32.json").read_text())
            settings.update(resolutions=[8], circles=[1], samples=2, flow_steps=3,
                ct_angles=[2], mri_accelerations=[], run_template="model", dataset_template="data.pt")
            settings["dlr"].update(rank=2, initialization_oversampling=0)
            model_path = root / "model"
            (model_path / "checkpoints").mkdir(parents=True)
            (model_path / "config.json").write_text(json.dumps({"dataset_metadata": {}}))
            checkpoint = model_path / "checkpoints/epoch_0200.pt"
            checkpoint.write_bytes(b"model-a")
            torch.save({"metadata": {}, "test": torch.ones(2, 1, 8, 8)}, root / "data.pt")
            manifest_path = root / "manifest.json"
            manifest = {"settings": settings, "source_sha256": {}, "protocol": {"smoke_only": False}}
            manifest_path.write_text(json.dumps(manifest))
            stack.enter_context(patch.object(run, "ROOT", root))
            stack.enter_context(patch.object(run, "model_ready", return_value=True))
            stack.enter_context(patch.object(run, "load_flow_model", return_value=(LinearVelocity(), {})))
            stack.enter_context(patch.object(run, "GraphedAProducts", return_value=None))
            stack.enter_context(patch.object(run, "plot_reconstruction_comparison", return_value=MagicMock()))
            stack.enter_context(patch.object(run.plt, "close"))
            stack.enter_context(patch.object(run, "solve", return_value={"image": None, "measurement": None, "history": []}))
            stack.enter_context(patch.object(torch, "set_num_interop_threads"))
            for name in ("set_device", "set_per_process_memory_fraction"):
                stack.enter_context(patch.object(torch.cuda, name))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            gpu_name = stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="test GPU"))
            for flag in ("HAND_JVP", "HAND_VJP", "SHARE_LINEARIZATION"):
                stack.enter_context(patch.object(core, flag, False))
            evolve = stack.enter_context(patch.object(run, "sensitivity_factors", wraps=run.sensitivity_factors))
            cache_path = root / "r8_n1_ct2/backend_validation_cache.json"
            for sample, mutation, expected_reuse in ((0, "none", False), (1, "none", True),
                    (1, "checkpoint", False), (1, "gpu", False), (1, "manifest", False),
                    (1, "corrupt", False), (1, "failed", False)):
                if mutation == "checkpoint":
                    checkpoint.write_bytes(b"model-b")
                elif mutation == "gpu":
                    gpu_name.return_value = "different GPU"
                elif mutation == "manifest":
                    manifest["settings"]["dlr"]["rank"] = 3
                    manifest_path.write_text(json.dumps(manifest))
                elif mutation == "corrupt":
                    cache_path.write_text("null")
                elif mutation == "failed":
                    cached = json.loads(cache_path.read_text())
                    cached["passed"] = False
                    cache_path.write_text(json.dumps(cached))
                before = evolve.call_count
                with patch("sys.argv", ["worker", "--manifest", str(manifest_path), "--job", "r8_n1_ct2",
                        "--device", "cpu", "--sample-start", str(sample), "--sample-stop", str(sample + 1)]):
                    run.main()
                self.assertEqual(evolve.call_count - before, 0 if expected_reuse else 4)
                record = json.loads((cache_path.parent / f"backend_validation_{sample:03d}.json").read_text())
                self.assertEqual(record["reused"], expected_reuse)
                if expected_reuse:
                    self.assertEqual(record["reference_sample"], 0)
            cache_path.unlink()
            with patch.object(run, "approx_source_gradient", return_value=torch.full((64,), float("nan"))), \
                    patch("sys.argv", ["worker", "--manifest", str(manifest_path), "--job", "r8_n1_ct2",
                                       "--device", "cpu", "--sample-start", "1", "--sample-stop", "2"]):
                with self.assertRaisesRegex(RuntimeError, "backend validation failed"):
                    run.main()
            self.assertFalse(cache_path.exists())
            self.assertFalse(json.loads((cache_path.parent / "backend_validation_001.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()

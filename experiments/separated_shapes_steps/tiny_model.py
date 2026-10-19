"""Test the existing 22k separated-shapes model under the unchanged inverse setup."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.queue import source_hashes


def main() -> None:
    root = Path("outputs/separated_shapes_steps_20261019")
    path = root / "tiny_model_steps_5_rank_32/production/manifest.json"
    if not path.exists():
        manifest = deepcopy(json.loads((root / "steps_10_rank_32/production/manifest.json").read_text()))
        manifest["created_at"] = datetime.now().astimezone().isoformat()
        settings = manifest["settings"]
        settings.update(flow_steps=5, resolutions=[64], epoch=1000,
                        run_template="outputs/fm_unet/separated-shapes-64-noattn-ch4-e1000")
        manifest["source_sha256"] = source_hashes()
        manifest["source_sha256"][str(Path(__file__).relative_to(run.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["protocol"]["purpose"] = "Existing 22k model, five Euler steps and rank 32; original history-100 LBFGS and source cap"
        manifest["protocol"]["selection"] = "64 MRI sample 1: selected failure case; model and training duration differ from the larger-model baseline"
        run.write_json(path, manifest)
    sys.argv = [sys.argv[0], "--manifest", str(path), "--job", "r64_shapes_mri4",
                "--sample-start", "1", "--sample-stop", "2", "--device", "cuda:0", "--memory-fraction", ".10"]
    run.main()


if __name__ == "__main__":
    main()

"""Train the 64x64 mixed-shape model with the existing circle FM settings."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/fm_unet/separated-shapes-64-noattn-ch16-b8-e200"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    data_path = root / "data/separated-shapes-64.pt"
    if not data_path.exists():
        parser.error(f"Missing dataset: {data_path}")
    dataset = torch.load(data_path, weights_only=True, mmap=True, map_location="cpu")
    if dataset.get("metadata", {}).get("supersample") != 8:
        parser.error("Separated-shape training requires the regenerated 8x8 antialiased dataset")
    del dataset
    args.out_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    command = [
        "train_fm_unet.py", "--data", str(data_path), "--out-dir", str(args.out_dir),
        "--device", args.device, "--epochs", "200", "--batch", "512", "--micro-batch", "64",
        "--lr", "0.0002", "--wd", "0.0001", "--ema-decay", "0.999", "--t-alpha", "1",
        "--sample-steps", "100", "--ch", "16", "--ch-mul", "1,2,2,2",
        "--att-channels", "0,0,0,0", "--groups", "4", "--no-mid-attn",
        "--num-workers", "2", "--threads", "4", "--compile",
    ]
    with data_path.open("rb") as dataset_file:
        dataset_sha256 = hashlib.file_digest(dataset_file, "sha256").hexdigest()
    launch = {
        "started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(),
        "seed": args.seed, "device": args.device, "command": command,
        "dataset_sha256": dataset_sha256,
        "trainer_sha256": hashlib.sha256((root / command[0]).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
    }
    (args.out_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch, indent=2), flush=True)
    sys.argv = command
    runpy.run_path(str(root / command[0]), run_name="__main__")


if __name__ == "__main__":
    main()

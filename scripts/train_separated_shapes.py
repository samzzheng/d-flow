"""Train mixed-shape models with the circle FM settings for each resolution."""

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


TRAINING_LAYOUT = {
    64: (4, 64), 96: (5, 64), 128: (5, 32),
    160: (5, 32), 200: (6, 16), 250: (6, 16),
}
DEEPER_MICROBATCHES = {64: 512, 96: 128, 128: 128, 160: 64, 200: 32, 250: 32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--resolution", type=int, choices=TRAINING_LAYOUT, default=64)
    parser.add_argument("--out-dir", type=Path)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--small-model", action="store_true", help="22,037-parameter capacity pilot")
    model_group.add_argument("--deeper-model", action="store_true", help="52,593 parameters; four levels and 32 time features")
    model_group.add_argument("--model-100k", action="store_true", help="102,793 parameters; four levels and 32 time features")
    parser.add_argument("--comparison-baseline", type=Path,
                        help="Override the reference run for the final paired evaluation")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--micro-batch", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    data_path = root / f"data/separated-shapes-{args.resolution}.pt"
    multiscale_model = args.deeper_model or args.model_100k
    compact_model = args.small_model or multiscale_model
    compact_multipliers = [1, 2, 4, 5] if args.model_100k else [1, 2, 3, 3] if args.deeper_model else [1, 2, 2]
    epochs = args.epochs or (500 if multiscale_model else 1000 if args.small_model else 200)
    if args.small_model and args.resolution != 64:
        parser.error("The small-model pilot currently targets resolution 64")
    if args.micro_batch is not None and not 1 <= args.micro_batch <= 512:
        parser.error("--micro-batch must be between1 and512")
    name = f"separated-shapes-{args.resolution}-noattn-ch4-e{epochs}" if args.small_model else f"separated-shapes-{args.resolution}-noattn-ch16-b8-e{epochs}"
    if multiscale_model:
        size = "100k" if args.model_100k else "53k"
        name = f"separated-shapes-{args.resolution}-{size}-time32-e{epochs}"
    out_dir = args.out_dir or Path("outputs/fm_unet") / name
    if not data_path.exists():
        parser.error(f"Missing dataset: {data_path}")
    dataset = torch.load(data_path, weights_only=True, mmap=True, map_location="cpu")
    if dataset.get("metadata", {}).get("supersample") != 8:
        parser.error("Separated-shape training requires the regenerated 8x8 antialiased dataset")
    for name, count in {"train": 80_000, "val": 10_000, "test": 10_000}.items():
        if tuple(dataset[name].shape) != (count, 1, args.resolution, args.resolution):
            parser.error(f"Unexpected {name} split shape: {dataset[name].shape}")
    del dataset
    checkpoint = out_dir / "checkpoints/training_state.pt"
    resume_epoch = 0
    if compact_model and checkpoint.exists():
        resume_epoch = int(torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)["epoch"])
        config = json.loads((out_dir / "config.json").read_text())
        if config["epochs"] != epochs or Path(config["data"]).resolve() != data_path.resolve():
            parser.error("Resume settings do not match the existing run")
        if (config["arch"]["ch"] != 4
                or config["arch"]["ch_mul"] != compact_multipliers
                or config["arch"].get("time_embed_dim", 4) != (32 if multiscale_model else 4)):
            parser.error("Resume architecture does not match the existing run")
        if resume_epoch >= epochs:
            parser.error("Training has already reached the requested epoch count")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    levels, micro_batch = TRAINING_LAYOUT[args.resolution]
    if compact_model:
        levels, micro_batch = len(compact_multipliers), DEEPER_MICROBATCHES[args.resolution] if multiscale_model else 512
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.micro_batch is not None:
        micro_batch = args.micro_batch
    command = [
        "train_fm_unet.py", "--data", str(data_path), "--out-dir", str(out_dir),
        "--device", args.device, "--epochs", str(epochs), "--batch", "512", "--micro-batch", str(micro_batch),
        "--lr", "0.0002", "--wd", "0.0001", "--ema-decay", "0.999", "--t-alpha", "1",
        "--sample-steps", "100", "--ch", "4" if compact_model else "16", "--ch-mul", ",".join(map(str, compact_multipliers if compact_model else [1] + [2] * (levels - 1))),
        "--att-channels", ",".join(["0"] * levels), "--groups", "2" if compact_model else "4", "--no-mid-attn",
        "--num-workers", "0" if compact_model else "2", "--threads", "4", "--no-compile" if compact_model else "--compile",
    ]
    if compact_model:
        command.extend(["--gpu-resident", "--channels-last", "--fused-adam", "--save-training-state"])
    if multiscale_model:
        command.extend(["--time-embed-dim", "32", "--graph-microbatches"])
    if resume_epoch:
        command.extend(["--resume-training-state", str(checkpoint), "--start-epoch", str(resume_epoch)])
    with data_path.open("rb") as dataset_file:
        dataset_sha256 = hashlib.file_digest(dataset_file, "sha256").hexdigest()
    launch = {
        "started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(),
        "seed": args.seed, "device": args.device, "command": command,
        "dataset_sha256": dataset_sha256,
        "trainer_sha256": hashlib.sha256((root / command[0]).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        "comparison_baseline": str(args.comparison_baseline) if args.comparison_baseline is not None else None,
    }
    (out_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch, indent=2), flush=True)
    sys.argv = command
    runpy.run_path(str(root / command[0]), run_name="__main__")
    if compact_model and (out_dir / "run_summary.json").exists():
        from scripts.compare_small_fm_models import compare_models
        compare_models(args.comparison_baseline or root / f"outputs/fm_unet/separated-shapes-{args.resolution}-noattn-ch16-b8-e200",
                       out_dir, args.device, micro_batch=min(128, micro_batch))


if __name__ == "__main__":
    main()

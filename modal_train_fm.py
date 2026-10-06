"""Modal runner for the attention-free ch16 "b8" (≈8x8 bottleneck) FM UNet sweep.

Each (resolution, n_circles) model trains in its own container on the shared
d-flow-data Volume under outputs/fm_unet/circles-<res>-noattn-ch16-b8-e200/n<n>.
A run resumes from its latest EMA checkpoint if its container is preempted or
times out, and the sweep is idempotent: rerunning it skips finished models.

    modal run modal_train_fm.py::upload_n1                 # once: local n1 datasets
    modal run modal_train_fm.py::probe --gpus L4,A10G      # epoch cost per GPU type
    modal run --detach modal_train_fm.py::sweep            # all 24 models
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent
GPU_TYPE = "A10G"  # probe: 244 s/epoch at 250px vs 360 on L4; cheaper per model
EPOCHS = 200
RESOLUTIONS = (64, 96, 128, 160, 200, 250)
CIRCLES = (1, 2, 3, 4)
SPLITS = {"train": 48_000, "val": 1_000, "test": 1_000}
VOL = "/vol"
RUN_TEMPLATE = "outputs/fm_unet/circles-{res}-noattn-ch16-b8-e200/n{n}"

# res -> (ch_mul, micro_batch), as in scripts/run_noattn_resolution_sweep.sh
CONFIG = {
    64: ("1,2,2,2", 64),
    96: ("1,2,2,2,2", 64),
    128: ("1,2,2,2,2", 32),
    160: ("1,2,2,2,2", 32),
    200: ("1,2,2,2,2,2", 16),
    250: ("1,2,2,2,2,2", 16),
}

app = modal.App("d-flow-fm-training")
volume = modal.Volume.from_name("d-flow-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("fonts-texgyre")
    .pip_install(
        "torch==2.11.0", "numpy", "matplotlib", "cmocean", "tqdm",
        extra_options="--index-url https://download.pytorch.org/whl/cu129 "
                      "--extra-index-url https://pypi.org/simple",
    )
    .add_local_file(REPO_ROOT / "train_fm_unet.py", "/repo/train_fm_unet.py")
    .add_local_dir(REPO_ROOT / "dataset", "/repo/dataset", ignore=["__pycache__"])
    .add_local_dir(REPO_ROOT / "models", "/repo/models", ignore=["__pycache__"])
    .add_local_dir(REPO_ROOT / "sampler", "/repo/sampler", ignore=["__pycache__"])
)


def train_args(res: int, n: int, data: str, out_dir: str, epochs: int) -> list[str]:
    ch_mul, micro = CONFIG[res]
    levels = len(ch_mul.split(","))
    return [
        sys.executable, "train_fm_unet.py",
        "--data", data, "--out-dir", out_dir,
        "--epochs", str(epochs), "--batch", "512", "--micro-batch", str(micro),
        "--device", "cuda:0", "--ch", "16", "--ch-mul", ch_mul,
        "--att-channels", ",".join(["0"] * levels), "--no-mid-attn", "--groups", "4",
    ]


@app.function(image=image, cpu=2, memory=16384, timeout=2 * 3600, volumes={VOL: volume})
def make_dataset(res: int, n: int) -> str:
    import torch
    sys.path.insert(0, "/repo")
    from dataset.circles_64 import generate_fixed_n_splits

    path = Path(f"{VOL}/data/circles-{res}-n{n}.pt")
    if path.exists():
        return f"{path.name}: exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Seed convention of dataset.circles_64.generate_all_resolutions (n1 files match it).
    splits = generate_fixed_n_splits(n, SPLITS, im_size=res, seed=42 + res * 10 + n * 100)
    tmp = path.with_suffix(".tmp")
    torch.save(splits, tmp)
    tmp.rename(path)
    volume.commit()
    return f"{path.name}: generated"


@app.function(
    image=image, gpu=GPU_TYPE, cpu=4, memory=16384, timeout=24 * 3600,
    volumes={VOL: volume}, retries=modal.Retries(max_retries=3, initial_delay=30.0),
)
def train(res: int, n: int) -> str:
    run_dir = Path(VOL) / RUN_TEMPLATE.format(res=res, n=n)
    if (run_dir / "run_summary.json").exists():
        return f"{res}/n{n}: already complete"
    cmd = train_args(res, n, f"{VOL}/data/circles-{res}-n{n}.pt", str(run_dir), EPOCHS)
    checkpoints = sorted(
        (int(m.group(1)), p)
        for p in (run_dir / "checkpoints").glob("epoch_*.pt")
        if (m := re.fullmatch(r"epoch_(\d+)\.pt", p.name)) and int(m.group(1)) < EPOCHS
    )
    if checkpoints and (run_dir / "loss_history.json").exists():
        start, ckpt = checkpoints[-1]
        cmd += ["--resume-checkpoint", str(ckpt), "--start-epoch", str(start)]
    run_dir.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    with (run_dir / "train.log").open("a") as log:
        proc = subprocess.Popen(
            cmd, cwd="/repo", stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "TQDM_MININTERVAL": "60"},
        )
        while proc.poll() is None:
            time.sleep(300)
            volume.commit()
    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"{res}/n{n}: train_fm_unet.py exited {proc.returncode}")
    return f"{res}/n{n}: complete"


@app.function(image=image, gpu=GPU_TYPE, cpu=4, memory=16384, timeout=3600, volumes={VOL: volume})
def probe_epoch(res: int) -> float:
    """Seconds per training epoch (second epoch, after compile) for circles-<res>-n1."""
    cmd = train_args(res, 1, f"{VOL}/data/circles-{res}-n1.pt", "/tmp/probe", 2)
    out = subprocess.run(
        cmd, cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ).stdout
    times = re.findall(r"Epoch\s+2: 100%.*?\[(\d+):(\d+)<", out.replace("\r", "\n"))
    if not times:
        raise RuntimeError(out[-3000:])
    minutes, seconds = map(int, times[-1])
    return 60.0 * minutes + seconds


@app.local_entrypoint()
def upload_n1():
    with volume.batch_upload(force=False) as batch:
        for res in RESOLUTIONS:
            if res in (64, 250):
                continue  # already on the Volume from modal_app.py::seed_volume
            batch.put_file(REPO_ROOT / f"data/circles-{res}-n1.pt", f"data/circles-{res}-n1.pt")
    print("uploaded n1 datasets")


@app.local_entrypoint()
def probe(gpus: str = "L4,A10G", res: int = 250):
    for gpu in gpus.split(","):
        seconds = probe_epoch.with_options(gpu=gpu).remote(res)
        print(f"{gpu}: {seconds:.0f} s/epoch at {res}px -> {seconds * EPOCHS / 3600:.1f} h per model")


@app.local_entrypoint()
def datasets():
    for message in make_dataset.starmap(
        [(res, n) for res in RESOLUTIONS for n in CIRCLES if n > 1]
    ):
        print(message, flush=True)


@app.local_entrypoint()
def sweep(max_rounds: int = 4):
    datasets()
    pending = [(res, n) for res in RESOLUTIONS for n in CIRCLES]
    for round_ in range(max_rounds):
        calls = {train.spawn(res, n): (res, n) for res, n in pending}
        print(f"round {round_}: launched {len(calls)} training jobs on {GPU_TYPE}", flush=True)
        pending = []
        for call, (res, n) in calls.items():
            try:
                print(call.get(), flush=True)
            except Exception as exc:  # noqa: BLE001 -- timeouts/preemptions resume next round
                print(f"{res}/n{n}: incomplete ({exc!r}); resuming next round", flush=True)
                pending.append((res, n))
                continue
            dest = REPO_ROOT / Path(RUN_TEMPLATE.format(res=res, n=n)).parent
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["modal", "volume", "get", "--force", "d-flow-data",
                 RUN_TEMPLATE.format(res=res, n=n), str(dest)],
                check=False,
            )
        if not pending:
            print("all models complete", flush=True)
            return
    print(f"still incomplete after {max_rounds} rounds: {pending}", flush=True)

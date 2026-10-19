"""Modal cloud-GPU runner for this repo's D-Flow / DLR-D-Flow benchmarks.

Lasting setup: one image (repo code baked in, rebuilt automatically when
scripts/dlr_dflow change), one persistent Volume holding data/ and the
outputs/fm_unet/ checkpoints (uploaded once, reused by every future run), and
one Modal function per benchmark sample so a whole sweep fans out across as
many GPUs as Modal will give you in parallel, instead of serializing on one
local GPU.

One-time setup (after `modal setup` has written ~/.modal.toml):
    modal run modal_app.py::seed_volume          # uploads data/ + checkpoints once

Fan out the Euler-vs-RK4 rerun across N remaining samples in parallel:
    modal run modal_app.py::euler_rk4_sweep --samples 5,6,7,8,10,11,12,13,14,22,23,24,26,27,28,29

Fan out the D-Flow vs DLR-D-Flow noise benchmark:
    modal run modal_app.py::dflow_dlr_sweep --samples 0,1,2,...,29

Generic: run any benchmark_dflow_dlr_sparse_ct_circles_gpu.py invocation for one
sample as a cloud function (used internally by the sweeps above, or directly):
    modal run modal_app.py::run_one_sample --sample 5 --extra-args "--methods forward_euler rk4"

Results land in the "d-flow-outputs" Modal Volume under the same out-dir layout
the local scripts use, and are pulled back to outputs/modal_runs/<label>/ after
each sweep. GPU type/count are set once below (default: A10G, one GPU per
sample) -- change GPU_TYPE if you want more headroom or lower cost.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "d-flow-benchmarks"
GPU_TYPE = "A10G"          # 24GB; circles-250 DLR peaked at ~4.6GB in earlier runs
TIMEOUT_S = 6 * 60 * 60    # 6h: covers a slow 300-iteration RK4 run at 250px
REPO_ROOT = Path(__file__).resolve().parent

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("d-flow-data", create_if_missing=True)
output_volume = modal.Volume.from_name("d-flow-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy", "matplotlib", "cmocean", "scipy",
        extra_options="--index-url https://download.pytorch.org/whl/cu121 "
                      "--extra-index-url https://pypi.org/simple",
    )
    .add_local_dir(REPO_ROOT / "scripts", remote_path="/repo/scripts")
    .add_local_dir(REPO_ROOT / "dlr_dflow", remote_path="/repo/dlr_dflow")
    .add_local_dir(REPO_ROOT / "models", remote_path="/repo/models")
)

DATA_MOUNT = "/data"      # -> data_volume:  data/, outputs/fm_unet/
OUT_MOUNT = "/outputs"    # -> output_volume: results, written back here


# ---------------------------------------------------------------------------
# One-time (and re-runnable) upload of data/ and model checkpoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def seed_volume(
    include_checkpoints: str = "circles-250-noattn-ch16-b8,circles-64",
    resolutions: str = "64,250",
):
    """Upload data/circles-<res>-n1.pt and the listed model run dirs into the
    d-flow-data Volume. Safe to re-run: existing files are skipped by content
    hash. Run this once after `modal setup`, and again if you add a new
    resolution or trained model."""
    with modal.Volume.from_name("d-flow-data", create_if_missing=True).batch_upload(
        force=False
    ) as batch:
        for res in resolutions.split(","):
            local = REPO_ROOT / f"data/circles-{res}-n1.pt"
            if local.exists():
                print(f"uploading {local} -> data/circles-{res}-n1.pt")
                batch.put_file(local, f"data/circles-{res}-n1.pt")
            else:
                print(f"skip (not found locally): {local}")
        for run_dir in include_checkpoints.split(","):
            local_dir = REPO_ROOT / "outputs/fm_unet" / run_dir
            if local_dir.exists():
                print(f"uploading {local_dir} -> outputs/fm_unet/{run_dir}/")
                batch.put_directory(local_dir, f"outputs/fm_unet/{run_dir}")
            else:
                print(f"skip (not found locally): {local_dir}")
    print("done. This Volume now persists across every future Modal run.")


# ---------------------------------------------------------------------------
# Worker: one benchmark invocation, one GPU, one sample
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_S,
    volumes={DATA_MOUNT: data_volume, OUT_MOUNT: output_volume},
)
def run_one_sample(sample: int, out_dir: str, extra_args: list[str]) -> str:
    """Runs benchmark_dflow_dlr_sparse_ct_circles_gpu.py for exactly one
    sample inside a fresh container with a dedicated GPU. `out_dir` is a path
    relative to the output Volume; `extra_args` are appended verbatim (e.g.
    --methods, --lbfgs-line-search, --armijo-step-cap, --resolutions)."""
    import os
    # Modal reuses warm containers across calls, so this setup must be
    # idempotent: skip anything already linked by a previous call on the
    # same container instead of failing on FileExistsError.
    if not os.path.lexists("/repo/data"):
        os.symlink(f"{DATA_MOUNT}/data", "/repo/data", target_is_directory=True)
    os.makedirs("/repo/outputs/fm_unet", exist_ok=True)
    for entry in os.listdir(f"{DATA_MOUNT}/outputs/fm_unet"):
        link = f"/repo/outputs/fm_unet/{entry}"
        if not os.path.lexists(link):
            os.symlink(f"{DATA_MOUNT}/outputs/fm_unet/{entry}", link, target_is_directory=True)

    full_out = f"{OUT_MOUNT}/{out_dir}"
    os.makedirs(full_out, exist_ok=True)
    cmd = [
        sys.executable, "/repo/scripts/benchmark_dflow_dlr_sparse_ct_circles_gpu.py",
        "--device", "cuda:0", "--sample-offset", str(sample), "--samples", "1",
        "--out-dir", full_out,
        *extra_args,
    ]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd="/repo", check=False)
    output_volume.commit()
    status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    return f"sample {sample}: {status}"


def _pull_results(out_dir: str, local_dest: Path) -> None:
    """Download one out_dir subtree from the outputs Volume to a local path."""
    local_dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "--force", "d-flow-outputs", out_dir, str(local_dest)],
        check=True,
    )


def _fan_out(label: str, samples: list[int], extra_args: list[str], out_dir_template: str) -> None:
    """Launch every sample in parallel and pull each one's figures/results to
    local disk as soon as that sample finishes, instead of waiting for the
    whole batch -- so you can check progress sample by sample."""
    started = time.time()
    dest = REPO_ROOT / "outputs/modal_runs" / label
    calls = {
        run_one_sample.spawn(sample=s, out_dir=out_dir_template.format(sample=s), extra_args=extra_args): s
        for s in samples
    }
    print(f"launched {len(calls)} containers in parallel, each on its own {GPU_TYPE} GPU")
    pending = set(calls)
    while pending:
        for call in list(pending):
            sample = calls[call]
            try:
                result = call.get(timeout=0)     # 0 = poll, don't block
            except TimeoutError:
                continue                          # not finished yet, check again next pass
            except Exception as exc:  # noqa: BLE001 -- surface every failure, keep collecting the rest
                print(f"sample {sample}: FAILED ({exc})")
                pending.discard(call)
                continue
            pending.discard(call)
            print(result)
            _pull_results(out_dir_template.format(sample=sample), dest)
            print(f"sample {sample}: pulled to {dest}")
        if pending:
            time.sleep(15)
    print(f"all {len(calls)} samples finished and pulled in {time.time() - started:.0f}s wall clock")
    print(f"done: {dest}")


@app.local_entrypoint()
def euler_rk4_sweep(
    samples: str,
    resolution: int = 250,
    noise: float = 0.01,
    armijo_cap: float = 0.01,
    outer_steps: int = 300,
):
    """Fan the Euler-vs-RK4 rerun out across the given samples, one GPU each.
    `samples` is a comma-separated list, e.g. the 16 still missing locally:
      modal run modal_app.py::euler_rk4_sweep \
        --samples 5,6,7,8,10,11,12,13,14,22,23,24,26,27,28,29
    """
    sample_list = [int(s) for s in samples.split(",")]
    extra_args = [
        "--resolutions", str(resolution), "--methods", "forward_euler", "rk4",
        "--euler-flow-steps", "100", "--rk4-flow-steps", "100",
        "--noise-level", str(noise),
        "--lbfgs-line-search", "armijo", "--armijo-step-cap", str(armijo_cap),
        "--outer-steps", str(outer_steps),
    ]
    _fan_out("euler_rk4_250_noise001_armijo", sample_list, extra_args,
             "euler_rk4/sample_{sample}")


@app.local_entrypoint()
def dflow_dlr_sweep(
    samples: str,
    resolution: int = 250,
    noise: float = 0.01,
    armijo_cap: float = 0.01,
    outer_steps: int = 300,
    integrator: str = "ksl",
):
    """Fan the D-Flow vs DLR-D-Flow comparison out across the given samples."""
    sample_list = [int(s) for s in samples.split(",")]
    extra_args = [
        "--resolutions", str(resolution), "--methods", "forward_euler", "dlr",
        "--noise-level", str(noise), "--trace-probes", "4",
        "--lbfgs-line-search", "armijo", "--armijo-step-cap", str(armijo_cap),
        "--outer-steps", str(outer_steps), "--dlr-integrator", integrator,
    ]
    _fan_out(f"dflow_dlr_{integrator}", sample_list, extra_args, "dflow_dlr/sample_{sample}")


@app.local_entrypoint()
def status():
    """List what's currently in the outputs Volume (across all past sweeps)."""
    subprocess.run(["modal", "volume", "ls", "d-flow-outputs"], check=False)

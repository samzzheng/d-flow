"""Extend the paired high-rank study without changing its existing sketch columns."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import MRI
from experiments.separated_shapes_117k.bounded_products import BoundedGraphedProducts


class ProfiledEvaluator(run.Evaluator):
    profile_path: Path

    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        started = time.perf_counter()
        gradient = super().gradient(source, method)
        if self.counts["gradients"] == 1:
            torch.cuda.synchronize(source.device)
            profile = dict(first_refresh_seconds=time.perf_counter()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated(source.device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(source.device)/2**30,
                gradient_norm=float(gradient.norm()), retained_rank=self.previous_v.shape[1])
            run.write_json(self.profile_path, profile)
            print("Initial rank/memory check", profile, flush=True)
        return gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=5096)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=.52)
    parser.add_argument("--reconstruction-directory", default="reconstruction")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    interrupted = [False]

    def stop(signum: int, frame: object) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    baseline = Path("outputs/separated_shapes_117k_noiseless_20261021/production")
    case = "r250_shapes_mri4/sample_001"
    inputs = torch.load(baseline/case/"inputs.pt", weights_only=True, map_location=device)
    protocol = inputs["protocol"]
    settings = json.loads((baseline/"manifest.json").read_text())["settings"]
    settings["dlr"].update(rank=args.rank, adaptive_rank=False)
    settings["execution"].update(memory_fraction=args.memory_fraction, dlr_full_step_graphs=False,
        dlr_jvp_columns=20, dlr_vjp_columns=16, dlr_fused_norm="native", dlr_fused_primal="native")
    folder = Path(settings["run_template"].format(resolution=250))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    assert settings["flow_steps"] == 5 and torch.equal(inputs["clean"], inputs["observed"])
    assert settings["dlr"]["trace_probes"] == settings["dlr"]["enrichment_probes"] == 0
    initial = inputs["initial"]
    assert 2048 < args.rank < initial.numel()
    model, _ = load_flow_model(250, 0, 500, str(device), folder)
    info = protocol["operator"]
    operator = MRI(250, 4, info["mask_seed"], device, info["pattern"])
    assert torch.equal(inputs["mask"], operator.mask)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = tangent_unet.GN_ACTION_BACKEND = tangent_unet.GN_PRIMAL_BACKEND = "native"
    # Regenerating a wider row-major random matrix would change shared columns.
    # Preserve all 2056 columns used by the original32/1024/2048 study exactly.
    sketch = torch.randint(0, 2, (initial.numel(), 2056), device=device,
        generator=torch.Generator(device=device).manual_seed(protocol["dlr_seed"]), dtype=torch.int8).float()
    sketch.mul_(2).sub_(1)
    original_sketch_hash = hashlib.sha256(sketch.cpu().numpy().tobytes()).hexdigest()
    reference = json.loads((args.out/"mri1_initial.json").read_text())
    assert original_sketch_hash == reference["sketch_sha256"]
    extra = torch.randint(0, 2, (initial.numel(), args.rank+8-2056), device=device,
        generator=torch.Generator(device=device).manual_seed(protocol["dlr_seed"]+1), dtype=torch.int8).float()
    extra.mul_(2).sub_(1)
    sketch = torch.cat((sketch, extra), dim=1)
    del extra
    products = BoundedGraphedProducts(model, initial, cache_size=4, share_linearization=True,
        full_step_graphs=False, max_jvp_columns=20, max_vjp_columns=16)
    evaluator = ProfiledEvaluator(model, operator, inputs["observed"], settings, protocol["dlr_seed"], products)
    evaluator.previous_v = sketch
    path = args.out/args.reconstruction_directory/case/f"rank{args.rank}"
    path.mkdir(parents=True, exist_ok=True)
    evaluator.profile_path = path/"first_refresh_profile.json"
    experiment = dict(settings=settings, original_sketch_sha256=original_sketch_hash,
        extended_sketch_sha256=hashlib.sha256(sketch.cpu().numpy().tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note=f"Existing2056 random sketch columns preserved; append independently seeded Rademacher columns. Same original source/model/operator and optimizer as32/1024/2048. No method change beyond rank. Memory cap {args.memory_fraction:.0%} after free-memory admission.")
    run.write_json(path/"config.json", experiment)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        result = run.solve(evaluator, "DLR D-Flow", initial, inputs["truth"], inputs["observed"],
            path/"dlr_d_flow.pt", {**protocol, "experiment": experiment}, interrupted)
        print(args.rank, result["status"], result["history"][-1], flush=True)
        status = result["status"]
    except torch.OutOfMemoryError as error:
        run.write_json(path/"memory_failure.json", dict(error=str(error), rank=args.rank,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        run.write_json(path/"memory_usage.json", dict(
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))
    run.write_json(path/"complete.json", dict(status=status, elapsed_seconds=time.perf_counter()-started))


if __name__ == "__main__":
    main()

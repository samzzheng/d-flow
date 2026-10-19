"""CPU-only fixed-source solver refinement for the 250px non-attention model.

This measures the learned ODE's discretization error, not inverse-solve quality.
No optimization and no CUDA calls are performed. Defaults use an existing
exact-Euler optimization iterate so the source is held fixed across solvers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_sparse_ct_circles_cpu as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--iterate", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--euler-steps", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--rk4-steps", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Reuse saved outputs after verifying their checkpoint and source")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/rk4_accuracy_2026_09_27/refinement")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("Use at most four CPU threads for this shared machine")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    previous_report = json.loads((args.out / "results.json").read_text()) if args.reuse_existing and (args.out / "results.json").exists() else None
    cache = ROOT / f"outputs/dlr_speed_2026_09_27/initial/points/res250_sample{args.sample}.pt"
    point = torch.load(cache, map_location="cpu", weights_only=True)
    source = point["points"][args.iterate].detach().contiguous()
    model = bench.load_model(250, 100, torch.device("cpu"),
                             "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
    assert not model.mid_attn and not any(model.att_channels)
    checkpoint = ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt"
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    assert checkpoint_hash == point["metadata"]["checkpoint_sha256"]
    report = {"protocol": {"device": "cpu", "threads": args.threads,
        "sample": args.sample, "iterate": args.iterate, "source_cache": str(cache),
        "checkpoint_sha256": checkpoint_hash, "torch": str(torch.__version__),
        "reference": f"rk4_{max(args.rk4_steps)}", "fixed_source": True,
        "purpose": "Discretization consistency, not reconstruction performance"}, "runs": {}}
    outputs = {}
    with torch.no_grad():
        start = time.perf_counter()
        model(source, torch.zeros(1))
        report["protocol"]["warmup_seconds"] = time.perf_counter() - start
        for method, counts in (("forward_euler", args.euler_steps), ("rk4", args.rk4_steps)):
            for steps in counts:
                name = f"{method}_{steps}"
                path = args.out / f"sample{args.sample}_iterate{args.iterate}_{name}.pt"
                if args.reuse_existing and path.exists():
                    if previous_report is None or name not in previous_report["runs"]:
                        raise ValueError(f"Missing timing record for saved output: {path}")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if saved["protocol"]["checkpoint_sha256"] != checkpoint_hash or not torch.equal(saved["source"], source):
                        raise ValueError(f"Source/checkpoint mismatch in saved output: {path}")
                    outputs[name] = saved["output"]
                    report["runs"][name] = previous_report["runs"][name] | {"reused": True}
                    print(json.dumps({"name": name, "reused": True}), flush=True)
                    continue
                start = time.perf_counter()
                x = bench.solve_flow(model, source, steps, method)
                elapsed = time.perf_counter() - start
                outputs[name] = x
                report["runs"][name] = {"steps": steps, "velocity_evaluations": steps * (4 if method == "rk4" else 1),
                    "seconds": elapsed, "output_norm": float(x.norm()), "finite": bool(x.isfinite().all())}
                torch.save({"output": x, "source": source, "protocol": report["protocol"]}, path)
                print(json.dumps({"name": name, **report["runs"][name]}), flush=True)
                (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    reference = outputs[report["protocol"]["reference"]]
    for name, x in outputs.items():
        report["runs"][name]["relative_to_reference"] = float((x-reference).norm()/reference.norm())
    report["successive_differences"] = {}
    for method, counts in (("forward_euler", args.euler_steps), ("rk4", args.rk4_steps)):
        for left, right in zip(sorted(counts), sorted(counts)[1:]):
            x, y = outputs[f"{method}_{left}"], outputs[f"{method}_{right}"]
            report["successive_differences"][f"{method}_{left}_vs_{right}"] = float((x-y).norm()/y.norm())
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

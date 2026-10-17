"""Independent native-Euler Jacobian spectra at saved D-Flow sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dlr_dflow.api import load_flow_model, make_terminal_flow_products
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import write_json
from experiments.multicircle_rank_hybrid.ranks import NativeProducts, spectrum
from p_flow import endpoint_gradient


def worker(case: Path, stage: str, out: Path, device: str) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = torch.load(case / "inputs.pt", map_location="cpu", weights_only=True)
    saved = torch.load(case / "d_flow.pt", map_location="cpu", weights_only=True)
    if saved["status"] not in {"step_limit", "stalled", "objective_plateau"}:
        raise ValueError("Select a completed D-Flow reconstruction")
    protocol = inputs["protocol"]
    if saved["protocol"] != protocol:
        raise ValueError("Input/reconstruction protocol mismatch")
    resolution = protocol["job"]["resolution"]
    run = Path(f"outputs/fm_unet/separated-shapes-{resolution}-noattn-ch16-b8-e200")
    checkpoint_hash = hashlib.sha256((run / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest()
    if checkpoint_hash != protocol["checkpoint_sha256"]:
        raise ValueError("Model checkpoint mismatch")
    manifest_path = case.parents[1] / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != protocol["manifest_sha256"]:
        raise ValueError("Manifest mismatch")
    manifest = json.loads(manifest_path.read_text())
    steps = manifest["settings"]["flow_steps"]
    source = (inputs["initial"] if stage == "initial" else saved["source"]).to(device)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"source": source.cpu(), "protocol": protocol, "stage": stage}, out / "source.pt")
    model, _ = load_flow_model(resolution, 0, 200, device, run)
    products = NativeProducts(model, source, steps, columns=2)
    raw_j, raw_jt = make_terminal_flow_products(model, source, steps)
    generator = torch.Generator(device=device).manual_seed(protocol["source_seed"] + 71000000)
    probe = torch.randn(source.numel(), 1, generator=generator, device=device)
    errors = []
    with torch.no_grad():
        for graph, native in ((products.apply, raw_j), (products.transpose, raw_jt)):
            reference = native(probe)
            errors.append(float((graph(probe) - reference).norm() / reference.norm()))
    if max(errors) > 1e-3:
        raise AssertionError(f"Native graph/full-map mismatch: {errors}")
    del raw_j, raw_jt, reference
    metadata = {"case": str(case), "stage": stage, "protocol": protocol,
                "flow_steps": steps, "device": device, "tf32": False,
                "source_sha256": hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native_graph_vs_full_map_relative_errors": errors,
                "final_image_relative_l2": saved["history"][-1]["image_relative_l2"],
                "definition": "J = derivative of the full discrete Euler map with respect to source; not the objective Hessian"}
    paired_initial = out.parent / f"r{resolution}_ct10_initial" / "spectrum.json"
    result = None
    if stage == "initial" and protocol["operator"]["kind"] == "mri" and paired_initial.exists():
        candidate = json.loads(paired_initial.read_text())
        if (candidate["status"] != "running" and candidate["source_sha256"] == metadata["source_sha256"]
                and candidate["protocol"]["checkpoint_sha256"] == checkpoint_hash
                and candidate["flow_steps"] == steps):
            result = {**candidate, **metadata, "reused_identical_source_spectrum": str(paired_initial)}
            write_json(out / "spectrum.json", result)
    if result is None:
        result = spectrum(products.apply, products.transpose, source, out / "spectrum.json",
                          [32, 64, 128], 3, 32, protocol["source_seed"] + 72000000, metadata)
    operator_info = protocol["operator"]
    operator = (CT(resolution, operator_info["angle_count"], torch.device(device))
                if operator_info["kind"] == "ct" else
                MRI(resolution, operator_info["nominal_acceleration"], operator_info["mask_seed"],
                    torch.device(device), operator_info["pattern"]))
    if operator_info["kind"] == "mri" and not torch.equal(operator.mask.cpu(), inputs["mask"]):
        raise ValueError("MRI mask mismatch")
    if not torch.equal(inputs["clean"], inputs["observed"]):
        raise ValueError("Expected noiseless data")
    _, gx = endpoint_gradient(products.states[-1], operator, inputs["observed"].to(device))
    exact = products.transpose(gx.reshape(-1, 1))
    # Independently refine a wider reference subspace; compare gradient changes
    # between iterations, rather than assuming spectral energy implies accuracy.
    width = min(64, result["latest"]["width"])
    q = torch.linalg.qr(products.apply(torch.randn(source.numel(), width,
        generator=generator, device=device)), mode="reduced")[0]
    previous: dict[int, torch.Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    for iteration in range(4):
        b = products.transpose(q)
        v, singular, ut = torch.linalg.svd(b.double(), full_matrices=False)
        u = q.double() @ ut.T
        residual = (products.apply(v[:, :16].float()).double() - u[:, :16] * singular[:16]).norm(dim=0) / singular[0]
        row: dict[str, Any] = {"refinement": iteration, "ranks": {}}
        for rank in (4, 8, 12, 16):
            truncated = v[:, :rank] @ (singular[:rank, None] * (u[:, :rank].T @ gx.reshape(-1, 1).double()))
            norm = float(exact.norm())
            change = float((truncated - previous[rank]).norm()) / norm if rank in previous and norm else None
            row["ranks"][str(rank)] = {
                "relative_gradient_error": float((truncated - exact).norm()) / norm if norm else None,
                "cosine": float(torch.nn.functional.cosine_similarity(truncated.flatten(), exact.double().flatten(), dim=0)) if norm and float(truncated.norm()) else None,
                "change_over_exact_norm": change,
                "triplet_residual": float(residual[:rank].max()),
                "reference_converged": change is not None and change < .01 and float(residual[:rank].max()) < 1e-3}
            previous[rank] = truncated
        diagnostics.append(row)
        write_json(out / "gradient.json", {"metadata": metadata, "exact_gradient_norm": float(exact.norm()),
                   "width": width, "stages": diagnostics, "latest": row})
        if all(item["reference_converged"] for item in row["ranks"].values()):
            break
        q = torch.linalg.qr(products.apply(torch.linalg.qr(b, mode="reduced")[0]), mode="reduced")[0]


def report(out: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    lines = ["# Separated-shapes Jacobian rank pilot", "",
             "Sample 0, noiseless CT10/MRI4×; native five-step Euler Jacobian at initial and final D-Flow sources. FP32, TF32 off. Energy ranks use squared singular values plus independently estimated omitted energy; they are not algebraic ranks. Unresolved cutoffs remain unresolved. Rank-truncated gradients are independent SVD references, not the evolved DLR approximation.", "",
             "| Case / source | 99% rank | 99.9% rank | 99.99% rank | Rank-8 energy (%) | Rank-8 gradient error (%) | Gradient reference converged | Spectrum status |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for path in sorted(out.glob("*/spectrum.json")):
        result = json.loads(path.read_text())
        latest = result["latest"]
        singular = latest["singular_values"]
        energy = 100 * sum(s*s for s in singular[:8]) / (sum(s*s for s in singular) + latest["omitted_energy_estimate"])
        ranks = [latest["effective_ranks"][key] or "unresolved" for key in ("0.99", "0.999", "0.9999")]
        axes[0].semilogy(range(1, len(singular) + 1), [s / singular[0] for s in singular], label=path.parent.name)
        total = sum(s*s for s in singular) + latest["omitted_energy_estimate"]
        axes[1].plot(range(1, len(singular) + 1), torch.tensor(singular).square().cumsum(0).numpy() / total)
        gradient_path = path.with_name("gradient.json")
        gradient = json.loads(gradient_path.read_text())["latest"]["ranks"]["8"] if gradient_path.exists() else None
        error = f"{100 * gradient['relative_gradient_error']:.2f}" if gradient and gradient["relative_gradient_error"] is not None else "pending"
        lines.append(f"| {path.parent.name} | {' | '.join(map(str, ranks))} | {energy:.6f} | {error} | {gradient['reference_converged'] if gradient else 'pending'} | {result['status']} |")
    lines.extend(["", "Only one paired sample per resolution: this is a diagnostic pilot, not a dataset-wide rank estimate. No intermediate optimization sources were saved by the original sweep. All sources used here are snapshotted alongside the results.", "",
                  "The overall spectral status requires the stricter 99.99% cutoff to resolve; a 99% rank estimate may still be available. Tail sampling uncertainty and individual triplet residuals are in spectrum.json. Large relative gradient errors near convergence can correspond to tiny absolute differences; exact gradient norms are saved in gradient.json. Neither spectral truncation error nor a single final-source diagnostic proves the cause of a reconstruction failure."])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    axes[0].set(xlabel="Singular value index", ylabel="Singular value / largest")
    axes[1].set(xlabel="Truncation rank", ylabel="Estimated captured squared-singular-value energy", ylim=(0, 1.01))
    for axis in axes:
        axis.axvline(8, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    figure.savefig(out / "spectra.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/separated_shapes_noiseless_20261016/production"))
    parser.add_argument("--out", type=Path, default=Path("outputs/separated_shapes_rank_20261017"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--stage", choices=["initial", "final"])
    args = parser.parse_args()
    if args.case:
        worker(args.case, args.stage, args.out, args.device)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    for resolution in (64, 128):
        for problem in ("ct10", "mri4"):
            case = args.root / f"r{resolution}_shapes_{problem}" / "sample_000"
            for stage in ("final", "initial"):
                destination = args.out / f"r{resolution}_{problem}_{stage}"
                if (destination / "gradient.json").exists():
                    continue
                with (args.out / f"{destination.name}.log").open("a") as log:
                    completed = subprocess.run([sys.executable, "-u", "-m", "experiments.separated_shapes_rank.run",
                        "--case", str(case), "--stage", stage, "--out", str(destination), "--device", args.device],
                        stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
                if completed.returncode:
                    failures.append({"case": str(case), "stage": stage, "exit_code": completed.returncode})
                write_json(args.out / "status.json", {"status": "running", "failures": failures})
                report(args.out)
    write_json(args.out / "status.json", {"status": "completed" if not failures else "completed_with_failures", "failures": failures})
    report(args.out)


if __name__ == "__main__":
    main()

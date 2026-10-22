"""CPU-only geometry stratification of paired, completed exact/proxy inversions."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from dataset.separated_shapes import Shape, rasterize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    geometry = {}
    rows = []
    terminal = {"stalled", "step_limit", "objective_plateau"}
    for directory in sorted(args.baseline.glob("r*/sample_*")):
        summaries = {}
        for method in ("d_flow", "p_flow"):
            path = directory/f"{method}.json"
            if path.exists():
                summary = json.loads(path.read_text())
                if summary["status"] in terminal:
                    summaries[method] = summary
        if len(summaries) != 2:
            continue
        inputs = torch.load(directory/"inputs.pt", map_location="cpu", weights_only=True, mmap=True)
        protocol = inputs["protocol"]
        resolution = protocol["job"]["resolution"]
        sample = int(directory.name.split("_")[-1])
        if resolution not in geometry:
            metadata = json.loads(Path(f"data/separated-shapes-{resolution}.json").read_text())
            start = metadata["split_sizes"]["train"] + metadata["split_sizes"]["val"]
            with Path(f"data/separated-shapes-{resolution}.geometry.jsonl").open() as source:
                geometry[resolution] = list(map(json.loads, itertools.islice(source, start, start+30)))
        record = geometry[resolution][sample]
        shapes = [Shape(**shape) for shape in record["shapes"]]
        components = [torch.from_numpy(np.rint(rasterize([shape], resolution)*255).astype(np.uint8)).double()/255
                      for shape in shapes]
        truth = inputs["truth"].double().squeeze()
        # The loader's FP32 scaling can round differently from CPU division;
        # compare the original quantized pixels exactly as well as the floats.
        assert torch.equal((sum(components)*255).round().to(torch.uint8), (truth*255).round().to(torch.uint8))
        torch.testing.assert_close(sum(components), truth, atol=torch.finfo(torch.float32).eps, rtol=0)
        assert torch.equal(inputs["observed"], inputs["clean"])
        energy = truth.square().sum()
        triangles = sum(shape.kind == "triangle" for shape in shapes)
        source_hash = hashlib.sha256(inputs["initial"].numpy().tobytes()).hexdigest()
        for method, summary in summaries.items():
            saved = torch.load(directory/f"{method}.pt", map_location="cpu", weights_only=True, mmap=True)
            assert saved["protocol"] == summary["protocol"] == protocol
            image = saved["image"].double().squeeze()
            error = (image-truth).square()
            total = error.sum()
            relative = float((total/energy).sqrt())
            assert abs(relative-summary["history"][-1]["image_relative_l2"]) < 1e-6
            foreground_errors = [float(error[component > 0].sum()) for component in components]
            background_error = float(error[truth == 0].sum())
            assert abs(sum(foreground_errors)+background_error-float(total)) < 1e-8*max(1.,float(total))
            amplitudes = [float((image*component).sum()/component.square().sum()) for component in components]
            observed = inputs["observed"].to(torch.complex128 if inputs["observed"].is_complex() else torch.float64)
            measurement = saved["measurement"].to(observed.dtype)
            residual = float((measurement-observed).norm()/observed.norm())
            assert abs(residual-summary["history"][-1]["measurement_relative_l2_noisy"]) < 1e-6
            rows.append(dict(case=str(directory.relative_to(args.baseline)), resolution=resolution,
                problem=protocol["job"]["problem"], sample=sample, seed=record["seed"],
                initial_source_sha256=source_hash, method=method, status=saved["status"],
                attempts=summary["history"][-1]["step"], objects=len(shapes), triangles=triangles,
                geometric_parameters=3*(len(shapes)-triangles)+6*triangles,
                image_error_percent=100*relative, residual_percent=100*residual,
                output_norm_ratio=float(image.norm()/truth.norm()),
                background_fraction_of_squared_error=background_error/float(total) if total else None,
                object_amplitudes=amplitudes, object_kinds=[shape.kind for shape in shapes],
                object_weak_at_truth_location=[amplitude < .5 for amplitude in amplitudes],
                object_squared_error_fractions=[value/float(total) if total else None for value in foreground_errors]))
    groups = []
    for resolution, problem in sorted({(row["resolution"], row["problem"]) for row in rows}):
        for count in range(1, 5):
            for method in ("d_flow", "p_flow"):
                selected = [row for row in rows if (row["resolution"], row["problem"], row["objects"], row["method"]) == (resolution, problem, count, method)]
                if selected:
                    groups.append(dict(resolution=resolution, problem=problem, objects=count, method=method, samples=len(selected),
                        median_image_error_percent=statistics.median(row["image_error_percent"] for row in selected),
                        median_residual_percent=statistics.median(row["residual_percent"] for row in selected),
                        above_50_percent=sum(row["image_error_percent"] > 50 for row in selected),
                        below_10_percent=sum(row["image_error_percent"] < 10 for row in selected)))
    quality = json.loads(Path("outputs/separated_shapes_117k_diagnosis_20261022/sampling_quality_summary.json").read_text())
    evidence = dict(status="complete", paired_cases=len(rows)//2,
        resolution_counts=dict(Counter(row["resolution"] for row in rows if row["method"] == "d_flow")),
        rows=rows, groups=groups,
        exclusions="Only completed paired D-Flow/P-Flow runs. Legacy DLR results excluded because derivative-backend defects were subsequently corrected. Incomplete and nonrandom coverage; descriptive strata, not population estimates or causal tests.",
        checks="Cached truths match separately rasterized saved geometry; observed measurements are noiseless; methods share exact saved input protocol; image/complex residual metrics and foreground/background error partition verified.",
        object_metric="Projection amplitude on each true object; below 0.5 means weak signal at its true location, not a semantic component-count classifier.")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"geometry_results.json").write_text(json.dumps(evidence, indent=2)+"\n")
    report = ["# Why separated shapes are harder: evidence audit", "",
        "This audit separates dataset/generator difficulty from DLR-specific approximation error. It does not establish a single cause.", "",
        "## Data complexity", "",
        "The generator samples 1–4 objects, each independently a disk or general triangle. A disk has 3 continuous geometric parameters (center and radius); a triangle has 6 (center, three side lengths, orientation). A fixed composition therefore has 3Nc+6Nt geometric parameters, up to 24, plus discrete count/type choices. This is a geometric description, not a prediction of the learned Jacobian's numerical rank. One mixed-shape model handles all compositions, whereas the circle study used separate models for fixed circle counts.", "",
        "## Five-step generator quality", "",
        "Existing paired unconditional checks use 128 source samples per resolution. Blank means the existing raster component detector finds no valid component; this is a detector-based diagnostic, not proof of an identically zero image.", "",
        "| Resolution | Blank at 5 Euler steps | Blank at 100 Euler steps |", "|---|---:|---:|"]
    for row in quality["rows"]:
        report.append(f"| {row['resolution']}² | {row['small_5']['histogram'].get('0',0)}/128 | {row['small_100']['histogram'].get('0',0)}/128 |")
    report += ["", "This shows substantial distortion of the learned sampling map at five steps, particularly at high resolution. It can create poor starting images for every inverse method. It does not prove that increasing steps alone will fix inversion; earlier paired controls found mixed reconstruction outcomes.", "",
        "## Completed paired inverse cases", "", f"Coverage: {len(rows)//2} paired cases; resolutions/counts: {evidence['resolution_counts']}. CT uses 10 projection angles; MRI uses 4x Cartesian sampling. All are noiseless, five Euler steps, with the established LBFGS/geometric Armijo setup. The partial sweep is not a random or balanced sample.", "",
        "| Resolution | Problem | Objects | Method | Cases | Median image error | Median residual | Image error > 50% |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in groups:
        report.append(f"| {row['resolution']}² | {row['problem'].upper()} | {row['objects']} | {row['method']} | {row['samples']} | {row['median_image_error_percent']:.2f}% | {row['median_residual_percent']:.2f}% | {row['above_50_percent']}/{row['samples']} |")
    mri_dflow = [row for row in rows if row["resolution"] == 250 and row["problem"] == "mri" and row["method"] == "d_flow"]
    mri_pflow = [row for row in rows if row["resolution"] == 250 and row["problem"] == "mri" and row["method"] == "p_flow"]
    mri_failed = [row for row in mri_dflow if row["image_error_percent"] > 50]
    if mri_failed:
        report += ["", f"At 250² MRI, P-Flow finishes below 10% image error on {sum(row['image_error_percent'] < 10 for row in mri_pflow)}/{len(mri_pflow)} paired cases, while D-Flow exceeds 50% on {len(mri_failed)}/{len(mri_dflow)}. The failed D-Flow cases have median output norm {statistics.median(row['output_norm_ratio'] for row in mri_failed):.4f} times the true image norm. All true-object projection amplitudes are below 0.5 in {sum(all(row['object_weak_at_truth_location']) for row in mri_failed)}/{len(mri_failed)} of these failures.", "",
            "Successful P-Flow images establish representability for those specific cases under the same five-step generator. They rule out insufficient generator expressivity as a complete explanation of the paired D-Flow failures. They do not prove a global capacity claim or identify one optimizer mechanism. The common setup differs in source-gradient estimator, which also changes the LBFGS curvature history and line searches."]
    report += ["", evidence["exclusions"], "", evidence["checks"], "",
        "Per-case records include true object types, geometric parameter count, signal amplitude at each true object's location, and the exact foreground/background partition of squared image error. These help distinguish weak/missing true objects from spurious background signal without relying on visual impressions.", "",
        "## What would discriminate the remaining causes", "",
        "- Fit the full image using the same generator and paired sources. Failure even with full-image observations would implicate the generator or its optimization; success would isolate difficulty added by CT/MRI.",
        "- Compare 5-step and 100-step full-image fits before changing the inverse problem. This separates coarse integration from measurement ambiguity.",
        "- At the true geometry, measure how CT/MRI attenuate small shape-parameter changes. Compare measurement directions with image directions; full-image Jacobian rank alone does not diagnose acquisition identifiability.",
        "- Compare exact and DLR gradients at identical sources, including missing-subspace and coefficient error. The ongoing high-start gradual-rank controls address one specific source of projection loss.", "",
        "A small data residual with a wrong image can indicate a measurement ambiguity or a nearly invisible image difference. A large residual with a wrong image instead indicates unresolved data fitting; it does not by itself establish non-identifiability. Neither classification alone proves model capacity is insufficient."]
    (args.out/"report.md").write_text("\n".join(report)+"\n")
    print(json.dumps({key:evidence[key] for key in ("paired_cases","resolution_counts","groups")},indent=2))


if __name__ == "__main__":
    main()

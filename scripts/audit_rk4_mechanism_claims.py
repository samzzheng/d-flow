"""Audit saved RK4 mechanism evidence; CPU-only, no learned-model evaluations.

Recomputes edge-statistic uncertainty, local-fit termination evidence, noise
decomposition, and low-rank Jacobian/metric comparisons from saved artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import investigate_rk4_mechanism as mechanism

BASE = ROOT / "outputs/rk4_mechanism_2026_09_27"
OUT = ROOT / "outputs/rk4_mechanism_claim_audit_2026_09_27"


def read(path):
    return json.loads(path.read_text())


def bootstrap_difference(a, b, paired=True):
    """Descriptive percentile interval; tiny samples are not population proof."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(20260927)
    ia = rng.integers(len(a), size=(20000, len(a)))
    ib = ia if paired else rng.integers(len(b), size=(20000, len(b)))
    draws = b[ib].mean(1) - a[ia].mean(1)
    return dict(mean_difference=float(b.mean() - a.mean()),
                bootstrap_95pct=np.quantile(draws, [.025, .975]).tolist(),
                paired=paired)


def edge_statistics():
    data = read(BASE / "samples/sample_stats.json")
    stats = {"data": data["data"], **data["generators"]}
    out = {"sample_count": {k: len(v["intermediate_fraction"]) for k, v in stats.items()},
           "means": {}, "paired_rk4100_minus_euler100": {}}
    for name, values in stats.items():
        means = {k: float(np.mean(v)) for k, v in values.items() if isinstance(v, list)}
        circumference = 62500 * np.array(values["intermediate_fraction"]) / np.array(
            values["intermediate_pixels_per_circumference"])
        means["circumference_proxy_pixels"] = float(circumference.mean())
        means["radius_proxy_pixels"] = float(circumference.mean() / (2 * np.pi))
        out["means"][name] = means
    for key in ("intermediate_fraction", "intermediate_pixels_per_circumference", "binary_distance"):
        out["paired_rk4100_minus_euler100"][key] = bootstrap_difference(
            stats["euler100"][key], stats["rk4100"][key])
        out.setdefault("unpaired_rk4100_minus_data", {})[key] = bootstrap_difference(
            stats["data"][key], stats["rk4100"][key], paired=False)
    return out


def oracle_fits():
    out = []
    for path in sorted((BASE / "fits").glob("*/results.json")):
        run = read(path)
        history = run["history"]
        last, ten_before = history[-1], history[max(0, len(history) - 11)]
        out.append(dict(name=path.parent.name, stop_reason=run["stop_reason"],
                        final_attempt=last["attempt"], final_gradient_norm=last["gradient_norm"],
                        clamped_image_error=last["image_relative_l2"],
                        raw_image_error=last["raw_image_relative_l2"],
                        final_source_step_norm=last["source_step_norm"],
                        last_ten_objective_reduction_fraction=(ten_before["objective"] - last["objective"])
                        / ten_before["objective"],
                        initial_source=run["protocol"]["init_source"],
                        source_prior=run["protocol"]["source_prior"]))
    return out


def noise_and_nullspace():
    out = {}
    for method in ("euler", "rk4"):
        row = read(BASE / f"decompose/pilot_s1_{method}.json")["rows"][-1]
        item = dict(raw_noise_projection_coefficient=row["raw_noise_fit_fraction"],
                    raw_clean_measurement_error_over_noise=row["raw_clean_residual_over_noise"],
                    raw_residual_noise_cosine=(row["raw_noise_fit_fraction"] - 1)
                    / row["raw_noisy_residual_over_noise"],
                    residual_identity_beta=.5 * (row["raw_clean_residual_over_noise"] ** 2 + 1
                                                  - row["raw_noisy_residual_over_noise"] ** 2))
        for kind in ("raw", "clamped"):
            total, visible, null = (row[f"{kind}_{name}_rel"] for name in ("image", "visible", "invisible"))
            item[kind] = dict(total=total, visible=visible, null=null,
                              null_norm_fraction=null / total, null_energy_fraction=(null / total) ** 2,
                              pythagorean_relative_error=(visible ** 2 + null ** 2 - total ** 2) / total ** 2)
        out[method] = item
    for kind in ("raw", "clamped"):
        a, b = out["euler"][kind], out["rk4"][kind]
        out.setdefault("excess_null_energy_fraction", {})[kind] = (b["null"] ** 2 - a["null"] ** 2) / (
            b["total"] ** 2 - a["total"] ** 2)
    return out


def jacobian_comparisons():
    problem = mechanism.problem(250, 1, torch.device("cpu"))
    out = {}
    entries = (("s1", "z0", "euler", 0), ("s1_solutions", "E96", "euler", 96),
               ("s1_solutions", "R94", "rk4", 94))
    for stem, source, method, step in entries:
        a, b = [torch.load(BASE / "spectra" / f"{stem}_{source}_{g}_vectors.pt", weights_only=True)
                for g in ("euler100", "rk4100")]
        ua, ub, va, vb, sa, sb = [v.double() for v in (a["U"], b["U"], a["V"], b["V"], a["s"], b["s"])]
        cu, cv = ua.T @ ub, va.T @ vb
        jdiff2 = (sa ** 2).sum() + (sb ** 2).sum() - 2 * (cu * cv * sa[:, None] * sb[None, :]).sum()
        kdiff2 = (sa ** 4).sum() + (sb ** 4).sum() - 2 * (cu ** 2 * sa[:, None] ** 2 * sb[None, :] ** 2).sum()
        row = dict(retained_rank=len(sa), relative_frobenius_J_difference=float(jdiff2.sqrt() / sa.norm()),
                   relative_frobenius_JJt_difference=float(kdiff2.sqrt() / (sa ** 4).sum().sqrt()),
                   top3_fraction_of_retained_squared_gain=float((sa[:3] ** 2).sum() / (sa ** 2).sum()),
                   principal_angle_min_cosines={})
        for k in (3, 8, 16, 22, 24):
            if k <= len(sa):
                row["principal_angle_min_cosines"][str(k)] = {
                    "image": float(torch.linalg.svdvals(cu[:k, :k])[-1]),
                    "source": float(torch.linalg.svdvals(cv[:k, :k])[-1])}
        state = torch.load(ROOT / f"outputs/rk4_noise_floor_2026_09_27/sample1_{method}/accepted/state_{step:05d}.pt",
                           map_location="cpu", weights_only=False)
        x = state["raw_reconstruction"].clone().requires_grad_(True)
        residual = mechanism.bench.sparse_radon(x, problem["angles"]) - problem["observed"]
        gx, = torch.autograd.grad(.5 * residual.square().sum(), x)
        gx = gx.flatten().double()
        ka, kb = ua @ (sa.square() * (ua.T @ gx)), ub @ (sb.square() * (ub.T @ gx))
        row["common_cotangent_metric_action"] = dict(
            cotangent_image=f"{method} state {step}", definition="gx = A^T(A x - y), same gx for both metrics",
            relative_difference=float((ka - kb).norm() / ka.norm()),
            cosine=float((ka @ kb) / (ka.norm() * kb.norm())), rk4_to_euler_norm=float(kb.norm() / ka.norm()))
        out[source] = row
    return out


def population():
    group = read(ROOT / "outputs/rk4_accuracy_2026_09_27/completed_runs.json")["groups"]["armijo_local"]
    pairs = []
    for pair in group["pairs"]:
        a, b = (float(pair[m]["image_relative_l2"]) for m in ("euler", "rk4"))
        pairs.append(dict(sample=pair["sample"], euler=a, rk4=b, both_below_5pct=a < .05 and b < .05))
    return dict(snapshot=group["summary"], pairs=pairs,
                caution="Completed partial cohort, old optimizer, 300-attempt endpoint, and outcome-selected <5% subset; not a population noise-floor study.")


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(provenance="Reanalysis of existing saved artifacts; no model evaluations, no GPU work.",
                  edge=edge_statistics(), oracle_fits=oracle_fits(), noise=noise_and_nullspace(),
                  jacobians=jacobian_comparisons(), population=population())
    path = OUT / "audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)
    print(json.dumps({"edge": result["edge"], "jacobians": result["jacobians"]}, indent=2))


if __name__ == "__main__":
    main()

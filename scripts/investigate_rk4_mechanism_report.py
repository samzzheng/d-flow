"""Tables and figures for the Euler-vs-RK4 mechanism study.

Reads outputs/rk4_mechanism_2026_09_27/{solves,solves64,swap,spectra,cross,
decompose} and writes summary.json plus figures next to them. Per-state error
decompositions (visible / invisible to A) are computed here for every solve
and cached as decompose/<run>.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import investigate_rk4_mechanism as M  # noqa: E402

OUT = M.OUT
GENERATORS = ["euler100", "rk425", "euler400", "rk4100"]
GEN_LABEL = {"euler100": "Euler-100", "rk425": "RK4-25", "euler400": "Euler-400", "rk4100": "RK4-100"}
GEN_COLOR = {"euler100": "#1f77b4", "rk425": "#9ecae1", "euler400": "#fdae6b", "rk4100": "#d62728"}
TAUS = ["1.2", "1.1", "1.05", "1.0"]


def load_results(folder: Path) -> dict[tuple[int, int, str], dict]:
    runs = {}
    for path in sorted(folder.glob("res*_s*_*/results.json")):
        r = json.loads(path.read_text())
        p = r["protocol"]
        runs[(p["resolution"], p["sample"], p["generator"])] = r | {"dir": path.parent}
    return runs


def decompose_run(run: dict, projectors: dict) -> list[dict]:
    """Per accepted state: visible / invisible split (cached)."""
    res, sample = run["protocol"]["resolution"], run["protocol"]["sample"]
    cache = OUT / "decompose" / f"{run['dir'].name}.json"
    states = M.list_states(run["dir"] / "accepted")
    if cache.exists():
        cached = json.loads(cache.read_text())
        if len(cached["rows"]) == len(states):
            return cached["rows"]
    if res not in projectors:
        projectors[res] = M.Projector(res, torch.device("cpu"))
    proj = projectors[res]
    P = M.problem(res, sample, torch.device("cpu"))
    truth = P["truth"].flatten()
    noise = P["noise"].flatten()
    noise_norm = noise.norm().item()
    rows = []
    for path in states:
        s = torch.load(path, map_location="cpu", weights_only=False)
        x = s["raw_reconstruction"].flatten().clamp(0, 1)
        e = x - truth
        vis, inv = proj.split(e)
        ae = proj.A @ e
        rows.append({"accepted_index": s["metrics"]["accepted_index"], "attempt": s["metrics"]["attempt"],
                     "residual_over_noise": s["metrics"]["residual_over_noise"],
                     "image_rel": e.norm().item() / truth.norm().item(),
                     "visible_rel": vis.norm().item() / truth.norm().item(),
                     "invisible_rel": inv.norm().item() / truth.norm().item(),
                     "noise_fit_fraction": (ae @ noise).item() / noise_norm ** 2})
    cache.parent.mkdir(exist_ok=True)
    M.write_json({"rows": rows}, cache)
    return rows


def at_crossing(history: list[dict], tau: float) -> dict | None:
    for row in history:
        if row["residual_over_noise"] <= tau:
            return row
    return None


def at_residual(history: list[dict], tau: float, key: str) -> float | None:
    row = at_crossing(history, tau)
    return None if row is None else row[key]


def best_below(history: list[dict], tau: float, key: str) -> float | None:
    """The best (lowest) value of `key` among states with residual/noise <= tau.
    An oracle number: what the generator could deliver at that data-fit level."""
    values = [row[key] for row in history if row["residual_over_noise"] <= tau]
    return min(values) if values else None


def summarize(runs: dict, res: int, decomps: dict) -> dict:
    samples = sorted({s for (r, s, g) in runs if r == res})
    table = {}
    for s in samples:
        table[s] = {}
        for g in GENERATORS:
            run = runs.get((res, s, g))
            if run is None:
                continue
            h = run["history"]
            last = h[-1]
            entry = {"stop_reason": run["stop_reason"], "attempts": last["attempt"],
                     "accepted": len(h), "elapsed_min": last["elapsed_seconds"] / 60,
                     "final_residual_over_noise": last["residual_over_noise"],
                     "final_image": last["image_relative_l2"],
                     "min_residual_over_noise": min(r["residual_over_noise"] for r in h),
                     "best_image_any": min(r["image_relative_l2"] for r in h),
                     "crossings": {tau: (at_crossing(h, float(tau)) or {}).get("attempt") for tau in TAUS},
                     "image_at_tau": {tau: at_residual(h, float(tau), "image_relative_l2") for tau in TAUS},
                     "best_image_below_tau": {tau: best_below(h, float(tau), "image_relative_l2") for tau in TAUS}}
            d = decomps.get((res, s, g))
            if d:
                by_idx = {row["accepted_index"]: row for row in d}
                entry["invisible_at_tau"] = {}
                entry["visible_at_tau"] = {}
                for tau in TAUS:
                    row = at_crossing(h, float(tau))
                    dd = by_idx.get(row["accepted_index"]) if row else None
                    entry["invisible_at_tau"][tau] = dd["invisible_rel"] if dd else None
                    entry["visible_at_tau"][tau] = dd["visible_rel"] if dd else None
                entry["final_invisible"] = d[-1]["invisible_rel"]
                entry["final_visible"] = d[-1]["visible_rel"]
            table[s][g] = entry
    # paired wins at matched residual, per tau, vs Euler-100
    wins = {}
    for tau in TAUS:
        for g in GENERATORS[1:]:
            e_better = g_better = 0
            diffs = []
            for s in samples:
                a = table[s].get("euler100", {}).get("image_at_tau", {}).get(tau)
                b = table[s].get(g, {}).get("image_at_tau", {}).get(tau)
                if a is None or b is None:
                    continue
                diffs.append(b - a)
                e_better += b > a
                g_better += b < a
            wins[f"{g}_vs_euler100@{tau}"] = {"euler100_better": e_better, f"{g}_better": g_better,
                                               "median_diff_pp": (100 * float(np.median(diffs))) if diffs else None,
                                               "n": len(diffs)}
    return {"samples": samples, "table": table, "paired": wins}


def fig_semiconvergence(runs: dict, res: int, decomps: dict, path: Path, samples: list[int]) -> None:
    """Per sample: full trajectory (left) and a zoom on the noise floor (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.9 * n), squeeze=False)
    for i, s in enumerate(samples):
        ax, axz = axes[i]
        for g in GENERATORS:
            run = runs.get((res, s, g))
            if run is None:
                continue
            h = run["history"]
            r = np.array([row["residual_over_noise"] for row in h])
            img = 100 * np.array([row["image_relative_l2"] for row in h])
            keep = r < 3
            label = GEN_LABEL[g] + ("" if run["stop_reason"] else " (running)")
            ax.plot(r[keep], img[keep], "-", color=GEN_COLOR[g], lw=1.6, label=label)
            ax.plot(r[keep][-1:], img[keep][-1:], "o", color=GEN_COLOR[g], ms=5)
            zoom = r < 1.12
            axz.plot(r[zoom], img[zoom], "-o", ms=2.5, color=GEN_COLOR[g], lw=1.4, label=label)
            d = decomps.get((res, s, g))
            if d:
                rr = np.array([row["residual_over_noise"] for row in d])
                inv = 100 * np.array([row["invisible_rel"] for row in d])
                z2 = rr < 1.12
                axz.plot(rr[z2], inv[z2], "--", color=GEN_COLOR[g], lw=1.0)
        for a in (ax, axz):
            a.invert_xaxis()
            a.axvline(1.0, color="k", ls=":", lw=1)
            a.grid(alpha=0.3)
            a.set_xlabel("residual / noise norm (optimization runs right to left)")
        ax.set_title(f"{res}px sample {s}: image error along the optimization")
        ax.set_ylabel("image rel. L2 error [%]")
        ax.set_ylim(0, min(ax.get_ylim()[1], 10))
        axz.set_title("zoom at the noise floor (solid: total, dashed: null-space part)")
        axz.set_xlim(1.12, 0.975)
        axz.set_ylim(0.6, 2.6)
        axz.set_ylabel("image rel. L2 error [%]")
        ax.legend(fontsize=8)
    fig.suptitle("Integrator family vs ODE accuracy, fp32, capped Armijo L-BFGS")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_spectra(path_json: Path, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(path_json.read_text())
    sources = sorted({k.split("|")[0] for k in data["entries"]})
    fig, axes = plt.subplots(2, len(sources), figsize=(4.6 * len(sources), 7), squeeze=False)
    for j, src in enumerate(sources):
        ax, ax2 = axes[0, j], axes[1, j]
        for g in GENERATORS:
            e = data["entries"].get(f"{src}|{g}")
            if e is None:
                continue
            s = np.array(e["singular_values"])
            ax.semilogy(np.arange(1, len(s) + 1), s, "o-", ms=3, color=GEN_COLOR[g], label=GEN_LABEL[g])
            if "invisible_fraction_per_direction" in e:
                ax2.plot(np.arange(1, len(s) + 1), e["invisible_fraction_per_direction"], "o-", ms=3,
                         color=GEN_COLOR[g], label=GEN_LABEL[g])
        ax.set_title(f"dG/dz singular values at source {src}")
        ax.set_xlabel("index")
        ax.set_ylabel("singular value")
        ax.grid(alpha=0.3, which="both")
        ax2.set_xlabel("index of left singular direction")
        ax2.set_ylabel("fraction of direction in null space of A")
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_aggregate64(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = summary["table"]
    fig, axes = plt.subplots(1, len(TAUS), figsize=(4 * len(TAUS), 4), squeeze=False)
    for j, tau in enumerate(TAUS):
        ax = axes[0, j]
        data = []
        for g in GENERATORS:
            vals = [100 * table[s][g]["image_at_tau"][tau] for s in table
                    if g in table[s] and table[s][g]["image_at_tau"][tau] is not None]
            data.append(vals)
        ax.boxplot(data, labels=[f"{GEN_LABEL[g]}\n(n={len(v)})" for g, v in zip(GENERATORS, data)], showfliers=True)
        ax.set_title(f"image error at first residual <= {tau} x noise")
        ax.set_ylabel("image rel. L2 error [%]")
        ax.grid(alpha=0.3)
    fig.suptitle("64px, attention-free model: image error at matched data misfit")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    projectors: dict = {}
    summary = {}
    for folder, res in (("solves", 250), ("solves64", 64)):
        runs = load_results(OUT / folder)
        if not runs:
            continue
        decomps = {}
        for key, run in runs.items():
            if key[0] == res and (OUT / "decompose").exists() or True:
                try:
                    decomps[key] = decompose_run(run, projectors)
                except Exception as exc:  # noqa: BLE001
                    print(f"decompose failed for {key}: {exc}")
        summary[str(res)] = summarize(runs, res, decomps)
        samples = summary[str(res)]["samples"]
        if res == 250:
            fig_semiconvergence(runs, res, decomps, OUT / "semiconvergence_250.png", samples)
        else:
            fig_aggregate64(summary[str(res)], OUT / "matched_misfit_64.png")
            fig_semiconvergence(runs, res, decomps, OUT / "semiconvergence_64.png", samples[:6])
    for spec in sorted((OUT / "spectra").glob("*.json")):
        fig_spectra(spec, OUT / f"spectra_{spec.stem}.png")
        summary[f"spectra_{spec.stem}"] = json.loads(spec.read_text())["entries"]
    swap = load_results(OUT / "swap")
    if swap:
        summary["swap"] = {f"{k[1]}_{k[2]}_{Path(v['protocol']['init_source']).parent.parent.name}":
                           {"history": [{kk: row[kk] for kk in ("attempt", "residual_over_noise",
                                                                "image_relative_l2", "noise_fit_fraction")}
                                        for row in v["history"]],
                            "stop_reason": v["stop_reason"]} for k, v in swap.items()}
    for cross in sorted((OUT / "cross").glob("*.json")):
        summary[f"cross_{cross.stem}"] = json.loads(cross.read_text())["images"]
    M.write_json(summary, OUT / "summary.json")
    # console tables
    for res in ("250", "64"):
        if res not in summary:
            continue
        t = summary[res]["table"]
        print(f"\n=== {res}px: image error [%] at first crossing of residual/noise <= tau (— = never) ===")
        print("sample  " + "  ".join(f"{GEN_LABEL[g]:>22s}" for g in GENERATORS))
        print("        " + "  ".join(f"{'1.2':>5s}{'1.1':>6s}{'1.05':>6s}{'1.0':>5s}" for _ in GENERATORS))
        for s in summary[res]["samples"]:
            cells = []
            for g in GENERATORS:
                e = t[s].get(g)
                if e is None:
                    cells.append(f"{'(running)':>22s}")
                    continue
                cells.append("".join(f"{(100*e['image_at_tau'][tau]):5.2f} " if e["image_at_tau"][tau] is not None else "  —   " for tau in TAUS))
            print(f"{s:5d}   " + "  ".join(cells))
        print("paired vs Euler-100:", json.dumps(summary[res]["paired"], indent=None))


if __name__ == "__main__":
    main()

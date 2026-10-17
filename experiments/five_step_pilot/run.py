"""Freeze, run and summarize five-step circle experiments."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from experiments.ct_mri_sweep.queue import gpu_processes, source_hashes, tasks
from experiments.ct_mri_sweep.run import ROOT, TERMINAL, jobs, model_ready, reviewed_source_hashes, write_json
from experiments.noisy_ct_mri.queue import Work, choose


def report(out: Path) -> None:
    manifest = json.loads((out / "production/manifest.json").read_text())
    settings = manifest["settings"]
    methods = manifest["protocol"]["methods"]
    sample_count = min(settings["samples"], settings["scheduling"].get("sample_limit", settings["samples"]))
    total_runs = len(jobs(settings)) * sample_count * len(methods)
    population = settings.get("dataset_label", f"circle count in {settings['circles']}")
    lines = ["# Five-step Euler pilot", "",
             f"{100*settings['noise_relative']:g}% relative measurement noise; CT10 and Cartesian MRI4; "
             f"{sample_count} test samples per acquisition at each resolution; {population}. "
             "Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed "
             "line searches stop a method. A step limit does not certify convergence. "
             f"DLR uses rank {settings['dlr']['rank']}, analytical alpha=1-t, no trace probes/enrichment and previous V.", "",
             "Final saved checkpoints, not retrospective best-image selection. "
             "Five-step Euler changes the generator being optimized; more optimizer steps "
             "do not remove its time-discretization error.", "",
             "| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    rows = []
    for job in sorted(jobs(manifest["settings"]), key=lambda job: (job.resolution, job.problem)):
        for method, stem in (("D-Flow", "d_flow"), ("P-Flow", "p_flow"), ("DLR D-Flow", "dlr_d_flow")):
            if method not in methods:
                continue
            for sample in range(sample_count):
                path = out / "production" / job.name / f"sample_{sample:03d}" / f"{stem}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                last = record["history"][-1]
                accepted = sum(row.get("source_step_norm", 0) > 0 for row in record["history"])
                row = {"case": job.name, "sample": sample, "method": method, "attempts": last["step"],
                       "accepted_updates": accepted, "image_percent": 100 * last["image_relative_l2"],
                       "residual_percent": 100 * last["measurement_relative_l2_noisy"], "status": record["status"]}
                rows.append(row)
                lines.append(f"| {job.name}/sample_{sample:03d} | {method} | {row['attempts']} | {accepted} | "
                             f"{row['image_percent']:.4f} | {row['residual_percent']:.4f} | {row['status']} |")
    complete = sum(row["status"] in TERMINAL for row in rows)
    lines += ["", f"Terminal method runs: {complete}/{total_runs}. Running rows are provisional.", "",
              "Each production acquisition saves `sample_NNN/comparison.png` and `.pdf`, "
              "including images and clean-reference measurements."]
    if settings.get("baseline_out"):
        baseline = Path(settings["baseline_out"])
        baseline_rank = json.loads((baseline / "production/manifest.json").read_text())["settings"]["dlr"]["rank"]
        rank = settings["dlr"]["rank"]
        paired = []
        for row in rows:
            path = baseline / "production" / row["case"] / f"sample_{row['sample']:03d}" / "dlr_d_flow.json"
            if row["status"] not in TERMINAL or not path.exists():
                continue
            old = json.loads(path.read_text())
            if old["status"] in TERMINAL:
                paired.append({"case": row["case"], "sample": row["sample"],
                               f"rank{baseline_rank}_image_percent": 100 * old["history"][-1]["image_relative_l2"],
                               f"rank{rank}_image_percent": row["image_percent"]})
        write_json(out / "paired_rank_comparison.json", paired)
        lines += ["", f"Original D/P controls and rank-{baseline_rank} results: {baseline}. Completed rank pairs: {len(paired)}.", "",
                  f"| Paired case | Rank {baseline_rank} image error % | Rank {rank} image error % |", "| --- | ---: | ---: |",
                  *[f"| {row['case']}/sample_{row['sample']:03d} | {row[f'rank{baseline_rank}_image_percent']:.4f} | {row[f'rank{rank}_image_percent']:.4f} |" for row in paired]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_json(out / "summary.json", {"terminal_runs": complete, "total_runs": total_runs, "rows": rows})
    if settings.get("rank_sweep_root"):
        from experiments.separated_shapes_rank.sweep_report import report as rank_report
        rank_report(Path(settings["rank_sweep_root"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--additional-out", type=Path, nargs="*", default=[],
                        help="Run additional frozen sweeps under the same GPU admission controller")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 96, 128, 160, 200, 250])
    parser.add_argument("--circles", type=int, nargs="+", default=[1])
    parser.add_argument("--dataset", choices=("circles", "separated_shapes"), default="circles")
    parser.add_argument("--run-circles", type=int, nargs="+",
                        help="Admit only these circle counts from frozen manifests")
    parser.add_argument("--priority-out", type=Path,
                        help="Give this sweep the next free slot when it has no active worker")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sample-limit", type=int,
                        help="Evaluate a prefix while retaining the baseline test-subset fingerprint")
    parser.add_argument("--noise-relative", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=1, help="Frozen at initialization")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dlr-only", action="store_true")
    parser.add_argument("--baseline-out", type=Path)
    parser.add_argument("--rank-sweep-root", type=Path)
    parser.add_argument("--shared-controller", type=int, nargs=2, action="append", default=[],
                        metavar=("PID", "MIB"))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "init":
        settings = json.loads((ROOT / "experiments/ct_mri_sweep/config_fp32.json").read_text())
        if args.samples < 1 or args.max_workers < 1 or not 0 <= args.noise_relative < 1:
            parser.error("Require positive sample/worker counts and noise in [0, 1)")
        settings.update(resolutions=args.resolutions, circles=args.circles, samples=args.samples,
                        flow_steps=5, noise_relative=args.noise_relative)
        if args.dataset == "separated_shapes":
            settings.update(circles=[0], dataset_label="separated shapes: 1–4 circles/triangles",
                            dataset_template="data/separated-shapes-{resolution}.pt",
                            run_template="outputs/fm_unet/separated-shapes-{resolution}-noattn-ch16-b8-e200")
        settings["stopping"].update(min_updates=1000, max_updates=1000)
        if args.rank < 1:
            parser.error("Rank must be positive")
        settings["dlr"]["rank"] = args.rank
        if args.baseline_out:
            if not args.dlr_only:
                parser.error("The paired baseline comparison requires --dlr-only")
            baseline_settings = json.loads((args.baseline_out / "production/manifest.json").read_text())["settings"]
            expected = deepcopy(baseline_settings)
            expected["dlr"]["rank"] = args.rank
            if any(settings[key] != value for key, value in expected.items() if key != "scheduling"):
                parser.error("Baseline settings differ beyond the DLR rank")
            settings["baseline_out"] = str(args.baseline_out.resolve())
        settings["scheduling"].update(samples_per_worker=1, max_workers_per_gpu=args.max_workers,
            worker_memory_mib={str(resolution): 9216 for resolution in settings["resolutions"]})
        if args.sample_limit is not None:
            if not 1 <= args.sample_limit <= args.samples:
                parser.error("Sample limit must lie between 1 and --samples")
            settings["scheduling"]["sample_limit"] = args.sample_limit
        if args.rank_sweep_root:
            settings["rank_sweep_root"] = str(args.rank_sweep_root.resolve())
        hashes = source_hashes()
        rank_worker = ROOT / "experiments/noisy_ct_mri/rank_worker.py"
        hashes[str(rank_worker.relative_to(ROOT))] = hashlib.sha256(rank_worker.read_bytes()).hexdigest()
        rank_report_path = ROOT / "experiments/separated_shapes_rank/sweep_report.py"
        hashes[str(rank_report_path.relative_to(ROOT))] = hashlib.sha256(rank_report_path.read_bytes()).hexdigest()
        hashes.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [*Path(__file__).parent.glob("*.py"), ROOT / "experiments/noisy_ct_mri/queue.py"]})
        for stage in ("smoke", "production"):
            path = out / stage / "manifest.json"
            if path.exists():
                raise FileExistsError(path)
            current = deepcopy(settings)
            if stage == "smoke":
                resolutions = [max(args.resolutions)]
                if args.dataset == "separated_shapes":
                    resolutions = sorted({min(args.resolutions), max(args.resolutions)})
                current.update(resolutions=resolutions, samples=1)
                current["stopping"].update(min_updates=2, max_updates=2, window=1)
                current["execution"]["checkpoint_every"] = 1
            write_json(path, {"created_at": datetime.now().astimezone().isoformat(), "settings": current,
                "source_sha256": hashes, "jobs": list(map(asdict, jobs(current))),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "protocol": {"smoke_only": stage == "smoke", "samples": list(range(min(current["samples"], current["scheduling"].get("sample_limit", current["samples"])))),
                    "methods": ["DLR D-Flow"] if args.dlr_only else ["D-Flow", "P-Flow", "DLR D-Flow"], "gpu": 1,
                    "pairing": "Same truth, acquisition mask and source seeds across methods; fresh optimizer states",
                    "stopping": "Up to 1000 attempts; preserve two-failed-search stop; no plateau stop before budget",
                    "noise": "Gaussian noise scaled to requested relative norm; complex on acquired MRI coefficients; zero means y=A(x_GT)",
                    "baseline": "Existing 100-step runs are historical, not a new paired control"}})
        report(out)
        return
    if args.mode == "report":
        report(out)
        return
    outputs = [out, *map(Path.resolve, args.additional_out)]
    priority_out = args.priority_out.resolve() if args.priority_out else None
    if priority_out is not None and priority_out not in outputs:
        parser.error("--priority-out must be one of the managed output directories")
    locks = []
    for root in outputs:
        lock = (root / "queue.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()))
        lock.flush()
        locks.append(lock)
    pending = []
    smoke_markers = []
    for path in (root / stage / "manifest.json" for root in outputs for stage in ("smoke", "production")):
        manifest = json.loads(path.read_text())
        for name, digest in reviewed_source_hashes(path, manifest).items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed: {name}")
        pending.extend(Work(path, task, manifest["settings"], False)
                       for task in sorted(tasks(manifest["settings"]), key=lambda task: (task.start, task.job.resolution, task.job.problem))
                       if (args.run_circles is None or task.job.circles in args.run_circles)
                       and not task.marker(path.parent).exists())
        if path.parent.name == "smoke":
            smoke_markers.extend(task.marker(path.parent) for task in tasks(manifest["settings"])
                                 if args.run_circles is None or task.job.circles in args.run_circles)
    pending.sort(key=lambda work: (work.task.start, work.task.job.resolution,
                                  work.task.job.circles, work.task.job.problem,
                                  outputs.index(work.manifest.parent.parent)))
    max_workers = manifest["settings"]["scheduling"]["max_workers_per_gpu"]
    shared = {pid: (mib, Path(f"/proc/{pid}/stat").read_text().split()[21])
              for pid, mib in args.shared_controller}
    stopped = [False]
    active: dict[str, tuple[Work, subprocess.Popen[bytes]]] = {}
    failures = json.loads((out / "failures.json").read_text()) if (out / "failures.json").exists() else {}

    def stop(signum: int, frame: Any) -> None:
        stopped[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (out / "logs").mkdir(exist_ok=True)
    try:
        while pending or active:
            if stopped[0]:
                break
            for name, (work, process) in list(active.items()):
                if process.poll() is None:
                    continue
                del active[name]
                if process.returncode != 0:
                    failures[work.name] = process.returncode
                    write_json(out / "failures.json", failures)
                elif not work.marker.exists():
                    pending.insert(0, work)
                report(work.manifest.parent.parent)
            if any("/smoke/" in name for name in failures):
                raise RuntimeError("Preflight failed; production remains blocked")
            pending = [work for work in pending if work.name not in failures]
            shared = {pid: entry for pid, entry in shared.items() if Path(f"/proc/{pid}/stat").exists()
                      and Path(f"/proc/{pid}/stat").read_text().split()[21] == entry[1]
                      and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"}
            allowed = set(shared)
            for pid in shared:
                children = Path(f"/proc/{pid}/task/{pid}/children")
                if children.exists():
                    allowed.update(map(int, children.read_text().split()))
            allowed.update(process.pid for _, process in active.values())
            external = gpu_processes()[1] - allowed
            total, used = map(int, subprocess.check_output(["nvidia-smi", "--id=1",
                "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], text=True).strip().split(","))
            if external and active:
                for _, process in active.values():
                    process.terminate()
            preflight_passed = all(path.exists() for path in smoke_markers)
            training_pending = {work.task.job for work in pending
                                if not model_ready(work.task.job, work.settings)}
            if pending and not external and shutil.disk_usage(out).free > 20 * 2**30:
                while len(active) < max_workers:
                    eligible = [work for work in pending if work.task.job not in training_pending
                                and (preflight_passed or work.manifest.parent.name == "smoke")]
                    if priority_out is not None:
                        priority_active = any(work.manifest.parent.parent == priority_out for work, _ in active.values())
                        eligible.sort(key=lambda work: (work.manifest.parent.parent == priority_out) == priority_active)
                    work = choose(eligible, [work for work, _ in active.values()], total, used,
                                  sum(mib for mib, _ in shared.values()))
                    if work is None:
                        break
                    worker = "experiments.ct_mri_sweep.run"
                    if json.loads(work.manifest.read_text())["protocol"]["methods"] == ["DLR D-Flow"]:
                        worker = "experiments.noisy_ct_mri.rank_worker"
                    with (out / "logs" / (work.name.replace("/", "_") + ".log")).open("ab") as log:
                        process = subprocess.Popen([sys.executable, "-m", worker,
                            "--manifest", str(work.manifest), "--job", work.task.job.name, "--device", "cuda:1",
                            "--sample-start", str(work.task.start), "--sample-stop", str(work.task.stop),
                            "--memory-fraction", str((9216 - 768) / total)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
                    active[work.name] = (work, process)
                    pending.remove(work)
            write_json(out / "queue_status.json", {"pid": os.getpid(), "gpu": 1,
                "updated_at": datetime.now().astimezone().isoformat(),
                "status": "running" if active else "waiting" if pending else "finished",
                "active": {name: {"pid": process.pid, "reserved_mib": 9216}
                           for name, (_, process) in active.items()}, "preflight_passed": preflight_passed,
                "pending": [work.name for work in pending], "failures": failures,
                "training_pending": sorted({f"r{job.resolution}_n{job.circles}" for job in training_pending}),
                "outputs": list(map(str, outputs)),
                "run_circles": args.run_circles, "priority_out": str(priority_out) if priority_out else None,
                "shared_controllers": shared, "external": sorted(external), "memory_total_used_mib": [total, used]})
            if pending or active:
                time.sleep(5)
    finally:
        for _, process in active.values():
            if process.poll() is None:
                process.terminate()
        for _, process in active.values():
            process.wait()
        if stopped[0]:
            write_json(out / "queue_status.json", {"status": "paused", "pid": os.getpid(), "gpu": 1})
        for root in outputs:
            report(root)
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    main()

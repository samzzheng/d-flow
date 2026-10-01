"""Independent GPU chunks with per-run checkpoints and a single report writer."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import signal
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from dlr_dflow.api import load_flow_model
from .diagnostics import diagnose
from .gpu_study import CanonicalProblem, TERMINAL, checkpoint, schedule
from .study import Radon, Record, Run, project, tangent

WORKER: Record = {}


def initialize_worker(args: argparse.Namespace, slots: Any) -> None:
    slot, gpu = slots.get()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    report = json.loads((args.out / "results.json").read_text())
    settings = report["settings"]
    torch.manual_seed(settings["seed"])
    model, _ = load_flow_model(64, epoch=100, run_dir=Path(settings["checkpoint"]).parents[1], device=device)
    inputs = torch.load(args.out / "inputs.pt", map_location="cpu", weights_only=True)
    operator = Radon()
    operator.grid = operator.grid.to(device)
    problems = {}
    for split, rows in (("validation", [inputs["validation"]]), ("test", inputs["test"])):
        problems[split] = [CanonicalProblem(model, row["truth"].to(device), row["source"].to(device), operator,
            row["clean"].to(device), row["noisy"].to(device), settings["seed"] + (20000 if split == "validation" else 21000) + i)
            for i, row in enumerate(rows)]
    interrupted = [False]

    def stop(signum: int, frame: Any) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    WORKER.update(args=args, slot=slot, gpu=gpu, device=device, problems=problems, interrupted=interrupted)
    (args.out / "workers" / f"{slot}.json").write_text(json.dumps({
        "pid": os.getpid(), "gpu": gpu, "cpu_threads": 1, "pool_token": args.pool_token,
        "started_at": datetime.now().astimezone().isoformat()}))
    print(f"Worker {slot}: GPU {gpu}, PID {os.getpid()}", flush=True)


def run_chunk(section: str, config: Record) -> Record:
    args = argparse.Namespace(**vars(WORKER["args"]))
    args.device, args.render_reports, args.one_chunk = WORKER["device"], False, True
    args.out = args.out / "shards" / config["label"]
    problems = WORKER["problems"]["validation" if section == "tuning" else "test"]
    source = problems[config["sample"]].source
    run = Run(**config, z=(project(source) if config["projected"] else source).cpu().clone())
    path = args.out / "results.json"
    report = json.loads(path.read_text()) if path.exists() else {"failures": []}
    try:
        schedule([run], problems, section, report, args, WORKER["interrupted"])
    except KeyboardInterrupt:
        pass  # schedule saved the last completed update before yielding.
    if run.status in TERMINAL and section != "tuning" and run.history:
        problem = problems[run.sample]
        z = run.z.to(args.device)
        x, loss, exact = problem.evaluate(z, "exact")
        norm = float(tangent(exact, z, run.projected).norm())
        initial_norm = run.history[1]["tangent_gradient_norm"]
        report["terminal_check"] = {"exact_tangent_gradient_norm": norm,
            "ratio_to_initial_estimator_gradient_norm": norm / initial_norm if initial_norm else None,
            "native_objective": loss, **problem.metrics(x)}
    report.update(gpu=WORKER["gpu"], slot=WORKER["slot"])
    checkpoint(args.out, report)
    return report


def diagnostic_point(entry: Record) -> Record:
    out = WORKER["args"].out
    saved = torch.load(out / "states" / f"{entry['source_run']}.pt", map_location="cpu", weights_only=True)
    sources = {**saved["snapshots"], saved["run"]["attempts"]: saved["source"]}
    diagnose(WORKER["problems"]["test"][0], sources[entry["iteration"]].to(WORKER["device"]),
             entry, math.inf, entry["projected"], product_batch_size=17,
             spectral_seconds=1800, max_refinements=12, refine_cutoff=True)
    entry.pop("seconds", None)
    entry.update(status="complete", gpu=WORKER["gpu"])
    path = out / "shards" / f"diagnostic_{entry['projected']}_{entry['iteration']}.json"
    path.with_suffix(".tmp").write_text(json.dumps(entry, indent=2, allow_nan=False))
    path.with_suffix(".tmp").replace(path)
    return entry


class ParallelStudy:
    def __init__(self, args: argparse.Namespace, report: Record) -> None:
        self.args = args
        self.lock = (args.out / "parallel.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.out / "workers").mkdir(exist_ok=True)
        (args.out / "shards").mkdir(exist_ok=True)
        self.context = multiprocessing.get_context("spawn")
        args.pool_token = uuid.uuid4().hex
        self.slots = self.context.Queue()
        for slot, gpu in enumerate(gpu for gpu in args.gpus for _ in range(args.workers_per_gpu)):
            self.slots.put((slot, gpu))
        self.pool = ProcessPoolExecutor(max_workers=len(args.gpus) * args.workers_per_gpu,
            mp_context=self.context, initializer=initialize_worker, initargs=(args, self.slots))
        self.closed = False
        report["parallel_execution"] = {"gpus": args.gpus, "workers_per_gpu": args.workers_per_gpu,
            "cpu_threads_per_worker": 1, "chunk_updates": 10, "report_writer": "coordinator only",
            "started_at": datetime.now().astimezone().isoformat(),
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        checkpoint(args.out, report)

    def close(self) -> None:
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.slots.close()
            self.lock.close()
            self.closed = True

    def completed(self, jobs: dict[Future, Any], interrupted: list[bool]) -> Iterator[tuple[Any, Record]]:
        pending = set(jobs)
        signaled = False
        while pending:
            if interrupted[0] and not signaled:
                for future in pending:
                    future.cancel()
                for path in (self.args.out / "workers").glob("*.json"):
                    record = json.loads(path.read_text())
                    if record.get("pool_token") != self.args.pool_token:
                        continue
                    try:
                        os.kill(record["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                signaled = True
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                if not future.cancelled():
                    yield jobs[future], future.result()
        if interrupted[0]:
            raise KeyboardInterrupt

    def run(self, groups: dict[str, list[Run]], report: Record, interrupted: list[bool]) -> None:
        labels = [run.label for runs in groups.values() for run in runs]
        if len(labels) != len(set(labels)):
            raise ValueError("Each concurrent run must have a unique checkpoint label")
        for section, runs in groups.items():
            for run in runs:
                shard = self.args.out / "shards" / run.label
                (shard / "states").mkdir(parents=True, exist_ok=True)
                local = shard / "states" / f"{run.label}.pt"
                canonical = self.args.out / "states" / f"{run.label}.pt"
                if not local.exists() and canonical.exists():
                    shutil.copy2(canonical, local)
                if local.exists():
                    saved = torch.load(local, map_location="cpu", weights_only=True)
                    for key, value in saved["run"].items():
                        setattr(run, key, value)
                    run.z = saved["source"]
                    # A worker may have saved its final chunk just before the coordinator stopped.
                    shutil.copy2(local, canonical.with_suffix(".tmp"))
                    canonical.with_suffix(".tmp").replace(canonical)
                    if (shard / "results.json").exists():
                        recovered = json.loads((shard / "results.json").read_text())
                        if "terminal_check" in recovered:
                            report.setdefault("terminal_checks", {})[run.label] = recovered["terminal_check"]
                        report.setdefault("stopping_evidence", {}).update(recovered.get("stopping_evidence", {}))
                        for failure in recovered.get("failures", []):
                            if failure not in report["failures"]:
                                report["failures"].append(failure)
                report[section] = [r.public() for r in runs]
        checkpoint(self.args.out, report)
        while any(run.status not in TERMINAL for runs in groups.values() for run in runs):
            jobs = {self.pool.submit(run_chunk, section, {
                key: getattr(run, key) for key in ("label", "estimator", "optimizer", "projected", "lr", "rank", "sample")}): (section, run)
                for section, runs in groups.items() for run in runs if run.status not in TERMINAL}
            for (section, run), result in self.completed(jobs, interrupted):
                for key, value in result[section][0].items():
                    setattr(run, key, value)
                local = self.args.out / "shards" / run.label / "states" / f"{run.label}.pt"
                canonical = self.args.out / "states" / f"{run.label}.pt"
                shutil.copy2(local, canonical.with_suffix(".tmp"))
                canonical.with_suffix(".tmp").replace(canonical)
                report[section] = [r.public() for r in groups[section]]
                report.setdefault("stopping_evidence", {}).update(result.get("stopping_evidence", {}))
                for failure in result.get("failures", []):
                    if failure not in report["failures"]:
                        report["failures"].append(failure)
                if "terminal_check" in result:
                    report.setdefault("terminal_checks", {})[run.label] = result["terminal_check"]
                report.setdefault("run_gpus", {}).setdefault(run.label, [])
                if result["gpu"] not in report["run_gpus"][run.label]:
                    report["run_gpus"][run.label].append(result["gpu"])
                checkpoint(self.args.out, report)
            from .gpu_report import render
            render(self.args.out)

    def diagnostics(self, entries: list[Record], report: Record, interrupted: list[bool]) -> None:
        jobs = {self.pool.submit(diagnostic_point, entry): entry for entry in entries}
        for _, entry in self.completed(jobs, interrupted):
            report.setdefault("diagnostics", []).append(entry)
            checkpoint(self.args.out, report)

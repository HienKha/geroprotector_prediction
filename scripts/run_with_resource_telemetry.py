#!/usr/bin/env python3
"""Run one immutable telemetry reproduction and record resource use.

This wrapper does not infer historical resource use.  It measures a new, explicitly
labelled telemetry rerun.  GPU figures are sampled once per second.  Process-specific
GPU memory is preferred; whole-device memory is retained as an auditable fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def nvidia_query(fields: str, query_type: str = "gpu") -> list[str]:
    cmd = ["nvidia-smi", f"--query-{query_type}={fields}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_sample() -> dict:
    device_rows = nvidia_query("index,name,driver_version,memory.total,memory.used,utilization.gpu")
    process_rows = nvidia_query("pid,used_memory", "compute-apps")
    process_mib = 0
    for row in process_rows:
        try:
            process_mib += int(row.rsplit(",", 1)[1].strip())
        except (IndexError, ValueError):
            continue
    used_mib = None
    if device_rows:
        try:
            used_mib = int(device_rows[0].split(",")[-2].strip())
        except (IndexError, ValueError):
            pass
    return {
        "device_rows": device_rows,
        "compute_process_rows": process_rows,
        "device_memory_used_mib": used_mib,
        "compute_process_memory_mib": process_mib,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--expected-output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.telemetry_dir.exists():
        raise SystemExit(f"Refusing to overwrite telemetry directory: {args.telemetry_dir}")
    if args.expected_output.exists():
        raise SystemExit(f"Refusing to overwrite scientific output: {args.expected_output}")

    args.telemetry_dir.mkdir(parents=True)
    stdout_path = args.telemetry_dir / "run.log"
    samples_path = args.telemetry_dir / "gpu_samples.jsonl"
    result_path = args.telemetry_dir / "RESOURCE_TELEMETRY.json"
    start_utc = utc_now()
    baseline = gpu_sample()
    start = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)

    stop = threading.Event()
    peaks = {"device_memory_used_mib": 0, "compute_process_memory_mib": 0,
             "gpu_utilization_percent": 0}

    def monitor() -> None:
        with samples_path.open("w", encoding="utf-8") as handle:
            while not stop.is_set():
                sample = gpu_sample()
                sample["captured_at_utc"] = utc_now()
                for key in ("device_memory_used_mib", "compute_process_memory_mib"):
                    value = sample.get(key)
                    if isinstance(value, int):
                        peaks[key] = max(peaks[key], value)
                if sample["device_rows"]:
                    try:
                        util = int(sample["device_rows"][0].split(",")[-1].strip())
                        peaks["gpu_utilization_percent"] = max(
                            peaks["gpu_utilization_percent"], util)
                    except (IndexError, ValueError):
                        pass
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
                stop.wait(1.0)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    with stdout_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at_utc: {start_utc}\n")
        log.write(f"command: {command_text(command)}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        return_code = process.wait()
    stop.set()
    watcher.join(timeout=5)

    end_utc = utc_now()
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    # Linux reports ru_maxrss in KiB.  The wrapper launches no child before the
    # scientific command except short nvidia-smi probes, whose RSS is negligible;
    # the value is therefore labelled as the child-process maximum for this rerun.
    peak_rss_kib = max(before.ru_maxrss, after.ru_maxrss)
    baseline_used = baseline.get("device_memory_used_mib")
    incremental_gpu = None
    if isinstance(baseline_used, int):
        incremental_gpu = max(0, peaks["device_memory_used_mib"] - baseline_used)

    record = {
        "schema_version": "geroprotector.resource_telemetry.v1",
        "scientific_role": "telemetry_rerun_not_historical_runtime_reconstruction",
        "family": args.family,
        "command": command,
        "command_shell_escaped": command_text(command),
        "started_at_utc": start_utc,
        "finished_at_utc": end_utc,
        "wall_clock_seconds": elapsed,
        "return_code": return_code,
        "expected_output": str(args.expected_output),
        "expected_output_exists": args.expected_output.exists(),
        "peak_child_rss_kib": peak_rss_kib,
        "peak_child_rss_gib": peak_rss_kib / 1024 / 1024,
        "gpu_sampling_interval_seconds": 1.0,
        "gpu_exclusivity_contract": "serialized_by_flock_across_project_telemetry_reruns",
        "gpu_baseline": baseline,
        "peak_whole_device_memory_used_mib": peaks["device_memory_used_mib"],
        "peak_incremental_device_memory_mib": incremental_gpu,
        "peak_compute_process_memory_mib": peaks["compute_process_memory_mib"],
        "peak_gpu_utilization_percent": peaks["gpu_utilization_percent"],
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "stdout_log": str(stdout_path),
        "gpu_samples": str(samples_path),
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    hashes = {
        "RESOURCE_TELEMETRY.json": sha256(result_path),
        "gpu_samples.jsonl": sha256(samples_path),
        "run.log": sha256(stdout_path),
    }
    (args.telemetry_dir / "TELEMETRY_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

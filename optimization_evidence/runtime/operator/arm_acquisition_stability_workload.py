"""Observe the actual public submission and verifier workload for twelve hours."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def descriptor(path):
    return {"path": str(path.relative_to(Path("/evidence"))),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=43200)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.seconds < 43200 or args.interval <= 0:
        raise ValueError("Twelve actual hours are required")
    root = Path("/evidence")
    cycles_root = root / "cycles"
    cycles_root.mkdir(exist_ok=True)
    cycles = []
    started = time.monotonic()
    artifact_bytes = 0
    log_bytes = 0

    def sample(elapsed):
        cgroup = Path("/sys/fs/cgroup")
        events = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
        value = {"elapsed_seconds": elapsed,
                 "memory_bytes": int((cgroup / "memory.current").read_text()),
                 "file_descriptors": len(list(Path("/proc/self/fd").iterdir())),
                 "disk_free_bytes": shutil.disk_usage("/harbor_task").free,
                 "log_bytes": log_bytes, "artifact_bytes": artifact_bytes,
                 "oom_kill": int(events["oom_kill"])}
        with (root / "telemetry.jsonl").open("a") as log:
            log.write(json.dumps(value) + "\n")
        if value["oom_kill"] or not value["disk_free_bytes"]:
            raise RuntimeError("Stability telemetry records resource exhaustion")

    sample(0)
    argv = ["/bin/sh", "-c",
            "./solution/solve.sh --output solution/artifacts/stability-current && /tests/test.sh"]
    try:
        while True:
            start = time.monotonic() - started
            index = len(cycles) + 1
            prefix = cycles_root / f"cycle-{index:04d}"
            log_path = prefix.with_suffix(".log")
            with log_path.open("wb") as log:
                result = subprocess.run(argv, cwd="/harbor_task", stdout=log,
                                        stderr=subprocess.STDOUT, timeout=120)
                log.write((json.dumps({"argv": argv, "exit_code": result.returncode}) + "\n").encode())
            if result.returncode:
                raise RuntimeError(f"Public workload cycle {index} exited {result.returncode}")
            artifacts = []
            for name, source in (("iteration", "stability-current"), ("verified", "verified")):
                value = json.loads(Path(f"/harbor_task/solution/artifacts/{source}/reward.json").read_text())
                expected = json.loads((root / "binding.json").read_text())["expected_score"]
                if value["score"] != expected or value["new_training_runs"] != 0:
                    raise ValueError("Cached workload changed score or started training")
                destination = prefix.with_name(prefix.name + "-" + name + ".json")
                write(destination, value)
                artifacts.append(descriptor(destination))
                artifact_bytes += destination.stat().st_size
            execution = prefix.with_suffix(".json")
            write(execution, {"argv": argv, "exit_code": result.returncode,
                              "command_elapsed_seconds": time.monotonic() - started - start,
                              "log": descriptor(log_path), "artifacts": artifacts})
            artifact_bytes += execution.stat().st_size
            log_bytes += log_path.stat().st_size
            end = time.monotonic() - started
            cycles.append({"start": start, "end": end, "execution": descriptor(execution)})
            write(root / "cycles.json", cycles)
            sample(end)
            write(root / "workload-progress.json", {"status": "running", "elapsed_seconds": end,
                "required_seconds": args.seconds, "completed_cycles": index, "new_training_runs": 0})
            print(json.dumps({"cycle": index, "elapsed_seconds": end, "score": expected}), flush=True)
            if end >= args.seconds:
                write(root / "workload-result.json", {"status": "passed", "elapsed_seconds": end,
                    "completed_cycles": index, "interrupted": False, "errors": [], "new_training_runs": 0})
                return
            time.sleep(min(args.interval, max(0, args.seconds - end)))
    except Exception as exc:
        write(root / "workload-result.json", {"status": "failed", "elapsed_seconds": time.monotonic() - started,
            "completed_cycles": len(cycles), "interrupted": False, "errors": [str(exc)]})
        raise


if __name__ == "__main__":
    main()

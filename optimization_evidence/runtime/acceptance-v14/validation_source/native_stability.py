"""Monitor an exact native container pair for the required twelve-hour window."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
COMPOSE = HERE.parent / "delivery/workspace/harbor_task/environment/docker-compose.yaml"
DIAGNOSTIC = """
import json
import shutil
from pathlib import Path
counts = {}
for path in Path('/proc').glob('[0-9]*/fd'):
    try:
        counts[path.parent.name] = len(list(path.iterdir()))
    except FileNotFoundError:
        continue
disk = shutil.disk_usage('/')
print(json.dumps({'fd_count': max(counts.values(), default=0),
                  'total_fd_count': sum(counts.values()), 'process_fd_counts': counts,
                  'disk_free_bytes': disk.free,
                  'shm_used_bytes': shutil.disk_usage('/dev/shm').used}))
"""


def docker(*arguments, timeout=60):
    result = subprocess.run([sys.executable, str(HERE / "docker_vm.py"), *arguments],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1500:])
    return result.stdout.decode()


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


parser = argparse.ArgumentParser()
parser.add_argument("--project", default="arm-native-v14")
parser.add_argument("--seconds", type=float, default=43200)
args = parser.parse_args()
output = HERE / (args.project + "-stability")
output.mkdir(exist_ok=True)
ids = {}
for service in ("main", "evaluator"):
    ids[service] = docker("compose", "-p", args.project, "-f", str(COMPOSE),
                          "ps", "-q", service).strip()
    if not ids[service]:
        raise RuntimeError("Required service is not running: " + service)
initial = json.loads(docker("inspect", *ids.values()))
images = {row["Id"]: row["Image"] for row in initial}
(output / "initial-inspect.json").write_text(json.dumps(initial, indent=2) + "\n")
started = time.monotonic()
summary = {"status": "running", "started_at": timestamp(), "required_seconds": args.seconds,
           "container_ids": ids, "images": images, "cycles": 0, "errors": [],
           "public_baseline_checks": 0,
           "monitor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
try:
    with (output / "samples.jsonl").open("a") as samples:
        while True:
            inspected = json.loads(docker("inspect", *ids.values()))
            rows = []
            for row in inspected:
                state, limits = row["State"], row["HostConfig"]
                assert state["Running"] and not state["OOMKilled"], state
                assert row["RestartCount"] == 0
                assert row["Image"] == images[row["Id"]]
                assert not row["Mounts"] and limits["NetworkMode"] == "none"
                assert not limits.get("DeviceRequests") and not limits.get("Devices")
                assert state.get("Health", {}).get("Status", "healthy") == "healthy"
                diagnostic = json.loads(docker("exec", row["Id"], "python3", "-c", DIAGNOSTIC))
                assert diagnostic["fd_count"] < 256, diagnostic
                assert diagnostic["total_fd_count"] < 1024, diagnostic
                assert diagnostic["disk_free_bytes"] > 2 * 1024**3, diagnostic
                assert diagnostic["shm_used_bytes"] < 48 * 1024**2, diagnostic
                rows.append({"id": row["Id"], "image": row["Image"], "state": state,
                             "nano_cpus": limits["NanoCpus"], "memory": limits["Memory"],
                             "logging": limits["LogConfig"], **diagnostic})
            assert sum(row["nano_cpus"] for row in rows) <= 8 * 10**9
            assert sum(row["memory"] for row in rows) == 36 * 1024**3
            stats = [json.loads(line) for line in docker("stats", "--no-stream", "--format", "{{json .}}",
                                                        *ids.values()).splitlines()]
            pending = int(docker("exec", ids["main"], "python3", "-c",
                "from pathlib import Path; print(len(list(Path('/opt/arm-evaluator/inbox').glob('*.json'))))").strip())
            # A full scientific evaluation already occupies the single slot.
            # Probe the public scoring path only while that slot is idle.
            if not pending:
                docker("exec", ids["main"], "python3", "/harbor_task/tests/grader.py", "--baseline",
                       "--output", "/harbor_task/solution/artifacts/stability-baseline", timeout=90)
                result = json.loads(docker("exec", ids["main"], "cat",
                    "/harbor_task/solution/artifacts/stability-baseline/reward.json"))
                assert result["status"] == "succeeded" and result["score"] == 0
                assert result["new_training_runs"] == 0
                summary["public_baseline_checks"] += 1
            elapsed = time.monotonic() - started
            samples.write(json.dumps({"at": timestamp(), "elapsed_seconds": elapsed,
                                      "containers": rows, "stats": stats,
                                      "pending_evaluations": pending}) + "\n")
            samples.flush()
            summary.update(cycles=summary["cycles"] + 1, elapsed_seconds=elapsed,
                           last_sample_at=timestamp())
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            if elapsed >= args.seconds:
                summary["status"] = "passed"
                break
            time.sleep(min(60, args.seconds - elapsed))
except Exception as error:
    summary["status"] = "failed"
    summary["errors"].append(str(error))
finally:
    summary.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - started)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
raise SystemExit(0 if summary["status"] == "passed" else 2)

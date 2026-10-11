"""Build both native services offline from a recorded, unchanged source snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
delivery = here.parent / "delivery"
compose = delivery / "workspace/harbor_task/environment/docker-compose.yaml"
parser = argparse.ArgumentParser()
parser.add_argument("--report-prefix", default="native-build")
args, build_arguments = parser.parse_known_args()
if Path(args.report_prefix).name != args.report_prefix:
    raise ValueError("Build report prefix must be a filename")
def snapshot():
    paths = [compose]
    for root in (delivery / "workspace/harbor_task",
                 delivery / "optimization_evidence/runtime/native"):
        paths.extend(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    result = {}
    for path in sorted(set(paths)):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                digest.update(block)
        result[str(path.relative_to(delivery))] = {"bytes": path.stat().st_size,
                                                 "sha256": digest.hexdigest()}
    return result

sources = snapshot()
command = [sys.executable, str(here / "docker_vm.py"), "compose", "-p", "arm-native-v14",
           "-f", str(compose), "build", *build_arguments]
started = time.monotonic()
with (here / (args.report_prefix + ".log")).open("wb") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
report = {"status": "passed" if not result.returncode else "failed",
          "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - started,
          "no_cache": "--no-cache" in build_arguments, "build_network": "none",
          "job_id": os.environ["SLURM_JOB_ID"], "source_at_start": sources}
if snapshot() != sources:
    report["status"] = "stale"
    report["reason"] = "Image inputs changed while this build was in progress"
(here / (args.report_prefix + ".json")).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "source_at_start"}))
raise SystemExit(result.returncode if report["status"] != "stale" else 3)

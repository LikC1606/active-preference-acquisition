"""Run Docker commands on this allocation, never on a stale local port."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

root = Path(__file__).resolve().parents[4] / "tmp/harborbuilder-kvm-test"
status = json.loads((root / f"services/job-{os.environ['SLURM_JOB_ID']}/status.json").read_text())
assert status["status"] == "running" and status["node"] == socket.gethostname()
tls = root / "deployment/tls"
command = [str(root / "deployment/docker"), "--host", f"tcp://127.0.0.1:{status['docker_api_port']}",
           "--tlsverify", "--tlscacert", str(tls / "ca.pem"),
           "--tlscert", str(tls / "client-cert.pem"), "--tlskey", str(tls / "client-key.pem")]
raise SystemExit(subprocess.run([*command, *sys.argv[1:]]).returncode)

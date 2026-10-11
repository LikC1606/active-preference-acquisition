"""Durable process-backed stability capture of the current public task snapshot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
PROJECT = WORKSPACE / "papers/active-reward-modeling"
PROOF = PROJECT / "acquisition-paper-aligned"
TASK = PROJECT / "harbor_task_acquisition"
IMAGE = "arm-acquisition-paper-aligned:prepared"
sys.path.insert(0, str(PROOF / "private"))
from broker import AcquisitionBroker
from evaluate import read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-method", action="store_true")
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    if int(os.environ["SLURM_CPUS_PER_TASK"]) != 8:
        raise ValueError("This capture requires exactly the declared 8 CPU, zero GPU resources")
    output = PROOF / "stability" / ("job-" + job)
    output.mkdir(mode=0o700, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    workload_source = output / "workload.py"
    shutil.copy2(ROOT / "arm_acquisition_stability_workload.py", workload_source)
    snapshot = output / "task"
    snapshot.mkdir()
    for source in sorted(TASK.rglob("*")):
        relative = source.relative_to(TASK)
        if "__pycache__" in relative.parts or relative.parts[:2] == ("solution", "artifacts"):
            continue
        if source.is_symlink():
            raise ValueError("Public snapshot contains a link")
        if source.is_file():
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".npy":
                os.link(source, destination)
            else:
                shutil.copy2(source, destination)
    definition = read(PROOF / "definition.json")
    selector_sha = definition["starter_sha256"]
    expected_score = 0.0
    scope = "Current public Starter submission and verifier using completed five-seed caches"
    if args.best_method:
        replay = read(PROOF / "best-replay-report.json")
        if replay["status"] != "best_replay_passed" or replay["new_training_runs"]:
            raise ValueError("Final stability requires successful native best replay")
        best = PROOF / "delivery/expert_evidence/best_method"
        for name in ("select.py", "solve.sh", "submission.json"):
            shutil.copy2(best / name, snapshot / "solution" / name)
        result = replay["trials"][replay["best_lane"]]
        selector_sha, expected_score = result["selector_sha256"], result["score"]
        if replay["image_id"] != read(PROOF / "native-report.json")["image_id"]:
            raise ValueError("Replay and final stability public image differ")
        scope = "Final best submission and verifier using completed five-seed caches"
    if sha(snapshot / "solution/select.py") != selector_sha:
        raise ValueError("Stability snapshot differs from its successful source")
    task_files = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    resources = tomllib.loads((snapshot / "task.toml").read_text())["metadata"]
    binding = {"task_files": task_files, "image_id": read(PROOF / "native-report.json")["image_id"],
               "resources": resources, "selector_sha256": selector_sha,
               "protocol_sha256": definition["protocol_sha256"], "expected_score": expected_score,
               "scope": scope,
               "workload_sha256": sha(workload_source), "runner_sha256": sha(Path(__file__)),
               "final_best_method_bound": args.best_method, "formal_platform_acceptance": False}
    write(evidence / "binding.json", binding)
    config = output / "docker-config"
    shutil.copytree(PROJECT / ".deepscientist/harborbuilder/sessions/hb-arm-paper-v4-3/.docker-runtime/.docker", config)
    os.environ["DOCKER_CONFIG"] = str(config)
    os.environ["DOCKER_CONTEXT"] = "harborbuilder-vm"
    prefix = [str(ROOT / "deployment/docker"), "--config", str(config)]

    def docker(*parts, timeout=30, **options):
        return subprocess.run([*prefix, *map(str, parts)], check=True, timeout=timeout, **options)

    def progress(status, **extra):
        value = {"status": status, "job_id": job, "node": socket.gethostname(),
                 "allocated_cpus": 8, "gpus": 0, "required_seconds": 43200,
                 "new_model_requests": 0, "new_training_runs": 0,
                 "final_best_method_bound": args.best_method, **extra}
        write(output / "progress.json", value)
        write(PROOF / ("stability/best-progress.json" if args.best_method else "stability/progress.json"), value)

    progress("starting")
    log = (output / "vm.log").open("wb")
    service = subprocess.Popen([str(ROOT / "engine-unit-venv/bin/python"), str(ROOT / "run_vm_service.py"),
        "--cpus", "8", "--memory-mib", "40960", "--disk-gib", "96", "--shared-root", str(WORKSPACE),
        "--paper-sandbox", "--docker-api-port", "23767"], stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "TMPDIR": "/tmp"})
    broker = None
    container = None
    workload = None
    try:
        began = time.monotonic()
        status_path = ROOT / f"services/job-{job}/status.json"
        while True:
            status = read(status_path) if status_path.exists() else {}
            if status.get("status") == "running":
                break
            if service.poll() is not None or time.monotonic() - began > 210:
                raise RuntimeError("Stability VM startup failed; inspect vm.log")
            time.sleep(1)
        if status["node"] != socket.gethostname() or status["vcpus"] != 8 or status["gpu_count"]:
            raise ValueError("Unexpected stability VM resources")
        tls = ROOT / "deployment/tls"
        endpoint = ",".join((f"host=tcp://127.0.0.1:{status['docker_api_port']}",
                            f"ca={tls/'ca.pem'}", f"cert={tls/'client-cert.pem'}", f"key={tls/'client-key.pem'}"))
        docker("context", "update", "harborbuilder-vm", "--docker", endpoint, capture_output=True)
        with (output / "image-load.log").open("wb") as load_log:
            docker("image", "load", "--input", PROOF / "stability/public-image.tar", timeout=1800,
                   stdout=load_log, stderr=subprocess.STDOUT)
        observed_image = json.loads(docker("image", "inspect", IMAGE, capture_output=True, text=True).stdout)[0]
        if observed_image["Id"] != binding["image_id"]:
            raise ValueError("Stability public image differs from native validation")
        channel = output / "channel"
        broker = AcquisitionBroker(channel, cached_only=True)
        broker.start()
        sandbox = read(ROOT / "deployment/arm-paper-runtime/sandbox.json")
        command = ["create", "--init", "--pull", "never", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--security-opt", "seccomp=" + sandbox["seccomp_path"],
            "--security-opt", "apparmor=" + sandbox["apparmor_profile"], "--security-opt", "systempaths=unconfined",
            "--cpus", "8", "--memory", "36864m", "--memory-swap", "36864m", "--pids-limit", "128",
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/harbor_task",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
            "--volume", f"{snapshot/'solution'}:/harbor_task/solution:rw",
            "--volume", f"{channel/'inbox'}:/opt/arm-evaluator/inbox:rw",
            "--volume", f"{channel/'outbox'}:/opt/arm-evaluator/outbox:ro",
            "--volume", f"{evidence}:/evidence:rw",
            "--volume", f"{workload_source}:/workload.py:ro",
            "--log-opt", "max-size=10m", "--log-opt", "max-file=2", IMAGE,
            "python3", "/workload.py", "--seconds", "43200", "--interval", "60"]
        container = docker(*command, capture_output=True, text=True).stdout.strip()
        with (evidence / "workload.log").open("wb") as workload_log:
            workload = subprocess.Popen([*prefix, "start", "--attach", container],
                                        stdout=workload_log, stderr=subprocess.STDOUT)
            while workload.poll() is None:
                if service.poll() is not None:
                    raise RuntimeError("Stability VM stopped unexpectedly")
                sample = evidence / "workload-progress.json"
                measurements = read(sample) if sample.exists() else {}
                measurements.pop("status", None)
                measurements.pop("new_training_runs", None)
                measurements.pop("required_seconds", None)
                progress("running" if sample.exists() else "workload_starting",
                         **measurements)
                time.sleep(5)
        if workload.returncode:
            raise RuntimeError("Stability workload failed; inspect evidence/workload.log")
        result = read(evidence / "workload-result.json")
        inspection = json.loads(docker("inspect", container, capture_output=True, text=True).stdout)[0]
        write(evidence / "container-inspect.json", inspection)
        state, host = inspection["State"], inspection["HostConfig"]
        if (result["status"] != "passed" or result["elapsed_seconds"] < 43200 or state["Running"]
                or state["ExitCode"] or state["OOMKilled"] or inspection["Image"] != binding["image_id"]
                or host["NanoCpus"] != 8 * 10**9 or host["Memory"] != 36864 * 2**20
                or host["NetworkMode"] != "none" or host.get("DeviceRequests") or host.get("Privileged")):
            raise ValueError("Stability lifecycle or resource validation failed")
        for name, identity in task_files.items():
            if sha(snapshot / name) != identity:
                raise ValueError("Bound task file changed: " + name)
        descriptor = lambda name: {"path": name, "sha256": sha(evidence / name)}
        report = {"schema": "deepscientist.harborbuilder.stability.v1", **binding, **result,
                  "container_id": container, "container_inspect": descriptor("container-inspect.json"),
                  "log": descriptor("workload.log"), "telemetry": descriptor("telemetry.jsonl"),
                  "cycles": descriptor("cycles.json"), "job_id": job}
        write(evidence / "stability-report.json", report)
        progress("snapshot_stability_passed", elapsed_seconds=result["elapsed_seconds"],
                 completed_cycles=result["completed_cycles"], final_best_method_bound=args.best_method,
                 formal_platform_acceptance=False)
        print(json.dumps({"job_id": job, "status": "snapshot_stability_passed", **result}), flush=True)
    except Exception as exc:
        progress("failed", error=str(exc)[:1000])
        if container:
            try:
                write(evidence / "container-inspect.json",
                      json.loads(docker("inspect", container, capture_output=True, text=True).stdout)[0])
            except Exception:
                pass
        raise
    finally:
        if container:
            subprocess.run([*prefix, "rm", "--force", "--volumes", container], capture_output=True, timeout=30)
        if workload is not None:
            try:
                workload.wait(timeout=10)
            except subprocess.TimeoutExpired:
                workload.kill()
                workload.wait()
        if broker:
            broker.close()
        service.terminate()
        try:
            service.wait(timeout=20)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()
        log.close()

    if args.best_method:
        subprocess.run([sys.executable, str(ROOT / "prepare_arm_acquisition_delivery.py"), "--research"],
                       check=True, timeout=180)
        subprocess.run([sys.executable, str(ROOT / "package_arm_acquisition.py"), "--output",
                        str(PROOF / "submission-final.zip")], check=True, timeout=900)


if __name__ == "__main__":
    main()

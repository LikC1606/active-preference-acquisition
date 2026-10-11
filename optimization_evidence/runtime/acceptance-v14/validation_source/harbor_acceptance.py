"""Fresh five-seed acceptance using Harbor's separate verifier and artifact APIs."""
import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent / "delivery"
TASK = DELIVERY / "workspace/harbor_task"
WORKSPACE = HERE.parents[3]
RUNTIME = WORKSPACE / "tmp/harborbuilder-kvm-test"
SOURCES = {
    "baseline": TASK / "environment/starter/select.py",
    "reference": DELIVERY / "optimization_evidence/reference_runs/select.py",
    "best": DELIVERY / "expert_evidence/best_method/select.py",
}
EXPECTED = {"baseline": 0.0, "reference": 0.2670160049935782,
            "best": 0.2513644502120524}


def docker(*arguments):
    result = subprocess.run([sys.executable, str(HERE / "docker_vm.py"), *arguments],
                            capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-2000:])
    return result.stdout.decode()


def write_summary(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def configure_client():
    status = json.loads((RUNTIME / ("services/job-" + os.environ["SLURM_JOB_ID"])
                         / "status.json").read_text())
    assert status["status"] == "running" and status["node"] == socket.gethostname()
    os.environ["PATH"] = str(RUNTIME / "deployment") + os.pathsep + os.environ["PATH"]
    os.environ["DOCKER_HOST"] = "tcp://127.0.0.1:%s" % status["docker_api_port"]
    os.environ["DOCKER_TLS_VERIFY"] = "1"
    client = HERE / "harbor-docker-client-tls"
    client.mkdir(exist_ok=True)
    for name, source in (("ca.pem", "ca.pem"), ("cert.pem", "client-cert.pem"),
                         ("key.pem", "client-key.pem")):
        link = client / name
        wanted = RUNTIME / "deployment/tls" / source
        if not link.is_symlink() or link.resolve() != wanted.resolve():
            link.unlink(missing_ok=True)
            link.symlink_to(wanted)
    os.environ["DOCKER_CERT_PATH"] = str(client)


async def run(args):
    evaluation_deadline = time.monotonic() + min(7500, getattr(args, "verifier_budget_seconds", 7500))
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import NetworkMode, NetworkPolicy
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
    from harbor.trial.artifact_handler import ArtifactHandler
    from harbor.verifier.verifier import Verifier

    output = Path(getattr(args, "output_root", None) or HERE) / (args.project + "-acceptance") / args.arm
    output.mkdir(parents=True, exist_ok=False)
    source = Path(getattr(args, "source", None) or SOURCES[args.arm])
    expected_score = getattr(args, "expected_score", None)
    if expected_score is None and not getattr(args, "source", None):
        expected_score = EXPECTED[args.arm]
    identity = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = hashlib.sha256((DELIVERY / "optimization_evidence/runtime/native/protocol.json")
                              .read_bytes()).hexdigest()
    submitted = output / "artifacts/solution"
    submitted.mkdir(parents=True)
    for name in ("select.py", "solve.sh", "submission.json"):
        shutil.copy2(source if name == "select.py" else TASK / "solution" / name, submitted / name)
    paths = TrialPaths(trial_dir=output / "trial")
    paths.mkdir()
    paths.chmod_dir()
    task = Task(TASK)
    record = {"status": "running", "arm": args.arm, "selector_sha256": identity,
              "protocol_sha256": protocol, "provider": "Harbor DockerEnvironment",
              "custom_provider": False, "verifier": "Harbor Verifier",
              "artifact_transfer": "Harbor ArtifactHandler", "environment_mode": "separate",
              "setup_started_unix": time.time()}
    summary = output / "summary.json"
    environment = None
    ids = {}
    state = "/var/lib/arm/runs/%s/%s" % (protocol, identity)
    try:
        environment = DockerEnvironment(environment_dir=TASK / "tests",
            environment_name="arm-native-v14", session_id=args.project,
            trial_paths=paths, task_env_config=task.config.environment,
            mounts=[{"type": "bind", "source": str(paths.verifier_dir.resolve()),
                     "target": "/logs/verifier"}],
            network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
        await environment.start(force_build=False)
        # The operator's unprivileged VirtioFS daemon can create host log files
        # only as its host UID. This remains a non-root verifier user.
        environment.default_user = os.getuid()
        record["validation_verifier_uid"] = os.getuid()
        for name in ("main", "evaluator"):
            result = await environment._run_docker_compose_command(["ps", "-q", name])
            ids[name] = result.stdout.strip()
        assert all(ids.values())
        first = args.cpu_start
        docker("update", "--cpuset-cpus", "%d-%d" % (first, first + 5), ids["evaluator"])
        docker("update", "--cpuset-cpus", "%d-%d" % (first + 6, first + 7), ids["main"])
        inspected = json.loads(docker("inspect", *ids.values()))
        assert sum(row["HostConfig"]["NanoCpus"] for row in inspected) == 8 * 10**9
        assert sum(row["HostConfig"]["Memory"] for row in inspected) == 36 * 1024**3
        for row in inspected:
            assert row["HostConfig"]["NetworkMode"] == "none"
            assert not row["HostConfig"].get("DeviceRequests")
            for mount in row["Mounts"]:
                assert (row["Id"] == ids["main"] and mount["Destination"] == "/logs/verifier"
                        and Path(mount["Source"]) == paths.verifier_dir.resolve())
        (output / "initial-inspect.json").write_text(json.dumps(inspected, indent=2) + "\n")
        checked = await environment.service_exec("test ! -e " + state, service="evaluator", user="root")
        assert checked.return_code == 0, "Candidate cache must be absent before fresh acceptance"
        handler = ArtifactHandler(artifacts=task.config.artifacts, logger=logging.getLogger(__name__))
        env_paths = EnvironmentPaths.for_os(environment.os)
        await handler.upload_artifacts(environment, output / "artifacts",
            source_artifacts_dir=env_paths.artifacts_dir, target_artifacts_dir=env_paths.artifacts_dir)
        transferred = await environment.exec("sha256sum solution/select.py", cwd="/harbor_task")
        assert transferred.stdout.split()[0] == identity
        writable = await environment.exec(
            "python3 -c 'from pathlib import Path; "
            "p=Path(\"/logs/verifier/write-probe\"); p.write_text(\"ok\"); p.unlink()'")
        assert writable.return_code == 0, {
            "verifier_log_write_exit": writable.return_code,
            "stdout": writable.stdout, "stderr": writable.stderr}
        started = time.monotonic()
        record.update(cache_absent_before_start=True, container_ids=ids, started_unix=time.time())
        record["operator_setup_elapsed_seconds"] = record["started_unix"] - record["setup_started_unix"]
        write_summary(summary, record)
        verifier = Verifier(task=task, trial_paths=paths, environment=environment, skip_tests_upload=True)
        remaining = evaluation_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Evaluation budget expired during environment setup")
        record["verifier_timeout_seconds"] = remaining
        result = await asyncio.wait_for(verifier.verify(), timeout=remaining)
        elapsed = time.monotonic() - started
        reward = json.loads((paths.verifier_dir / "public-score.json").read_text())
        assert reward["status"] == "succeeded" and reward["new_training_runs"] == 5
        assert reward["selector_sha256"] == identity and reward["protocol_sha256"] == protocol
        assert reward["seeds"] == [101, 202, 303, 404, 505]
        assert reward["mean_error"] == math.fsum(reward["raw_values"]) / 5
        assert reward["score"] == (reward["baseline_mean"] - reward["mean_error"]) / reward["baseline_mean"]
        assert result.rewards == {"score": reward["score"]}
        if expected_score is not None:
            assert reward["score"] == expected_score, "Scientific metric differs from the frozen result"
        assert elapsed <= 7200, ("Complete five-seed verifier exceeded the budget", elapsed)
        docker("cp", ids["evaluator"] + ":" + state, str(output / "runs"))
        record.update(status="passed", elapsed_seconds=elapsed, reward=reward,
                      scientific_metric_unchanged=True, within_7200_seconds=True,
                      formal_platform_acceptance=False)
    except Exception as error:
        message = str(error) or repr(error)
        if isinstance(error, TimeoutError):
            message = "Five-seed verifier timed out before the proposal-to-score deadline"
        record.update(status="failed", error_type=type(error).__name__, error=message[-4000:])
        if "started" in locals():
            record["elapsed_seconds"] = time.monotonic() - started
        if ids.get("evaluator"):
            try:
                docker("cp", ids["evaluator"] + ":" + state, str(output / "failed-runs"))
            except Exception as export_error:
                record["export_error"] = str(export_error)[-1500:]
    finally:
        write_summary(summary, record)
        if environment is not None:
            try:
                await environment.stop(delete=True)
            except Exception as error:
                record["cleanup_error"] = str(error)[-1500:]
                write_summary(summary, record)
    print(json.dumps({key: value for key, value in record.items() if key != "reward"}), flush=True)
    return 0 if record["status"] == "passed" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(SOURCES))
    parser.add_argument("--project", required=True)
    parser.add_argument("--cpu-start", type=int, required=True, choices=(0, 8))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expected-score", type=float)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verifier-budget-seconds", type=float, default=7500)
    arguments = parser.parse_args()
    configure_client()
    logging.basicConfig(filename=HERE / (arguments.project + ".log"), level=logging.INFO)
    raise SystemExit(asyncio.run(run(arguments)))

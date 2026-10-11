"""Verify acquisition in native Harbor using only completed five-seed caches."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
PROJECT = WORKSPACE / "papers/active-reward-modeling"
PROOF = PROJECT / "acquisition-paper-aligned"
PRIVATE = PROOF / "private"
TASK = PROJECT / "harbor_task_acquisition"
IMAGE = "arm-acquisition-paper-aligned:prepared"
sys.path.insert(0, str(PRIVATE))
from broker import AcquisitionBroker
from evaluate import IMAGE as ORACLE_IMAGE, ORACLE, PYTHON, SEEDS, create, read, sha, write


def stage_context():
    staging = TASK / "environment/task"
    staging.mkdir(exist_ok=True)
    for source, name in ((TASK / "instruction.md", "instruction.md"),
                         (TASK / "task.toml", "task.toml"),
                         (TASK / "solution/solve.sh", "solve.sh"),
                         (TASK / "solution/submission.json", "submission.json")):
        shutil.copyfile(source, staging / name)
    (staging / "tests").mkdir(exist_ok=True)
    for name in ("grader.py", "test.sh", "acquisition_test_contract.py"):
        shutil.copyfile(TASK / "tests" / name, staging / "tests" / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-lanes", action="store_true")
    parser.add_argument("--rebuild-image", action="store_true")
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    if not 1 <= int(os.environ["SLURM_CPUS_PER_TASK"]) <= 8:
        raise ValueError("CPU allocation exceeds the authorized limit")
    definition = read(PROOF / "definition.json")
    for path, key in ((PROOF / "protocol.json", "protocol_sha256"),
                      (PRIVATE / "worker.py", "worker_sha256"),
                      (PRIVATE / "reference_select.py", "reference_sha256"),
                      (TASK / "environment/starter/select.py", "starter_sha256")):
        if sha(path) != definition[key]:
            raise ValueError("Frozen scientific configuration changed")
    output = PROOF / ("native-" + job)
    output.mkdir(mode=0o700, exist_ok=False)
    if not args.best_lanes or args.rebuild_image:
        stage_context()
    best = {}
    if args.best_lanes:
        for lane in ("gpt", "seed"):
            state_path = PROOF / "agent-research/acquisition-gpt-seed-v1" / lane / "state.json"
            state = read(state_path)
            if state["completed_loops"] < 3 or state["effective_seconds"] < 36000 or state.get("pending_attempt"):
                raise ValueError("Research lane is incomplete: " + lane)
            selector = Path(state["best_selector"])
            run = PROOF / "runs" / definition["protocol_sha256"] / sha(selector)
            for seed in SEEDS:
                receipt = read(run / f"seed_{seed}/evaluation.json")
                if (receipt["status"] != "succeeded" or receipt["protocol_sha256"] != definition["protocol_sha256"]
                        or receipt["selector_sha256"] != sha(selector)
                        or sha(run / f"seed_{seed}/checkpoint") != receipt["checkpoint_sha256"]):
                    raise ValueError("Best lane lacks matching successful checkpoints")
            best[lane] = {"selector": selector, "run": run, "score": state["best_score"]}
    config = output / "docker-config"
    shutil.copytree(PROJECT / ".deepscientist/harborbuilder/sessions/hb-arm-paper-v4-3/.docker-runtime/.docker", config)
    settings = read(config / "config.json") if (config / "config.json").exists() else {}
    settings["cliPluginsExtraDirs"] = [str(ROOT / "deployment/native-tools/cli-plugins")]
    write(config / "config.json", settings)
    os.environ["DOCKER_CONFIG"] = str(config)
    os.environ["PATH"] = str(ROOT / "deployment") + os.pathsep + os.environ["PATH"]
    os.environ["TMPDIR"] = "/tmp"
    docker = [str(ROOT / "deployment/docker"), "--config", str(config)]

    def command(args, timeout=30):
        return subprocess.run([*docker, *args], check=True, capture_output=True, text=True, timeout=timeout)

    vm_log = (output / "vm.log").open("wb")
    service = subprocess.Popen([str(ROOT / "engine-unit-venv/bin/python"), str(ROOT / "run_vm_service.py"),
        "--cpus", "8", "--memory-mib", "40960", "--disk-gib", "96", "--shared-root", str(WORKSPACE),
        "--paper-sandbox", "--reuse-data-disk", str(ROOT / "services/job-114749/data.raw")],
        stdout=vm_log, stderr=subprocess.STDOUT, env={**os.environ, "TMPDIR": "/tmp"})
    containers = set()
    started = time.monotonic()
    try:
        status_file = ROOT / f"services/job-{job}/status.json"
        while True:
            status = read(status_file) if status_file.exists() else {}
            if status.get("status") == "running":
                break
            if service.poll() is not None or time.monotonic() - started > 210:
                raise RuntimeError("VM startup failed; inspect vm.log")
            time.sleep(1)
        if status["node"] != socket.gethostname() or status["vcpus"] != 8 or status["gpu_count"]:
            raise ValueError("Unexpected VM resources")
        tls = ROOT / "deployment/tls"
        endpoint = ",".join((f"host=tcp://127.0.0.1:{status['docker_api_port']}",
                             f"ca={tls/'ca.pem'}", f"cert={tls/'client-cert.pem'}", f"key={tls/'client-key.pem'}"))
        action = "update" if subprocess.run([*docker, "context", "inspect", "harborbuilder-vm"], capture_output=True).returncode == 0 else "create"
        command(["context", action, "harborbuilder-vm", "--docker", endpoint])
        command(["context", "use", "harborbuilder-vm"])
        os.environ["DOCKER_CONTEXT"] = "harborbuilder-vm"
        if not args.best_lanes or args.rebuild_image:
            with (output / "build.log").open("wb") as log:
                subprocess.run([*docker, "build", "--network", "none", "--tag", IMAGE, str(TASK / "environment")],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        image = json.loads(command(["image", "inspect", IMAGE]).stdout)[0]
        if args.best_lanes and not args.rebuild_image and image["Id"] != read(PROOF / "native-report.json")["image_id"]:
            raise ValueError("Best replay must use the validated public image")
        write(output / "image-inspect.json", image)

        reloads = {}
        for lane, candidate in best.items():
            creation = create() + [
                "--workdir", "/workspace", "--volume", f"{ORACLE}:/opt/paper:ro",
                "--volume", f"{TASK/'environment/public_assets'}:/workspace/environment/public_assets:ro",
                "--volume", f"{PROOF/'protocol.json'}:/protocol.json:ro",
                "--volume", f"{candidate['run']}:/checkpoints:ro",
                "--volume", f"{ROOT/'arm_acquisition_reload.py'}:/reload.py:ro",
                ORACLE_IMAGE, PYTHON, "/reload.py"]
            container = command(creation).stdout.strip()
            containers.add(container)
            write(output / (lane + "-reload-inspect.json"), json.loads(command(["inspect", container]).stdout)[0])
            with (output / (lane + "-reload.log")).open("wb") as log:
                subprocess.run([*docker, "start", "--attach", container], check=True,
                               stdout=log, stderr=subprocess.STDOUT, timeout=600)
            rows = [json.loads(line) for line in (output / (lane + "-reload.log")).read_text().splitlines()
                    if line.startswith("{")]
            if len(rows) != len(SEEDS) or {r["seed"] for r in rows} != set(SEEDS) or not all(r["matched"] for r in rows):
                raise ValueError("Fresh checkpoint reload evidence is incomplete")
            reloads[lane] = rows
            command(["rm", "--force", "--volumes", container])
            containers.discard(container)

        from harbor import Trial, TrialConfig
        from harbor.trial.hooks import TrialEvent

        async def run_trial(arm, selector):
            channel = output / (arm + "-channel")
            broker = AcquisitionBroker(channel, cached_only=True)
            broker.start()
            config = TrialConfig.model_validate({
                "task": {"path": str(TASK)}, "trial_name": f"arm-acq-{job}-{arm}",
                "trials_dir": str(output / "trials"), "agent": {"name": "nop"},
                "environment": {"import_path": "arm_acquisition_native:AcquisitionDocker", "delete": False,
                                "override_cpus": 8, "override_memory_mb": 36864, "override_gpus": 0,
                                "kwargs": {"broker_root": str(channel), "keep_containers": True}},
            })
            trial = await Trial.create(config)

            async def inject(_event):
                await trial.agent_environment.upload_file(selector, "/harbor_task/solution/select.py")
                result = await trial.agent_environment.exec(
                    command="cd /harbor_task && ./solution/solve.sh --output solution/artifacts/current",
                    timeout_sec=120)
                if result.return_code:
                    raise RuntimeError("Agent-facing acquisition evaluation failed")
                await trial.agent_environment.download_file(
                    "/harbor_task/solution/artifacts/current/reward.json", output / (arm + "-iteration.json"))

            trial.add_hook(TrialEvent.AGENT_START, inject)
            inspections = {}
            pending = asyncio.create_task(trial.run())
            try:
                while not pending.done():
                    listing = await asyncio.to_thread(command, ["ps", "--all", "--format", "{{.ID}} {{.Names}}"])
                    ids = [line.split()[0] for line in listing.stdout.splitlines() if f"arm-acq-{job}-{arm}" in line]
                    if ids:
                        containers.update(ids)
                        observed = json.loads((await asyncio.to_thread(command, ["inspect", *ids])).stdout)
                        for row in observed:
                            if row["State"]["Running"]:
                                phase = "verifier" if "verifier" in row["Name"] else "agent"
                                inspections[phase] = row
                    await asyncio.sleep(0.2)
                result = await pending
                if result.exception_info:
                    raise RuntimeError("Native Harbor trial failed; inspect trial result.json")
                rewards = read(trial.paths.verifier_dir / "reward.json")
                public = read(trial.paths.verifier_dir / "public-score.json")
                expected = best[arm]["score"] if arm in best else read(PROOF / "comparison.json")[arm]["score"]
                if rewards != {"score": expected} or result.verifier_result.rewards != rewards:
                    raise ValueError("Native Harbor score disagrees with scientific calibration")
                if public["selector_sha256"] != sha(selector) or public["new_training_runs"] != 0:
                    raise ValueError("Native evaluation reused the wrong source or started training")
                if read(output / (arm + "-iteration.json"))["score"] != expected:
                    raise ValueError("Iteration and final evaluation scores disagree")
                if set(inspections) != {"agent", "verifier"} or len(broker.receipts) != 2:
                    raise ValueError("Missing native phases or evaluation requests")
                for row in inspections.values():
                    host = row["HostConfig"]
                    mounts = {m["Destination"]: m for m in row["Mounts"]}
                    if (host["NanoCpus"] != 8 * 10**9 or host["Memory"] != 36864 * 2**20
                            or host["NetworkMode"] != "none" or host.get("DeviceRequests")
                            or row["Image"] != image["Id"]
                            or mounts["/opt/arm-evaluator/outbox"]["RW"]
                            or not mounts["/opt/arm-evaluator/inbox"]["RW"]
                            or any(not target.startswith("/logs/") and target not in {
                                "/opt/arm-evaluator/inbox", "/opt/arm-evaluator/outbox"} for target in mounts)):
                        raise ValueError("Native environment boundary changed")
                write(output / (arm + "-native-inspect.json"), inspections)
                return {"score": expected, "selector_sha256": sha(selector), "agent": "nop",
                        "result_path": str(trial.paths.result_path.relative_to(PROOF)),
                        "iteration_evaluation_passed": True, "separate_verifier_passed": True,
                        "new_training_runs": 0, "requests": len(broker.receipts)}
            finally:
                broker.close()
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

        trials = {}
        arms = ([(lane, candidate["selector"]) for lane, candidate in best.items()] if args.best_lanes else
                [("baseline", TASK / "environment/starter/select.py"), ("reference", PRIVATE / "reference_select.py")])
        if args.best_lanes and args.rebuild_image:
            arms = [("baseline", TASK / "environment/starter/select.py"),
                    ("reference", PRIVATE / "reference_select.py"), *arms]
        for arm, selector in arms:
            trials[arm] = asyncio.run(run_trial(arm, selector))
        report = {"status": "best_replay_passed" if args.best_lanes else "native_acquisition_passed",
                  "job_id": job, "node": socket.gethostname(),
                  "image_id": image["Id"], "protocol_sha256": definition["protocol_sha256"],
                  "resources": {"allocated_cpus": 8, "gpus": 0, "vm_memory_mib": 40960},
                  "trials": trials, "new_training_runs": 0, "elapsed_seconds": time.monotonic() - started,
                  "formal_platform_acceptance": False,
                  "runtime_adapter": "arm_acquisition_native:AcquisitionDocker",
                  "runtime_requirement": "Operator broker and cached dependency image must be supplied",
                  "runner_sha256": sha(Path(__file__)), "broker_sha256": sha(PRIVATE / "broker.py")}
        if args.best_lanes:
            report.update(checkpoint_reloads=reloads, new_model_requests=0,
                          clean_replay_scope="Fresh native agent/verifier containers and fresh checkpoint metric reload; successful training reused",
                          best_lane=max(best, key=lambda lane: best[lane]["score"]),
                          independent_retraining=False)
        write(PROOF / ("best-replay-report.json" if args.best_lanes else "native-report.json"), report)
        if args.best_lanes and args.rebuild_image:
            native = {**report, "status": "native_acquisition_passed",
                      "trials": {arm: trials[arm] for arm in ("baseline", "reference")}}
            write(PROOF / "native-report.json", native)
            with (output / "image-save.log").open("wb") as log:
                subprocess.run([*docker, "image", "save", "--output",
                                str(PROOF / "stability/public-image.tar.tmp"), IMAGE],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            (PROOF / "stability/public-image.tar.tmp").replace(PROOF / "stability/public-image.tar")
        write(output / "report.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)
    except Exception as exc:
        write(output / "failure.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        for container in containers:
            subprocess.run([*docker, "rm", "--force", "--volumes", container], capture_output=True, timeout=20)
        service.terminate()
        try:
            service.wait(timeout=20)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()
        vm_log.close()


if __name__ == "__main__":
    main()

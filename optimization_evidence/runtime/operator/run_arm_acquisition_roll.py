"""Durable Codex/Seed acquisition lanes with bounded model requests."""
import argparse
import ast
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
PROJECT = WORKSPACE / "papers/active-reward-modeling"
PROOF = PROJECT / "acquisition-paper-aligned"
TASK = PROJECT / "harbor_task_acquisition"
PRIVATE = PROOF / "private"
IMAGE = "arm-acquisition-paper-aligned:prepared"
sys.path.insert(0, str(PRIVATE))
sys.path.insert(0, str(WORKSPACE / "harborbuilder/packages/harborbuilder-engine/src"))
from evaluate import evaluate, read, sha, write
from arm_model_proxy import ModelProxy
from deepscientist_harborbuilder_engine.harborbuilder.paper_timing import CodexWorkClock
from deepscientist_harborbuilder_engine.harborbuilder.paper_executor import codex_model_catalog


def parse_candidate(raw_response):
    raw_response = raw_response.strip()
    if raw_response.startswith("```"):
        raw_response = raw_response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(raw_response)
    if (not isinstance(result, dict) or set(result) != {"source", "method_summary"}
            or not isinstance(result["source"], str) or not 0 < len(result["source"].encode()) <= 65536
            or not isinstance(result["method_summary"], str) or len(result["method_summary"]) > 4096):
        raise ValueError("Invalid candidate response contract")
    ast.parse(result["source"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="acquisition-gpt-seed-v1")
    parser.add_argument("--lane", choices=("gpt", "seed", "both"), default="both")
    args = parser.parse_args()
    lanes = ("gpt", "seed") if args.lane == "both" else (args.lane,)
    model_port = 18632 if args.lane == "seed" else 18631
    docker_port = 23766 if args.lane == "seed" else 23765
    disk = ROOT / ("services/arm-seed/data.raw" if args.lane == "seed" else "services/job-114749/data.raw")
    job = os.environ["SLURM_JOB_ID"]
    if not 1 <= int(os.environ["SLURM_CPUS_PER_TASK"]) <= 8:
        raise ValueError("CPU allocation exceeds the authorized limit")
    policy = read(PROOF / "model-request-policy.json")
    run = PROOF / "agent-research" / args.run_id
    run.mkdir(parents=True, mode=0o700, exist_ok=True)
    allocation = run / ("allocation-" + job)
    allocation.mkdir(mode=0o700, exist_ok=False)
    token = secrets.token_hex(24)
    proxy = ModelProxy(WORKSPACE / ".codex/config.toml", WORKSPACE / ".codex/auth.json",
                       policy, allocation / "model-requests.jsonl", token, port=model_port,
                       relay_root=PROOF / "model-relay")
    proxy.start()
    config = allocation / "docker-config"
    shutil.copytree(PROJECT / ".deepscientist/harborbuilder/sessions/hb-arm-paper-v4-3/.docker-runtime/.docker", config)
    prefix = [str(ROOT / "deployment/docker"), "--config", str(config)]
    containers = set()

    def docker(*parts, timeout=30, **options):
        return subprocess.run([*prefix, *map(str, parts)], timeout=timeout, check=True, **options)

    def report(status, **extra):
        progress = run / "progress.json" if args.lane == "both" else run / args.lane / "progress.json"
        progress.parent.mkdir(exist_ok=True)
        write(progress, {"status": status, "run_id": args.run_id, "job_id": job,
            "node": socket.gethostname(), "cpus": 8, "gpus": 0, "policy": policy,
            "lanes": {lane: read(run / lane / "state.json") for lane in lanes
                      if (run / lane / "state.json").exists()}, **extra})

    report("starting")
    log = (allocation / "vm.log").open("wb")
    service = subprocess.Popen([str(ROOT / "engine-unit-venv/bin/python"), str(ROOT / "run_vm_service.py"),
        "--cpus", "8", "--memory-mib", "40960", "--disk-gib", "96", "--shared-root", str(WORKSPACE),
        "--paper-sandbox", "--reuse-data-disk", str(disk), "--docker-api-port", str(docker_port),
        "--model-proxy-port", str(model_port)], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "TMPDIR": "/tmp"})
    try:
        began = time.monotonic()
        status_path = ROOT / f"services/job-{job}/status.json"
        while True:
            status = read(status_path) if status_path.exists() else {}
            if status.get("status") == "running":
                break
            if service.poll() is not None or time.monotonic() - began > 210:
                raise RuntimeError("VM startup failed; inspect allocation vm.log")
            time.sleep(1)
        if status["node"] != socket.gethostname() or status["vcpus"] != 8 or status["gpu_count"]:
            raise ValueError("Unexpected VM resources")
        tls = ROOT / "deployment/tls"
        endpoint = ",".join((f"host=tcp://127.0.0.1:{status['docker_api_port']}",
            f"ca={tls/'ca.pem'}", f"cert={tls/'client-cert.pem'}", f"key={tls/'client-key.pem'}"))
        docker("context", "update", "harborbuilder-vm", "--docker", endpoint, capture_output=True)
        docker("context", "use", "harborbuilder-vm", capture_output=True)
        os.environ["DOCKER_CONFIG"] = str(config)
        os.environ["DOCKER_CONTEXT"] = "harborbuilder-vm"
        image = json.loads(docker("image", "inspect", IMAGE, capture_output=True, text=True).stdout)[0]
        if image["Id"] != read(PROOF / "native-report.json")["image_id"]:
            raise ValueError("Validated public image changed")
        binaries = WORKSPACE / ".local/node22/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin"
        write(allocation / "runtime.json", {"image_id": image["Id"], "cpus": 8, "gpus": 0,
            "protocol_sha256": sha(PROOF / "protocol.json"), "codex_sha256": sha(binaries / "codex"),
            "model_policy_sha256": sha(PROOF / "model-request-policy.json"), "private_oracle_visible_to_models": False})

        states = {}
        for lane in lanes:
            directory = run / lane
            directory.mkdir(exist_ok=True)
            state_path = directory / "state.json"
            if state_path.exists():
                states[lane] = read(state_path)
                expected_model = proxy.providers[lane]["model"]
                if states[lane]["model"] != expected_model:
                    if states[lane]["completed_loops"] or states[lane].get("pending_attempt"):
                        raise ValueError("Use a separate model lane when completed or pending method work exists")
                    for previous in states[lane]["history"]:
                        previous.setdefault("model", states[lane]["model"])
                    states[lane]["model"] = expected_model
                    states[lane]["consecutive_failures"] = 0
                    write(state_path, states[lane])
            else:
                provider = proxy.providers[lane]
                states[lane] = {"model": provider["model"], "provider": "my_codex" if lane == "gpt" else "ark",
                    "completed_loops": 0, "effective_seconds": 0.0, "attempts": 0, "history": [],
                    "best_score": 0.0, "best_selector": None, "baseline": read(PROOF / "baseline.json")}
                write(state_path, states[lane])
            policy_sha = sha(PROOF / "model-request-policy.json")
            if states[lane].get("request_policy_sha256") != policy_sha:
                states[lane]["consecutive_failures"] = 0
                states[lane]["request_policy_sha256"] = policy_sha
                write(state_path, states[lane])
            state = states[lane]
            if state["history"] and not state.get("pending_attempt"):
                previous = state["history"][-1]
                response = run / lane / f"attempt-{previous['attempt']:03d}/model-output/response.json"
                if previous["status"] == "failed" and response.is_file() and not previous.get("selector_sha256"):
                    try:
                        parse_candidate(response.read_text())
                    except (ValueError, SyntaxError):
                        pass
                    else:
                        state["attempts"] += 1
                        state["pending_attempt"] = state["attempts"]
                        state["consecutive_failures"] = 0
                        recovered = run / lane / f"attempt-{state['attempts']:03d}"
                        (recovered / "model-output").mkdir(parents=True, mode=0o700)
                        shutil.copyfile(response, recovered / "model-output/response.json")
                        write(recovered / "recovery.json", {"source_attempt": previous["attempt"],
                            "new_model_requests": 0, "reason": "Parse retained completed model response"})
                        write(state_path, state)

        def model_turn(lane, state, attempt):
            proxy.allow_invocation(lane)
            home = attempt / "home"
            home.mkdir(mode=0o700)
            catalog = json.loads(codex_model_catalog("gpt_seed_dual"))
            write(home / "models.json", catalog)
            write(home / "auth.json", {"OPENAI_API_KEY": token})
            (home / "auth.json").chmod(0o600)
            current = Path(state["best_selector"]) if state["best_selector"] else TASK / "environment/starter/select.py"
            feedback = [{k: row.get(k) for k in ("round", "score", "mean_error", "method_summary", "error")}
                        for row in state["history"][-3:]]
            prompt = ("You are the independent method author for an active preference-acquisition Harbor task. "
                      "Propose ONE new, concrete algorithm within select_pairs. Return ONLY a JSON object with "
                      "source (complete Python source) and method_summary (1-3 sentences describing the change and "
                      "your decision from previous feedback). Do not use any tools or run commands. The operator "
                      "will write your source and evaluate all five seeds. Do not invent scores. Keep the source "
                      "under 120 lines. Avoid duplicate proposals and copying a rejected method.\n\n" +
                      (TASK / "instruction.md").read_text() + "\nCurrent best selector:\n" + current.read_text() +
                      "\nYour recent feedback:\n" + json.dumps(feedback, separators=(",", ":")))
            (attempt / "prompt.txt").write_text(prompt)
            command = ["/opt/codex/bin/codex", "exec", "--skip-git-repo-check", "--color", "never", "--json",
                "--sandbox", "danger-full-access", "-c", 'approval_policy="never"',
                "-c", 'model_provider="operator"', "-c", 'model_providers.operator.name="operator"',
                "-c", f'model_providers.operator.base_url="http://10.0.2.100:{model_port}/{lane}/v1"',
                "-c", 'model_providers.operator.wire_api="responses"',
                "-c", 'model_providers.operator.env_key="OPENAI_API_KEY"',
                "-c", 'model_providers.operator.requires_openai_auth=false',
                "-c", 'model_providers.operator.request_max_retries=0',
                "-c", 'model_providers.operator.stream_max_retries=0',
                "-c", 'model_reasoning_effort="low"', "-c", 'model_reasoning_summary="none"',
                "-c", 'model_context_window=24576', "-c", 'model_catalog_json="/codex-home/models.json"',
                "--model", state["model"], "--output-last-message", "/output/response.json", "-"]
            sandbox = read(ROOT / "deployment/arm-paper-runtime/sandbox.json")
            output = attempt / "model-output"
            output.mkdir()
            creation = ["create", "--init", "--pull", "never", "--network", "host", "--read-only",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
                "--security-opt", "seccomp=" + sandbox["seccomp_path"],
                "--security-opt", "apparmor=" + sandbox["apparmor_profile"], "--security-opt", "systempaths=unconfined",
                "--cpus", "2", "--memory", "4096m", "--memory-swap", "4096m", "--pids-limit", "128",
                "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/harbor_task",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--env", "CODEX_HOME=/codex-home",
                "--env", "HOME=/tmp", "--env", "OPENAI_API_KEY=" + token,
                "--volume", f"{binaries}:/opt/codex/bin:ro", "--volume", f"{home}:/codex-home:rw",
                "--volume", f"{output}:/output:rw", IMAGE, "sleep", "1800"]
            container = docker(*creation, capture_output=True, text=True).stdout.strip()
            containers.add(container)
            write(attempt / "model-container-inspect.json", json.loads(docker("inspect", container, capture_output=True, text=True).stdout)[0])
            clock = CodexWorkClock(max_tool_calls=policy["maximum_shell_calls_per_invocation"])
            started = time.monotonic()
            try:
                docker("start", container, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                from deepscientist_harborbuilder_engine.harborbuilder.agent_runner import _run_command
                code, stdout, stderr, elapsed = _run_command(
                    [*prefix, "exec", "-i", container, *command], cwd=ROOT,
                    env={**os.environ, "TMPDIR": "/tmp"}, timeout=1200,
                    stdin_bytes=prompt.encode(), observe_stdout=clock.observe)
                for key in (token, proxy.providers[lane]["key"]):
                    stdout, stderr = stdout.replace(key, "[REDACTED]"), stderr.replace(key, "[REDACTED]")
                timing = clock.finish(elapsed, success=code == 0)
                write(attempt / "model-log.json", {"exit_code": code, "stdout": stdout, "stderr": stderr, "work_clock": timing})
                if code:
                    raise RuntimeError("Model invocation failed; inspect retained redacted model-log.json")
                if not (output / "response.json").is_file():
                    raise ValueError("Model returned no candidate")
                result = parse_candidate((output / "response.json").read_text())
                if clock.tool_call_count:
                    raise ValueError("This low-token invocation must not execute tools")
                return result, sum(i["end"] - i["start"] for i in timing["intervals"] if i["kind"] == "method")
            finally:
                docker("rm", "--force", "--volumes", container, capture_output=True)
                containers.discard(container)
                # Keep actual model keys and the temporary relay credential out
                # of retained evidence and delivery packages.
                (home / "auth.json").unlink(missing_ok=True)
                for name in ("model-container-inspect.json",):
                    path = attempt / name
                    path.write_text(path.read_text().replace(token, "[REDACTED]"))

        while True:
            pending = [lane for lane in states if states[lane]["effective_seconds"] < policy["minimum_effective_seconds_per_lane"]
                       or states[lane]["completed_loops"] < policy["minimum_completed_loops_per_lane"]]
            if not pending:
                break
            pending = [lane for lane in pending if states[lane].get("consecutive_failures", 0) <
                       policy["maximum_failed_invocations_per_round"]]
            if not pending:
                report("model_repair_needed")
                return
            lane = min(pending, key=lambda item: (states[item]["effective_seconds"], states[item]["attempts"]))
            state = states[lane]
            recovering = bool(state.get("pending_attempt"))
            if not recovering:
                state["attempts"] += 1
                state["pending_attempt"] = state["attempts"]
            attempt = run / lane / f"attempt-{state['attempts']:03d}"
            attempt.mkdir(mode=0o700, exist_ok=recovering)
            state_path = run / lane / "state.json"
            if recovering and (attempt / "state-after.json").exists():
                states[lane] = read(attempt / "state-after.json")
                write(state_path, states[lane])
                continue
            write(state_path, state)
            row = {"round": state["completed_loops"] + 1, "attempt": state["attempts"], "model": state["model"]}
            try:
                report("authoring", active_lane=lane, active_attempt=state["attempts"])
                candidate_path = attempt / "candidate.json"
                if candidate_path.exists():
                    candidate = read(candidate_path)
                    result, author_seconds = candidate["result"], candidate["author_seconds"]
                elif recovering:
                    response = attempt / "model-output/response.json"
                    if not response.is_file():
                        raise RuntimeError("Interrupted author request retained; no automatic request repeated")
                    result, author_seconds = parse_candidate(response.read_text()), 0.0
                else:
                    result, author_seconds = model_turn(lane, state, attempt)
                row["method_summary"] = result["method_summary"]
                selector = attempt / "select.py"
                selector.write_text(result["source"])
                ast.parse(result["source"])
                row["selector_sha256"] = sha(selector)
                if any(old.get("selector_sha256") == row["selector_sha256"] for old in state["history"]):
                    raise ValueError("Duplicate candidate; no training repeated")
                source_run = PROOF / "runs" / sha(PROOF / "protocol.json") / row["selector_sha256"]
                if not candidate_path.exists():
                    prior = {}
                    for seed in (101, 202, 303, 404, 505):
                        directory = source_run / f"seed_{seed}"
                        receipt = directory / "metrics.json"
                        if not receipt.exists():
                            receipt = directory / "progress.json"
                        prior[str(seed)] = read(receipt)["elapsed_seconds"] if receipt.exists() else 0.0
                    candidate = {"result": result, "author_seconds": author_seconds, "prior_elapsed_seconds": prior}
                    write(candidate_path, candidate)
                report("evaluating", active_lane=lane, selector_sha256=row["selector_sha256"])
                measured = evaluate(selector, attempt / "evaluation")
                row.update({key: measured[key] for key in ("score", "mean_error", "raw_values", "new_training_runs")})
                training_seconds = sum(max(0, read(source_run / f"seed_{seed}/metrics.json")["elapsed_seconds"] -
                    candidate["prior_elapsed_seconds"][str(seed)]) for seed in measured["seeds"])
                row.update(status="succeeded", effective_seconds=training_seconds + author_seconds)
                state["effective_seconds"] += row["effective_seconds"]
                state["completed_loops"] += 1
                if measured["score"] > state["best_score"]:
                    state["best_score"] = measured["score"]
                    state["best_selector"] = str(selector)
                    best = run / lane / "best_method"
                    best.mkdir(exist_ok=True)
                    shutil.copyfile(selector, best / "select.py")
                    shutil.copyfile(TASK / "solution/submission.json", best / "submission.json")
                    shutil.copyfile(TASK / "solution/solve.sh", best / "solve.sh")
                row["decision"] = "retain" if state["best_selector"] == str(selector) else "reject"
                state["consecutive_failures"] = 0
            except Exception as exc:
                row.update(status="failed", error=str(exc)[:1000])
                state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            state["history"].append(row)
            state.pop("pending_attempt", None)
            write(attempt / "result.json", row)
            write(attempt / "state-after.json", state)
            write(state_path, state)
            write(run / lane / "trajectory.json", [{"round": i + 1, "method_summary": r.get("method_summary", "") +
                (" Evaluation failed: " + r["error"] if r["status"] == "failed" else
                 f" Measured score {r['score']:.6f}; {r['decision']}.")}
                for i, r in enumerate(state["history"])])
            report("running")
            print(json.dumps({"lane": lane, **row}), flush=True)
        report("lanes_completed_replay_pending")
    except Exception as exc:
        report("interrupted", error=str(exc)[:1000])
        raise
    finally:
        for container in containers:
            subprocess.run([*prefix, "rm", "--force", "--volumes", container], capture_output=True, timeout=20)
        service.terminate()
        try:
            service.wait(timeout=20)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()
        proxy.close()
        log.close()


if __name__ == "__main__":
    main()

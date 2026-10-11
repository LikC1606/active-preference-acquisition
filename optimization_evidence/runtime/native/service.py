"""Private scoring sidecar using the frozen trainer and fresh selector jails."""
import base64
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
STATE = Path("/var/lib/arm")
CHANNEL = Path("/dev/shm/arm-evaluator")
PROTOCOL = Path("/protocol.json")
SEEDS = (101, 202, 303, 404, 505)
SPLIT = os.environ.get("ARM_SCORING_SPLIT", "dev")
if SPLIT not in ("dev", "hidden"):
    raise ValueError("Unknown scoring split")
spec = importlib.util.spec_from_file_location("public_contract", "/harbor_task/tests/grader.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
EvaluationError = contract.EvaluationError


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            checksum.update(block)
    return checksum.hexdigest()


def write(path, data, mode=0o600):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(data, stream, allow_nan=False)
            os.fchmod(stream.fileno(), mode)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def kill(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def indices(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 32768:
            raise EvaluationError("format_error", "Selector output must be a bounded regular JSON file")
        values = json.loads(os.read(descriptor, 32769))
    finally:
        os.close(descriptor)
    if (not isinstance(values, list) or len(values) != 500
            or any(type(v) is not int or not 0 <= v < 22500 for v in values)
            or len(set(values)) != 500):
        raise EvaluationError("hard_gate", "Selector must return 500 unique in-range Python integers")
    return values


@functools.lru_cache(maxsize=1)
def selector_cpu_capability():
    # cpuinfo cannot inspect /proc after chroot. Preserve the trusted trainer's
    # automatic ATen dispatch without exposing /proc to candidate code.
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import torch; print(torch.backends.cpu.get_cpu_capability())"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
    capability = result.stdout.decode().strip()
    modes = {"NO AVX": "default", "AVX2": "avx2", "AVX512": "avx512"}
    if capability not in modes:
        raise EvaluationError("infrastructure_error", "Unsupported trusted CPU dispatch mode")
    return modes[capability]


def select(source, pool, destination, deadline):
    jail = Path("/opt/selector-root")
    try:
        for target, original in (("selector.py", source), ("pool.npz", pool)):
            temporary = jail / (target + ".pending")
            shutil.copyfile(original, temporary)
            os.replace(temporary, jail / target)
            (jail / target).chmod(0o444)
        result = destination / "selector-indices.json"
        result.unlink(missing_ok=True)
        with (destination / "selector.log").open("wb") as log, result.open("wb") as output:
            process = subprocess.Popen([sys.executable, "-I", str(HERE / "selector_sandbox.py"),
                                        str(jail), str(output.fileno()), selector_cpu_capability()],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       close_fds=True, pass_fds=(output.fileno(),), start_new_session=True)
            try:
                process.wait(timeout=min(60, max(0.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired as exc:
                kill(process)
                raise EvaluationError("timeout", "Selector exceeded its 60-second or scoring deadline") from exc
            finally:
                kill(process)
        if process.returncode:
            kind = "resource_limit" if process.returncode in (-9, -24, -25, 137) else "format_error"
            raise EvaluationError(kind, "Selector execution failed; check its source and resource use")
        return indices(result)
    finally:
        for name in ("selector.py", "pool.npz"):
            (jail / name).unlink(missing_ok=True)


def worker_setup():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:6])
    resource.setrlimit(resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_seed(source, destination, seed, deadline):
    destination.mkdir(parents=True, exist_ok=True)
    exchange = destination / "exchange"
    exchange.mkdir(exist_ok=True)
    for path in exchange.iterdir():
        path.unlink()
    environment = dict(os.environ, ARM_RESULT_ROOT=str(destination), ARM_EXCHANGE_ROOT=str(exchange))
    with (destination / "worker.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, str(HERE / "worker.py"), "train", "--seed", str(seed)],
                                   cwd="/workspace", env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True, preexec_fn=worker_setup)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise EvaluationError("timeout", "Five-seed scoring exceeded 7,200 seconds")
                request = exchange / "request.json"
                if request.exists():
                    if sha(exchange / "pool.npz") != read(request)["pool_sha256"]:
                        raise EvaluationError("infrastructure_error", "Trainer pool hash changed")
                    write(exchange / "indices.json", select(source, exchange / "pool.npz", destination, deadline))
                    while request.exists() and process.poll() is None:
                        if time.monotonic() >= deadline:
                            raise EvaluationError("timeout", "Trainer did not consume selector output")
                        time.sleep(0.05)
                time.sleep(0.05)
            if process.returncode:
                raise EvaluationError("infrastructure_error", "Trusted trainer failed; inspect its private worker log")
        finally:
            kill(process)


def validate_row(row, identity, seed, destination):
    if (row.get("selector_sha256") != identity or row.get("seed") != seed
            or row.get("protocol_sha256") != sha(PROTOCOL)
            or row.get("status") != "succeeded" or row.get("actual_updates") != 40000
            or row.get("evaluation_split", "dev") != SPLIT
            or row.get("selected_preferences") != 20000
            or row.get("checkpoint_reload_match") is not True
            or type(row.get("raw_metric")) not in (int, float) or not math.isfinite(row["raw_metric"])
            or sha(destination / "checkpoint") != row.get("checkpoint_sha256")):
        raise EvaluationError("infrastructure_error", "Completed seed evidence differs from the fixed protocol")


def evaluate(source, identity, deadline, baseline_check=False):
    baseline = read(HERE / ("baseline-hidden.json" if SPLIT == "hidden" else "baseline.json"))
    definition = read(HERE / "definition.json")
    if (baseline["protocol_sha256"] != sha(PROTOCOL)
            or baseline["selector_sha256"] != definition["starter_sha256"]
            or baseline["seeds"] != list(SEEDS)
            or baseline["mean_error"] != math.fsum(baseline["raw_values"]) / len(SEEDS)):
        raise EvaluationError("infrastructure_error", "Baseline calibration identity changed")
    if baseline_check and identity != definition["starter_sha256"]:
        raise EvaluationError("format_error", "Baseline check must use the protected Starter")
    if identity == definition["starter_sha256"]:
        root = HERE / "calibration"
    else:
        root = STATE / "runs" / sha(PROTOCOL) / identity
        if SPLIT == "hidden":
            root = root / "hidden"
        root.mkdir(parents=True, exist_ok=True)
        (root / "selector.py").write_bytes(source)
    rows = []
    trained = 0
    for seed in SEEDS:
        if time.monotonic() >= deadline:
            raise EvaluationError("timeout", "Five-seed scoring exceeded 7,200 seconds")
        destination = root / f"seed_{seed}"
        receipt = destination / ("hidden-evaluation.json" if root == HERE / "calibration" and SPLIT == "hidden"
                                 else "evaluation.json")
        if not receipt.exists():
            if root == HERE / "calibration":
                raise EvaluationError("infrastructure_error", "Baseline calibration is incomplete")
            run_seed(root / "selector.py", destination, seed, deadline)
            row = read(receipt)
            row["selector_sha256"] = identity
            write(receipt, row)
            trained += 1
        row = read(receipt)
        validate_row(row, identity, seed, destination)
        rows.append(row)
        print(json.dumps({"selector_sha256": identity, "seed": seed,
                          "raw_metric": row["raw_metric"], "baseline_check": baseline_check}), flush=True)
    mean = math.fsum(row["raw_metric"] for row in rows) / len(rows)
    result = {"schema": "arm.acquisition.reward.v1", "status": "succeeded",
              "evaluation_split": SPLIT,
              "score": (baseline["mean_error"] - mean) / baseline["mean_error"],
              "baseline_mean": baseline["mean_error"], "baseline_score": 0.0,
              "mean_error": mean, "seeds": list(SEEDS), "selector_sha256": identity,
              "raw_values": [r["raw_metric"] for r in rows], "protocol_sha256": sha(PROTOCOL),
              "new_training_runs": trained, "formal_harbor_acceptance": False}
    if SPLIT == "hidden":
        result["hidden_manifest_sha256"] = baseline["hidden_manifest_sha256"]
    if not baseline_check and root != HERE / "calibration":
        write(root / "reward.json", result)
    return result


def request_at(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 131072:
            raise EvaluationError("format_error", "Request must be a bounded regular file")
        request = json.loads(os.read(descriptor, 131073))
    finally:
        os.close(descriptor)
    source = base64.b64decode(request["source_base64"], validate=True)
    identity = hashlib.sha256(source).hexdigest()
    if (request.get("request_id") != path.stem or request.get("selector_sha256") != identity
            or not 0 < len(source) <= 65536 or type(request.get("baseline_check", False)) is not bool):
        raise EvaluationError("format_error", "Malformed selector request")
    expires = request.get("expires_unix")
    if type(expires) not in (int, float) or not math.isfinite(expires) or not 0 < expires - time.time() <= 7200:
        raise EvaluationError("timeout", "Request expired before evaluation")
    contract.validate_source(source)
    return source, identity, request.get("baseline_check", False), time.monotonic() + expires - time.time()


def main():
    subprocess.run([sys.executable, str(HERE / "validate_calibration.py")],
                   cwd="/workspace", check=True, timeout=600)
    for name, mode in (("inbox", 0o1777), ("outbox", 0o755)):
        (CHANNEL / name).mkdir(parents=True, exist_ok=True)
        (CHANNEL / name).chmod(mode)
    (STATE / "runs").mkdir(exist_ok=True)
    write(STATE / "runtime.json", {"protocol_sha256": sha(PROTOCOL),
          "worker_sha256": sha(HERE / "worker.py"), "service_sha256": sha(__file__),
          "network": "none", "trainer_threads": 6, "selector_cpus": 2})
    (STATE / "ready").touch()
    while True:
        for path in (CHANNEL / "inbox").glob("*.json"):
            if not re.fullmatch(r"[0-9a-f]{32}\.json", path.name):
                continue
            response = {"request_id": path.stem, "status": "failed"}
            try:
                source, identity, baseline_check, deadline = request_at(path)
                response["selector_sha256"] = identity
                result = evaluate(source, identity, deadline, baseline_check)
                if time.monotonic() >= deadline:
                    raise EvaluationError("timeout", "Result arrived after its scoring deadline")
                response.update(result)
            except Exception as error:
                if not isinstance(error, EvaluationError):
                    error = EvaluationError("infrastructure_error", "Private scoring failed; inspect operator logs")
                response.update(error.record())
                traceback.print_exc()
            write(CHANNEL / "outbox" / path.name, response, 0o644)
            path.unlink(missing_ok=True)
        time.sleep(0.1)


if __name__ == "__main__":
    main()

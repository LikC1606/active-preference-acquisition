"""Host evaluator: private training and untrusted selection use separate containers."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

PRIVATE = Path(__file__).resolve().parent
PACKAGE = PRIVATE.parents[2] if (PRIVATE.parents[2] / "workspace/harbor_task").is_dir() else None
PROOF = Path(os.environ.get("ARM_PROOF_ROOT", str(PACKAGE.parent / "acquisition-runtime" if PACKAGE else PRIVATE.parent))).resolve()
PROJECT = PROOF.parent
WORKSPACE = PROJECT.parent.parent
TOOLS = Path(os.environ.get("ARM_TOOLS_ROOT", str(WORKSPACE / "tmp/harborbuilder-kvm-test"))).resolve()
TASK = Path(os.environ.get("ARM_TASK_ROOT", str(PACKAGE / "workspace/harbor_task" if PACKAGE else PROJECT / "harbor_task_acquisition"))).resolve()
ORACLE = Path(os.environ.get("ARM_ORACLE_ROOT", str(TOOLS / "deployment/arm-private-oracle-v14/source"))).resolve()
IMAGE = os.environ.get("ARM_ORACLE_IMAGE", "127.0.0.1:5000/arm-oracle-boundary-v12@sha256:ca8328f2612111d04786b096121f2bc833b75c166d632a4600d2b7ef664531e8")
SEEDS = (101, 202, 303, 404, 505)
PYTHON = os.environ.get("ARM_CONTAINER_PYTHON", "/opt/arm-python/bin/python3.11")
PROTOCOL = Path(os.environ.get("ARM_PROTOCOL_FILE", str(PACKAGE / "optimization_evidence/experiment_plan.json" if PACKAGE else PROOF / "protocol.json"))).resolve()
DEFINITION = PACKAGE / "optimization_evidence/runtime/definition.json" if PACKAGE else PROOF / "definition.json"
BASELINE = TASK / "environment/baseline.json" if PACKAGE else PROOF / "baseline.json"
DOCKER_BIN = os.environ.get("ARM_DOCKER_BIN", "docker" if PACKAGE else str(TOOLS / "deployment/docker"))
SANDBOX = Path(os.environ.get("ARM_SANDBOX_CONFIG", str(TOOLS / "deployment/arm-paper-runtime/sandbox.json"))).resolve()
# The operator loads its protected public contract, never submitted code.
_contract_spec = importlib.util.spec_from_file_location("arm_contract", TASK / "tests/grader.py")
_contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
EvaluationError = _contract.EvaluationError
validate_source = _contract.validate_source
SCORE_TIMEOUT_SECONDS = _contract.SCORE_TIMEOUT_SECONDS


if PACKAGE and PROOF.is_relative_to(PACKAGE):
    raise ValueError("ARM_PROOF_ROOT must be outside the immutable submission package")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_selector(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Selector must be a regular file of at most 64 KiB")
    with path.open("rb") as stream:
        source = stream.read(65537)
    if len(source) > 65536:
        raise ValueError("Selector exceeds the 64 KiB source limit")
    return source


def write(path, value, *, mode=None):
    path = Path(path)
    if mode is None:
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o600
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            os.fchmod(stream.fileno(), mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def docker_prefix():
    config = os.environ.get("DOCKER_CONFIG")
    if not config and not PACKAGE:
        config = PROJECT / ".deepscientist/harborbuilder/sessions/hb-arm-paper-v4-3/.docker-runtime/.docker"
    return [DOCKER_BIN] + (["--config", str(config)] if config else [])


def docker(*args, timeout=30, **kwargs):
    return subprocess.run([*docker_prefix(), *map(str, args)],
                          timeout=timeout, check=True, **kwargs)


def create(selector=False):
    sandbox = read(SANDBOX)
    return ["create", "--init", "--pull", "never", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--security-opt", "seccomp=" + sandbox["seccomp_path"],
            "--security-opt", "apparmor=" + sandbox["apparmor_profile"],
            "--security-opt", "systempaths=unconfined", "--pids-limit", "128",
            "--cpus", "2" if selector else "6", "--memory", "4096m" if selector else "32768m",
            "--memory-swap", "4096m" if selector else "32768m", "--user", f"{os.getuid()}:{os.getgid()}",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1"]


def validate_container(observed, *, selector=False):
    host = observed["HostConfig"]
    memory = (4 if selector else 32) * 1024**3
    if (host.get("NanoCpus") != (2 if selector else 6) * 10**9
            or host.get("Memory") != memory or host.get("MemorySwap") != memory
            or host.get("PidsLimit") != 128 or not host.get("ReadonlyRootfs")
            or host.get("NetworkMode") != "none" or host.get("DeviceRequests")):
        raise EvaluationError("infrastructure_error", "Container resources or isolation differ from the declared boundary")


def select(selector, pool, output, deadline):
    command = create(selector=True) + [
        "--tmpfs", "/output:rw,noexec,nosuid,size=4m,mode=1777",
        "--volume", f"{selector}:/selector.py:ro", "--volume", f"{pool}:/pool.npz:ro",
        "--volume", f"{PRIVATE / 'selector_entry.py'}:/entry.py:ro", IMAGE, "sleep", "600"]
    container = docker(*command, capture_output=True, text=True).stdout.strip()
    try:
        observed = json.loads(docker("inspect", container, capture_output=True, text=True).stdout)[0]
        validate_container(observed, selector=True)
        if ({m["Destination"] for m in observed["Mounts"]} != {"/selector.py", "/pool.npz", "/entry.py"}
                or any(m["RW"] for m in observed["Mounts"])):
            raise ValueError("Selector container boundary changed")
        write(output / "last-selector-inspect.json", observed)
        docker("start", container, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EvaluationError("timeout", "Scoring budget expired before selection")
        try:
            docker("exec", container, PYTHON, "/entry.py", timeout=min(60, remaining),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise EvaluationError("timeout", "Selector exceeded its call or scoring deadline") from exc
        except subprocess.CalledProcessError as exc:
            kind = "resource_limit" if exc.returncode == 137 else "format_error"
            raise EvaluationError(kind, "Selector execution failed; check source and resource use") from exc
        # Docker's archive API cannot copy from this tmpfs; read a bounded
        # regular file in a separate trusted invocation instead.
        extract = ("import os,stat,sys; f=os.open('/output/indices.json',os.O_RDONLY|os.O_NOFOLLOW); "
                   "s=os.fstat(f); assert stat.S_ISREG(s.st_mode) and s.st_size<=32768; "
                   "b=os.read(f,32769); assert len(b)<=32768; sys.stdout.buffer.write(b)")
        response = docker("exec", container, PYTHON, "-I", "-c", extract, capture_output=True).stdout
        try:
            indices = json.loads(response, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except (ValueError, UnicodeError) as exc:
            raise EvaluationError("format_error", "Selector output must be a finite JSON list of indices") from exc
        if (not isinstance(indices, list) or len(indices) != 500
                or any(type(i) is not int or not 0 <= i < 22500 for i in indices)
                or len(set(indices)) != 500):
            raise EvaluationError("hard_gate", "Selector must return 500 unique in-range Python integers")
        return indices
    finally:
        docker("rm", "--force", "--volumes", container, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_worker(operation, selector, destination, seed=None, deadline=None, protocol_file=None):
    destination.mkdir(parents=True, exist_ok=True)
    exchange = destination / "exchange"
    exchange.mkdir(exist_ok=True)
    for path in exchange.iterdir():
        if path.name not in {"pool.npz", "pending.npz", "request.json", "indices.json", "indices.json.tmp"} or not path.is_file():
            raise ValueError("Unexpected exchange file")
        path.unlink()
    command = create() + ["--workdir", "/workspace",
        "--volume", f"{ORACLE}:/opt/paper:ro",
        "--volume", f"{TASK / 'environment/public_assets'}:/workspace/environment/public_assets:ro",
        "--volume", f"{PRIVATE / 'worker.py'}:/worker.py:ro",
        "--volume", f"{protocol_file or PROTOCOL}:/protocol.json:ro",
        "--volume", f"{destination}:/result:rw", "--volume", f"{exchange}:/exchange:rw"]
    command += [IMAGE, PYTHON, "/worker.py", operation]
    if seed is not None:
        command += ["--seed", str(seed)]
    container = docker(*command, capture_output=True, text=True).stdout.strip()
    process = None
    deadline = deadline if deadline is not None else time.monotonic() + SCORE_TIMEOUT_SECONDS
    try:
        observed = json.loads(docker("inspect", container, capture_output=True, text=True).stdout)[0]
        validate_container(observed)
        write(destination / "trainer-inspect.json", observed)
        with (destination / "worker.log").open("wb") as log:
            process = subprocess.Popen([*docker_prefix(), "start", "--attach", container],
                                       stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("Experiment exceeded its deadline")
                request = exchange / "request.json"
                if request.exists():
                    expected = read(request)["pool_sha256"]
                    if sha(exchange / "pool.npz") != expected:
                        raise ValueError("Pool changed before isolated selection")
                    indices = select(selector, exchange / "pool.npz", destination, deadline)
                    write(exchange / "indices.json", indices)
                    while request.exists() and process.poll() is None:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Trainer did not consume selector response")
                        time.sleep(0.05)
                time.sleep(0.1)
            if process.returncode:
                state = json.loads(docker("inspect", container, capture_output=True, text=True).stdout)[0]["State"]
                kind = "resource_limit" if state.get("OOMKilled") else "infrastructure_error"
                raise EvaluationError(kind, "Trusted trainer failed; the operator must inspect its private worker log")
    finally:
        docker("rm", "--force", "--volumes", container, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def validate_receipt(row, identity, protocol_sha, seed, protocol):
    updates = (protocol["annotation_budget"] // protocol["annotation_batch"]
               * protocol["training_iterations_per_round"])
    if (row.get("selector_sha256") != identity or row.get("status") != "succeeded"
            or row.get("protocol_sha256") != protocol_sha
            or type(row.get("seed")) is not int or row["seed"] != seed
            or type(row.get("raw_metric")) not in (int, float)
            or not math.isfinite(row["raw_metric"])
            or type(row.get("selected_preferences")) is not int
            or row["selected_preferences"] != protocol["annotation_budget"]
            or type(row.get("actual_updates")) is not int or row["actual_updates"] != updates
            or row.get("checkpoint_reload_match") is not True):
        raise ValueError("Completed seed evidence does not match the frozen protocol")


def validate_baseline(anchor, protocol_sha, definition):
    values = anchor.get("raw_values")
    if (anchor.get("protocol_sha256") != protocol_sha
            or anchor.get("selector_sha256") != definition["starter_sha256"]
            or anchor.get("seeds") != list(SEEDS)
            or type(anchor.get("score")) not in (int, float) or anchor["score"] != 0.0
            or type(anchor.get("mean_error")) not in (int, float)
            or not math.isfinite(anchor["mean_error"]) or anchor["mean_error"] <= 0
            or not isinstance(values, list) or len(values) != len(SEEDS)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or anchor["mean_error"] != math.fsum(values) / len(SEEDS)):
        raise ValueError("Baseline calibration does not match the frozen protocol or recorded metrics")


def restore_packaged_seed(destination, identity, protocol_sha, seed, protocol):
    if not PACKAGE:
        return
    for directory in ("baseline_runs", "reference_runs", "best_method_runs"):
        source = PACKAGE / "optimization_evidence" / directory / f"seed_{seed}"
        receipt = source / ("evaluation.json" if directory == "best_method_runs" else "records/evaluation.json")
        if not receipt.is_file():
            continue
        row = read(receipt)
        if row.get("selector_sha256") != identity or row.get("protocol_sha256") != protocol_sha:
            continue
        validate_receipt(row, identity, protocol_sha, seed, protocol)
        checkpoint = source / "model/checkpoint"
        if not checkpoint.is_file() or sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("Packaged completed seed evidence changed")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint, destination / "checkpoint")
        write(destination / "evaluation.json", row)
        return


def _evaluate(source, protocol_source, output, cached_only=False, *, deadline):
    identity = hashlib.sha256(source).hexdigest()
    published = Path(output).resolve()
    protocol_sha = hashlib.sha256(protocol_source).hexdigest()
    protocol = json.loads(protocol_source)
    if protocol["seeds"] != list(SEEDS):
        raise ValueError("Frozen protocol seed set changed")
    output = PROOF / "runs" / protocol_sha / identity
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    snapshot = output / "selector.py"
    if snapshot.exists() and snapshot.read_bytes() != source:
        raise ValueError("Use a fresh output directory for a different selector")
    snapshot.write_bytes(source)
    protocol_snapshot = output / "protocol.json"
    protocol_snapshot.write_bytes(protocol_source)
    definition = read(DEFINITION)
    baseline = identity == definition["starter_sha256"]
    if not baseline and not BASELINE.exists():
        raise ValueError("Calibrate the new-protocol Baseline before candidate scoring")
    if PACKAGE or not baseline:
        validate_baseline(read(BASELINE), protocol_sha, definition)
    rows = []
    new_training_runs = 0
    for seed in SEEDS:
        if time.monotonic() >= deadline:
            raise EvaluationError("timeout", "Five-seed scoring exceeded 7,200 seconds")
        destination = output / f"seed_{seed}"
        receipt = destination / "evaluation.json"
        if not receipt.exists():
            restore_packaged_seed(destination, identity, protocol_sha, seed, protocol)
        cached = receipt.exists()
        if cached:
            row = read(receipt)
        else:
            if cached_only:
                raise ValueError(f"No completed cache for seed {seed}; cached-only evaluation cannot train")
            run_worker("train", snapshot, destination, seed=seed, deadline=deadline,
                       protocol_file=protocol_snapshot)
            new_training_runs += 1
            row = {**read(receipt), "selector_sha256": identity, "new_training_runs": 1}
        validate_receipt(row, identity, protocol_sha, seed, protocol)
        checkpoint = destination / "checkpoint"
        if not checkpoint.is_file() or sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("Completed seed checkpoint is missing or changed")
        if not cached:
            write(receipt, row)
        rows.append(row)
        write(output / "observations.json", rows)
        print(json.dumps({"seed": seed, "raw_metric": row["raw_metric"],
                          "cached": cached, "new_training_runs": 0 if cached else 1}), flush=True)
    if time.monotonic() >= deadline:
        raise EvaluationError("timeout", "Scoring budget expired before publication")
    mean = math.fsum(row["raw_metric"] for row in rows) / len(SEEDS)
    if baseline and not PACKAGE:
        anchor = {"mean_error": mean, "raw_values": [r["raw_metric"] for r in rows], "seeds": list(SEEDS),
                  "protocol_sha256": protocol_sha, "selector_sha256": identity, "score": 0.0}
        write(PROOF / "baseline.json", anchor)
        write(TASK / "environment/baseline.json", anchor)
    anchor = read(BASELINE)
    validate_baseline(anchor, protocol_sha, definition)
    B = anchor["mean_error"]
    result = {"schema": "arm.acquisition.reward.v1", "status": "succeeded", "score": (B - mean) / B,
              "baseline_mean": B, "baseline_score": 0.0, "mean_error": mean, "seeds": list(SEEDS),
              "selector_sha256": identity, "raw_values": [r["raw_metric"] for r in rows],
              "protocol_sha256": protocol_sha,
              "new_training_runs": new_training_runs,
              "formal_harbor_acceptance": False}
    write(published / "observations.json", rows)
    if output != published:
        write(output / "reward.json", result)
    write(published / "reward.json", result)
    return result


def lock_until(stream, deadline):
    while True:
        if time.monotonic() >= deadline:
            raise EvaluationError("timeout", "Scoring budget expired while waiting for the single evaluation slot")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))


def evaluate(selector, output, *, cached_only=False, timeout_seconds=SCORE_TIMEOUT_SECONDS):
    deadline = time.monotonic() + min(timeout_seconds, SCORE_TIMEOUT_SECONDS)
    PROOF.mkdir(parents=True, exist_ok=True)
    # One evaluation slot across all selectors, brokers and output directories.
    with (PROOF / ".evaluation.lock").open("a") as global_lock:
        lock_until(global_lock, deadline)
        return _evaluate_locked(selector, output, cached_only=cached_only, deadline=deadline)


def _evaluate_locked(selector, output, *, cached_only, deadline):
    published = Path(output).resolve()
    if PACKAGE and published.is_relative_to(PACKAGE):
        raise ValueError("Evaluation output must be outside the immutable submission package")
    runs = (PROOF / "runs").resolve()
    source = protocol_source = None
    if published.is_relative_to(runs):
        source = read_selector(selector)
        protocol_source = PROTOCOL.read_bytes()
        identity = hashlib.sha256(source).hexdigest()
        protocol_sha = hashlib.sha256(protocol_source).hexdigest()
        if published != (runs / protocol_sha / identity).resolve():
            raise ValueError("Evaluation output must not overwrite another selector's cache or seed files")
    published.mkdir(parents=True, exist_ok=True)
    # Different selectors may share an output directory; publish one result at a time.
    with (published / ".evaluation.lock").open("a") as publication_lock:
        lock_until(publication_lock, deadline)
        for name in ("reward.json", "observations.json"):
            (published / name).unlink(missing_ok=True)
        if source is None:
            source = read_selector(selector)
            protocol_source = PROTOCOL.read_bytes()
        validate_source(source)
        identity = hashlib.sha256(source).hexdigest()
        protocol_sha = hashlib.sha256(protocol_source).hexdigest()
        cache = PROOF / "runs" / protocol_sha / identity
        cache.mkdir(parents=True, mode=0o700, exist_ok=True)
        # Both lanes may discover the same source; only one may populate its cache.
        with (cache / "evaluation.lock").open("a") as lock:
            lock_until(lock, deadline)
            (cache / "reward.json").unlink(missing_ok=True)
            return _evaluate(source, protocol_source, published, cached_only=cached_only, deadline=deadline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached-only", action="store_true", help="Reuse completed evidence without Docker or training")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.selector, args.output, cached_only=args.cached_only), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

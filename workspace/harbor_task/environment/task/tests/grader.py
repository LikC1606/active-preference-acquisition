"""Validate submissions and publish only the trusted evaluator's results."""
import argparse
import ast
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import uuid

SCORE_TIMEOUT_SECONDS = 7200
REPAIRS = {
    "format_error": "Restore the submission schema and submit valid Python source of at most 64 KiB.",
    "hard_gate": "Use permitted imports and return exactly 500 unique in-range Python integers.",
    "timeout": "Reduce selector cost and retry within the 7,200-second scoring budget; completed seeds are retained.",
    "resource_limit": "Reduce selector memory or process use; limits are 2 CPUs, 4 GiB and 128 processes.",
    "infrastructure_error": "Ask the operator to restore the trusted evaluation service and inspect its private logs, then retry.",
}
ALLOWED_IMPORTS = frozenset({"numpy", "scipy", "torch", "math", "random", "statistics",
                           "itertools", "functools", "collections", "heapq", "operator",
                           "typing", "dataclasses", "copy", "enum", "warnings", "time"})


class EvaluationError(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind

    def record(self):
        record = {"schema": "arm.acquisition.failure.v1", "status": "failed",
                  "rankable": False,
                  "error": {"type": self.kind, "message": str(self), "repair": REPAIRS[self.kind]}}
        if self.kind == "hard_gate":
            record.update(status="invalid", score=-1)
        return record


def validate_source(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise EvaluationError("format_error", "Selector source is not valid Python") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [(node.module or "").split(".")[0]] if not node.level else [""]
        else:
            modules = []
        if any(module not in ALLOWED_IMPORTS for module in modules):
            raise EvaluationError("hard_gate", "Selector imports a module outside the permitted list")
        if isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec", "compile", "open", "input"}
                or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"):
            raise EvaluationError("hard_gate", "Dynamic imports and external file access are not permitted")


def write_json(path, value):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            os.fchmod(stream.fileno(), 0o644)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def clear_result(output):
    for name in ("reward.json", "observations.json", "failure.json", "public-score.json"):
        (output / name).unlink(missing_ok=True)


def broker_evaluate(selector, baseline, *, deadline, baseline_check=False):
    channel = Path("/opt/arm-evaluator")
    if not (channel / "inbox").is_dir() or not (channel / "outbox").is_dir():
        raise EvaluationError("infrastructure_error", "Trusted evaluation channel is unavailable")
    source = selector.read_bytes()
    identity = hashlib.sha256(source).hexdigest()
    request_id = uuid.uuid4().hex
    request = {"request_id": request_id, "selector_sha256": identity,
               "source_base64": base64.b64encode(source).decode("ascii"),
               "baseline_check": baseline_check,
               "expires_unix": time.time() + max(0, deadline - time.monotonic())}
    write_json(channel / "inbox" / (request_id + ".json"), request)
    response = channel / "outbox" / (request_id + ".json")
    while not response.exists():
        if time.monotonic() >= deadline:
            raise EvaluationError("timeout", "Trusted evaluation exceeded its scoring budget")
        time.sleep(0.2)
    if time.monotonic() >= deadline:
        raise EvaluationError("timeout", "Trusted response arrived after the scoring deadline")
    if (response.is_symlink() or not stat.S_ISREG(response.stat().st_mode)
            or response.stat().st_size > 65536):
        raise EvaluationError("infrastructure_error", "Trusted response is not a bounded regular file")
    result = json.loads(response.read_text())
    if not isinstance(result, dict) or result.get("request_id") != request_id or result.get("selector_sha256") != identity:
        raise EvaluationError("infrastructure_error", "Trusted response does not match the submitted source")
    if result.get("status") != "succeeded":
        error = result.get("error", {})
        kind = error.get("type", "infrastructure_error") if isinstance(error, dict) else "infrastructure_error"
        if kind not in REPAIRS:
            kind = "infrastructure_error"
        message = error.get("message", "Trusted evaluation failed") if isinstance(error, dict) else "Trusted evaluation failed"
        raise EvaluationError(kind, message)
    values = result.get("raw_values")
    if (type(result.get("score")) not in (int, float) or not math.isfinite(result["score"])
            or result.get("schema") != "arm.acquisition.reward.v1"
            or result.get("protocol_sha256") != baseline["protocol_sha256"]
            or result.get("seeds") != baseline["seeds"]
            or not isinstance(values, list) or len(values) != len(baseline["seeds"])
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or type(result.get("mean_error")) not in (int, float)
            or not math.isfinite(result["mean_error"])
            or type(result.get("baseline_mean")) not in (int, float)
            or result["baseline_mean"] != baseline["mean_error"]
            or type(result.get("baseline_score")) not in (int, float)
            or result.get("baseline_score") != 0.0):
        raise EvaluationError("infrastructure_error", "Trusted metric schema or frozen protocol changed")
    mean = math.fsum(values) / len(values)
    expected = (baseline["mean_error"] - mean) / baseline["mean_error"]
    if result["mean_error"] != mean or result["score"] != expected:
        raise EvaluationError("infrastructure_error", "Trusted score differs from five-seed normalization")
    if baseline_check and (result["score"] != 0.0
                           or type(result.get("new_training_runs")) is not int
                           or result["new_training_runs"] != 0):
        raise EvaluationError("infrastructure_error", "Baseline check must reuse validated calibration and score zero")
    return result


def publish_verifier(output):
    logs = Path("/logs/verifier")
    if not logs.is_dir():
        return
    clear_result(logs)
    try:
        for name in ("reward.json", "failure.json"):
            path = output / name
            if not path.is_file():
                continue
            result = json.loads(path.read_text())
            if name == "failure.json":
                write_json(logs / name, result)
            else:
                write_json(logs / "public-score.json", result)
                write_json(logs / "reward.json", {"score": result["score"]})
    except (OSError, ValueError) as exc:
        clear_result(logs)
        error = EvaluationError("infrastructure_error", "Verifier result publication failed")
        write_json(logs / "failure.json", error.record())
        raise error from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("solution/artifacts/current"))
    parser.add_argument("--baseline", action="store_true", help="Check the protected Starter against cached calibration")
    parser.add_argument("--verifier", action="store_true", help="Publish the trusted result to Harbor's verifier logs")
    args = parser.parse_args()
    deadline = time.monotonic() + SCORE_TIMEOUT_SECONDS
    root = Path("/harbor_task") if Path("/harbor_task").is_dir() else Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to((root / "solution").resolve() / "artifacts"):
        raise EvaluationError("format_error", "Evaluation output must be inside solution/artifacts/")
    output.mkdir(parents=True, exist_ok=True)
    clear_result(output)
    try:
        if not args.baseline:
            manifest = root / "solution/submission.json"
            try:
                submission = json.loads(manifest.read_text()) if manifest.is_file() and not manifest.is_symlink() and manifest.stat().st_size <= 4096 else None
            except (ValueError, UnicodeError) as exc:
                raise EvaluationError("format_error", "Submission manifest must be valid JSON") from exc
            if (manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 4096
                    or submission != {"schema": "arm.acquisition.submission.v1",
                                                            "selector_path": "solution/select.py"}):
                raise EvaluationError("format_error", "Restore the provided regular solution/submission.json")
        selector = root / ("environment/starter/select.py" if args.baseline else "solution/select.py")
        if selector.is_symlink() or not selector.is_file() or selector.stat().st_size > 65536:
            raise EvaluationError("format_error", "Submit a regular selector of at most 64 KiB")
        validate_source(selector.read_bytes())
        baseline = json.loads((root / "environment/baseline.json").read_text())
        result = broker_evaluate(selector, baseline, deadline=deadline, baseline_check=args.baseline)
        write_json(output / "observations.json", [
            {"seed": seed, "raw_metric": error}
            for seed, error in zip(result["seeds"], result["raw_values"])])
        if time.monotonic() >= deadline:
            raise EvaluationError("timeout", "Scoring budget expired before final score publication")
        # The score is the commit marker and is always published last.
        write_json(output / "reward.json", result)
    except (EvaluationError, OSError, ValueError) as exc:
        if not isinstance(exc, EvaluationError):
            exc = EvaluationError("infrastructure_error", "Evaluation data or result publication failed")
        clear_result(output)
        write_json(output / "failure.json", exc.record())
        if exc.kind == "hard_gate":
            write_json(output / "reward.json", exc.record())
        raise exc
    finally:
        if args.verifier:
            publish_verifier(output)


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as exc:
        print(json.dumps(exc.record(), allow_nan=False), file=sys.stderr)
        raise SystemExit(2)

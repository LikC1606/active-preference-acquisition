"""Operator-only filesystem bridge to the isolated acquisition evaluator."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import traceback
import time

from evaluate import evaluate, write, EvaluationError, SCORE_TIMEOUT_SECONDS, DEFINITION, read


def read_request(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 131072:
            raise ValueError("Request must be a bounded regular file")
        content = os.read(descriptor, 131073)
        if len(content) > 131072:
            raise ValueError("Request exceeds the limit")
    finally:
        os.close(descriptor)
    request = json.loads(content)
    if request.get("request_id") != path.stem or not re.fullmatch(r"[0-9a-f]{32}", path.stem):
        raise ValueError("Invalid request ID")
    source = base64.b64decode(request["source_base64"], validate=True)
    if not 0 < len(source) <= 65536:
        raise ValueError("Invalid source length")
    identity = hashlib.sha256(source).hexdigest()
    if request.get("selector_sha256") != identity:
        raise ValueError("Selector source hash mismatch")
    if type(request.get("baseline_check", False)) is not bool:
        raise ValueError("Invalid baseline-check flag")
    expires = request.get("expires_unix")
    if type(expires) not in (int, float):
        raise ValueError("Invalid scoring deadline")
    return source, identity, request.get("baseline_check", False), expires


class AcquisitionBroker:
    def __init__(self, root, *, cached_only=False):
        self.root = Path(root)
        self.cached_only = cached_only
        for name in ("inbox", "outbox", "private"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.receipts = []

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Acquisition evaluation is still running")

    def serve(self):
        handled = set()
        while not self.stop_event.is_set():
            for path in (self.root / "inbox").glob("*.json"):
                if path.name in handled:
                    continue
                handled.add(path.name)
                if not re.fullmatch(r"[0-9a-f]{32}\.json", path.name):
                    continue
                response = {"request_id": path.stem, "status": "failed"}
                try:
                    source, identity, baseline_check, expires = read_request(path)
                    response["selector_sha256"] = identity
                    if not 0 < expires - time.time() <= SCORE_TIMEOUT_SECONDS:
                        raise EvaluationError("timeout", "Request expired before its evaluation slot became available")
                    private = self.root / "private" / path.stem
                    private.mkdir(mode=0o700)
                    selector = private / "select.py"
                    selector.write_bytes(source)
                    if baseline_check and identity != read(DEFINITION)["starter_sha256"]:
                        raise EvaluationError("format_error", "Baseline check must use the protected Starter source")
                    result = evaluate(selector, private / "result", cached_only=self.cached_only or baseline_check,
                                      timeout_seconds=expires - time.time())
                    public_fields = ("schema", "status", "score", "baseline_mean", "baseline_score",
                                     "mean_error", "seeds", "selector_sha256", "raw_values",
                                     "protocol_sha256", "new_training_runs", "formal_harbor_acceptance")
                    response.update({key: result[key] for key in public_fields})
                    self.receipts.append(response.copy())
                except Exception as exc:
                    if not isinstance(exc, EvaluationError):
                        kind = "timeout" if isinstance(exc, TimeoutError) else "infrastructure_error"
                        exc = EvaluationError(kind, "Trusted evaluation failed; inspect the operator's private logs")
                    response.update(exc.record())
                    with (self.root / "private/errors.log").open("a") as log:
                        traceback.print_exc(file=log)
                write(self.root / "outbox" / path.name, response, mode=0o644)
            self.stop_event.wait(0.1)

"""Rate-limited, host-only Responses bridge; never publish upstream credentials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import io
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
import tomllib
import urllib.error
import urllib.request
import uuid


class ModelProxy:
    def __init__(self, config_path, auth_path, policy, log_path, token, port=18631, relay_root=None):
        config = tomllib.loads(Path(config_path).read_text())
        auth = json.loads(Path(auth_path).read_text())
        self.providers = {
            "gpt": {"url": config["model_providers"]["my_codex"]["base_url"],
                    "model": "gpt-5.6-sol", "key": auth["OPENAI_API_KEY"]},
            "seed": {"url": config["model_providers"]["ark"]["base_url"],
                     "model": "doubao-seed-2.1-turbo", "key": os.environ["ARK_API_KEY"]},
        }
        self.policy = policy
        self.log_path = Path(log_path)
        self.token = token
        self.relay_root = Path(relay_root) if relay_root else None
        self.locks = {lane: threading.Lock() for lane in self.providers}
        self.quota_lock = threading.Lock()
        self.request_quota = {lane: 0 for lane in self.providers}
        self.last_start = {lane: 0.0 for lane in self.providers}
        self.cooldown_until = {lane: 0.0 for lane in self.providers}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                route = self.path.strip("/").split("/")
                if (len(route) != 3 or route[1:] != ["v1", "responses"]
                        or route[0] not in owner.providers
                        or self.headers.get("Authorization") != "Bearer " + owner.token):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= owner.policy["maximum_input_bytes"]:
                        self.send_error(413)
                        return
                    payload = json.loads(self.rfile.read(length))
                    provider = owner.providers[route[0]]
                    if payload.get("model") != provider["model"]:
                        self.send_error(400, "Model differs from the frozen lane")
                        return
                    payload["max_output_tokens"] = min(
                        payload.get("max_output_tokens", owner.policy["maximum_output_tokens"]),
                        owner.policy["maximum_output_tokens"])
                    payload["store"] = False
                    reasoning = payload.setdefault("reasoning", {})
                    reasoning["effort"] = "low"
                    reasoning.pop("summary", None)
                    payload["tools"] = []
                    payload["tool_choice"] = "none"
                    payload["parallel_tool_calls"] = False
                    with owner.quota_lock:
                        if owner.request_quota[route[0]] <= 0:
                            self.send_error(429, "Candidate invocation request budget exhausted")
                            return
                        owner.request_quota[route[0]] -= 1
                    owner.forward(self, route[0], payload)
                except (ValueError, TypeError, KeyError):
                    self.send_error(400)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def allow_invocation(self, lane):
        with self.quota_lock:
            self.request_quota[lane] = self.policy["maximum_requests_per_invocation"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def forward(self, handler, lane, payload):
        if lane == "seed":
            # Ark thinking consumes the same output budget as candidate code.
            payload.pop("reasoning", None)
            payload["thinking"] = {"type": "disabled"}
        if self.relay_root:
            identity = uuid.uuid4().hex
            request = self.relay_root / "inbox" / (identity + ".json")
            temporary = request.with_suffix(".tmp")
            temporary.write_text(json.dumps({"lane": lane, "payload": payload}))
            temporary.replace(request)
            response = self.relay_root / "outbox" / (identity + ".json")
            deadline = time.monotonic() + 600
            while not response.is_file():
                if time.monotonic() > deadline:
                    handler.send_error(502, "Login model relay timed out")
                    return
                time.sleep(0.1)
            result = json.loads(response.read_text())
            body_path = self.relay_root / "outbox" / (identity + ".body")
            maximum_body = self.policy["maximum_response_bytes"]
            if body_path.stat().st_size > maximum_body:
                handler.send_error(502, "Login model relay response exceeded the output limit")
                return
            handler.send_response(result["status"])
            handler.send_header("Content-Type", result["content_type"])
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(body_path.read_bytes())
            handler.wfile.flush()
            return
        provider = self.providers[lane]
        with self.locks[lane]:
            wait = max(0.0, self.last_start[lane] + self.policy["minimum_request_interval_seconds"] - time.monotonic(),
                       self.cooldown_until[lane] - time.monotonic())
            if wait:
                time.sleep(wait)
            self.last_start[lane] = time.monotonic()
            record = {"lane": lane, "model": payload["model"], "started_at": time.time(),
                      "rate_wait_seconds": wait, "maximum_output_tokens": payload["max_output_tokens"]}
            response = None
            sent_headers = False
            try:
                body = json.dumps(payload, separators=(",", ":")).encode()
                record["input_bytes"] = len(body)
                request = urllib.request.Request(provider["url"].rstrip("/") + "/responses", data=body,
                    headers={"Authorization": "Bearer " + provider["key"],
                             "Content-Type": "application/json", "Accept": "text/event-stream"})
                try:
                    response = urllib.request.urlopen(request, timeout=180)
                except urllib.error.HTTPError as error:
                    response = error
                record["http_status"] = response.status
                if response.status in (429, 500, 502, 503, 504):
                    self.cooldown_until[lane] = time.monotonic() + self.policy["infrastructure_retry_cooldown_seconds"]
                handler.send_response(response.status)
                handler.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                handler.send_header("Connection", "close")
                handler.end_headers()
                sent_headers = True
                count = 0
                while True:
                    chunk = response.read1(8192)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > self.policy["maximum_response_bytes"]:
                        record["output_limit_exceeded"] = True
                        raise OSError("Upstream response exceeded the output limit")
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                record["output_bytes"] = count
            except (OSError, http.client.HTTPException) as exc:
                record["error_type"] = type(exc).__name__
                if not sent_headers or hasattr(handler, "maximum_body"):
                    handler.send_error(502, "Operator model upstream unavailable")
                self.cooldown_until[lane] = time.monotonic() + self.policy["infrastructure_retry_cooldown_seconds"]
            finally:
                if response:
                    response.close()
                record["elapsed_seconds"] = time.monotonic() - self.last_start[lane]
                with self.log_path.open("a") as log:
                    log.write(json.dumps(record) + "\n")


def serve_files(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    lock = (root / "lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ("inbox", "outbox"):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name).chmod(0o700)
    workspace = Path(__file__).resolve().parent.parent.parent
    policy = json.loads((workspace / "papers/active-reward-modeling/acquisition-paper-aligned/model-request-policy.json").read_text())
    proxy = ModelProxy(workspace / ".codex/config.toml", workspace / ".codex/auth.json",
                       policy, root / "requests.jsonl", "unused", port=0)
    processed = set()
    (root / "pid").write_text(str(os.getpid()))
    for lane in proxy.providers:
        rate_path = root / ("rate-" + lane + ".json")
        if not rate_path.exists():
            rate_path = root / "rate.json"
        if rate_path.exists():
            remaining = max(0, json.loads(rate_path.read_text())["next_allowed_at"] - time.time())
            proxy.cooldown_until[lane] = time.monotonic() + remaining

    class Capture:
        def __init__(self):
            self.maximum_body = policy["maximum_response_bytes"]
            self.wfile = io.BytesIO()
            self.status = 502
            self.content_type = "application/json"
        def send_response(self, status):
            self.status = status
        def send_header(self, name, value):
            if name == "Content-Type":
                self.content_type = value
        def end_headers(self):
            pass
        def send_error(self, status, message):
            self.status = status
            self.wfile.seek(0)
            self.wfile.truncate()
            self.wfile.write(json.dumps({"error": {"message": message, "type": "operator_relay"}}).encode())

    def consume(request, lane, payload):
        capture = Capture()
        started = request.with_suffix(".started")
        rate_path = root / ("rate-" + lane + ".json")
        if started.exists():
            capture.send_error(502, "Previous relay interrupted; request was not repeated")
        else:
            started.touch()
            # Keep each lane's conservative cooldown across operator restarts.
            rate_path.write_text(json.dumps({"next_allowed_at": time.time() +
                max(0, proxy.cooldown_until[lane] - time.monotonic()) +
                policy["infrastructure_retry_cooldown_seconds"]}))
            proxy.forward(capture, lane, payload)
            remaining = max(proxy.last_start[lane] + policy["minimum_request_interval_seconds"],
                            proxy.cooldown_until[lane]) - time.monotonic()
            rate_path.write_text(json.dumps({"next_allowed_at": time.time() + max(0, remaining)}))
        (root / "outbox" / (request.stem + ".body")).write_bytes(capture.wfile.getvalue())
        response = root / "outbox" / (request.stem + ".json")
        temporary = response.with_suffix(".tmp")
        temporary.write_text(json.dumps({"status": capture.status, "content_type": capture.content_type}))
        temporary.replace(response)

    workers = ThreadPoolExecutor(max_workers=len(proxy.providers))
    active = {}
    try:
        while True:
            for lane, future in list(active.items()):
                if future.done():
                    future.result()
                    del active[lane]
            for request in sorted((root / "inbox").glob("*.json"), key=lambda p: p.stat().st_mtime_ns):
                if request.stem in processed:
                    continue
                response = root / "outbox" / (request.stem + ".json")
                body = root / "outbox" / (request.stem + ".body")
                if response.is_file() and body.is_file():
                    processed.add(request.stem)
                    continue
                if request.is_symlink() or request.stat().st_size > policy["maximum_input_bytes"] + 4096:
                    processed.add(request.stem)
                    continue
                try:
                    value = json.loads(request.read_text())
                    lane = value["lane"]
                    payload = value["payload"]
                    if lane not in proxy.providers or not isinstance(payload, dict):
                        processed.add(request.stem)
                        continue
                    if payload.get("model") != proxy.providers[lane]["model"]:
                        processed.add(request.stem)
                        continue
                    if len(json.dumps(payload, separators=(",", ":")).encode()) > policy["maximum_input_bytes"]:
                        processed.add(request.stem)
                        continue
                except (OSError, TypeError, ValueError, KeyError):
                    processed.add(request.stem)
                    continue
                if lane in active:
                    continue
                processed.add(request.stem)
                active[lane] = workers.submit(consume, request, lane, payload)
            time.sleep(0.2)
    finally:
        workers.shutdown(wait=True)
        proxy.server.server_close()
        lock.close()


def relay(port):
    import selectors
    import socket
    import sys
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        connection.settimeout(None)
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin.buffer, selectors.EVENT_READ, "input")
            selector.register(connection, selectors.EVENT_READ, "socket")
            while True:
                for key, _ in selector.select():
                    if key.data == "input":
                        content = os.read(sys.stdin.fileno(), 65536)
                        if not content:
                            return
                        connection.sendall(content)
                    else:
                        content = connection.recv(65536)
                        if not content:
                            return
                        sys.stdout.buffer.write(content)
                        sys.stdout.buffer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-port", type=int)
    parser.add_argument("--serve-files", type=Path)
    args = parser.parse_args()
    if args.serve_files:
        serve_files(args.serve_files)
    elif args.relay_port:
        relay(args.relay_port)
    else:
        parser.error("Specify --serve-files or --relay-port")

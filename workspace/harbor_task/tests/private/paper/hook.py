"""Frozen CPU scientific hook; candidate methods are data, never executed code."""
from __future__ import annotations

import hashlib
import errno
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA = Path("environment/public_assets")
METHOD_KEYS = {"schema", "model", "normalize", "optimizer", "learning_rate", "l2"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path, limit=262144):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Expected bounded regular JSON file: " + str(path))
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def method_at(path):
    method = read_json(path, 16384)
    if not isinstance(method, dict) or set(method) != METHOD_KEYS:
        raise ValueError("Method must contain exactly the six documented fields")
    if method["schema"] != "arm.quickstart.method.v1":
        raise ValueError("Unsupported method schema")
    if method["model"] not in ("mlp", "linear") or type(method["normalize"]) is not bool:
        raise ValueError("Invalid model or normalization")
    if method["optimizer"] not in ("adam", "lbfgs"):
        raise ValueError("Invalid optimizer")
    for key, low, high in (("learning_rate", 1e-5, 1.0), ("l2", 0.0, 0.1)):
        value = method[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid " + key)
    return method


def protocol():
    return read_json(ROOT / "protocol.json")


def safe_task_path(value):
    path = Path(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("Expected canonical task-relative path")
    target = Path.cwd() / path
    if not target.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Task path escaped the snapshot")
    return target


def model_for(method):
    import torch
    from torch import nn
    from bradleyterry import MLP
    if method["model"] == "mlp":
        class RewardMLP(MLP):
            def embd(self, x):
                if method["normalize"]:
                    x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
                return super().embd(x)
        return RewardMLP(input_dim=2048, hidden_dim=64)
    class RewardLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2048, 1, bias=False)
        def forward(self, x):
            if method["normalize"]:
                x = torch.nn.functional.layer_norm(x, (x.shape[-1],))
            return self.projection(x)[:, 0]
    return RewardLinear()


def state_digest(model):
    """Identity of the actual parameter tensors, independent of torch ZIP metadata."""
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        result.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def fit(model, a, b, labels, method, updates, max_updates):
    import torch
    from torch.nn import functional as F

    def record():
        if len(updates) >= max_updates:
            raise ValueError("Optimizer update budget exceeded")
        updates.append({"step": len(updates) + 1, "state_sha256": state_digest(model)})

    if method["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=method["learning_rate"])
        for _ in range(30):
            optimizer.zero_grad()
            # Preserve the upstream Baseline's sigmoid/BCE arithmetic.
            loss = F.binary_cross_entropy(torch.sigmoid(model(a) - model(b)), labels.float())
            loss = loss + method["l2"] * sum(p.square().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            record()
    else:
        class AccountedLBFGS(torch.optim.LBFGS):
            # The pinned Torch implementation uses _add_grad for both accepted
            # iterations and temporary line-search trials. Count committed
            # updates only, retaining each real resulting parameter state.
            in_trial = False
            def _directional_evaluate(self, *args, **kwargs):
                self.in_trial = True
                try:
                    return super()._directional_evaluate(*args, **kwargs)
                finally:
                    self.in_trial = False
            def _add_grad(self, *args, **kwargs):
                result = super()._add_grad(*args, **kwargs)
                if not self.in_trial:
                    record()
                return result
        optimizer = AccountedLBFGS(model.parameters(), lr=method["learning_rate"],
                                  max_iter=30, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(a) - model(b), labels.float())
            loss = loss + method["l2"] * sum(p.square().sum() for p in model.parameters())
            loss.backward()
            return loss
        optimizer.step(closure)


def arrays():
    from oracle_data import arrays as load_arrays
    return load_arrays(ROOT, DATA)


def trainer_files():
    return {str(ROOT / name): digest_file(ROOT / name) for name in (
        "hook.py", "oracle_data.py", "oracle_sampling.py",
        "bradleyterry.py", "rules_pre_paired.py", "protocol.json",
    )}


def oracle_audit(request):
    from oracle_sampling import audit_oracle
    seed = request["seed"]
    if type(seed) is not int or seed not in protocol()["seeds"]:
        raise ValueError("Oracle audit requires a frozen executor seed")
    if request["training_accounting"] != read_json(ROOT / "accounting-policy.json"):
        raise ValueError("Oracle audit accounting policy changed")
    if request["training_accounting"]["trainer_files"] != trainer_files():
        raise ValueError("Oracle audit trainer identity changed")
    result = audit_oracle(arrays(), seed, model_for, state_digest,
                          method_at("solution/method.json"))
    (Path(request["output_directory"]) / "oracle-audit.json").write_bytes(canonical(result))
    return {key: value for key, value in result.items() if key != "rounds"}


def evaluate_model(model, data, split="dev"):
    import numpy as np
    import torch
    from scipy.stats import rankdata
    if split not in ("dev", "hidden"):
        raise ValueError("Unknown evaluation split")
    embeddings, rewards = data[split + "-embeddings"], data[split + "-rewards"]
    order = np.random.default_rng(1729).permutation(4188)
    correlations, chosen = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), 8):
            ids = order[start:start + 8]
            x = np.asarray(embeddings[:, ids, :], dtype=np.float32)
            prediction = model(torch.from_numpy(x.reshape(-1, 2048))).numpy().reshape(10, len(ids))
            truth = np.asarray(rewards[:, ids])
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite predictions")
            for column in range(len(ids)):
                a, b = rankdata(prediction[:, column]), rankdata(truth[:, column])
                a, b = a - a.mean(), b - b.mean()
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                correlations.append(float(a @ b / denominator) if denominator else 0.0)
                chosen.append(float(truth[np.argmax(prediction[:, column]), column]))
    return {"one_minus_spearman": float(1 - np.mean(correlations)),
            "best_of_10": float(np.mean(chosen)), "evaluated_prompts": 4188}


def train(request):
    import numpy as np
    import torch
    from rules_pre_paired import random_candidate, random_rule
    seed = request["seed"]
    fixed = protocol()
    if type(seed) is not int or seed not in fixed["seeds"] or request.get("resume_checkpoint") is not None:
        raise ValueError("Independent training requires a declared executor seed and no resume")
    if request["operation"] == "train":
        if read_json(safe_task_path(request["config"])) != fixed:
            raise ValueError("Calibration configuration differs from frozen protocol")
        method = method_at(safe_task_path(request["implementation"]))
    else:
        method = method_at("solution/method.json")
    policy = request["training_accounting"]
    if policy["mode"] != "trusted_updates" or policy["candidate_execution"] != "declarative_only":
        raise ValueError("Unsupported accounting policy")
    expected_files = trainer_files()
    if policy["trainer_files"] != expected_files or policy["max_updates"] != 480:
        raise ValueError("Accounting policy does not bind the complete trusted trainer")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output = Path(request["output_directory"])
    if (output / "checkpoint").exists() or (output / "accounting.json").exists():
        raise ValueError("Training output must be fresh")
    data = arrays()
    # Inputs are immutable; a private copy keeps upstream Torch operations
    # away from the NumPy read-only mmap and stays within the memory budget.
    embeddings = torch.from_numpy(np.array(data["train-embeddings"], copy=True))
    rewards = torch.from_numpy(np.array(data["train-rewards"], copy=True))
    scale = float(np.std(data["train-rewards"])) + 1e-8
    past_a = past_b = labels = None
    updates, evaluated_steps, history = [], [], []
    started = time.monotonic()
    with (output / "train.jsonl").open("x") as log:
        for round_index in range(16):
            from oracle_sampling import sample_round
            chosen_a, chosen_b, selected_labels = sample_round(
                embeddings, rewards, scale, seed, round_index)
            past_a = chosen_a if past_a is None else torch.cat((past_a, chosen_a))
            past_b = chosen_b if past_b is None else torch.cat((past_b, chosen_b))
            labels = selected_labels if labels is None else torch.cat((labels, selected_labels))
            torch.manual_seed(seed)
            model = model_for(method)
            fit(model, past_a, past_b, labels, method, updates, policy["max_updates"])
            if not updates or updates[-1]["state_sha256"] != state_digest(model):
                raise ValueError("Final trained state is missing from the update ledger")
            evaluated_steps.append(len(updates))
            metrics = evaluate_model(model, data)
            row = {"round": round_index + 1, "labelled_pairs": len(labels),
                   "actual_updates": len(updates), **metrics}
            history.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
    checkpoint = output / "checkpoint"
    torch.save({"state_dict": model.state_dict(), "method": method, "seed": seed,
                "selected_step": len(updates)}, checkpoint)
    ledger = {"schema": "deepscientist.training-accounting.v1",
              "policy_sha256": hashlib.sha256(canonical(policy)).hexdigest(),
              "seed": seed, "training_runs": 1, "resume_checkpoint_sha256": None,
              "actual_updates": len(updates), "selected_step": len(updates),
              "selected_state_sha256": state_digest(model),
              "checkpoint_sha256": digest_file(checkpoint), "updates": updates,
              "evaluated_steps": sorted(set(evaluated_steps))}
    (output / "accounting.json").write_bytes(canonical(ledger))
    (output / "metrics.json").write_bytes(canonical({
        "schema": "arm.quickstart.training.v1", "seed": seed, "method": method,
        "history": history, "elapsed_seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "formal_acceptance": False}))
    return {"status": "succeeded"}


def evaluate(request):
    import torch
    torch.set_num_threads(2)
    checkpoint = safe_task_path(request["checkpoint"])
    accounting = read_json(checkpoint.parent / "accounting.json")
    if digest_file(checkpoint) != accounting["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from its ledger")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["seed"] != request["seed"] or saved["selected_step"] != accounting["selected_step"]:
        raise ValueError("Checkpoint seed or selection changed")
    # Validate the stored method with the same strict interface, without
    # running any candidate Python code or trusting reported metric values.
    method = saved["method"]
    if set(method) != METHOD_KEYS:
        raise ValueError("Checkpoint method schema changed")
    model = model_for(method)
    model.load_state_dict(saved["state_dict"], strict=True)
    state = state_digest(model)
    if state != accounting["selected_state_sha256"]:
        raise ValueError("Decoded model state differs from the selected update")
    metrics = evaluate_model(model, arrays())
    return {"status": "succeeded", "raw_metric": metrics["one_minus_spearman"],
            "actual_updates": accounting["actual_updates"],
            "selected_step": accounting["selected_step"], "selected_state_sha256": state,
            "accounting_sha256": digest_file(checkpoint.parent / "accounting.json")}


def validate(request):
    import difflib
    import subprocess
    import tomllib
    for name in ("instruction.md", "task.toml", "environment/Dockerfile",
                 "tests/submission_contract.json", "tests/method_test_contract.py"):
        if not Path(name).is_file():
            raise ValueError("Missing public task file: " + name)
    if any(Path("tests/hidden_assets").iterdir()) or any(Path("output").iterdir()):
        raise ValueError("Initial public hidden/output directories must be empty")
    if read_json(DATA / "protocol.json") != protocol():
        raise ValueError("Public and immutable protocol differ")
    config = tomllib.loads(Path("task.toml").read_text())
    if config["resources"] != {"cpu_cores": 8, "memory_mb": 32768, "gpu_count": 0,
                               "network_access": False, "max_concurrent_jobs": 1}:
        raise ValueError("Resource contract changed")
    subprocess.run(["/opt/arm-python/bin/python3.11", "tests/method_test_contract.py"],
                   check=True, stdout=sys.stderr, stderr=sys.stderr)
    starter = method_at("environment/starter/method.json")
    candidate = method_at("solution/method.json")
    diff = "".join(difflib.unified_diff(
        json.dumps(starter, sort_keys=True, indent=2).splitlines(True),
        json.dumps(candidate, sort_keys=True, indent=2).splitlines(True),
        fromfile="starter/method.json", tofile="solution/method.json"))
    for item in read_json(DATA / "data_manifest.json")["files"]:
        if digest_file(DATA / item["file"]) != item["sha256"]:
            raise ValueError("Dataset digest changed: " + item["file"])
    if os.geteuid() == 0 or "CapEff:\t0000000000000000" not in Path("/proc/self/status").read_text():
        raise ValueError("Scientific hook requires non-root execution without capabilities")
    try:
        with Path("task.toml").open("r+"):
            raise ValueError("Task snapshot is writable")
    except PermissionError:
        pass
    except OSError as exc:
        import errno
        if exc.errno != errno.EROFS:
            raise
    probe = Path(request["output_directory"]) / "write-probe"
    probe.write_text("verified")
    probe.unlink()
    return {"status": "succeeded",
            "checks": {key: "passed" for key in (
                "layout", "instruction", "test_contract", "continuous_score", "invalid_status",
                "isolation", "surface_validation", "resource_limits")},
            "access_manifest": read_json(DATA / "access_manifest.json"),
            "surface_validation": {"status": "passed", "scope": "public_declarative_method",
                                   "allowed_paths": ["solution/method.json"]},
            "surface_diff": diff}


def main():
    try:
        request = read_json(sys.argv[1])
        operation = request["operation"]
        if operation == "validate":
            result = validate(request)
        elif operation in ("train", "candidate_train"):
            result = train(request)
        elif operation == "oracle_audit":
            result = oracle_audit(request)
        elif operation in ("evaluate", "reload", "candidate_evaluate"):
            result = evaluate(request)
        else:
            raise ValueError("Unknown scientific operation")
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        print(json.dumps({"status": "failed", "error": str(exc), "raw_metric": None}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

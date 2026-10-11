"""Reload all packaged Baseline checkpoints in the actual scoring environment."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "/opt/paper")
import hook
import torch
from bradleyterry import MLP

torch.set_num_threads(6)
torch.use_deterministic_algorithms(True)
root = Path("/opt/arm-private")
split = os.environ.get("ARM_SCORING_SPLIT", "dev")
if split == "hidden":
    from oracle_data import hidden_arrays
    data = hidden_arrays("/opt/hidden")
else:
    data = hook.arrays()
baseline = hook.read_json(root / ("baseline-hidden.json" if split == "hidden" else "baseline.json"))
if split == "hidden":
    assert baseline["hidden_manifest_sha256"] == hook.digest_file("/opt/hidden/manifest.json")
rows = []
for seed, expected in zip(baseline["seeds"], baseline["raw_values"]):
    directory = root / "calibration" / f"seed_{seed}"
    receipt = hook.read_json(directory / ("hidden-evaluation.json" if split == "hidden" else "evaluation.json"))
    assert receipt["seed"] == seed and receipt["raw_metric"] == expected
    assert receipt["protocol_sha256"] == hook.digest_file("/protocol.json")
    assert receipt["selector_sha256"] == baseline["selector_sha256"]
    assert receipt["checkpoint_sha256"] == hook.digest_file(directory / "checkpoint")
    if split == "hidden":
        assert receipt["hidden_manifest_sha256"] == baseline["hidden_manifest_sha256"]
    model = MLP(input_dim=2048, hidden_dim=64)
    saved = torch.load(directory / "checkpoint", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    metric = hook.evaluate_model(model, data, split)
    assert metric["one_minus_spearman"] == expected, (seed, metric, expected)
    rows.append({"seed": seed, "raw_metric": expected, "checkpoint_reload_match": True})
    print(json.dumps({"event": "calibration_reload", "seed": seed, "evaluation_split": split,
                      "checkpoint_sha256": receipt["checkpoint_sha256"], **metric}), flush=True)
Path("/var/lib/arm/calibration-reload.json").write_bytes(hook.canonical({
    "status": "passed", "protocol_sha256": hook.digest_file("/protocol.json"), "seeds": rows}))

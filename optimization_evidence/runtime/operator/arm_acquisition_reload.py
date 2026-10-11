"""Recompute frozen dev metrics from successful checkpoints without training."""
import json
from pathlib import Path
import sys

sys.path.insert(0, "/opt/paper")
import hook
import torch
from bradleyterry import MLP


def main():
    protocol = hook.read_json("/protocol.json")
    identity = hook.digest_file("/protocol.json")
    torch.set_num_threads(protocol["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    data = hook.arrays()
    for seed in protocol["seeds"]:
        root = Path("/checkpoints") / f"seed_{seed}"
        receipt = hook.read_json(root / "evaluation.json")
        if receipt["status"] != "succeeded" or receipt["protocol_sha256"] != identity:
            raise ValueError("Checkpoint receipt does not match the frozen protocol")
        digest = hook.digest_file(root / "checkpoint")
        if digest != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint identity changed")
        saved = torch.load(root / "checkpoint", map_location="cpu", weights_only=True)
        if saved["seed"] != seed or saved["protocol_sha256"] != identity:
            raise ValueError("Checkpoint seed or protocol changed")
        model = MLP(input_dim=2048, hidden_dim=protocol["hidden_units"])
        model.load_state_dict(saved["state_dict"])
        metric = hook.evaluate_model(model, data)
        if metric["one_minus_spearman"] != receipt["raw_metric"] or metric["best_of_10"] != receipt["best_of_10"]:
            raise ValueError("Reloaded checkpoint metric differs from successful evaluation")
        print(json.dumps({"seed": seed, "matched": True, "checkpoint_sha256": digest,
                          "raw_metric": metric["one_minus_spearman"], "best_of_10": metric["best_of_10"],
                          "new_training_runs": 0}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

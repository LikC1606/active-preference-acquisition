"""Evaluate existing B/R checkpoints on Hidden; never train or tune a method."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, "/opt/paper")
import hook
from oracle_data import hidden_arrays
import torch
from bradleyterry import MLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    data = hidden_arrays("/opt/hidden")
    baseline = hook.read_json("/opt/arm-private/baseline.json")
    rows = {"baseline": [], "reference": []}
    args.output.mkdir(parents=True, exist_ok=False)
    for arm, root in (("baseline", Path("/opt/arm-private/calibration")), ("reference", args.reference)):
        for seed in baseline["seeds"]:
            directory = root / f"seed_{seed}"
            previous = hook.read_json(directory / "evaluation.json")
            checkpoint = directory / "checkpoint"
            assert previous["checkpoint_sha256"] == hook.digest_file(checkpoint)
            model = MLP(input_dim=2048, hidden_dim=64)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["seed"] == seed and saved["protocol_sha256"] == baseline["protocol_sha256"]
            model.load_state_dict(saved["state_dict"], strict=True)
            metric = hook.evaluate_model(model, data, "hidden")
            receipt = {**previous, "evaluation_split": "hidden", "raw_metric": metric["one_minus_spearman"],
                       "best_of_10": metric["best_of_10"], "evaluated_prompts": metric["evaluated_prompts"],
                       "hidden_manifest_sha256": hook.digest_file("/opt/hidden/manifest.json"),
                       "operation": "evaluate_existing_checkpoint", "new_training_runs": 0}
            dest = args.output / arm / f"seed_{seed}"
            dest.mkdir(parents=True)
            (dest / "hidden-evaluation.json").write_bytes(hook.canonical(receipt))
            rows[arm].append(receipt)
            print(json.dumps({"event": "hidden_checkpoint_evaluated", "arm": arm, **receipt}), flush=True)
    b = [r["raw_metric"] for r in rows["baseline"]]
    r = [r["raw_metric"] for r in rows["reference"]]
    bm, rm = math.fsum(b) / len(b), math.fsum(r) / len(r)
    hidden_baseline = {**baseline, "evaluation_split": "hidden", "raw_values": b, "mean_error": bm,
                       "hidden_manifest_sha256": hook.digest_file("/opt/hidden/manifest.json")}
    (args.output / "baseline-hidden.json").write_bytes(hook.canonical(hidden_baseline))
    summary = {"evaluation_split": "hidden", "new_training_runs": 0,
               "baseline_values": b, "reference_values": r, "baseline_mean": bm,
               "reference_mean": rm, "baseline_sample_std": statistics.stdev(b),
               "reference_score": (bm - rm) / bm,
               "improvement_in_baseline_std": (bm - rm) / statistics.stdev(b),
               "protocol_sha256": baseline["protocol_sha256"],
               "hidden_manifest_sha256": hidden_baseline["hidden_manifest_sha256"]}
    (args.output / "comparison.json").write_bytes(hook.canonical(summary))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

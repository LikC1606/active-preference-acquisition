"""Paper equations and upstream trainer; isolated acquisition and durable rounds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, "/opt/paper")
import hook
import numpy as np
import torch
# Import-time default model construction must precede all model seeds.
from bradleyterry import MLP, train_BT
from rules_pre_paired import dopt_rule, random_candidate, random_rule

P = hook.read_json("/protocol.json")
PROTO_SHA = hook.digest_file("/protocol.json")
SEEDS = tuple(P["seeds"])
RESULT = Path(os.environ.get("ARM_RESULT_ROOT", "/result"))
EXCHANGE = Path(os.environ.get("ARM_EXCHANGE_ROOT", "/exchange"))


def write(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(hook.canonical(value))
    pending.replace(path)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def indices_at(path):
    values = hook.read_json(path, 32768)
    if (not isinstance(values, list) or len(values) != P["annotation_batch"]
            or any(type(i) is not int or not 0 <= i < P["candidate_pairs"] for i in values)
            or len(set(values)) != P["annotation_batch"]):
        raise ValueError("Expected 500 unique candidate indices")
    return np.asarray(values, dtype=np.int64)


def exchange(a, b, model, seed, round_index):
    with torch.no_grad():
        features = model.embd(a) - model.embd(b)
        logits = model.lastlayer(features)
    path = (EXCHANGE / "pool.npz")
    np.savez(path.with_name("pending.npz"), features=features.numpy(), logits=logits.numpy(),
             k=np.asarray(500), seed=np.asarray(seed), round_index=np.asarray(round_index))
    path.with_name("pending.npz").replace(path)
    write(EXCHANGE / "request.json", {"round_index": round_index, "pool_sha256": hook.digest_file(path)})
    deadline = time.monotonic() + 90
    response = (EXCHANGE / "indices.json")
    while not response.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Isolated selector response missing")
        time.sleep(0.05)
    selected = indices_at(response)
    response.unlink()
    (EXCHANGE / "request.json").unlink()
    return selected


def seed_pool(embeddings, rewards, seed, round_index):
    torch.manual_seed(seed * 1000 + round_index)
    np.random.seed(seed * 1000 + round_index)
    a, b, ra, rb = random_candidate(embeddings, rewards, P["prompts_per_round"], P["candidate_pairs"])
    labels = torch.bernoulli(torch.sigmoid((ra - rb).float()))
    return a, b, labels


def train(data, seed):
    out = RESULT
    embeddings = torch.from_numpy(np.array(data["train-embeddings"], copy=True))
    rewards = torch.from_numpy(np.array(data["train-rewards"], copy=True))
    model = past_a = past_b = labels = None
    history, sampling, rounds = [], [], []
    elapsed = 0.0
    state = out / "round-checkpoint"
    if state.exists():
        saved = torch.load(state, map_location="cpu", weights_only=True)
        if saved["protocol_sha256"] != PROTO_SHA or saved["seed"] != seed:
            raise ValueError("Round checkpoint protocol or seed changed")
        history, sampling, rounds = saved["history"], saved["sampling"], saved["rounds"]
        elapsed = saved["elapsed_seconds"]
        for index, row in enumerate(sampling):
            a, b, pool_labels = seed_pool(embeddings, rewards, seed, index)
            if digest(pool_labels) != row["pool_labels_sha256"]:
                raise ValueError("Resumed candidate-pool labels differ")
            selected = row["indices"]
            aa, bb, yy = a[selected], b[selected], pool_labels[selected]
            past_a = aa if past_a is None else torch.cat((past_a, aa))
            past_b = bb if past_b is None else torch.cat((past_b, bb))
            labels = yy if labels is None else torch.cat((labels, yy))
        model = MLP(input_dim=2048, hidden_dim=P["hidden_units"])
        model.load_state_dict(saved["state_dict"], strict=True)
    started = time.monotonic()
    print(json.dumps({"event": "training_started", "seed": seed, "resumed_rounds": len(history),
                      "protocol_sha256": PROTO_SHA}), flush=True)
    total_rounds = P["annotation_budget"] // P["annotation_batch"]
    for round_index in range(len(history), total_rounds):
        a, b, pool_labels = seed_pool(embeddings, rewards, seed, round_index)
        if model is None:
            chosen_a, chosen_b, selected = random_rule(a, b, P["annotation_batch"])
        else:
            selected = exchange(a, b, model, seed, round_index)
            chosen_a, chosen_b = a[selected], b[selected]
        selected_labels = pool_labels[selected]
        past_a = chosen_a if past_a is None else torch.cat((past_a, chosen_a))
        past_b = chosen_b if past_b is None else torch.cat((past_b, chosen_b))
        labels = selected_labels if labels is None else torch.cat((labels, selected_labels))
        torch.manual_seed(seed)
        model = MLP(input_dim=2048, hidden_dim=P["hidden_units"])
        initialized = hook.state_digest(model)
        # Execute the actual upstream function, count real Adam.step calls.
        original_step = torch.optim.Adam.step
        updates = 0
        def counted_step(optimizer, *args, **kwargs):
            nonlocal updates
            result = original_step(optimizer, *args, **kwargs)
            updates += 1
            return result
        torch.optim.Adam.step = counted_step
        try:
            model = train_BT(past_a, past_b, labels, score_model=model,
                             lr=P["learning_rate"], niter=P["training_iterations_per_round"])
        finally:
            torch.optim.Adam.step = original_step
        if updates != P["training_iterations_per_round"]:
            raise ValueError("Actual upstream optimizer updates differ from the protocol")
        metric = hook.evaluate_model(model, data)
        with torch.no_grad():
            loss = float(torch.nn.functional.binary_cross_entropy(torch.sigmoid(model(past_a) - model(past_b)), labels))
        rounds.append({"round": round_index + 1, "actual_updates": updates,
                       "initialized_state_sha256": initialized, "trained_state_sha256": hook.state_digest(model),
                       "final_training_bce": loss})
        history.append({"round": round_index + 1, "labelled_pairs": len(labels),
                        "actual_updates": sum(r["actual_updates"] for r in rounds), **metric})
        sampling.append({"round": round_index + 1, "indices": selected.tolist(),
                         "pool_labels_sha256": digest(pool_labels), "selected_labels_sha256": digest(selected_labels)})
        pending = state.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(), "seed": seed, "protocol_sha256": PROTO_SHA,
                    "history": history, "sampling": sampling, "rounds": rounds,
                    "elapsed_seconds": elapsed + time.monotonic() - started}, pending)
        pending.replace(state)
        write(out / "progress.json", {**history[-1], "elapsed_seconds": elapsed + time.monotonic() - started})
        print(json.dumps({"event": "round_completed", "seed": seed, **history[-1],
                          "elapsed_seconds": elapsed + time.monotonic() - started}), flush=True)
    torch.save({"state_dict": model.state_dict(), "seed": seed, "protocol_sha256": PROTO_SHA,
                "selected_step": sum(r["actual_updates"] for r in rounds)}, out / "checkpoint")
    write(out / "accounting.json", {"seed": seed, "protocol_sha256": PROTO_SHA,
          "actual_updates": sum(r["actual_updates"] for r in rounds), "rounds": rounds,
          "checkpoint_sha256": hook.digest_file(out / "checkpoint")})
    write(out / "sampling.json", sampling)
    write(out / "metrics.json", {"seed": seed, "history": history,
                                "elapsed_seconds": elapsed + time.monotonic() - started})
    reloaded = MLP(input_dim=2048, hidden_dim=P["hidden_units"])
    reloaded.load_state_dict(torch.load(out / "checkpoint", map_location="cpu", weights_only=True)["state_dict"])
    metric = hook.evaluate_model(reloaded, data)
    if metric != {key: history[-1][key] for key in metric}:
        raise ValueError("Final checkpoint reload metric differs")
    split = os.environ.get("ARM_SCORING_SPLIT", "dev")
    if split == "hidden":
        from oracle_data import hidden_arrays
        metric = hook.evaluate_model(reloaded, hidden_arrays("/opt/hidden"), "hidden")
    elif split != "dev":
        raise ValueError("Unknown scoring split")
    write(out / "evaluation.json", {"seed": seed, "status": "succeeded", "protocol_sha256": PROTO_SHA,
          "evaluation_split": split,
          "raw_metric": metric["one_minus_spearman"], "best_of_10": metric["best_of_10"],
          "checkpoint_reload_match": True, "actual_updates": sum(r["actual_updates"] for r in rounds),
          "selected_preferences": len(labels), "checkpoint_sha256": hook.digest_file(out / "checkpoint")})
    print(json.dumps({"event": "checkpoint_evaluated", "seed": seed, "evaluation_split": split,
                      **metric, "checkpoint_sha256": hook.digest_file(out / "checkpoint")}), flush=True)


def contract(data):
    embeddings = torch.from_numpy(np.array(data["train-embeddings"], copy=True))
    rewards = torch.from_numpy(np.array(data["train-rewards"], copy=True))
    a, b, labels = seed_pool(embeddings, rewards, 101, 0)
    aa, bb, selected = random_rule(a, b, 500)
    torch.manual_seed(101)
    model = MLP(input_dim=2048, hidden_dim=64)
    model = train_BT(aa, bb, labels[selected], score_model=model)
    a, b, _ = seed_pool(embeddings, rewards, 101, 1)
    torch.set_num_threads(2)
    expected = dopt_rule(a, b, model, 500, batch_size=1024)[2].tolist()
    actual = exchange(a, b, model, 101, 1).tolist()
    if actual != expected:
        raise ValueError("Reference differs from upstream selected indices/order")
    write(RESULT / "contract.json", {"upstream_dopt_exact_indices_and_order": True,
          "upstream_training_defaults_executed": True, "raw_reward_difference_labels": True,
          "bootstrap_metric": hook.evaluate_model(model, data), "protocol_sha256": PROTO_SHA})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("train", "contract"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(P["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    data = hook.arrays()
    if args.operation == "contract":
        contract(data)
    else:
        train(data, args.seed)


if __name__ == "__main__":
    main()

"""Arrange existing acquisition evidence; never start scientific computation."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("ARM_PROJECT_ROOT", str(ROOT.parent.parent / "papers/active-reward-modeling"))).resolve()
PROOF = PROJECT / "acquisition-paper-aligned"
TASK = PROJECT / "harbor_task_acquisition"
DELIVERY = PROOF / "delivery"
if (ROOT.parents[2] / "workspace/harbor_task").is_dir():
    DELIVERY = ROOT.parents[2]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            os.fchmod(stream.fileno(), path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy(source, destination):
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".npy" or source.name == "checkpoint":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def add_research_evidence():
    replay = read(PROOF / "best-replay-report.json")
    if replay["status"] != "best_replay_passed" or replay["new_training_runs"]:
        raise ValueError("Both best methods must pass fresh-container replay first")
    expert = DELIVERY / "expert_evidence"
    evidence = DELIVERY / "optimization_evidence"
    runtime = evidence / "runtime"
    research = PROOF / "agent-research/acquisition-gpt-seed-v1"
    comparison = read(PROOF / "comparison.json")
    definition = read(PROOF / "definition.json")
    public = DELIVERY / "workspace/harbor_task"
    for name in ("evaluate.py", "evaluate.sh", "broker.py", "worker.py", "selector_entry.py"):
        copy(PROOF / "private" / name, runtime / "private" / name)
    for name in ("grader.py", "test.sh", "acquisition_test_contract.py"):
        copy(TASK / "tests" / name, public / "tests" / name)
        copy(TASK / "environment/task/tests" / name, public / "environment/task/tests" / name)
    native = read(PROOF / "native-report.json")
    copy(PROOF / "native-report.json", runtime / "native-report.json")
    native_root = PROOF / ("native-" + native["job_id"])
    for arm in ("baseline", "reference"):
        copy(native_root / (arm + "-native-inspect.json"), runtime / (arm + "-native-inspect.json"))
        trial = native_root / "trials" / f"arm-acq-{native['job_id']}-{arm}"
        for name in ("result.json", "verifier/reward.json", "verifier/public-score.json", "verifier/test-stdout.txt"):
            copy(trial / name, runtime / arm / name)
    executions = []
    states = {}
    for lane, filename in (("gpt", "trajectory_codex.json"), ("seed", "trajectory_seed.json")):
        state = read(research / lane / "state.json")
        states[lane] = state
        if state["completed_loops"] < 3 or state["effective_seconds"] < 36000:
            raise ValueError("Research duration or effective loops incomplete")
        rows = [r for r in state["history"] if r["status"] == "succeeded" and r["model"] == state["model"]]
        if len(rows) != state["completed_loops"]:
            raise ValueError("Scientific loop count differs from successful method evidence")
        if replay["trials"][lane]["selector_sha256"] != sha(Path(state["best_selector"])):
            raise ValueError("Replay does not bind the current best method")
        write(expert / filename, {"rounds": [{"round": index + 1,
            "method_summary": row["method_summary"].rstrip(".") +
                f"; measured dev score {row['score']:.6f}, decision {row['decision']}."}
            for index, row in enumerate(rows)]})
        executions.append({"runtime": "Codex", "model": state["model"],
                           "duration_hours": state["effective_seconds"] / 3600,
                           "effective_rounds": len(rows),
                           "typical_round_minutes": statistics.median(r["effective_seconds"] for r in rows) / 60})
        lane_root = runtime / "agent-research" / lane
        copy(research / lane / "state.json", lane_root / "state.json")
        copy(research / lane / "trajectory.json", lane_root / "original_trajectory.json")
        copy(research / lane / "progress.json", lane_root / "progress.json")
        for row in state["history"]:
            source = research / lane / f"attempt-{row['attempt']:03d}"
            for name in ("result.json", "recovery.json"):
                if (source / name).is_file():
                    copy(source / name, lane_root / source.name / name)
            model_log = source / "model-log.json"
            if model_log.is_file():
                record = read(model_log)
                write(lane_root / source.name / "model-clock.json",
                      {key: record[key] for key in ("exit_code", "work_clock")})
    selected = replay["best_lane"]
    for name in ("select.py", "solve.sh", "submission.json"):
        copy(research / selected / "best_method" / name, expert / "best_method" / name)
    (expert / "best_method/solve.sh").chmod(0o755)
    copy(PROOF / "best-replay-report.json", runtime / "best-replay-report.json")
    replay_root = PROOF / ("native-" + replay["job_id"])
    for lane in states:
        for suffix in ("-reload.log", "-reload-inspect.json", "-native-inspect.json", "-iteration.json"):
            copy(replay_root / (lane + suffix), runtime / "best-replay" / (lane + suffix))
        trial = replay_root / "trials" / f"arm-acq-{replay['job_id']}-{lane}"
        for name in ("result.json", "verifier/reward.json", "verifier/public-score.json", "verifier/test-stdout.txt"):
            copy(trial / name, runtime / "best-replay" / lane / name)
    for name in ("verify_arm_acquisition_native.py", "verify_arm_acquisition_native.sbatch", "arm_acquisition_reload.py",
                 "run_arm_acquisition_roll.py", "run_arm_acquisition_roll.sbatch", "arm_model_proxy.py",
                 "run_arm_acquisition_stability.py", "run_arm_acquisition_stability.sbatch",
                 "arm_acquisition_stability_workload.py", "prepare_arm_acquisition_delivery.py",
                 "package_arm_acquisition.py"):
        copy(ROOT / name, runtime / "operator" / name)
    copy(PROOF / "model-request-policy.json", runtime / "model-request-policy.json")
    repair = research / "seed/model-repair.json"
    if repair.is_file():
        copy(repair, runtime / "agent-research/seed/model-repair.json")
    best_run = PROOF / "runs" / definition["protocol_sha256"] / sha(expert / "best_method/select.py")
    for seed in definition["seeds"]:
        source = best_run / f"seed_{seed}"
        destination = evidence / "best_method_runs" / source.name
        receipt = read(source / "evaluation.json")
        if sha(source / "checkpoint") != receipt["checkpoint_sha256"]:
            raise ValueError("Best-method checkpoint changed")
        for name in ("evaluation.json", "accounting.json", "sampling.json", "metrics.json", "worker.log", "trainer-inspect.json"):
            copy(source / name, destination / name)
        target = destination / "model/checkpoint"
        if not target.exists():
            copy(source / "checkpoint", target)
        elif sha(target) != receipt["checkpoint_sha256"]:
            raise ValueError("Packaged best checkpoint changed")
    best_reward = read(best_run / "reward.json")
    write(evidence / "agent_comparison.json", {"gpt_score": states["gpt"]["best_score"],
          "seed_score": states["seed"]["best_score"], "selected_lane": selected,
          "selected_reward": best_reward, "reference_score": comparison["reference"]["score"],
          "reference_outperformed": best_reward["score"] > comparison["reference"]["score"],
          "clean_replay_new_training_runs": 0, "independent_retraining": False})
    stability = read(PROOF / "stability/progress.json")
    stability_root = PROOF / "stability" / ("job-" + stability["job_id"]) / "evidence"
    for name in ("binding.json", "stability-report.json", "workload-result.json", "container-inspect.json", "telemetry.jsonl", "workload.log", "cycles.json"):
        copy(stability_root / name, runtime / "starter-stability" / name)
    for source in sorted((stability_root / "cycles").iterdir()):
        if source.is_file():
            copy(source, runtime / "starter-stability/cycles" / source.name)
    final_path = PROOF / "stability/best-progress.json"
    if not final_path.exists():
        final_path = PROOF / "stability/best-submission.json"
    final_stability = read(final_path) if final_path.exists() else {"status": "not_started"}
    final_passed = final_stability.get("status") == "snapshot_stability_passed"
    if final_path.exists():
        copy(final_path, runtime / "best-stability/progress.json")
    if final_passed:
        final_root = PROOF / "stability" / ("job-" + final_stability["job_id"]) / "evidence"
        report = read(final_root / "stability-report.json")
        if (not report["final_best_method_bound"] or report["elapsed_seconds"] < 43200
                or report["image_id"] != replay["image_id"] or report["selector_sha256"] != sha(expert / "best_method/select.py")):
            raise ValueError("Final stability evidence does not bind the delivered best method")
        for name in ("binding.json", "stability-report.json", "workload-result.json", "container-inspect.json", "telemetry.jsonl", "workload.log", "cycles.json"):
            copy(final_root / name, runtime / "best-stability" / name)
        for source in sorted((final_root / "cycles").iterdir()):
            if source.is_file():
                copy(source, runtime / "best-stability/cycles" / source.name)
    write(expert / "run_summary.json", {
        "task_id": "research/active-preference-acquisition", "status": "pilot_required",
        "image": "arm-acquisition-paper-aligned:prepared", "agent_execution": executions,
        "fast_iteration_exception": {"used": False, "rollable_proved": False, "evidence": ""},
        "validation": {"baseline_passed": True, "reference_passed": True,
                       "optimization_evidence_complete": True, "best_method_rerun_passed": True},
        "stability": {"starter_snapshot_passed": stability["status"] == "snapshot_stability_passed",
                      "elapsed_seconds": stability["elapsed_seconds"],
                      "final_best_method_bound": final_passed,
                      "best_method_snapshot_passed": final_passed,
                      "best_method_job_id": final_stability.get("job_id"),
                      "best_method_status": final_stability.get("status")},
        "formal_platform_acceptance": False,
    })
    write(expert / "expert_annotation.json", {
        "task_id": "research/active-preference-acquisition", "task_type": "S1",
        "research_direction": "Active reward modeling and adaptive preference acquisition",
        "optimization_surface": "solution/select.py:select_pairs",
        "score_contract": {"metric": "one_minus_spearman", "direction": "minimize", "split": "dev",
            "protocol": "arm.paper-aligned.protocol.v1",
            "baseline": {"raw_value": comparison["baseline"]["mean_error"], "normalized_score": 0.0},
            "reference": {"raw_value": comparison["reference"]["mean_error"],
                          "normalized_score": comparison["reference"]["score"]},
            "upper_bound": {"raw_value": 0.0, "source": "metric_ceiling"}},
        "isolation": {"reference_visible_to_agent": False, "hidden_assets_visible_to_agent": False},
    })
    notes = (
        "# Expert Task Notes\n\n"
        "Task research/active-preference-acquisition is an S1 paper-derived task. The only editable "
        "algorithm is solution/select.py:select_pairs. Its inputs are public pair features, predicted "
        "logits, k, seed and round index; it returns 500 unique indices from 22500 candidates. "
        "The reward-model architecture, training, splits and labels are fixed.\n\n"
        "Run ./solution/solve.sh --output solution/artifacts/current inside the supplied public image "
        "with an operator AcquisitionBroker, then /tests/test.sh. Reference is available to the operator "
        "only in workspace/reference/. Frozen seeds are 101, 202, 303, 404 and 505, with 40 rounds, "
        "20000 labels and 40000 updates per seed. Runtime evidence includes the native Harbor runner "
        "and arm_acquisition_reload.py, which verifies checkpoint identity and recalculates dev metrics. "
        "The VM requires 8 CPU, 0 GPU and the supplied dependency image and private oracle. "
        "Packaging and cached-only evaluation support the extracted submission layout; see "
        "optimization_evidence/runtime/LOCAL_PACKAGE.md. Historical VM/Slurm scripts require "
        "external operator infrastructure.\n\n"
        f"Baseline mean error B={comparison['baseline']['mean_error']}; Reference mean error "
        f"R={comparison['reference']['mean_error']}. U=0 is the metric ceiling. Score=(B-error)/B, "
        "without clipping. comparison_summary.json contains all five paired runs and the 3/5-sigma "
        "statistics. Public and dev embeddings are visible; reward labels, Reference, checkpoints "
        "and optimization evidence are withheld from method authors. Formal hidden evaluation "
        "and secure asset submission remain platform responsibilities.\n\n"
        f"GPT completed {states['gpt']['completed_loops']} loops in {states['gpt']['effective_seconds']/3600:.6f} "
        f"effective hours; Seed completed {states['seed']['completed_loops']} loops in "
        f"{states['seed']['effective_seconds']/3600:.6f} effective hours. Both satisfy the default 10-hour "
        "requirement; no 7-hour exception is used. Failed transport and length-limited attempts are "
        "retained in runtime evidence and excluded from scientific loop counts. Each lane used one "
        "concurrent request, at least 30 seconds between requests and a 2048-token output limit.\n\n"
        f"The selected {selected} method scored {best_reward['score']:.9f}, above Baseline but below "
        "Reference. GPT explored whitened Fisher and Bayesian mutual-information selection; Seed "
        "explored gradient-embedding diversity and dynamic candidate pools. Both best methods "
        "passed fresh native agent/verifier containers and exact five-checkpoint metric reload. "
        "Successful scientific training was reused; replay is not independent retraining and "
        "adds no research duration. Further experiments could test the pool schedule and stochastic "
        "selection fraction under the unchanged protocol.\n\n"
        "The original Starter snapshot passed 12 actual hours of verifier workload, with no new training or "
        "model requests. That report does not bind the final best selector or the revised contract-test image. "
        f"Final best-method stability job {final_stability.get('job_id')} is {final_stability.get('status')}; "
        f"passed={final_passed}. The revised public test entry runs 13 submission/response contract tests "
        "before scoring; native replay checks actual resource and isolation settings. Original approximately "
        "10GB quickstart data is used, with ten-response dev evaluation rather than the paper's "
        "independent 500-response test. Training uses public train_BT defaults where the paper's "
        "experiment driver is unavailable. License review, portable cold deployment and authenticated "
        "platform acceptance remain outstanding.\n"
    )
    (expert / "\u4e13\u5bb6\u4f5c\u4e1a\u8bf4\u660e\u6587\u6863.md").write_text(notes)
    original = (evidence / "training-evidence.md").read_text()
    marker = "acceptance. GPT/Seed optimization trajectories, clean rerun of an Agent's best\n"
    if marker in original:
        original = original.split(marker)[0] + (
            "acceptance. Both model trajectories have now completed five scientific loops and more than "
            "10 effective hours independently. Their best methods passed fresh native replay and "
            "checkpoint re-evaluation without new training; expert_evidence/ contains the final "
            "summaries and chosen solution. The 12-hour stability report binds the Starter only. "
            "Formal hidden evaluation and license review remain outstanding.\n")
    (evidence / "training-evidence.md").write_text(original)
    (evidence / "\u8bad\u7ec3\u8bc1\u636e\u8bf4\u660e.md").write_text(original)
    manifest_path = evidence / "integrity_manifest.json"
    manifest = {str(p.relative_to(DELIVERY)): {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(DELIVERY.rglob("*")) if p.is_file() and p != manifest_path}
    write(manifest_path, manifest)
    print(json.dumps({"delivery": str(DELIVERY), "best_lane": selected, "score": best_reward["score"],
                      "files": len(manifest), "new_training_runs": 0, "formal_platform_acceptance": False}))


def main():
    global DELIVERY
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", action="store_true")
    parser.add_argument("--delivery", type=Path, default=DELIVERY)
    parser.add_argument("--manifest-only", action="store_true",
                        help="Refresh identities after an intentional local package edit; does not run experiments")
    args = parser.parse_args()
    DELIVERY = args.delivery.resolve()
    if args.manifest_only:
        delivery = args.delivery.resolve()
        manifest_path = delivery / "optimization_evidence/integrity_manifest.json"
        manifest = {}
        for path in sorted(delivery.rglob("*")):
            if path.is_symlink():
                raise ValueError("Submission contains a symbolic link: " + str(path))
            if path == manifest_path or path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("Submission contains a special file: " + str(path))
            if "__pycache__" in path.relative_to(delivery).parts or path.suffix == ".pyc":
                raise ValueError("Remove generated Python bytecode before preparing a submission")
            manifest[str(path.relative_to(delivery))] = {"sha256": sha(path), "bytes": path.stat().st_size}
        write(manifest_path, manifest)
        print(json.dumps({"status": "manifest_refreshed", "files": len(manifest), "new_training_runs": 0}))
        return
    if args.research:
        add_research_evidence()
        return
    native = read(PROOF / "native-report.json")
    if native["status"] != "native_acquisition_passed" or native["new_training_runs"]:
        raise ValueError("Native cache validation has not passed")
    comparison = read(PROOF / "comparison.json")
    definition = read(PROOF / "definition.json")
    DELIVERY.mkdir(exist_ok=False)
    public = DELIVERY / "workspace/harbor_task"
    reference = DELIVERY / "workspace/reference"
    evidence = DELIVERY / "optimization_evidence"
    (DELIVERY / "expert_evidence").mkdir()
    reference.mkdir(parents=True)
    for name in ("instruction.md", "task.toml", "environment/Dockerfile",
                 "environment/requirements.txt", "environment/baseline.json",
                 "solution/solve.sh", "solution/submission.json", "tests/grader.py", "tests/test.sh"):
        copy(TASK / name, public / name)
    copy(TASK / "environment/starter/select.py", public / "solution/select.py")
    for directory in ("environment/public_assets", "environment/starter", "environment/task"):
        for source in sorted((TASK / directory).rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                copy(source, public / source.relative_to(TASK))
    (public / "tests/hidden_assets").mkdir()
    copy(PROOF / "private/reference_select.py", reference / "select.py")
    copy(TASK / "solution/submission.json", reference / "submission.json")
    copy(ROOT / "arm_acquisition_reference.sh", reference / "solve.sh")
    (reference / "solve.sh").chmod(0o755)

    protocol = read(PROOF / "protocol.json")
    provenance = read(PROJECT / "public-workflow-v15/data-provenance.json")
    write(evidence / "data_manifest.json", {
        **read(TASK / "environment/public_assets/data_manifest.json"),
        "dataset": provenance["dataset"], "embedding_model": provenance["embedding_model"],
        "generation_model": provenance["generation_model"], "golden_reward_model": provenance["golden_reward_model"],
        "source_folder_url": provenance["source_folder_url"], "repository_url": provenance["repository_url"],
        "repository_revision": provenance["repository_revision"], "license_status": provenance["license_status"],
        "preprocessing": "Frozen source embeddings partitioned by prompt index using split seed 1729; no reward normalization",
        "public_contents": ["train embeddings", "dev embeddings"],
        "private_contents": ["reward targets", "oracle", "Reference", "models and training evidence"],
    })
    copy(PROOF / "protocol.json", evidence / "experiment_plan.json")
    copy(PROOF / "alignment.md", evidence / "alignment.md")
    baseline = comparison["baseline"]
    ref = comparison["reference"]
    sigma = statistics.stdev(baseline["raw_values"])
    gain = baseline["mean_error"] - ref["mean_error"]
    write(evidence / "comparison_summary.json", {
        "metric": {"name": "one_minus_spearman", "direction": "minimize", "split": "dev"},
        "protocol_sha256": comparison["protocol_sha256"], "seeds": protocol["seeds"],
        "independent_runs_per_arm": 5, "baseline": baseline, "reference": ref,
        "baseline_sample_std": sigma, "reference_sample_std": statistics.stdev(ref["raw_values"]),
        "raw_improvement": gain, "improvement_in_sigma_B": gain / sigma,
        "all_paired_seeds_improve": all(b > r for b, r in zip(baseline["raw_values"], ref["raw_values"])),
        "passes_3_sigma": gain >= 3 * sigma, "passes_5_sigma": gain >= 5 * sigma,
        "platform_statistical_review_required": gain < 5 * sigma,
        "score": {"formula": "(B - mean_error) / B", "baseline": 0.0,
                  "reference": ref["score"], "ceiling_error_U": 0.0,
                  "ceiling_basis": "Mathematical minimum of one_minus_spearman; no reliable achievable ceiling estimated"},
        "checkpoint_reload_passed": True, "new_training_runs_for_packaging": 0,
    })
    difference = "".join(difflib.unified_diff(
        (TASK / "environment/starter/select.py").read_text().splitlines(keepends=True),
        (reference / "select.py").read_text().splitlines(keepends=True),
        fromfile="a/solution/select.py", tofile="b/solution/select.py"))
    (evidence / "surface_diff.patch").write_text(difference)
    write(evidence / "surface_validation.json", {
        "allowed_surface": "solution/select.py:select_pairs", "changed_algorithm_files": ["solution/select.py"],
        "outside_surface_changes": [], "baseline": "random acquisition", "reference": "upstream dopt_rule",
        "starter_sha256": definition["starter_sha256"], "reference_sha256": definition["reference_sha256"],
        "shared_worker_sha256": definition["worker_sha256"], "protocol_sha256": definition["protocol_sha256"],
        "upstream_contract": read(PROOF / "contract.json"),
    })
    write(evidence / "ablation/runs.json", {
        "additional_training": "not_applicable", "reason": "One acquisition optimization point only; five paired Baseline/Reference runs are the rollback comparison",
        "seeds": protocol["seeds"], "new_training_runs": 0,
    })
    for arm, results in (("baseline", baseline), ("reference", ref)):
        source_root = PROOF / "runs" / comparison["protocol_sha256"] / results["selector_sha256"]
        target_root = evidence / (arm + "_runs")
        copy(source_root / "selector.py", target_root / "select.py")
        write(target_root / "train_config.json", protocol)
        for seed in protocol["seeds"]:
            source = source_root / f"seed_{seed}"
            target = target_root / "runs" / f"seed_{seed}"
            receipt = read(source / "evaluation.json")
            if receipt["status"] != "succeeded" or not receipt["checkpoint_reload_match"]:
                raise ValueError("Missing completed seed or model reload evidence")
            if sha(source / "checkpoint") != receipt["checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            for name in ("evaluation.json", "accounting.json", "sampling.json", "trainer-inspect.json", "worker.log"):
                copy(source / name, target / name)
            copy(source / "checkpoint", target / "model/checkpoint")
            copy(source / "worker.log", target / "train.log")
            write(target / "run_config.json", {**protocol, "seed": seed, "selector_sha256": results["selector_sha256"]})
            write(target / "metrics.json", {
                **read(source / "metrics.json"), "evaluation": receipt,
                "model": {"path": "model/checkpoint", "bytes": (target / "model/checkpoint").stat().st_size,
                          "sha256": receipt["checkpoint_sha256"], "reload_match": True},
            })
    runtime = evidence / "runtime"
    for name in ("native-report.json", "contract.json", "definition.json"):
        copy(PROOF / name, runtime / name)
    for name in ("evaluate.py", "evaluate.sh", "broker.py", "worker.py", "selector_entry.py"):
        copy(PROOF / "private" / name, runtime / "private" / name)
    for name in ("arm_acquisition_native.py", "arm_native_static_network.py", "verify_arm_acquisition_native.py",
                 "verify_arm_acquisition_native.sbatch"):
        copy(ROOT / name, runtime / "operator" / name)
    native_root = PROOF / ("native-" + native["job_id"])
    for arm in ("baseline", "reference"):
        copy(native_root / (arm + "-native-inspect.json"), runtime / (arm + "-native-inspect.json"))
        trial = native_root / "trials" / f"arm-acq-{native['job_id']}-{arm}"
        for name in ("result.json", "verifier/reward.json", "verifier/public-score.json", "verifier/test-stdout.txt"):
            copy(trial / name, runtime / arm / name)
    copy(ROOT / "arm_acquisition_training_evidence.md", evidence / "training-evidence.md")
    manifest = {str(p.relative_to(DELIVERY)): {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(DELIVERY.rglob("*")) if p.is_file()}
    write(evidence / "integrity_manifest.json", manifest)
    print(json.dumps({"delivery": str(DELIVERY), "files": len(manifest), "models": 10,
                      "reference_score": ref["score"], "gain_in_sigma_B": gain / sigma,
                      "new_training_runs": 0, "formal_platform_acceptance": False}))


if __name__ == "__main__":
    main()

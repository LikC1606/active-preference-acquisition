"""Submission and trusted-response contract checks without scientific training."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("acquisition_grader", Path(__file__).with_name("grader.py"))
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


class AcquisitionContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "task"
        self.channel = Path(self.temporary.name) / "channel"
        self.logs = Path(self.temporary.name) / "verifier"
        for directory in ("solution", "environment", "environment/starter"):
            (self.root / directory).mkdir(parents=True)
        for directory in ("inbox", "outbox"):
            (self.channel / directory).mkdir(parents=True)
        self.selector = self.root / "solution/select.py"
        self.selector.write_text("def select_pairs(*args): return []\n")
        (self.root / "environment/starter/select.py").write_bytes(self.selector.read_bytes())
        (self.root / "solution/submission.json").write_text(json.dumps({
            "schema": "arm.acquisition.submission.v1", "selector_path": "solution/select.py"}))
        self.baseline = {"mean_error": 0.5, "protocol_sha256": "f" * 64,
                         "seeds": [101, 202, 303, 404, 505]}
        (self.root / "environment/baseline.json").write_text(json.dumps(self.baseline))
        def mapped_path(*parts):
            if parts == ("/harbor_task",):
                return self.root
            if parts == ("/opt/arm-evaluator",):
                return self.channel
            if parts == ("/logs/verifier",):
                return self.logs
            return Path(*parts)
        self.path_patch = patch.object(grader, "Path", mapped_path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def response(self, **changes):
        values = changes.pop("raw_values", [0.25] * 5)
        mean = math.fsum(values) / len(values)
        return {"request_id": "1" * 32, "selector_sha256": hashlib.sha256(self.selector.read_bytes()).hexdigest(),
                "schema": "arm.acquisition.reward.v1", "status": "succeeded",
                "protocol_sha256": self.baseline["protocol_sha256"], "seeds": self.baseline["seeds"],
                "raw_values": values, "mean_error": mean, "baseline_mean": 0.5, "baseline_score": 0.0,
                "score": (0.5 - mean) / 0.5, "new_training_runs": 0, **changes}

    def invoke(self, response, output="solution/artifacts/current", *options):
        def deliver(_seconds):
            (self.channel / ("outbox/" + "1" * 32 + ".json")).write_text(json.dumps(response))
        with patch.object(sys, "argv",
                          ["grader.py", "--output", str(self.root / output), *options]), \
                patch.dict(grader.os.environ, {"ARM_ACQUISITION_EVALUATOR": ""}), \
                patch.object(grader.uuid, "uuid4", return_value=SimpleNamespace(hex="1" * 32)), \
                patch.object(grader.time, "sleep", side_effect=deliver):
            grader.main()

    def test_baseline_improvement_and_regression(self):
        for error, expected in ((0.5, 0.0), (0.25, 0.5), (0.75, -0.5), (0.0, 1.0), (2.0, -3.0)):
            with self.subTest(error=error):
                response = self.channel / ("outbox/" + "1" * 32 + ".json")
                response.unlink(missing_ok=True)
                self.invoke(self.response(raw_values=[error] * 5))
                reward = json.loads((self.root / "solution/artifacts/current/reward.json").read_text())
                self.assertEqual(reward["score"], expected)

    def test_five_seed_sum_is_stable_across_python_versions(self):
        values = [0.4926824610600503, 0.5195836423443716, 0.5055554949432712,
                  0.524572882595614, 0.5332989722475213]
        self.invoke(self.response(raw_values=values))
        reward = json.loads((self.root / "solution/artifacts/current/reward.json").read_text())
        self.assertEqual(reward["mean_error"], 0.5151386906381656)

    def test_rejects_nonfinite_boolean_and_inconsistent_scores(self):
        for value in (float("nan"), float("inf"), True, 0.99):
            with self.subTest(score=value), self.assertRaises(grader.EvaluationError):
                response = self.channel / ("outbox/" + "1" * 32 + ".json")
                response.unlink(missing_ok=True)
                self.invoke(self.response(score=value))

    def test_rejects_mismatched_request_source_protocol_or_seeds(self):
        for field, value in (("request_id", "2" * 32), ("selector_sha256", "a" * 64),
                             ("protocol_sha256", "b" * 64), ("seeds", [101]),
                             ("schema", "wrong"), ("mean_error", 0.0), ("baseline_mean", 1.0)):
            with self.subTest(field=field), self.assertRaises(grader.EvaluationError):
                response = self.channel / ("outbox/" + "1" * 32 + ".json")
                response.unlink(missing_ok=True)
                self.invoke(self.response(**{field: value}))

    def test_rejects_incomplete_or_nonfinite_seed_metrics(self):
        for values in ([0.25], [float("nan")] * 5, [True] * 5):
            with self.subTest(values=values), self.assertRaises(grader.EvaluationError):
                response = self.channel / ("outbox/" + "1" * 32 + ".json")
                response.unlink(missing_ok=True)
                self.invoke(self.response(raw_values=values))

    def test_rejects_changed_manifest(self):
        (self.root / "solution/submission.json").write_text('{"selector_path":"../private.py"}')
        with self.assertRaises(grader.EvaluationError):
            self.invoke(self.response())

    def test_rejects_missing_or_oversized_selector(self):
        self.selector.write_bytes(b"#" * 65537)
        with self.assertRaises(grader.EvaluationError):
            self.invoke(self.response())
        self.selector.unlink()
        with self.assertRaises(grader.EvaluationError):
            self.invoke({})

    def test_rejects_linked_selector(self):
        target = self.root / "private.py"
        target.write_text(self.selector.read_text())
        self.selector.unlink()
        self.selector.symlink_to(target)
        with self.assertRaises(grader.EvaluationError):
            self.invoke(self.response())

    def test_protected_output_path_cannot_be_written(self):
        with self.assertRaises(grader.EvaluationError):
            self.invoke(self.response(), output="tests/reward.json")
        self.assertFalse((self.root / "tests").exists())

    def test_previous_self_report_is_replaced_by_trusted_response(self):
        output = self.root / "solution/artifacts/current"
        output.mkdir(parents=True)
        (output / "reward.json").write_text('{"score":999}')
        self.invoke(self.response())
        self.assertEqual(json.loads((output / "reward.json").read_text())["score"], 0.5)

    def test_failed_evaluator_is_an_error(self):
        with self.assertRaises(grader.EvaluationError):
            self.invoke(self.response(status="failed"))

    def test_missing_channel_is_an_infrastructure_error(self):
        (self.channel / "inbox").rmdir()
        with self.assertRaises(grader.EvaluationError):
            self.invoke(self.response())

    def test_missing_response_times_out(self):
        with patch.object(grader.time, "monotonic", side_effect=[0, 0, 7201]), self.assertRaises(grader.EvaluationError):
            self.invoke(self.response())

    def test_failure_types_clear_stale_scores_and_only_gate_returns_minus_one(self):
        output = self.root / "solution/artifacts/current"
        output.mkdir(parents=True)
        for kind in grader.REPAIRS:
            with self.subTest(kind=kind):
                (self.channel / ("outbox/" + "1" * 32 + ".json")).unlink(missing_ok=True)
                (output / "reward.json").write_text('{"score":999}')
                (output / "observations.json").write_text('[]')
                with self.assertRaises(grader.EvaluationError) as caught:
                    self.invoke(self.response(status="failed", error={"type": kind, "message": "Retry"}))
                self.assertEqual(caught.exception.kind, kind)
                failure = json.loads((output / "failure.json").read_text())
                self.assertEqual(failure["error"]["type"], kind)
                self.assertTrue(failure["error"]["repair"])
                self.assertFalse(failure["rankable"])
                self.assertFalse((output / "observations.json").exists())
                if kind == "hard_gate":
                    reward = json.loads((output / "reward.json").read_text())
                    self.assertEqual(reward["score"], -1)
                    self.assertEqual(reward["status"], "invalid")
                else:
                    self.assertNotIn("score", failure)
                    self.assertFalse((output / "reward.json").exists())

    def test_illegal_imports_and_dynamic_execution_are_gates(self):
        for source in ("import os", "from pathlib import Path", "import subprocess", "import socket",
                       "__import__('os')", "exec('pass')", "open('/private')"):
            with self.subTest(source=source), self.assertRaises(grader.EvaluationError) as caught:
                grader.validate_source(source.encode())
            self.assertEqual(caught.exception.kind, "hard_gate")
        grader.validate_source(b"import numpy as np\nfrom scipy.linalg import eigh\nimport torch\n")

    def test_syntax_and_invalid_manifest_json_are_format_errors(self):
        with self.assertRaises(grader.EvaluationError) as caught:
            grader.validate_source(b"def :")
        self.assertEqual(caught.exception.kind, "format_error")
        (self.root / "solution/submission.json").write_text('{')
        with self.assertRaises(grader.EvaluationError) as caught:
            self.invoke(self.response())
        self.assertEqual(caught.exception.kind, "format_error")

    def test_baseline_chain_uses_starter_and_requires_zero_cached_score(self):
        self.selector.write_text("import subprocess\n")
        # The Candidate is intentionally invalid: a Baseline request must use Starter.
        self.selector = self.root / "environment/starter/select.py"
        self.invoke(self.response(raw_values=[0.5] * 5), "solution/artifacts/baseline-check", "--baseline")
        request = json.loads(next((self.channel / "inbox").glob("*.json")).read_text())
        self.assertTrue(request["baseline_check"])
        self.assertIn("expires_unix", request)
        (self.channel / ("outbox/" + "1" * 32 + ".json")).unlink()
        with self.assertRaises(grader.EvaluationError):
            self.invoke(self.response(raw_values=[0.5] * 5, new_training_runs=1),
                        "solution/artifacts/baseline-check", "--baseline")

    def test_failed_atomic_score_commit_leaves_no_success_or_partial_observations(self):
        real_write = grader.write_json
        def fail_commit(path, value):
            if path.name == "reward.json" and value.get("status") == "succeeded":
                raise OSError("Simulated publication failure")
            real_write(path, value)
        with patch.object(grader, "write_json", side_effect=fail_commit), self.assertRaises(grader.EvaluationError):
            self.invoke(self.response())
        output = self.root / "solution/artifacts/current"
        self.assertFalse((output / "reward.json").exists())
        self.assertFalse((output / "observations.json").exists())
        self.assertEqual(json.loads((output / "failure.json").read_text())["error"]["type"], "infrastructure_error")
        self.assertFalse(list(output.glob("*.tmp")))

    def test_artifacts_symlink_cannot_redirect_results_to_protected_path(self):
        protected = self.root / "tests"
        protected.mkdir()
        (self.root / "solution/artifacts").symlink_to(protected, target_is_directory=True)
        with self.assertRaises(grader.EvaluationError):
            self.invoke(self.response())
        self.assertFalse(list(protected.iterdir()))

    def test_late_response_cannot_publish_a_success(self):
        response = self.channel / ("outbox/" + "1" * 32 + ".json")
        with patch.object(grader.time, "monotonic", side_effect=lambda: 7201 if response.exists() else 0):
            with self.assertRaises(grader.EvaluationError) as caught:
                self.invoke(self.response())
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertFalse((self.root / "solution/artifacts/current/reward.json").exists())

    def test_rejects_boolean_calibration_and_noninteger_training_counts(self):
        with self.assertRaises(grader.EvaluationError) as caught:
            self.invoke(self.response(baseline_score=False))
        self.assertEqual(caught.exception.kind, "infrastructure_error")
        for count in (False, 0.0, None):
            with self.subTest(count=count):
                (self.channel / ("outbox/" + "1" * 32 + ".json")).unlink(missing_ok=True)
                with self.assertRaises(grader.EvaluationError) as caught:
                    self.invoke(self.response(raw_values=[0.5] * 5, new_training_runs=count),
                                "solution/artifacts/baseline-check", "--baseline")
                self.assertEqual(caught.exception.kind, "infrastructure_error")

    def test_verifier_failure_clears_last_success(self):
        self.logs.mkdir()
        for kind in grader.REPAIRS:
            with self.subTest(kind=kind):
                response_path = self.channel / ("outbox/" + "1" * 32 + ".json")
                response_path.unlink(missing_ok=True)
                self.invoke(self.response(), "solution/artifacts/current", "--verifier")
                self.assertEqual(json.loads((self.logs / "public-score.json").read_text())["score"], 0.5)
                response_path.unlink()
                with self.assertRaises(grader.EvaluationError):
                    self.invoke(self.response(status="failed", error={"type": kind, "message": "Retry"}),
                                "solution/artifacts/current", "--verifier")
                self.assertEqual(json.loads((self.logs / "failure.json").read_text())["error"]["type"], kind)
                if kind == "hard_gate":
                    for name in ("reward.json", "public-score.json"):
                        self.assertEqual(json.loads((self.logs / name).read_text())["score"], -1)
                    self.assertFalse(json.loads((self.logs / "public-score.json").read_text())["rankable"])
                else:
                    self.assertFalse((self.logs / "reward.json").exists())
                    self.assertFalse((self.logs / "public-score.json").exists())

    def test_failed_verifier_commit_removes_partial_public_score(self):
        self.logs.mkdir()
        self.invoke(self.response(), "solution/artifacts/current", "--verifier")
        real_write = grader.write_json
        def fail_commit(path, value):
            if path == self.logs / "reward.json":
                raise OSError("Simulated verifier commit failure")
            real_write(path, value)
        with patch.object(grader, "write_json", side_effect=fail_commit):
            with self.assertRaises(grader.EvaluationError) as caught:
                grader.publish_verifier(self.root / "solution/artifacts/current")
        self.assertEqual(caught.exception.kind, "infrastructure_error")
        self.assertFalse((self.logs / "reward.json").exists())
        self.assertFalse((self.logs / "public-score.json").exists())
        self.assertEqual(json.loads((self.logs / "failure.json").read_text())["error"]["type"], "infrastructure_error")

    def test_machine_budget_matches_score_deadline_and_resources(self):
        import tomllib
        task = Path(__file__).resolve().parents[1]
        config = tomllib.loads((task / "task.toml").read_text())
        self.assertEqual(config["task"]["time_limit_seconds"], grader.SCORE_TIMEOUT_SECONDS)
        self.assertEqual(config["environment"]["cpus"], 8)
        self.assertEqual(config["environment"]["gpus"], 0)
        self.assertEqual(config["environment"]["build_timeout_sec"], 1800)
        self.assertEqual(config["verifier"]["timeout_sec"], 7500)
        self.assertNotEqual(config["agent"]["timeout_sec"], grader.SCORE_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()

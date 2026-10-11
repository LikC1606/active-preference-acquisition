"""Bind the monitored containers to packaged files without restarting them."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from harbor_acceptance import docker

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent / "delivery"
TASK = DELIVERY / "workspace/harbor_task"
NATIVE = DELIVERY / "optimization_evidence/runtime/native"


def contract(text):
    return {
        "headings": re.findall(r"^## (.+)$", text, re.M),
        "code_blocks": re.findall(r"^```([^\n]*)\n(.*?)^```", text, re.M | re.S),
        "boundaries": text.split("## Hard Boundaries\n\n", 1)[1].split("\n\n## ", 1)[0],
        "interface": text.split("| Argument | Value |", 1)[1].split("\n\n", 1)[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="arm-native-v14")
    parser.add_argument("--report", default="stability-runtime-bindings.json")
    args = parser.parse_args()
    assert Path(args.project).name == args.project and Path(args.report).name == args.report
    summary = json.loads((HERE / (args.project + "-stability/summary.json")).read_text())
    ids = summary["container_ids"]
    inspected = json.loads(docker("inspect", *ids.values()))
    assert all(row["State"]["Running"] for row in inspected)
    assert {row["Id"]: row["Image"] for row in inspected} == summary["images"]
    mappings = {"main": {}, "evaluator": {}}

    def add(service, source, target):
        mappings[service][str(source.relative_to(DELIVERY))] = target

    for source in (TASK / "environment/public_assets").rglob("*"):
        if source.is_file():
            suffix = source.relative_to(TASK / "environment/public_assets").as_posix()
            add("main", source, "/harbor_task/environment/public_assets/" + suffix)
            add("evaluator", source, "/workspace/environment/public_assets/" + suffix)
    for source in (TASK / "environment/starter").rglob("*"):
        if source.is_file():
            add("main", source, "/harbor_task/environment/starter/" + source.name)
    for source in (TASK / "environment/task/tests").rglob("*"):
        if source.is_file():
            suffix = source.relative_to(TASK / "environment/task/tests").as_posix()
            add("main", source, "/harbor_task/tests/" + suffix)
    for name in ("instruction.md", "task.toml"):
        add("main", TASK / "environment/task" / name, "/harbor_task/" + name)
    for name in ("baseline.json", "requirements.txt"):
        add("main", TASK / "environment" / name, "/harbor_task/environment/" + name)
    for name in ("service.py", "worker.py", "selector_entry.py", "selector_sandbox.py",
                 "validate_calibration.py", "baseline.json", "definition.json"):
        add("evaluator", NATIVE / name, "/opt/arm-private/" + name)
    add("evaluator", NATIVE / "protocol.json", "/protocol.json")
    for source in (NATIVE / "paper").rglob("*"):
        if source.is_file():
            add("evaluator", source, "/opt/paper/" + source.relative_to(NATIVE / "paper").as_posix())
    for source in (NATIVE / "calibration").rglob("*"):
        if source.is_file():
            add("evaluator", source, "/opt/arm-private/calibration/" + source.relative_to(NATIVE / "calibration").as_posix())

    bindings, document_changes = {}, []
    for service, mapping in mappings.items():
        code = """import hashlib, importlib.metadata, json
from pathlib import Path
mapping = %r
records = {}
for relative, target in mapping.items():
    path = Path(target)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    records[relative] = {'image_path': target, 'bytes': path.stat().st_size, 'sha256': digest}
print(json.dumps({'files': records, 'dependencies': {name: importlib.metadata.version(name)
                 for name in ['numpy', 'scipy', 'torch']}}))
""" % mapping
        result = json.loads(docker("exec", ids[service], "python3", "-c", code))
        bindings[service] = result
        for relative, actual in result["files"].items():
            source = DELIVERY / relative
            with source.open("rb") as stream:
                expected = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual["sha256"] != expected:
                assert relative.endswith("/instruction.md"), relative
                before = docker("exec", ids[service], "cat", actual["image_path"])
                assert contract(before) == contract(source.read_text()), relative
                document_changes.append({"path": relative, "image_sha256": actual["sha256"],
                                         "package_sha256": expected, "contract_unchanged": True})
    report = {"status": "passed", "container_ids": ids, "images": summary["images"],
              "runtime_and_data_match": True, "document_only_changes": document_changes,
              "bindings": bindings}
    (HERE / args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "bindings"}))


if __name__ == "__main__":
    main()

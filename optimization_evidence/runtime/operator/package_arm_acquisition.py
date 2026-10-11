"""Build a reviewable local submission archive using Harbor's public file set."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import tomllib
import zipfile

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2] / "papers/active-reward-modeling/acquisition-paper-aligned"
DELIVERY = ROOT / "delivery"
if (Path(__file__).resolve().parents[3] / "workspace/harbor_task").is_dir():
    DELIVERY = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, default=DELIVERY,
                        help="Extracted submission root containing workspace/ and optimization_evidence/")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Verify files without creating an archive")
    parser.add_argument("--require-author-acceptance", action="store_true",
                        help="Reject packaging until current author validation, cold build, budget and stability pass")
    parser.add_argument("--compression", choices=("fast", "stored"), default="fast")
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=1,
                        help="DEFLATE level for fast compression (1-9; default: 1)")
    args = parser.parse_args()
    delivery = args.delivery.resolve()
    output = (args.output or delivery.parent / "submission-local.zip").resolve()
    report_path = output.with_suffix(".json")
    if not args.check_only and output.suffix.lower() != ".zip":
        raise ValueError("Choose an archive output with a .zip extension")
    if not args.check_only and any(path.exists() or path.is_symlink() for path in (output, report_path)):
        raise ValueError("Choose fresh archive and report paths; completed files are not overwritten")
    if not args.check_only and output.is_relative_to(delivery):
        raise ValueError("Write the archive outside the submission directory")
    from harbor.publisher.packager import Packager

    manifest_path = delivery / "optimization_evidence/integrity_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files, directories = [], []
    for path in sorted(delivery.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            directories.append(path)
        elif stat.S_ISREG(mode):
            files.append(path)
        else:
            raise ValueError("Submission contains a link or special file: " + str(path))
    expected = {str(p.relative_to(delivery)) for p in files if p != manifest_path}
    if expected != set(manifest):
        raise ValueError("Submission file set differs from the integrity manifest")
    public = delivery / "workspace/harbor_task"
    published = Packager.collect_files(public)
    digest = hashlib.sha256()
    for path in published:
        record = manifest[str(path.relative_to(delivery))]
        digest.update(f"{path.relative_to(public).as_posix()}\0{record['sha256']}\n".encode())
    for relative in ("instruction.md", "task.toml", "environment/Dockerfile", "environment/requirements.txt",
                     "solution/select.py", "solution/solve.sh", "tests/Dockerfile", "tests/grader.py", "tests/test.sh",
                     "tests/acquisition_test_contract.py"):
        if not (public / relative).is_file():
            raise ValueError("Required public task file is absent: " + relative)
    hidden = public / "tests/hidden_assets"
    if not (hidden / "manifest.json").is_file():
        raise ValueError("Independent Verifier Hidden manifest is absent")
    if (public / "solution/select.py").read_bytes() != (public / "environment/starter/select.py").read_bytes():
        raise ValueError("Publishable solution must retain the unchanged Starter")
    config = tomllib.loads((public / "task.toml").read_text())
    if config["metadata"]["cpu_cores"] != 8 or config["metadata"]["gpu_count"] != 0:
        raise ValueError("Public resource contract changed")
    for path in files:
        if path.suffix == ".sh":
            subprocess.run(["sh", "-n", str(path)], check=True, capture_output=True)
    contents = set(published)
    contents.update(p for p in files if not p.is_relative_to(public))
    if set(contents) != set(files):
        raise ValueError("Harbor's file collection would omit submission files")
    if args.check_only:
        for path in files:
            record = manifest.get(str(path.relative_to(delivery)))
            if record:
                with path.open("rb") as stream:
                    identity = hashlib.file_digest(stream, "sha256").hexdigest()
                if path.stat().st_size != record["bytes"] or identity != record["sha256"]:
                    raise ValueError("Submission file identity changed: " + str(path.relative_to(delivery)))
        print(json.dumps({"status": "local_file_preflight_passed", "files": len(files),
                          "public_files": len(published), "public_content_sha256": digest.hexdigest(),
                          "formal_platform_acceptance": False}))
        return
    summary = json.loads((delivery / "expert_evidence/run_summary.json").read_text())
    current = summary.get("current_snapshot", {})
    author_checks = (all(summary.get("validation", {}).get(key) is True for key in
                         ("baseline_passed", "reference_passed", "optimization_evidence_complete",
                          "best_method_rerun_passed"))
                     and all(current.get(key) is True for key in
                             ("cold_build_passed", "fresh_score_within_budget_passed", "stability_12h_passed")))
    if args.require_author_acceptance and not author_checks:
        raise ValueError("Current author acceptance is incomplete; no final archive was created")
    final_stability = current.get("stability_12h_passed", False)
    compression = zipfile.ZIP_DEFLATED if args.compression == "fast" else zipfile.ZIP_STORED
    level = args.compression_level if args.compression == "fast" else None
    with tempfile.TemporaryDirectory(prefix=".arm-package-", dir=output.parent) as staging:
        temporary = Path(staging) / "submission.zip"
        with zipfile.ZipFile(temporary, "x", compression=compression, compresslevel=level, allowZip64=True) as archive:
            for path in directories:
                name = path.relative_to(delivery).as_posix() + "/"
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
                archive.writestr(info, b"")
            for path in sorted(contents):
                name = path.relative_to(delivery).as_posix()
                info = zipfile.ZipInfo(name)
                info.external_attr = path.stat().st_mode << 16
                info.compress_type = compression
                info._compresslevel = level
                record = manifest.get(name)
                size = path.stat().st_size
                if record and size != record["bytes"]:
                    raise ValueError("Submission file size changed: " + name)
                identity = hashlib.sha256()
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    while chunk := source.read(4 * 1024 * 1024):
                        identity.update(chunk)
                        target.write(chunk)
                if record and identity.hexdigest() != record["sha256"]:
                    raise ValueError("Submission file identity changed: " + name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Submission archive CRC verification failed")
        with temporary.open("rb") as stream:
            identity = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {"status": "local_package_preflight_passed", "archive": str(output),
                  "archive_sha256": identity, "archive_bytes": temporary.stat().st_size,
                  "files": len(contents), "public_files": len(published),
                  "compression": args.compression, "compression_level": level,
                  "public_content_sha256": digest.hexdigest(), "crc_passed": True,
                  "file_collection": "harbor.publisher.packager.Packager.collect_files",
                  "final_stability_passed": final_stability, "author_self_checks_passed": author_checks,
                  "strict_author_acceptance_required": args.require_author_acceptance,
                  "formal_platform_acceptance": False}
        temporary_report = Path(staging) / "report.json"
        temporary_report.write_text(json.dumps(report, indent=2) + "\n")
        # Publish verified files without replacing existing results.
        output.hardlink_to(temporary)
        try:
            report_path.hardlink_to(temporary_report)
        except BaseException:
            output.unlink()
            raise
    print(json.dumps(report))


if __name__ == "__main__":
    main()

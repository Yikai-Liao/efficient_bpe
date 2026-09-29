"""Check the frozen source capsule and measured rows without rerunning training."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    environment = json.loads((OUT / "screen.jsonl.environment.json").read_text())
    shared = json.loads((OUT / "shared-source-provenance.json").read_text())
    source_hashes = json.loads((OUT / "new-source-hashes.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    rows = [json.loads(line) for line in (OUT / "screen.jsonl").read_text().splitlines()]
    assert environment["status"] == summary["status"] == "passed"
    assert len(rows) == environment["measured_calls"] == summary["rows"] == 56
    assert all(row["full_trace_match"] for row in rows)
    assert all(row["cpu_affinity"] == environment["cpu_budget"][str(row["workers"])]
               for row in rows)
    assert all(row["cpu_budget"] == row["workers"] for row in rows)
    assert all(row["call_seconds"] > 0 and row["call_cpu_seconds"] > 0
               and row["train_vm_hwm_mib"] > 0 for row in rows)
    expected_counts = {
        "micro": 24, "atomic": 16, "ordered": 12, "serial": 4,
    }
    actual_counts = {family: sum(row["mode_label"].startswith(prefix)
                                 for row in rows)
                     for family, prefix in (("micro", "micro_"), ("atomic", "atomic_"),
                                            ("ordered", "ordered_"), ("serial", "serial_"))}
    assert actual_counts == expected_counts
    assert len(summary["groups"]) == 32
    assert environment["source_sha256"] == source_hashes
    for path, digest in source_hashes.items():
        assert sha((ROOT / path).read_bytes()) == digest, path
    capsule = OUT / "new-source-snapshot.tar.gz"
    assert sha(capsule.read_bytes()) == environment["source_snapshot_sha256"]
    with tarfile.open(capsule, "r:gz") as archive:
        members = archive.getmembers()
        assert set(member.name for member in members) == set(source_hashes)
        for member in members:
            assert member.isfile() and sha(archive.extractfile(member).read()) == source_hashes[member.name]
    assert environment["shared_source_provenance_sha256"] == sha(
        (OUT / "shared-source-provenance.json").read_bytes())
    assert shared["all_shared_files_match_base_commit_byte_for_byte"]
    assert shared["git_base_commit"] == environment.get("git_base_commit", shared["git_base_commit"])
    for path, digest in shared["shared_files_sha256"].items():
        assert sha((ROOT / path).read_bytes()) == digest, path
        base = subprocess.run(["git", "show", f"{shared['git_base_commit']}:{path}"],
                              cwd=ROOT, check=True, capture_output=True).stdout
        assert sha(base) == digest, path
    binary_paths = {
        "micro": "rust/target/reruns/radical-region-tasks-gate-v1/radical-owned-region-tasks",
        "atomic": "rust/target/reruns/radical-atomic-old-gate-v1/radical-owned-atomic-old",
        "ordered": "rust/target/reruns/radical-ordered-posting-gate-v1/radical-owned-ordered-posting",
        "serial": "rust/target/reruns/radical-serial-integer-gate-v1/serial_integer_hash",
    }
    for family, path in binary_paths.items():
        assert sha((ROOT / path).read_bytes()) == environment["binary_sha256"][family]
    assert any(row["atomic_old_calls"] > 0 for row in rows
               if row["mode_label"] == "atomic_producer-atomic")
    assert any(row["snapshot_deferred_stores"] > 0 for row in rows
               if row["mode_label"].startswith("micro_snapshot_"))
    assert all(row["aa_sort_elided_batches"] > 0 and row["aa_sort_seconds"] == 0
               for row in rows if row["mode_label"] == "ordered_global"
               and row["case_id"] == "single-piece-ab-65536")
    result = {
        "status": "passed",
        "measured_calls": len(rows),
        "full_trace_matches": len(rows),
        "group_count": len(summary["groups"]),
        "family_call_counts": actual_counts,
        "source_file_count": len(source_hashes),
        "shared_file_count": len(shared["shared_files_sha256"]),
        "source_capsule_sha256": environment["source_snapshot_sha256"],
        "binary_sha256": environment["binary_sha256"],
        "base_commit": shared["git_base_commit"],
        "cpu_affinity": environment["cpu_budget"],
        "memory_metric": "train_vm_hwm_mib",
        "cpu_metric": "call_cpu_seconds",
        "verification": "offline hashes, manifest, measured-row invariants; no training rerun",
    }
    with (OUT / "checks.json").open("w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

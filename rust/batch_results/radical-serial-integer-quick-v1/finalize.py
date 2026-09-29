"""Verify frozen binary/source provenance and exact quick-screen completeness."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BINS = {
    "serial": RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash",
    "owner": RUST / "target/reruns/radical-local-hash-v1/owned_integer_hash",
    "native": RUST / "target/reruns/radical-fixed-budget-v1/ablation",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = json.loads((OUT / "source-provenance.json").read_text())
    environment = json.loads((OUT / "screen.jsonl.environment.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    rows = [json.loads(line) for line in (OUT / "screen.jsonl").read_text().splitlines()]
    assert source["all_files_match_base_commit_byte_for_byte"]
    for name, digest in source["files_sha256"].items():
        current = (ROOT / name).read_bytes()
        assert sha(current) == digest, (name, "source changed")
        committed = subprocess.check_output(
            ["git", "show", f"{source['git_base_commit']}:{name}"], cwd=ROOT)
        assert current == committed, (name, "commit differs")
    for name, path in source["prior_archive_paths"].items():
        assert sha((ROOT / path).read_bytes()) == source["prior_archive_sha256"][name]
    assert {name: sha(path.read_bytes()) for name, path in BINS.items()} == environment["binary_sha256"]
    assert environment["cpu_budget"] == {"1": [5], "4": [0, 1, 2, 5]}
    assert environment["manifest_sha256"] == sha((ROOT / environment["manifest"]).read_bytes())
    assert len(rows) == summary["rows"] == environment["rows"] == 22
    assert all(row["full_trace_match"] for row in rows)
    assert len({(row["case_id"], row["version"], row["workers"]) for row in rows}) == 22
    assert all(row["call_seconds"] > 0 and row["call_cpu_seconds"] > 0
               and row["train_vm_hwm_mib"] > 0 for row in rows)
    for case_id in summary["cases"]:
        group = [row for row in rows if row["case_id"] == case_id]
        assert len(group) == 11
        assert sum(row["version"] == "native_direct" for row in group) == 1
        assert sum(row["version"] == "owner_ahash" for row in group) == 2
        assert sum(row["version"].startswith("cf") for row in group) == 8
    checks = {
        "status": "passed", "rows": 22, "cases": 2, "repeats": 1,
        "full_trace_and_fingerprint_matches": 22,
        "serial_modes": "CF32/CF16 × std/ahash × checked/unchecked, W1",
        "owner_modes": "aHash, W1/W4", "native_modes": "EN CF16/ZH CF32, checked W1",
        "cpu_budget": environment["cpu_budget"],
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds",
        "git_base_commit": source["git_base_commit"],
        "source_provenance": "source-provenance.json",
        "binary_sha256": environment["binary_sha256"],
        "release_profile": environment["release_profile"],
        "limitations": [
            "One run per cell is a light screen, not a stable speed ranking.",
            "Only within-backend std/ahash pairs use the same serial kernel; native and owner are different trainer paths.",
            "The prior aHash build uses its software fallback, not the x86 AES branch; archived build-context evidence is linked by source-provenance.json.",
            "Process HWM includes parsing and transient allocation, not just corpus buffers.",
        ],
    }
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "passed", "rows": 22, "traces": 22}))


if __name__ == "__main__":
    main()

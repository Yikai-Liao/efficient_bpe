"""Verify the frozen exact-certificate diagnostic without rerunning training."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-neighbor-certificate-v1/owned_neighbor_sketch"
NATIVE = RUST / "target/reruns/radical-fixed-budget-v1/ablation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    modes = json.loads((OUT / "modes.json").read_text())
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    shared = json.loads((OUT / "shared-source-provenance.json").read_text())
    oracle = json.loads((OUT / "differential.json").read_text())
    quick = json.loads((OUT / "summary.json").read_text())
    environment = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, (name, "source changed")
    assert environment["new_source_sha256"] == sources
    snapshot_sha = sha(OUT / "new-source-snapshot.tar.gz")
    assert environment["new_source_snapshot_sha256"] == snapshot_sha
    assert shared["all_files_match_base_commit_byte_for_byte"]
    for name, expected in shared["files_sha256"].items():
        assert sha(ROOT / name) == expected, (name, "shared source changed")
    assert len(modes["oracle"]) == len(modes["quick"]) == 2
    assert oracle["status"] == "passed" and oracle["compared_runs"] == 80
    assert oracle["full_rule_trace_and_final_tokens_match"]
    assert len(rows) == quick["rows"] == 8
    assert all(row["full_trace_match"] for row in rows)
    assert len({(row["case_id"], row["workers"], row["version"]) for row in rows}) == 8
    assert sha(BIN) == oracle["binary_sha256"]["owned_neighbor_sketch"]
    assert all(sha(BIN) == digest for digest in environment["binary_sha256"].values())
    assert sha(NATIVE) == environment["native_reference_binary_sha256"]
    assert environment["cpu_budget"] == {"1": [5], "4": [0, 1, 2, 5]}
    checks = {
        "status": "passed",
        "debug_lib_tests": "14/14",
        "strict_clippy_all_targets": "passed -D warnings after CLI recursion-limit fix",
        "release_build": "passed",
        "oracle": {"cases": 20, "modes": 2, "workers": [1, 4],
                   "full_trace_matches": 80},
        "quick": {"rows": 8, "cases": 2, "modes": 2, "workers": [1, 4],
                  "repeats": 1, "full_trace_matches": 8},
        "limitations": [
            "This is a one-repeat 256 KiB mechanism screen; no 4 MiB performance claim is made.",
            "Different exact batch widths can legitimately change intermediate birth and posting counts; complete rule traces and final tokens are the semantic criterion.",
            "Extra sketch visits and the 32-to-40-byte Entry tuple are real costs; process HWM also includes unrelated and transient allocations.",
        ],
        "new_source_snapshot_sha256": snapshot_sha,
        "new_source_hashes": "new-source-hashes.json",
        "shared_source_provenance": "shared-source-provenance.json",
        "binary_sha256": sha(BIN),
        "binary_path": str(BIN.relative_to(ROOT)),
        "native_reference_binary_sha256": sha(NATIVE),
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
    }
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "passed", "oracle": 80, "quick": 8}))


if __name__ == "__main__":
    main()

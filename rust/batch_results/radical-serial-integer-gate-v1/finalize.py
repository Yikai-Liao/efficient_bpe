"""Verify serial integer-hash exactness and source provenance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    modes = json.loads((OUT / "modes.json").read_text())["oracle"]
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    shared = json.loads((OUT / "shared-source-provenance.json").read_text())
    oracle = json.loads((OUT / "differential.json").read_text())
    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, (name, "source changed")
    assert shared["all_files_match_base_commit_byte_for_byte"]
    for name, expected in shared["files_sha256"].items():
        assert sha(ROOT / name) == expected, (name, "shared source changed")
    assert len(modes) == 8
    assert oracle["status"] == "passed" and oracle["compared_runs"] == 156
    assert oracle["full_rule_trace_and_final_tokens_match"]
    assert len(oracle["expected_rejections"]) == 4
    assert all(row["case_id"] == "initial-alphabet-65536"
               and row["version"].startswith("cf16_")
               for row in oracle["expected_rejections"])
    assert sha(BIN) == oracle["binary_sha256"]
    snapshot_sha = sha(OUT / "new-source-snapshot.tar.gz")
    checks = {
        "status": "passed",
        "debug_lib_tests": "6/6 after final Clippy-only source edit",
        "strict_clippy_all_targets": "passed -D warnings",
        "release_build": "passed after offline Cargo.lock generation",
        "oracle": {"cases": 20, "modes": 8, "workers": [1],
                   "full_trace_matches": 156,
                   "expected_halfword_domain_rejections": 4},
        "limitations": [
            "This archive tests correctness only; no performance timing is claimed.",
            "Halfword rejects the initial alphabet of 65,536 IDs; CF32 succeeds for that case.",
            "The serial kernel requires workers=1.",
        ],
        "new_source_snapshot_sha256": snapshot_sha,
        "new_source_hashes": "new-source-hashes.json",
        "shared_source_provenance": "shared-source-provenance.json",
        "binary_sha256": sha(BIN),
        "binary_path": str(BIN.relative_to(ROOT)),
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
    }
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "passed", "oracle_matches": 156,
                      "expected_rejections": 4}))


if __name__ == "__main__":
    main()

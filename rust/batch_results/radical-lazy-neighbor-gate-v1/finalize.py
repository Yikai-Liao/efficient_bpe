"""Verify lazy-certificate oracle, quick screen, and frozen provenance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-lazy-neighbor-gate-v1/owned_lazy_neighbor"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    modes = json.loads((OUT / "modes.json").read_text())["oracle"]
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
    assert len(modes) == 5 and oracle["status"] == "passed"
    assert oracle["compared_runs"] == 200
    assert oracle["full_rule_trace_and_final_tokens_match"]
    assert len(rows) == quick["rows"] == environment["rows"] == 14
    assert len({(row["case_id"], row["version"], row["workers"]) for row in rows}) == 14
    assert all(row["full_trace_match"] for row in rows)
    assert sha(BIN) == oracle["binary_sha256"]["owned_lazy_neighbor"]
    assert sha(BIN) == environment["binary_sha256"]
    assert environment["cpu_budget"] == {"1": [5], "4": [0, 1, 2, 5]}
    checks = {
        "status": "passed", "debug_lib_tests": "17/17",
        "strict_clippy_all_targets": "passed -D warnings",
        "release_build": "passed",
        "oracle": {"cases": 20, "modes": 5, "workers": [1, 4],
                   "full_trace_matches": 200},
        "quick": {"rows": 14, "cases": 2, "workers": [1, 4],
                  "repeats": 1, "full_trace_matches": 14,
                  "w4_force_serial_controls": 2},
        "cpu_budget": environment["cpu_budget"],
        "default_parallel_builds_on_quick_fixtures": quick["default_parallel_builds_on_these_fixtures"],
        "source_snapshot_sha256": snapshot_sha,
        "source_hashes": "new-source-hashes.json",
        "shared_source_provenance": "shared-source-provenance.json",
        "binary_sha256": sha(BIN), "binary_path": str(BIN.relative_to(ROOT)),
        "limitations": [
            "One repeat per 256 KiB cell, so timing is diagnostic only.",
            "Default threshold caused no parallel mask builds on these fixtures; the quick screen cannot measure that path's performance.",
            "Batch boundaries and intermediate birth counts may change between exact certificates; complete output traces are the semantic criterion.",
            "Capacity-scaled byte counters are payload proxies, not actual HashMap allocation or process RSS.",
        ],
    }
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "passed", "oracle": 200, "quick": 14}))


if __name__ == "__main__":
    main()

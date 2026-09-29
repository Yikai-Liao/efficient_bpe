"""Check exactness, provenance and completeness of the controlled long screen."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN_DIR = RUST / "target/reruns/radical-controlled-longscreen-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    modes = json.loads((OUT / "modes.json").read_text())
    oracle = json.loads((OUT / "differential.json").read_text())
    summary = json.loads((OUT / "screen-4m-summary.json").read_text())
    environment = json.loads((OUT / "screen-4m.jsonl.environment.json").read_text())
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    shared = json.loads((OUT / "shared-source-provenance.json").read_text())
    rows = [json.loads(line) for line in (OUT / "screen-4m.jsonl").read_text().splitlines()]

    for name, digest in sources.items():
        assert sha(ROOT / name) == digest, (name, "new source changed")
    assert environment["new_source_sha256"] == sources
    assert sha(OUT / "new-source-snapshot.tar.gz") == environment["new_source_snapshot_sha256"]
    assert shared["all_files_match_base_commit_byte_for_byte"]
    for name, digest in shared["files_sha256"].items():
        assert sha(ROOT / name) == digest, (name, "shared source changed")

    assert oracle["status"] == "passed"
    assert oracle["compared_runs"] == 318
    assert len(oracle["documented_u16_limit_rejections"]) == 2
    assert oracle["full_rule_trace_and_final_tokens_match"]
    assert oracle["compared_runs"] + 2 == 20 * 2 * len(modes["oracle"])
    assert len(rows) == summary["rows"] == 2 * 2 * len(modes["screen_4m"]) == 32
    assert not summary["deterministic_work_mismatches"]
    assert all(row["full_training_fingerprint_match"] for row in rows)
    assert all(row["call_seconds"] > 0 and row["call_cpu_seconds"] > 0
               and row["train_vm_hwm_mib"] > 0 for row in rows)
    assert len({(row["case_id"], row["workers"], row["version"]) for row in rows}) == 32
    assert environment["cpu_budget"] == {"1": [5], "4": [0, 1, 2, 5]}
    assert environment["repeats"] == 1
    for name, config in modes["screen_4m"].items():
        assert sha(BIN_DIR / config["binary"]) == environment["binary_sha256"][name], name

    checks = {
        "status": "passed",
        "debug_lib_tests": {"owned_narrow_exact": "11/11", "owned_route_cache_reset": "12/12"},
        "strict_clippy_all_targets": "both new crates passed -D warnings",
        "release_builds": "both new crates passed",
        "complete_python_oracle": {
            "matching_full_traces": 318,
            "expected_u16_domain_rejections": 2,
            "cases": 20, "modes": 8, "workers": [1, 4],
        },
        "long_screen": {
            "rows": 32, "cases": 2, "modes": 8, "workers": [1, 4],
            "repeats": 1, "fingerprints_match_prior_reference": True,
            "deterministic_work_mismatches": 0,
            "worker1_affinity": [5], "worker4_affinity": [0, 1, 2, 5],
            "memory_metric": "train_vm_hwm_mib",
            "process_cpu_metric": "call_cpu_seconds",
        },
        "limitations": [
            "One repeat per cell is diagnostic and does not establish a stable speed ordering.",
            "u16 and u32-exact both exact-allocate the endpoint Vec; u32 inherits spare input capacity.",
            "Cache victim phase is reset in both lifetimes, but four-worker dynamic task ownership still changes cache hit counts.",
            "CPU/wall is average occupied cores, not useful-compute utilization; C4/C1 is accounting, not a causal overhead estimate.",
            "train_vm_hwm_mib is a process high-water mark that includes parsing and transient allocation, not just endpoint payload.",
        ],
        "new_source_snapshot_sha256": environment["new_source_snapshot_sha256"],
        "new_source_hashes": "new-source-hashes.json",
        "shared_source_provenance": "shared-source-provenance.json",
        "binary_sha256": environment["binary_sha256"],
        "binary_paths": {name: str((BIN_DIR / config["binary"]).relative_to(ROOT))
                         for name, config in modes["screen_4m"].items()},
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
    }
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "passed", "oracle_matches": 318,
                      "documented_rejections": 2, "screen_rows": 32}))


if __name__ == "__main__":
    main()

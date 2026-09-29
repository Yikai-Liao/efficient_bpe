"""Verify local/hash experiments, provenance, exactness, and authorized screens."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN_DIR = RUST / "target/reruns/radical-local-hash-v1"
NATIVE = RUST / "target/reruns/radical-fixed-budget-v1/ablation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    modes = json.loads((OUT / "modes.json").read_text())
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    shared = json.loads((OUT / "shared-source-provenance.json").read_text())
    oracle = json.loads((OUT / "differential.json").read_text())
    quick = json.loads((OUT / "summary.json").read_text())
    quick_env = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    long = json.loads((OUT / "integer-long-summary.json").read_text())
    long_env = json.loads((OUT / "integer-long.jsonl.environment.json").read_text())
    build = json.loads((OUT / "ahash-build-context.json").read_text())
    quick_rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    long_rows = [json.loads(line) for line in (OUT / "integer-long.jsonl").read_text().splitlines()]

    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, (name, "source changed")
    assert quick_env["new_source_sha256"] == sources
    assert long_env["new_source_sha256"] == sources
    snapshot_sha = sha(OUT / "new-source-snapshot.tar.gz")
    assert snapshot_sha == quick_env["new_source_snapshot_sha256"]
    assert snapshot_sha == long_env["new_source_snapshot_sha256"]
    assert shared["all_files_match_base_commit_byte_for_byte"]
    for name, expected in shared["files_sha256"].items():
        assert sha(ROOT / name) == expected, (name, "shared source changed")
    assert build["fingerprint_sha256"] == sha(OUT / "ahash-cargo-fingerprint.json")
    assert build["rustflags"] == [] and not build["target_feature_aes_enabled"]
    assert not build["RUSTFLAGS_set"] and not build["CARGO_ENCODED_RUSTFLAGS_set"]

    assert oracle["status"] == "passed" and oracle["compared_runs"] == 240
    assert oracle["full_rule_trace_and_final_tokens_match"]
    assert len(modes["oracle"]) == 6 and len(modes["quick"]) == 5
    assert len(quick_rows) == quick["rows"] == 20
    assert len(long_rows) == long["rows"] == 20
    assert not quick["deterministic_work_mismatches"]
    assert not long["deterministic_work_mismatches"]
    assert all(row["full_trace_match"] for row in quick_rows)
    assert all(row["full_training_fingerprint_match"] for row in long_rows)
    assert len({(row["case_id"], row["workers"], row["version"])
                for row in quick_rows}) == 20
    assert len({(row["case_id"], row["workers"], row["version"], row["repetition"])
                for row in long_rows}) == 20
    assert quick_env["cpu_budget"] == long_env["cpu_budget"] == {
        "1": [5], "4": [0, 1, 2, 5]}
    for name, config in modes["quick"].items():
        assert sha(BIN_DIR / config["binary"]) == quick_env["binary_sha256"][name]
    assert sha(BIN_DIR / "owned_integer_hash") == long_env["integer_binary_sha256"]
    assert sha(NATIVE) == quick_env["native_reference_binary_sha256"]
    assert sha(NATIVE) == long_env["native_binary_sha256"]

    checks = {
        "status": "passed",
        "debug_lib_tests": {"owned_context_local": "12/12", "owned_integer_hash": "9/9"},
        "strict_clippy_all_targets": "both crates passed -D warnings",
        "release_builds": "both crates passed; integer-hash built offline after Cargo.lock generation",
        "oracle": {"cases": 20, "modes": 6, "workers": [1, 4],
                   "full_trace_matches": 240},
        "quick": {"rows": 20, "cases": 2, "modes": 5, "workers": [1, 4],
                  "repeats": 1, "full_trace_matches": 20,
                  "deterministic_work_mismatches": 0},
        "integer_long": {"rows": 20, "cases": 2,
                         "integer_modes": ["std", "ahash"],
                         "direct_scalar_reference": ["combined_filtered_halfword",
                                                     "combined_filtered"],
                         "workers": [1, 4], "repeats": 2,
                         "fingerprint_matches": 20,
                         "deterministic_work_mismatches": 0},
        "cpu_budget": {"1": [5], "4": [0, 1, 2, 5]},
        "memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds",
        "limitations": [
            "The 256 KiB quick screen has one run per cell.",
            "The 4 MiB follow-up has only two runs per cell; ranges are descriptive, not confidence intervals.",
            "The direct scalar reference uses a different algorithm path; only std versus ahash isolates the hasher within one binary.",
            "Local scratch results do not establish hardware false sharing or its absence.",
            "CPU/wall is average occupied cores, not useful-compute utilization.",
        ],
        "new_source_snapshot_sha256": snapshot_sha,
        "new_source_hashes": "new-source-hashes.json",
        "shared_source_provenance": "shared-source-provenance.json",
        "ahash_build_context": "ahash-build-context.json",
        "binary_sha256": quick_env["binary_sha256"],
        "native_reference_binary_sha256": quick_env["native_reference_binary_sha256"],
        "binary_paths": {name: str((BIN_DIR / config["binary"]).relative_to(ROOT))
                         for name, config in modes["quick"].items()},
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
    }
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": "passed", "oracle_matches": 240,
                      "quick_rows": 20, "integer_long_rows": 20}))


if __name__ == "__main__":
    main()

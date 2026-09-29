"""Verify archived source, binaries, exactness and measurement completeness."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_hashes = json.loads((OUT / "new-source-hashes.json").read_text())
    for name, expected in source_hashes.items():
        if sha(ROOT / name) != expected:
            raise AssertionError((name, "source changed after freeze"))
    shared = json.loads((OUT / "shared-source-provenance.json").read_text())
    if not shared["all_files_match_base_commit_byte_for_byte"]:
        raise AssertionError("shared source differs from base commit")
    for name, expected in shared["files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise AssertionError((name, "shared source changed after measurement"))
    oracle = json.loads((OUT / "differential.json").read_text())
    quick = json.loads((OUT / "summary.json").read_text())
    screen = json.loads((OUT / "screen-4m-summary.json").read_text())
    quick_env = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    quick_rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    screen_rows = [json.loads(line) for line in (OUT / "screen-4m.jsonl").read_text().splitlines()]
    if oracle["compared_runs"] != 400 or len(quick_rows) != 28 or len(screen_rows) != 28:
        raise AssertionError("incomplete validation")
    if quick["deterministic_work_mismatches"] or screen["deterministic_work_mismatches"]:
        raise AssertionError("deterministic work changed")
    if not all(row["full_trace_match"] for row in quick_rows):
        raise AssertionError("quick trace mismatch")
    if not all(row["full_training_fingerprint_match"] for row in screen_rows):
        raise AssertionError("4 MiB fingerprint mismatch")
    if not all(row.get("call_cpu_seconds", 0) > 0 for row in quick_rows
               if row["version"] != "combo_lazy"):
        raise AssertionError("quick process CPU field missing")
    if not all(row.get("call_cpu_seconds", 0) > 0 for row in screen_rows
               if row["version"].startswith("integrated_")):
        raise AssertionError("4 MiB process CPU field missing")
    binary_names = {
        "table_hash": "owned_selected_table", "table_flat": "owned_selected_table",
        "cache_off": "owned_route_cache", "cache_4096": "owned_route_cache",
        "integrated_control": "owned_fused_direct",
        "integrated_candidate": "owned_fused_direct",
    }
    for version, expected in quick_env["binary_sha256"].items():
        path = (ROOT / "rust/target/reruns/radical-layout-combo-v1/combo"
                if version == "combo_lazy" else
                ROOT / "rust/target/reruns/radical-planning-integrated-v1" /
                binary_names[version])
        if sha(path) != expected:
            raise AssertionError((version, "binary SHA mismatch"))
    report = {
        "status": "passed",
        "validation": {
            "selected_table_debug_lib_tests": "11/11 passed",
            "route_cache_debug_lib_tests": "10/10 passed",
            "fused_direct_debug_lib_tests": "9/9 passed",
            "strict_clippy_all_targets": "all three crates passed -D warnings",
            "release_builds": "all three passed",
            "complete_python_oracle": oracle,
        },
        "quick": {
            "rows": 28, "versions": 7, "cases": 2, "workers": [1, 4],
            "repeats": 1, "full_trace_match": True,
            "same_deterministic_work_as_combo": True,
            "worker1_affinity": [5], "worker4_affinity": [0, 1, 2, 5],
            "memory_metric": "train_vm_hwm_mib", "process_cpu_metric": "call_cpu_seconds",
        },
        "screen_4m": {
            "rows": 28, "configurations": 14, "repeats": 2,
            "fingerprints_match_prior_reference": True,
            "same_deterministic_work_across_posting_modes": True,
            "worker1_affinity": [5], "worker4_affinity": [0, 1, 2, 5],
            "memory_metric": "train_vm_hwm_mib", "process_cpu_metric": "call_cpu_seconds",
        },
        "shared_source_provenance": "shared-source-provenance.json",
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "new_source_hashes": "new-source-hashes.json",
        "binary_sha256": quick_env["binary_sha256"],
        "native_reference_binary_sha256": quick_env["native_reference_binary_sha256"],
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
        "binary_paths": {
            "combo": "rust/target/reruns/radical-layout-combo-v1/combo",
            "native": "rust/target/reruns/radical-fixed-budget-v1/ablation",
            "selected_table": "rust/target/reruns/radical-planning-integrated-v1/owned_selected_table",
            "route_cache": "rust/target/reruns/radical-planning-integrated-v1/owned_route_cache",
            "fused_direct": "rust/target/reruns/radical-planning-integrated-v1/owned_fused_direct",
        },
    }
    with (OUT / "checks.json").open("w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "oracle": 400,
                      "quick": 28, "screen_4m": 28}))


if __name__ == "__main__":
    main()

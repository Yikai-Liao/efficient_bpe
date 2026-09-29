"""Verify archived provenance and summarize completed validation checks."""

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
    oracle = json.loads((OUT / "differential.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    screen_summary = json.loads((OUT / "screen-4m-summary.json").read_text())
    environment = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    if oracle["compared_runs"] != 280 or len(rows) != 32:
        raise AssertionError("incomplete validation")
    if summary["deterministic_work_mismatches"]:
        raise AssertionError("deterministic work changed")
    if not all(row["full_trace_match"] for row in rows):
        raise AssertionError("trace mismatch")
    if not all(row.get("call_cpu_seconds", 0) > 0 for row in rows
               if row["version"] != "combo_lazy"):
        raise AssertionError("missing process CPU measure")
    screen_rows = [json.loads(line) for line in (OUT / "screen-4m.jsonl").read_text().splitlines()]
    if len(screen_rows) != 28 or screen_summary["deterministic_work_mismatches"]:
        raise AssertionError("4 MiB diagnostic incomplete or changed work")
    if not all(row["full_training_fingerprint_match"] and
               row["call_cpu_seconds"] > 0 for row in screen_rows):
        raise AssertionError("4 MiB fingerprint or CPU metric failed")
    for version, expected in environment["binary_sha256"].items():
        if version == "combo_lazy":
            binary = ROOT / "rust/target/reruns/radical-layout-combo-v1/combo"
        else:
            binary_name = {"fused": "owned_fused", "scatter": "owned_scatter",
                           "direct": "owned_direct"}[version.split("_")[0]]
            binary = ROOT / "rust/target/reruns/radical-fused-scatter-v1" / binary_name
        if sha(binary) != expected:
            raise AssertionError((version, "binary SHA mismatch"))
    report = {
        "status": "passed",
        "validation": {
            "fused_debug_lib_tests": "9/9 passed",
            "scatter_debug_lib_tests": "11/11 passed after final Clippy fix",
            "direct_debug_lib_tests": "10/10 passed",
            "strict_clippy_all_targets": "all three crates passed -D warnings",
            "release_builds": "all three passed",
            "complete_python_oracle": oracle,
        },
        "quick": {
            "rows": 32, "versions": 8, "cases": 2, "workers": [1, 4],
            "repeats": 1, "full_trace_match": True,
            "same_deterministic_work_as_combo": True,
            "worker1_affinity": environment["cpu_budget"]["1"],
            "worker4_affinity": environment["cpu_budget"]["4"],
            "memory_metric": "train_vm_hwm_mib",
            "process_cpu_metric": "call_cpu_seconds",
            "scatter_default_threshold": 4096,
            "scatter_heavy_keys_all_zero": all(row.get("scatter_heavy_keys", 0) == 0
                                           for row in rows if row["version"].startswith("scatter_")),
        },
        "screen_4m": {
            "rows": 28, "versions": 7, "cases": 2, "workers": [1, 4],
            "repeats": 1, "fingerprints_match_prior_reference": True,
            "same_deterministic_work_across_new_modes": True,
            "worker1_affinity": [5], "worker4_affinity": [0, 1, 2, 5],
            "memory_metric": "train_vm_hwm_mib",
            "process_cpu_metric": "call_cpu_seconds",
            "scatter_heavy_keys_en_w4": next(row["scatter_heavy_keys"] for row in screen_rows
                                               if row["case_id"] == "en-4m-continuous" and
                                               row["version"] == "scatter_scatter" and
                                               row["workers"] == 4),
            "scatter_heavy_keys_zh_w4": next(row["scatter_heavy_keys"] for row in screen_rows
                                               if row["case_id"] == "zh-4m-continuous" and
                                               row["version"] == "scatter_scatter" and
                                               row["workers"] == 4),
        },
        "source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "source_hashes_file": "new-source-hashes.json",
        "binary_sha256": environment["binary_sha256"],
        "native_reference_binary_sha256": environment["native_reference_binary_sha256"],
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
        "binary_paths": {
            "combo": "rust/target/reruns/radical-layout-combo-v1/combo",
            "native": "rust/target/reruns/radical-fixed-budget-v1/ablation",
            "fused": "rust/target/reruns/radical-fused-scatter-v1/owned_fused",
            "scatter": "rust/target/reruns/radical-fused-scatter-v1/owned_scatter",
            "direct": "rust/target/reruns/radical-fused-scatter-v1/owned_direct",
        },
    }
    with (OUT / "checks.json").open("w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "oracle": 280, "quick": 32,
                      "screen_4m": 28}))


if __name__ == "__main__":
    main()

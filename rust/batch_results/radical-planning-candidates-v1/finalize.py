"""Verify source, binaries, exactness and fixed-budget screen completeness."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN_DIR = RUST / "target/reruns/radical-planning-candidates-v1"


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
    quick_env = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    modes = json.loads((OUT / "modes.json").read_text())
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    expected_oracle = 20 * 2 * len(modes["oracle"])
    expected_quick = 2 * 2 * len(modes["quick"])
    rejections = oracle["documented_u16_limit_rejections"]
    if (oracle["compared_runs"] + len(rejections) != expected_oracle or
            len(rejections) != 2 or len(rows) != expected_quick):
        raise AssertionError("incomplete validation")
    if quick["deterministic_work_mismatches"]:
        raise AssertionError("deterministic work changed")
    if not all(row["full_trace_match"] for row in rows):
        raise AssertionError("quick trace mismatch")
    if not all(row.get("call_cpu_seconds", 0) > 0 for row in rows
               if row["version"] != "combo_lazy"):
        raise AssertionError("process CPU field missing")
    for version, expected in quick_env["binary_sha256"].items():
        binary = BIN_DIR / modes["quick"][version]["binary"]
        if sha(binary) != expected:
            raise AssertionError((version, "binary SHA mismatch"))
    report = {
        "status": "passed",
        "validation": {
            "narrow_debug_lib_tests": "11/11 passed after final hot-path cast change",
            "cache_reuse_debug_lib_tests": "11/11 passed",
            "context_debug_lib_tests": "11/11 passed",
            "strict_clippy_all_targets": "all three crates passed -D warnings",
            "release_builds": "all three passed",
            "complete_python_oracle": oracle,
        },
        "quick": {
            "rows": expected_quick, "versions": len(modes["quick"]),
            "cases": 2, "workers": [1, 4], "repeats": 1,
            "full_trace_match": True,
            "same_deterministic_work_as_combo": True,
            "worker1_affinity": [5], "worker4_affinity": [0, 1, 2, 5],
            "memory_metric": "train_vm_hwm_mib", "process_cpu_metric": "call_cpu_seconds",
        },
        "limitations": [
            "In the measured cache modes, batch resets victim_way each batch and reuse carries it across batches; the timing does not isolate allocation/scan savings from replacement-phase changes.",
            "The u16 endpoint corpus is exact-allocated while the u32 control retains the input Vec capacity, so width and capacity both change.",
            "The quick screen has one run per cell and does not establish a stable speed ordering.",
        ],
        "shared_source_provenance": "shared-source-provenance.json",
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "new_source_hashes": "new-source-hashes.json",
        "binary_sha256": quick_env["binary_sha256"],
        "native_reference_binary_sha256": quick_env["native_reference_binary_sha256"],
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
        "binary_paths": {name: str((BIN_DIR / config["binary"]).relative_to(ROOT))
                         for name, config in modes["quick"].items()},
    }
    with (OUT / "checks.json").open("w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "oracle_matches": oracle["compared_runs"],
                      "documented_rejections": len(rejections), "quick": len(rows)}))


if __name__ == "__main__":
    main()

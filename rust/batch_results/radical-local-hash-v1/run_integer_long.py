"""Authorized n=2 4 MiB follow-up for std versus ahash and direct scalar."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN_DIR = RUST / "target/reruns/radical-local-hash-v1"
INTEGER = BIN_DIR / "owned_integer_hash"
NATIVE = RUST / "target/reruns/radical-fixed-budget-v1/ablation"
MANIFEST = RUST / "ablation_results/fixtures.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_fingerprints():
    reference = {}
    old = RUST / "batch_results/radical-layout-combo-v1/screen.jsonl"
    for line in old.read_text().splitlines():
        row = json.loads(line)
        prior = reference.setdefault(row["case_id"], row["fingerprint"])
        if prior != row["fingerprint"]:
            raise AssertionError((row["case_id"], "prior reference disagreement"))
    return reference


def main():
    fixtures = [row for row in json.loads(MANIFEST.read_text())
                if row["case_id"] in ("en-4m-continuous", "zh-4m-continuous")]
    assert len(fixtures) == 2
    modes = json.loads((OUT / "modes.json").read_text())["quick"]
    source_hashes = json.loads((OUT / "new-source-hashes.json").read_text())
    for name, digest in source_hashes.items():
        assert sha(ROOT / name) == digest, name
    quick_env = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    assert sha(INTEGER) == quick_env["binary_sha256"]["integer_std"]
    assert sha(NATIVE) == quick_env["native_reference_binary_sha256"]
    reference = expected_fingerprints()
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())
    first = [(row, mode, workers) for row in fixtures
             for mode, workers in (("integer_std", 1), ("integer_std", 4),
                                   ("integer_ahash", 1), ("integer_ahash", 4),
                                   ("best_direct_scalar", 1))]
    random.Random(20260930).shuffle(first)
    jobs = [(0, *job) for job in first]
    jobs.extend((1, *job) for job in reversed(first))
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST),
        "integer_binary_sha256": sha(INTEGER), "native_binary_sha256": sha(NATIVE),
        "modes_sha256": sha(OUT / "modes.json"),
        "new_source_sha256": source_hashes,
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
        "seed": 20260930, "repeats": 2, "rows": len(jobs),
        "schedule": "seeded shuffle then exact reverse",
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds (CLOCK_PROCESS_CPUTIME_ID)",
        "semantic_check": "prior full-training SHA-256 fingerprint of every rule and final tokens",
    }
    with (OUT / "integer-long.jsonl").open("x") as output, \
            (OUT / "integer-long.jsonl.environment.json").open("x") as meta:
        meta.write(json.dumps(sidecar, indent=2) + "\n")
        for index, (repetition, row, mode, workers) in enumerate(jobs, 1):
            fixture = RUST / row["file"]
            assert sha(fixture) == row["fixture_sha256"]
            common = ["--input", str(fixture), "--workers", str(workers),
                      "--rules", str(row["rules"]),
                      "--min-frequency", str(row["min_frequency"])]
            if mode == "best_direct_scalar":
                variant = ("combined_filtered_halfword" if row["case_id"].startswith("en-")
                           else "combined_filtered")
                binary = NATIVE
                command = [str(binary), *common, "--variant", variant, "--bounds", "checked"]
            else:
                binary = INTEGER
                command = [str(binary), *common, "--chunk-size", "4096",
                           "--heap-policy", "lazy", *modes[mode]["args"]]
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            assert observed["fixture_sha256"] == row["fixture_sha256"]
            assert observed["fingerprint"] == reference[row["case_id"]]
            for field in ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"):
                assert observed[field] > 0, (mode, field)
            observed.update({
                "case_id": row["case_id"], "version": mode, "repetition": repetition,
                "requested_rules": row["rules"], "min_frequency": row["min_frequency"],
                "outer_call_seconds": outer_seconds, "binary_sha256": sha(binary),
                "cpu_affinity": budgets[workers], "cpu_budget": workers,
                "mean_occupied_cores": observed["call_cpu_seconds"] / observed["call_seconds"],
                "command": command, "full_training_fingerprint_match": True,
            })
            output.write(json.dumps(observed) + "\n")
            output.flush()
            print(f"{index}/{len(jobs)}", row["case_id"], mode, workers, repetition,
                  f"{observed['call_seconds']:.4f}s", flush=True)
    print(f"verified {len(jobs)} exact training fingerprints")


if __name__ == "__main__":
    main()

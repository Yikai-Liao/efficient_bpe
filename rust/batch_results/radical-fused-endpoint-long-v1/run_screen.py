"""Authorized 20-call, n=2, fixed-CPU 4 MiB fused-endpoint follow-up."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
QUICK = RUST / "batch_results/radical-fused-bitmap-quick-v1"
MANIFEST = RUST / "ablation_results/fixtures.json"
FUSED = RUST / "target/reruns/radical-fused-endpoint-gate-v1/radical-owned-fused-endpoint"
SERIAL = RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_fingerprints():
    reference = {}
    for line in (RUST / "batch_results/radical-layout-combo-v1/screen.jsonl").read_text().splitlines():
        row = json.loads(line)
        old = reference.setdefault(row["case_id"], row["fingerprint"])
        if old != row["fingerprint"]:
            raise AssertionError((row["case_id"], "prior fingerprint disagreement"))
    return reference


def main():
    fixtures = [row for row in json.loads(MANIFEST.read_text())
                if row["case_id"] in ("en-4m-continuous", "zh-4m-continuous")]
    assert len(fixtures) == 2
    quick_check = json.loads((QUICK / "checks.json").read_text())
    assert quick_check["status"] == "passed" and quick_check["quick_rows"] == 30
    source_hashes = json.loads((QUICK / "new-source-hashes.json").read_text())
    for name, expected in source_hashes.items():
        assert sha(ROOT / name) == expected, name
    binary_hashes = quick_check["binary_sha256"]
    assert sha(FUSED) == binary_hashes["fused"]
    assert sha(SERIAL) == binary_hashes["serial"]
    reference_fingerprints = expected_fingerprints()
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())
    first = [(row, mode, workers) for row in fixtures
             for mode, workers in (("two-pass", 1), ("two-pass", 4),
                                   ("tagged-fused", 1), ("tagged-fused", 4),
                                   ("serial-cf32-ahash-checked", 1))]
    random.Random(20260930).shuffle(first)
    jobs = [(0, *job) for job in first]
    jobs.extend((1, *job) for job in reversed(first))
    assert len(jobs) == 20
    sidecar = {
        "rows": len(jobs), "repetitions": 2, "seed": 20260930,
        "schedule": "seeded shuffle then exact reverse",
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST),
        "binary_sha256": {"fused": sha(FUSED), "serial": sha(SERIAL)},
        "new_source_sha256": source_hashes,
        "new_source_snapshot_sha256": sha(QUICK / "new-source-snapshot.tar.gz"),
        "shared_source_provenance_sha256": sha(QUICK / "shared-source-provenance.json"),
        "quick_checks_sha256": sha(QUICK / "checks.json"),
        "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds",
        "semantic_check": "prior reference fingerprint plus full merge-rule and final-token JSON equality",
    }
    traces = {}
    with tempfile.TemporaryDirectory(prefix="fused-endpoint-long-") as temp_name, \
            (OUT / "screen.jsonl").open("x") as output:
        temp = Path(temp_name)
        for index, (repetition, row, mode, workers) in enumerate(jobs, 1):
            fixture = RUST / row["file"]
            assert sha(fixture) == row["fixture_sha256"]
            trace_path = temp / "trace.json"
            common = ["--input", str(fixture), "--workers", str(workers),
                      "--rules", str(row["rules"]),
                      "--min-frequency", str(row["min_frequency"]),
                      "--trace", str(trace_path)]
            if mode.startswith("serial-"):
                binary = SERIAL
                command = [str(binary), *common, "--backend", "combined_filtered",
                           "--bounds", "checked", "--integer-hash", "ahash"]
            else:
                binary = FUSED
                command = [str(binary), *common, "--chunk-size", "4096",
                           "--heap-policy", "lazy", "--integer-hash", "ahash",
                           "--endpoint-plan", mode]
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            assert observed["fixture_sha256"] == row["fixture_sha256"]
            assert observed["fingerprint"] == reference_fingerprints[row["case_id"]]
            actual_trace = json.loads(trace_path.read_text())
            prior = traces.setdefault(row["case_id"], actual_trace)
            assert actual_trace == prior, (row["case_id"], mode, workers, repetition)
            assert all(observed[field] > 0 for field in
                       ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"))
            if mode in ("two-pass", "tagged-fused"):
                assert observed["endpoint_plan_requested"] == mode
                assert observed["endpoint_plan_effective"] == mode
            observed.update({
                "case_id": row["case_id"], "version": mode, "repetition": repetition,
                "requested_rules": row["rules"], "min_frequency": row["min_frequency"],
                "outer_call_seconds": outer_seconds, "binary_sha256": sha(binary),
                "cpu_affinity": budgets[workers], "cpu_budget": workers,
                "mean_occupied_cores": observed["call_cpu_seconds"] / observed["call_seconds"],
                "command": command, "full_training_fingerprint_match": True,
                "full_trace_match": True,
            })
            output.write(json.dumps(observed) + "\n")
            output.flush()
            print(f"{index}/20", row["case_id"], mode, workers, repetition,
                  f"{observed['call_seconds']:.6f}s", flush=True)
    with (OUT / "screen.jsonl.environment.json").open("x") as meta:
        json.dump(sidecar, meta, indent=2)
        meta.write("\n")
    print("verified 20 complete training traces and reference fingerprints")


if __name__ == "__main__":
    main()

"""Authorized twelve-call fixed-budget snapshot-region screen."""

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
BIN = RUST / "target/reruns/radical-region-snapshot-gate-v1/radical-owned-region-snapshot"
MANIFEST = RUST / "batch_results/quick-fixtures-262144-512.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    oracle = json.loads((RUST / "batch_results/radical-region-snapshot-gate-v1/differential.json").read_text())
    assert oracle["status"] == "passed"
    assert oracle["standard_fulltrace_matches"] == 160
    assert oracle["directed_fulltrace_matches"] == 10
    assert oracle["dynamic_smoke_fulltrace_matches"] == 1
    assert sha(BIN) == oracle["binary_sha256"]
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    assert all(sha(ROOT / path) == digest for path, digest in sources.items())
    fixtures = json.loads(MANIFEST.read_text())
    assert {row["case_id"] for row in fixtures} == {
        "quick-en-continuous-262144", "quick-zh-continuous-262144"}
    assert all(sha(RUST / row["file"]) == row["fixture_sha256"] for row in fixtures)
    jobs = [(row, mode, workers) for row in fixtures
            for mode in ("dynamic", "region", "snapshot") for workers in (1, 4)]
    assert len(jobs) == 12
    random.Random(20260930).shuffle(jobs)
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())
    observations = []
    with tempfile.TemporaryDirectory(prefix="region-snapshot-screen-") as temp_name:
        temp = Path(temp_name)
        for index, (row, mode, workers) in enumerate(jobs, 1):
            trace = temp / f"trace-{index}.json"
            command = [str(BIN), "--input", str(RUST / row["file"]),
                       "--workers", str(workers), "--chunk-size", "4096",
                       "--rules", str(row["rules"]),
                       "--min-frequency", str(row["min_frequency"]),
                       "--heap-policy", "lazy", "--integer-hash", "ahash",
                       "--endpoint-plan", "tagged-fused", "--region-mode", mode,
                       "--trace", str(trace)]
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            result = json.loads(process.stdout.strip().splitlines()[-1])
            assert result["fixture_sha256"] == row["fixture_sha256"]
            assert result["region_mode_effective"] == mode
            assert result["endpoint_plan_effective"] == "tagged-fused"
            assert all(result[field] > 0 for field in
                       ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"))
            result.update({"case_id": row["case_id"], "version": mode,
                           "requested_rules": row["rules"],
                           "min_frequency": row["min_frequency"],
                           "outer_call_seconds": outer_seconds,
                           "binary_sha256": sha(BIN), "cpu_affinity": budgets[workers],
                           "cpu_budget": workers,
                           "mean_occupied_cores": result["call_cpu_seconds"] / result["call_seconds"],
                           "command": command})
            observations.append((result, json.loads(trace.read_text())))
            print(f"{index}/12", row["case_id"], mode, workers,
                  f"{result['call_seconds']:.6f}s", flush=True)
        references = {(result["case_id"]): (result["fingerprint"], trace)
                      for result, trace in observations if result["version"] == "dynamic"}
        assert len(references) == 2
        for result, trace in observations:
            fingerprint, expected = references[result["case_id"]]
            assert result["fingerprint"] == fingerprint and trace == expected
            result["full_trace_match"] = True
    sidecar = {"status": "passed", "rows": 12, "repetitions": 1,
               "seed": 20260930, "manifest_sha256": sha(MANIFEST),
               "binary_sha256": sha(BIN), "new_source_sha256": sources,
               "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
               "shared_source_provenance_sha256": sha(OUT / "shared-source-provenance.json"),
               "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
               "training_memory_metric": "train_vm_hwm_mib",
               "process_cpu_metric": "call_cpu_seconds",
               "semantic_check": "full merge-rule and final-token trace plus SHA-256 fingerprint"}
    with (OUT / "screen.jsonl").open("x") as output:
        for result, _ in observations:
            output.write(json.dumps(result) + "\n")
    with (OUT / "screen.jsonl.environment.json").open("x") as output:
        json.dump(sidecar, output, indent=2)
        output.write("\n")
    print("verified twelve exact dynamic/region/snapshot traces")


if __name__ == "__main__":
    main()

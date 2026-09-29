"""Authorized 14-call lazy-neighbor certificate mechanism screen."""

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
BIN = RUST / "target/reruns/radical-lazy-neighbor-gate-v1/owned_lazy_neighbor"
MANIFEST = RUST / "batch_results/quick-fixtures-262144-512.json"
MAX_USIZE = "18446744073709551615"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    fixtures = json.loads(MANIFEST.read_text())
    assert len(fixtures) == 2
    oracle = json.loads((OUT / "differential.json").read_text())
    assert oracle["status"] == "passed" and oracle["compared_runs"] == 200
    assert sha(BIN) == oracle["binary_sha256"]["owned_lazy_neighbor"]
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, name
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())
    jobs = [(row, mode, workers) for row in fixtures
            for mode in ("type", "birth_neighbor64", "lazy_default")
            for workers in (1, 4)]
    jobs.extend((row, "lazy_force_serial", 4) for row in fixtures)
    assert len(jobs) == 14
    random.Random(20260930).shuffle(jobs)
    observations = []
    with tempfile.TemporaryDirectory(prefix="radical-lazy-neighbor-quick-") as temp_name:
        temp = Path(temp_name)
        for index, (row, mode, workers) in enumerate(jobs, 1):
            fixture = RUST / row["file"]
            assert sha(fixture) == row["fixture_sha256"]
            trace = temp / f"trace-{index}.json"
            certificate = ("lazy-neighbor64" if mode.startswith("lazy_") else
                           "birth-neighbor64" if mode == "birth_neighbor64" else "type")
            extra = (["--mask-parallel-threshold", MAX_USIZE]
                     if mode == "lazy_force_serial" else [])
            command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
                       "--chunk-size", "4096", "--rules", str(row["rules"]),
                       "--min-frequency", str(row["min_frequency"]),
                       "--heap-policy", "lazy", "--batch-certificate", certificate,
                       *extra, "--trace", str(trace)]
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            assert observed["fixture_sha256"] == row["fixture_sha256"]
            assert observed["batch_certificate"] == certificate
            for metric in ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"):
                assert observed[metric] > 0, metric
            observed.update({
                "case_id": row["case_id"], "version": mode,
                "requested_rules": row["rules"], "min_frequency": row["min_frequency"],
                "outer_call_seconds": outer_seconds, "binary_sha256": sha(BIN),
                "cpu_affinity": budgets[workers], "cpu_budget": workers,
                "mean_occupied_cores": observed["call_cpu_seconds"] / observed["call_seconds"],
                "command": command,
            })
            observations.append((observed, json.loads(trace.read_text())))
            print(f"{index}/14", row["case_id"], mode, workers,
                  f"{observed['call_seconds']:.6f}s", flush=True)
        references = {observed["case_id"]: (observed, trace)
                      for observed, trace in observations
                      if observed["version"] == "type" and observed["workers"] == 1}
        assert len(references) == 2
        for observed, trace in observations:
            reference, expected_trace = references[observed["case_id"]]
            assert trace == expected_trace, (observed["case_id"], observed["version"], "trace")
            assert observed["fingerprint"] == reference["fingerprint"]
            observed["full_trace_match"] = True
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST),
        "binary_sha256": sha(BIN), "new_source_sha256": sources,
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
        "seed": 20260930, "repeats": 1, "rows": 14,
        "heap_policy": "lazy", "chunk_size": 4096,
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds",
        "semantic_check": "complete merge-rule and final-token trace plus SHA-256 fingerprint",
    }
    with (OUT / "quick.jsonl").open("x") as output:
        for observed, _ in observations:
            output.write(json.dumps(observed) + "\n")
    with (OUT / "quick.jsonl.environment.json").open("x") as output:
        json.dump(sidecar, output, indent=2)
        output.write("\n")
    print("verified 14 exact lazy-neighbor training traces")


if __name__ == "__main__":
    main()

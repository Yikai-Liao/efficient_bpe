"""Authorized 12-call AA-sort mechanism screen, aHash fixed."""

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
BIN = RUST / "target/reruns/radical-aa-radix-gate-v1/owned_aa_radix"
MANIFESTS = (RUST / "ablation_results/fixtures.json",
             RUST / "batch_results/quick-fixtures-262144-512.json")
CASES = {"single-run-a-65536", "single-piece-ab-65536",
         "quick-en-continuous-262144"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    fixtures = [row for manifest in MANIFESTS for row in json.loads(manifest.read_text())
                if row["case_id"] in CASES]
    assert len(fixtures) == len(CASES) == 3
    oracle = json.loads((OUT / "differential.json").read_text())
    assert oracle["status"] == "passed" and sha(BIN) == oracle["binary_sha256"]["owned_aa_radix"]
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, name
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())
    jobs = [(row, aa_sort, workers) for row in fixtures
            for aa_sort in ("std", "radix") for workers in (1, 4)]
    assert len(jobs) == 12
    random.Random(20260930).shuffle(jobs)
    observations = []
    with tempfile.TemporaryDirectory(prefix="radical-aa-radix-quick-") as temp_name:
        temp = Path(temp_name)
        for index, (row, aa_sort, workers) in enumerate(jobs, 1):
            fixture = RUST / row["file"]
            assert sha(fixture) == row["fixture_sha256"]
            trace = temp / f"trace-{index}.json"
            command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
                       "--chunk-size", "4096", "--rules", str(row["rules"]),
                       "--min-frequency", str(row["min_frequency"]),
                       "--heap-policy", "lazy", "--integer-hash", "ahash",
                       "--aa-sort", aa_sort, "--trace", str(trace)]
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            assert observed["fixture_sha256"] == row["fixture_sha256"]
            assert observed["aa_sort"] == aa_sort and observed["integer_hash"] == "ahash"
            for metric in ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"):
                assert observed[metric] > 0, metric
            observed.update({
                "case_id": row["case_id"], "version": aa_sort,
                "requested_rules": row["rules"], "min_frequency": row["min_frequency"],
                "outer_call_seconds": outer_seconds, "binary_sha256": sha(BIN),
                "cpu_affinity": budgets[workers], "cpu_budget": workers,
                "mean_occupied_cores": observed["call_cpu_seconds"] / observed["call_seconds"],
                "aa_sort_share_of_call": observed["aa_sort_seconds"] / observed["call_seconds"],
                "command": command,
            })
            observations.append((observed, json.loads(trace.read_text())))
            print(f"{index}/12", row["case_id"], aa_sort, workers,
                  f"{observed['call_seconds']:.6f}s",
                  f"AA share {observed['aa_sort_share_of_call']:.3f}", flush=True)
        references = {observed["case_id"]: (observed, trace) for observed, trace in observations
                      if observed["version"] == "std" and observed["workers"] == 1}
        assert len(references) == 3
        for observed, trace in observations:
            reference, expected_trace = references[observed["case_id"]]
            assert trace == expected_trace, (observed["case_id"], observed["version"], "trace")
            assert observed["fingerprint"] == reference["fingerprint"]
            observed["full_trace_match"] = True
    sidecar = {
        "manifest_sha256": {str(manifest.relative_to(ROOT)): sha(manifest)
                            for manifest in MANIFESTS},
        "case_ids": sorted(CASES), "binary_sha256": sha(BIN),
        "new_source_sha256": sources,
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
        "seed": 20260930, "repeats": 1, "rows": 12,
        "integer_hash": "ahash", "heap_policy": "lazy", "chunk_size": 4096,
        "std_sort_implementation": "Rayon pool.install(par_sort_unstable)",
        "radix_sort_implementation": "serial in-place u32 radix",
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds",
        "semantic_check": "full rule trace and final tokens plus SHA-256 fingerprint",
    }
    with (OUT / "quick.jsonl").open("x") as output:
        for observed, _ in observations:
            output.write(json.dumps(observed) + "\n")
    with (OUT / "quick.jsonl.environment.json").open("x") as output:
        json.dump(sidecar, output, indent=2)
        output.write("\n")
    print("verified 12 exact AA-sort training traces")


if __name__ == "__main__":
    main()

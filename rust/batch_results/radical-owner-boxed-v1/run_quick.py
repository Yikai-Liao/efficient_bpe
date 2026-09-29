"""Fixed process-CPU-budget quick screen for owned and boxed batches."""

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
MANIFEST = RUST / "batch_results/quick-fixtures-262144-512.json"
NATIVE = RUST / "target/reruns/radical-fixed-budget-v1/ablation"
BINARIES = {
    "v4": RUST / "target/reruns/radical-fixed-budget-v1/v4",
    "boxed": RUST / "target/reruns/radical-fixed-budget-v1/boxed",
    "owned_lazy": RUST / "target/reruns/radical-owner-boxed-v1/owned",
    "owned_eager": RUST / "target/reruns/radical-owner-boxed-v1/owned",
    "batched_boxed": RUST / "target/reruns/radical-owner-boxed-v1/batched_boxed",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    rows = json.loads(MANIFEST.read_text())
    native_rows = [json.loads(line) for line in (OUT / "native-quick.jsonl").read_text().splitlines()]
    references = {row["case_id"]: row for row in native_rows
                  if row["variant"] == "combined_filtered"}
    native_env = json.loads((OUT / "native-quick.jsonl.environment.json").read_text())
    original_affinity = set(os.sched_getaffinity(0))
    cpu_order = [native_env["scalar_cpu"]] + sorted(original_affinity - {native_env["scalar_cpu"]})
    budgets = {1: cpu_order[:1], 4: cpu_order[:4]}
    if sorted(original_affinity) != native_env["initial_affinity"]:
        raise RuntimeError("current affinity differs from native reference")
    if native_env["parallel_core_budget_policy"] != "workers":
        raise RuntimeError("native reference did not use a fixed worker CPU budget")
    jobs = [(v, row, workers) for row in rows for v in BINARIES for workers in (1, 4)]
    random.Random(20260930).shuffle(jobs)
    source_hashes = json.loads((OUT / "native-source-hashes.json").read_text())
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha(MANIFEST),
        "native_reference_binary_sha256": sha(NATIVE),
        "radical_binary_sha256": {v: sha(b) for v, b in BINARIES.items()},
        "source_snapshot": "rust/batch_results/radical-owner-boxed-v1/native-source-snapshot.tar.gz",
        "native_source_sha256": source_hashes,
        "initial_affinity": sorted(original_affinity),
        "cpu_budget": budgets,
        "seed": 20260930, "repeats": 1,
        "fingerprint": "complete rule trace plus final token sequence",
        "training_memory_metric": "train_vm_hwm_mib",
    }
    with tempfile.TemporaryDirectory(prefix="radical-fixed-quick-") as temp:
        temp = Path(temp)
        reference_traces = {}
        for row in rows:
            fixture = RUST / row["file"]
            if sha(fixture) != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], "fixture SHA mismatch"))
            trace = temp / f"reference-{row['case_id']}.json"
            command = [str(NATIVE), "--input", str(fixture),
                       "--variant", "combined_filtered", "--workers", "1",
                       "--rules", str(row["rules"]),
                       "--min-frequency", str(row["min_frequency"]),
                       "--bounds", "checked", "--trace", str(trace)]
            try:
                os.sched_setaffinity(0, budgets[1])
                observed = run(command)
            finally:
                os.sched_setaffinity(0, original_affinity)
            if observed["fingerprint"] != references[row["case_id"]]["fingerprint"]:
                raise AssertionError((row["case_id"], "reference fingerprint mismatch"))
            reference_traces[row["case_id"]] = json.loads(trace.read_text())
        with (OUT / "quick.jsonl").open("x") as output, \
                (OUT / "quick.jsonl.environment.json").open("x") as meta:
            meta.write(json.dumps(sidecar, indent=2) + "\n")
            for version, row, workers in jobs:
                fixture = RUST / row["file"]
                trace = temp / f"{version}-{row['case_id']}-{workers}.json"
                command = [str(BINARIES[version]), "--input", str(fixture),
                           "--workers", str(workers), "--chunk-size", "4096",
                           "--rules", str(row["rules"]),
                           "--min-frequency", str(row["min_frequency"]),
                           "--trace", str(trace)]
                if version.startswith("owned_"):
                    command.extend(["--heap-policy", version.removeprefix("owned_")])
                try:
                    os.sched_setaffinity(0, budgets[workers])
                    started = time.perf_counter()
                    observed = run(command)
                    outer_seconds = time.perf_counter() - started
                finally:
                    os.sched_setaffinity(0, original_affinity)
                if observed["fixture_sha256"] != row["fixture_sha256"]:
                    raise AssertionError((version, row["case_id"], "fixture SHA mismatch"))
                if observed["fingerprint"] != references[row["case_id"]]["fingerprint"]:
                    raise AssertionError((version, row["case_id"], workers, "fingerprint mismatch"))
                if json.loads(trace.read_text()) != reference_traces[row["case_id"]]:
                    raise AssertionError((version, row["case_id"], workers, "complete trace mismatch"))
                if "train_vm_hwm_mib" not in observed:
                    raise AssertionError((version, "training-time memory metric missing"))
                if version.startswith("owned_") and observed.get("heap_policy") != version.removeprefix("owned_"):
                    raise AssertionError((version, "heap policy mismatch"))
                observed.update({"case_id": row["case_id"], "version": version,
                                 "requested_rules": row["rules"],
                                 "min_frequency": row["min_frequency"],
                                 "outer_call_seconds": outer_seconds,
                                 "binary_sha256": sha(BINARIES[version]),
                                 "cpu_affinity": sorted(budgets[workers]),
                                 "cpu_budget": workers,
                                 "command": command, "full_trace_match": True})
                output.write(json.dumps(observed) + "\n")
                output.flush()
                print(version, row["case_id"], workers,
                      f"{observed['call_seconds']:.6f}s",
                      f"{observed['train_vm_hwm_mib']:.2f}MiB", flush=True)
    print(f"wrote {len(jobs)} verified rows")


if __name__ == "__main__":
    main()

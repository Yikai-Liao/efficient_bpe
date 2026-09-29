"""Fixed-budget 256 KiB quick screen of exact birth-neighbor certificate."""

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
BIN_DIR = RUST / "target/reruns/radical-neighbor-certificate-v1"
MANIFEST = RUST / "batch_results/quick-fixtures-262144-512.json"
NATIVE = RUST / "target/reruns/radical-fixed-budget-v1/ablation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    process = subprocess.run(command, text=True, capture_output=True, check=True)
    return json.loads(process.stdout.strip().splitlines()[-1])


def main():
    rows = json.loads(MANIFEST.read_text())
    modes = json.loads((OUT / "modes.json").read_text())["quick"]
    initial_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    if not all(set(cpus) <= initial_affinity for cpus in budgets.values()):
        raise RuntimeError(("required CPU affinity unavailable", sorted(initial_affinity)))
    for config in modes.values():
        if not (BIN_DIR / config["binary"]).is_file():
            raise FileNotFoundError(BIN_DIR / config["binary"])
    if not NATIVE.is_file():
        raise FileNotFoundError(NATIVE)
    source_hashes = json.loads((OUT / "new-source-hashes.json").read_text())
    for name, digest in source_hashes.items():
        if sha(ROOT / name) != digest:
            raise AssertionError((name, "source changed after freeze"))
    jobs = [(version, row, workers) for row in rows for version in modes
            for workers in (1, 4)]
    random.Random(20260930).shuffle(jobs)
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST),
        "modes_sha256": sha(OUT / "modes.json"),
        "native_reference_binary_sha256": sha(NATIVE),
        "binary_sha256": {name: sha(BIN_DIR / config["binary"])
                          for name, config in modes.items()},
        "new_source_sha256": source_hashes,
        "new_source_snapshot": "rust/batch_results/radical-neighbor-certificate-v1/new-source-snapshot.tar.gz",
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "initial_affinity": sorted(initial_affinity), "cpu_budget": budgets,
        "seed": 20260930, "repeats": 1,
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds (CLOCK_PROCESS_CPUTIME_ID)",
        "mean_occupied_cores": "call_cpu_seconds / call_seconds; not useful-computation utilization",
        "fingerprint": "complete rule trace plus final token sequence",
    }
    with tempfile.TemporaryDirectory(prefix="radical-neighbor-quick-") as temp_name:
        temp = Path(temp_name)
        references = {}
        with (OUT / "native-reference.jsonl").open("x") as reference_output:
            for row in rows:
                fixture = RUST / row["file"]
                if sha(fixture) != row["fixture_sha256"]:
                    raise AssertionError((row["case_id"], "fixture SHA mismatch"))
                trace = temp / f"native-{row['case_id']}.json"
                command = [str(NATIVE), "--input", str(fixture), "--variant",
                           "combined_filtered", "--workers", "1", "--bounds", "checked",
                           "--rules", str(row["rules"]), "--min-frequency",
                           str(row["min_frequency"]), "--trace", str(trace)]
                try:
                    os.sched_setaffinity(0, budgets[1])
                    observed = run(command)
                finally:
                    os.sched_setaffinity(0, initial_affinity)
                if observed["fixture_sha256"] != row["fixture_sha256"]:
                    raise AssertionError((row["case_id"], "native fixture SHA mismatch"))
                observed["case_id"] = row["case_id"]
                observed["cpu_affinity"] = budgets[1]
                observed["binary_sha256"] = sha(NATIVE)
                reference_output.write(json.dumps(observed) + "\n")
                references[row["case_id"]] = (observed, json.loads(trace.read_text()))
        with (OUT / "quick.jsonl").open("x") as output, \
                (OUT / "quick.jsonl.environment.json").open("x") as meta:
            meta.write(json.dumps(sidecar, indent=2) + "\n")
            for version, row, workers in jobs:
                config = modes[version]
                binary = BIN_DIR / config["binary"]
                fixture = RUST / row["file"]
                trace = temp / f"{version}-{row['case_id']}-{workers}.json"
                command = [str(binary), "--input", str(fixture), "--workers",
                           str(workers), "--chunk-size", "4096", "--rules",
                           str(row["rules"]), "--min-frequency",
                           str(row["min_frequency"]), "--heap-policy", "lazy",
                           "--trace", str(trace), *config["args"]]
                try:
                    os.sched_setaffinity(0, budgets[workers])
                    started = time.perf_counter()
                    observed = run(command)
                    outer_seconds = time.perf_counter() - started
                finally:
                    os.sched_setaffinity(0, initial_affinity)
                reference, reference_trace = references[row["case_id"]]
                if observed["fixture_sha256"] != row["fixture_sha256"]:
                    raise AssertionError((version, row["case_id"], "fixture SHA mismatch"))
                if observed["fingerprint"] != reference["fingerprint"]:
                    raise AssertionError((version, row["case_id"], workers, "fingerprint mismatch"))
                if json.loads(trace.read_text()) != reference_trace:
                    raise AssertionError((version, row["case_id"], workers, "trace mismatch"))
                if "train_vm_hwm_mib" not in observed:
                    raise AssertionError((version, "training memory missing"))
                if "call_cpu_seconds" not in observed:
                    raise AssertionError((version, "process CPU metric missing"))
                if observed.get("heap_policy") != "lazy":
                    raise AssertionError((version, "heap policy mismatch"))
                observed["mean_occupied_cores"] = (
                    observed["call_cpu_seconds"] / observed["call_seconds"])
                observed.update({"case_id": row["case_id"], "version": version,
                                 "requested_rules": row["rules"],
                                 "min_frequency": row["min_frequency"],
                                 "outer_call_seconds": outer_seconds,
                                 "binary_sha256": sha(binary),
                                 "cpu_affinity": budgets[workers], "cpu_budget": workers,
                                 "command": command, "full_trace_match": True})
                output.write(json.dumps(observed) + "\n")
                output.flush()
                print(version, row["case_id"], workers,
                      f"{observed['call_seconds']:.6f}s",
                      f"{observed['train_vm_hwm_mib']:.2f}MiB", flush=True)
    print(f"wrote {len(jobs)} verified rows")


if __name__ == "__main__":
    main()

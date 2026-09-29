"""One-run 256 KiB exact screen under the same fixed process CPU budget."""

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
    "combo_lazy": (RUST / "target/reruns/radical-layout-combo-v1/combo", []),
    "table_hash": (RUST / "target/reruns/radical-planning-integrated-v1/owned_selected_table",
                   ["--selected-lookup", "hash"]),
    "table_flat": (RUST / "target/reruns/radical-planning-integrated-v1/owned_selected_table",
                   ["--selected-lookup", "flat"]),
    "cache_off": (RUST / "target/reruns/radical-planning-integrated-v1/owned_route_cache",
                  ["--route-cache-slots", "0"]),
    "cache_4096": (RUST / "target/reruns/radical-planning-integrated-v1/owned_route_cache",
                   ["--route-cache-slots", "4096"]),
    "integrated_control": (RUST / "target/reruns/radical-planning-integrated-v1/owned_fused_direct",
                           ["--commit-mode", "separate", "--reduce-mode", "combined"]),
    "integrated_candidate": (RUST / "target/reruns/radical-planning-integrated-v1/owned_fused_direct",
                             ["--commit-mode", "owner-fused", "--reduce-mode", "direct-old"]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    process = subprocess.run(command, text=True, capture_output=True, check=True)
    return json.loads(process.stdout.strip().splitlines()[-1])


def main():
    rows = json.loads(MANIFEST.read_text())
    initial_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    if not all(set(cpus) <= initial_affinity for cpus in budgets.values()):
        raise RuntimeError(("required CPU affinity unavailable", sorted(initial_affinity)))
    for binary, _ in BINARIES.values():
        if not binary.is_file():
            raise FileNotFoundError(binary)
    if not NATIVE.is_file():
        raise FileNotFoundError(NATIVE)
    source_hashes = json.loads((OUT / "new-source-hashes.json").read_text())
    for name, digest in source_hashes.items():
        if sha(ROOT / name) != digest:
            raise AssertionError((name, "source changed after freeze"))
    jobs = [(version, row, workers) for row in rows
            for version in BINARIES for workers in (1, 4)]
    random.Random(20260930).shuffle(jobs)
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST),
        "native_reference_binary_sha256": sha(NATIVE),
        "binary_sha256": {name: sha(binary) for name, (binary, _) in BINARIES.items()},
        "new_source_sha256": source_hashes,
        "new_source_snapshot": "rust/batch_results/radical-planning-integrated-v1/new-source-snapshot.tar.gz",
        "combo_source_snapshot": "rust/batch_results/radical-layout-combo-v1/new-source-snapshot.tar.gz",
        "initial_affinity": sorted(initial_affinity), "cpu_budget": budgets,
        "seed": 20260930, "repeats": 1,
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds (CLOCK_PROCESS_CPUTIME_ID)",
        "mean_occupied_cores": "call_cpu_seconds / call_seconds; not useful-computation utilization",
        "fingerprint": "complete rule trace plus final token sequence",
    }
    with tempfile.TemporaryDirectory(prefix="radical-planning-quick-") as temp_name:
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
                binary, extra_args = BINARIES[version]
                fixture = RUST / row["file"]
                trace = temp / f"{version}-{row['case_id']}-{workers}.json"
                command = [str(binary), "--input", str(fixture), "--workers",
                           str(workers), "--chunk-size", "4096", "--rules",
                           str(row["rules"]), "--min-frequency",
                           str(row["min_frequency"]), "--heap-policy", "lazy",
                           "--trace", str(trace), *extra_args]
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
                if version != "combo_lazy" and "call_cpu_seconds" not in observed:
                    raise AssertionError((version, "process CPU metric missing"))
                if observed.get("heap_policy") != "lazy":
                    raise AssertionError((version, "heap policy mismatch"))
                if version != "combo_lazy":
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

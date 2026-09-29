"""Authorized 22-call fixed-budget serial integer-hash quick screen."""

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
SERIAL = RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash"
OWNER = RUST / "target/reruns/radical-local-hash-v1/owned_integer_hash"
NATIVE = RUST / "target/reruns/radical-fixed-budget-v1/ablation"
SERIAL_MODES = (
    ("cf32_std_checked", "combined_filtered", "std", "checked"),
    ("cf32_std_unchecked", "combined_filtered", "std", "unchecked"),
    ("cf32_ahash_checked", "combined_filtered", "ahash", "checked"),
    ("cf32_ahash_unchecked", "combined_filtered", "ahash", "unchecked"),
    ("cf16_std_checked", "combined_filtered_halfword", "std", "checked"),
    ("cf16_std_unchecked", "combined_filtered_halfword", "std", "unchecked"),
    ("cf16_ahash_checked", "combined_filtered_halfword", "ahash", "checked"),
    ("cf16_ahash_unchecked", "combined_filtered_halfword", "ahash", "unchecked"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_for(row, mode, workers, trace):
    fixture = RUST / row["file"]
    common = ["--input", str(fixture), "--workers", str(workers),
              "--rules", str(row["rules"]), "--min-frequency",
              str(row["min_frequency"]), "--trace", str(trace)]
    if mode == "native_direct":
        backend = ("combined_filtered_halfword" if row["case_id"].startswith("quick-en-")
                   else "combined_filtered")
        return NATIVE, [str(NATIVE), *common, "--variant", backend, "--bounds", "checked"]
    if mode == "owner_ahash":
        return OWNER, [str(OWNER), *common, "--chunk-size", "4096",
                       "--heap-policy", "lazy", "--integer-hash", "ahash"]
    for name, backend, integer_hash, bounds in SERIAL_MODES:
        if name == mode:
            return SERIAL, [str(SERIAL), *common, "--backend", backend,
                            "--integer-hash", integer_hash, "--bounds", bounds]
    raise ValueError(mode)


def main():
    fixtures = json.loads(MANIFEST.read_text())
    assert len(fixtures) == 2
    assert {row["case_id"] for row in fixtures} == {
        "quick-en-continuous-262144", "quick-zh-continuous-262144"}
    previous_serial = json.loads((RUST / "batch_results/radical-serial-integer-gate-v1/checks.json").read_text())
    previous_owner = json.loads((RUST / "batch_results/radical-local-hash-v1/checks.json").read_text())
    assert previous_serial["status"] == previous_owner["status"] == "passed"
    hashes = {"serial": sha(SERIAL), "owner": sha(OWNER), "native": sha(NATIVE)}
    assert hashes["serial"] == previous_serial["binary_sha256"]
    assert hashes["owner"] == previous_owner["binary_sha256"]["integer_ahash"]
    assert hashes["native"] == previous_owner["native_reference_binary_sha256"]
    for row in fixtures:
        assert sha(RUST / row["file"]) == row["fixture_sha256"]
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())
    jobs = [(row, mode, 1) for row in fixtures for mode, *_ in SERIAL_MODES]
    jobs.extend((row, "owner_ahash", workers) for row in fixtures for workers in (1, 4))
    jobs.extend((row, "native_direct", 1) for row in fixtures)
    assert len(jobs) == 22
    random.Random(20260930).shuffle(jobs)
    observations = []
    with tempfile.TemporaryDirectory(prefix="radical-serial-integer-quick-") as temp_name:
        temp = Path(temp_name)
        for index, (row, mode, workers) in enumerate(jobs, 1):
            trace = temp / f"trace-{index}.json"
            binary, command = command_for(row, mode, workers, trace)
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            assert observed["fixture_sha256"] == row["fixture_sha256"]
            for field in ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"):
                assert observed[field] > 0, (mode, field)
            observed.update({
                "case_id": row["case_id"], "version": mode, "workers": workers,
                "requested_rules": row["rules"], "min_frequency": row["min_frequency"],
                "outer_call_seconds": outer_seconds,
                "binary_sha256": sha(binary), "cpu_affinity": budgets[workers],
                "cpu_budget": workers,
                "mean_occupied_cores": observed["call_cpu_seconds"] / observed["call_seconds"],
                "command": command,
            })
            observations.append((observed, json.loads(trace.read_text())))
            print(f"{index}/22", row["case_id"], mode, workers,
                  f"{observed['call_seconds']:.6f}s", flush=True)
        references = {observed["case_id"]: (observed, trace)
                      for observed, trace in observations if observed["version"] == "native_direct"}
        assert len(references) == 2
        for observed, trace in observations:
            reference, expected_trace = references[observed["case_id"]]
            assert trace == expected_trace, (observed["case_id"], observed["version"], "trace")
            assert observed["fingerprint"] == reference["fingerprint"], (
                observed["case_id"], observed["version"], "fingerprint")
            observed["full_trace_match"] = True
    sidecar = {
        "git_head_during_measurement": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST),
        "binary_sha256": hashes, "source_archives": {
            "serial": "rust/batch_results/radical-serial-integer-gate-v1/new-source-snapshot.tar.gz",
            "owner": "rust/batch_results/radical-local-hash-v1/new-source-snapshot.tar.gz",
            "native": "rust/batch_results/radical-fixed-budget-v1",
        },
        "cpu_budget": budgets, "initial_affinity": sorted(original_affinity),
        "seed": 20260930, "repeats": 1, "rows": len(observations),
        "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds (CLOCK_PROCESS_CPUTIME_ID)",
        "semantic_check": "full merge-rule and final-token trace plus SHA-256 fingerprint",
    }
    with (OUT / "screen.jsonl").open("x") as output:
        for observed, _ in observations:
            output.write(json.dumps(observed) + "\n")
    with (OUT / "screen.jsonl.environment.json").open("x") as output:
        json.dump(sidecar, output, indent=2)
        output.write("\n")
    print("verified 22 exact training traces")


if __name__ == "__main__":
    main()

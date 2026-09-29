"""Authorized 16 measured calls for pending owner entries and a serial control."""

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
PENDING = RUST / "target/reruns/radical-pending-entry-gate-v1/radical-owned-pending-entry"
SERIAL = RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash"
MODES = ("staged", "fused-direct-combined", "fused-direct-pending")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_for(row, mode, workers, minimum, trace):
    common = ["--input", str(RUST / row["file"]), "--workers", str(workers),
              "--rules", str(row["rules"]), "--min-frequency", str(minimum),
              "--trace", str(trace)]
    if mode == "serial_cf32":
        return SERIAL, [str(SERIAL), *common, "--backend", "combined_filtered",
                        "--integer-hash", "ahash", "--bounds", "checked"]
    return PENDING, [str(PENDING), *common, "--chunk-size", "4096",
                     "--heap-policy", "lazy", "--integer-hash", "ahash",
                     "--owner-commit", mode]


def invoke(command):
    process = subprocess.run(command, text=True, capture_output=True, check=True)
    return json.loads(process.stdout.strip().splitlines()[-1])


def main():
    oracle = json.loads((RUST / "batch_results/radical-pending-entry-gate-v1/differential.json").read_text())
    assert oracle["status"] == "passed" and oracle["standard_fulltrace_matches"] == 240
    assert oracle["directed_fulltrace_matches"] == 24
    serial_checks = json.loads((RUST / "batch_results/radical-serial-integer-gate-v1/checks.json").read_text())
    hashes = {"pending": sha(PENDING), "serial": sha(SERIAL)}
    assert hashes["pending"] == oracle["binary_sha256"]
    assert hashes["serial"] == serial_checks["binary_sha256"]
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    assert all(sha(ROOT / path) == digest for path, digest in sources.items())
    fixtures = json.loads(MANIFEST.read_text())
    assert {row["case_id"] for row in fixtures} == {
        "quick-en-continuous-262144", "quick-zh-continuous-262144"}
    assert all(sha(RUST / row["file"]) == row["fixture_sha256"] for row in fixtures)
    english = next(row for row in fixtures if row["case_id"].startswith("quick-en-"))
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())

    # A separate, untimed correctness reference is necessary for the min=16
    # variant; it is not one of the 16 measured calls below.
    with tempfile.TemporaryDirectory(prefix="pending-entry-screen-") as temp_name:
        temp = Path(temp_name)
        high_trace = temp / "serial-high-min.json"
        _, high_command = command_for(english, "serial_cf32", 1, 16, high_trace)
        try:
            os.sched_setaffinity(0, budgets[1])
            high_reference = invoke(high_command)
        finally:
            os.sched_setaffinity(0, original_affinity)
        assert high_reference["fixture_sha256"] == english["fixture_sha256"]
        high_trace_value = json.loads(high_trace.read_text())
        high_oracle = {"case_id": english["case_id"], "minimum": 16,
                       "unmeasured_correctness_call": 1, "command": high_command,
                       "binary_sha256": hashes["serial"],
                       "fingerprint": high_reference["fingerprint"],
                       "trace_sha256": sha(high_trace),
                       "rules": high_reference["rules"]}
        with (OUT / "high-min-serial-oracle.json").open("x") as output:
            json.dump(high_oracle, output, indent=2)
            output.write("\n")

        jobs = [(row, mode, workers, row["min_frequency"])
                for row in fixtures for mode in MODES for workers in (1, 4)]
        jobs.extend((english, mode, 4, 16)
                    for mode in ("fused-direct-combined", "fused-direct-pending"))
        jobs.extend((row, "serial_cf32", 1, row["min_frequency"]) for row in fixtures)
        assert len(jobs) == 16
        random.Random(20260930).shuffle(jobs)
        observations = []
        for index, (row, mode, workers, minimum) in enumerate(jobs, 1):
            trace = temp / f"trace-{index}.json"
            binary, command = command_for(row, mode, workers, minimum, trace)
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                result = invoke(command)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            assert result["fixture_sha256"] == row["fixture_sha256"]
            assert all(result[field] > 0 for field in
                       ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"))
            if mode != "serial_cf32":
                assert result["owner_commit"] == mode
            result.update({"case_id": row["case_id"], "version": mode,
                           "requested_rules": row["rules"], "min_frequency": minimum,
                           "outer_call_seconds": outer_seconds,
                           "binary_sha256": sha(binary), "cpu_affinity": budgets[workers],
                           "cpu_budget": workers,
                           "mean_occupied_cores": result["call_cpu_seconds"] / result["call_seconds"],
                           "command": command})
            observations.append((result, json.loads(trace.read_text())))
            print(f"{index}/16", row["case_id"], mode, workers, f"min={minimum}",
                  f"{result['call_seconds']:.6f}s", flush=True)
        references = {(result["case_id"], result["min_frequency"]):
                      (result["fingerprint"], trace)
                      for result, trace in observations if result["version"] == "serial_cf32"}
        assert len(references) == 2
        references[(english["case_id"], 16)] = (high_reference["fingerprint"], high_trace_value)
        for result, trace in observations:
            fingerprint, expected = references[(result["case_id"], result["min_frequency"])]
            assert result["fingerprint"] == fingerprint and trace == expected
            result["full_trace_match"] = True
    sidecar = {"status": "passed", "rows": 16, "repetitions": 1, "seed": 20260930,
               "high_min_untimed_serial_oracle_calls": 1,
               "high_min_oracle_sha256": sha(OUT / "high-min-serial-oracle.json"),
               "manifest_sha256": sha(MANIFEST), "binary_sha256": hashes,
               "new_source_sha256": sources,
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
    print("verified sixteen measured exact traces and one untimed high-min serial oracle")


if __name__ == "__main__":
    main()

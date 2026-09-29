"""Authorized 30-call fixed-CPU endpoint and AA-bitmap mechanism screen."""

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
BINARIES = {
    "fused": RUST / "target/reruns/radical-fused-endpoint-gate-v1/radical-owned-fused-endpoint",
    "bitmap": RUST / "target/reruns/radical-aa-bitmap-gate-v1/radical-owned-aa-bitmap",
    "owner": RUST / "target/reruns/radical-local-hash-v1/owned_integer_hash",
    "serial": RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash",
}
MANIFESTS = (RUST / "batch_results/quick-fixtures-262144-512.json",
             RUST / "ablation_results/fixtures.json")
QUICK = {"quick-en-continuous-262144", "quick-zh-continuous-262144"}
AA = {"single-run-a-65536", "single-piece-ab-65536", "quick-en-continuous-262144"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_command(kind, mode, workers, row, trace):
    basic = ["--input", str(RUST / row["file"]), "--workers", str(workers),
             "--rules", str(row["rules"]), "--min-frequency", str(row["min_frequency"]),
             "--trace", str(trace)]
    if kind == "fused":
        return [str(BINARIES[kind]), *basic, "--chunk-size", "4096",
                "--heap-policy", "lazy", "--integer-hash", "ahash",
                "--endpoint-plan", mode]
    if kind == "bitmap":
        return [str(BINARIES[kind]), *basic, "--chunk-size", "4096",
                "--heap-policy", "lazy", "--integer-hash", "ahash", "--aa-order", mode]
    if kind == "owner":
        return [str(BINARIES[kind]), *basic, "--chunk-size", "4096",
                "--heap-policy", "lazy", "--integer-hash", "ahash"]
    if kind == "serial":
        return [str(BINARIES[kind]), *basic, "--backend", "combined_filtered",
                "--bounds", "checked", "--integer-hash", "ahash"]
    raise ValueError(kind)


def main():
    fused_oracle = json.loads((RUST / "batch_results/radical-fused-endpoint-gate-v1/differential.json").read_text())
    bitmap_oracle = json.loads((RUST / "batch_results/radical-aa-bitmap-gate-v1/differential.json").read_text())
    assert fused_oracle["status"] == bitmap_oracle["status"] == "passed"
    assert fused_oracle["standard_fulltrace_matches"] == 240
    assert bitmap_oracle["standard_fulltrace_matches"] == 160
    assert sha(BINARIES["fused"]) == fused_oracle["binary_sha256"]
    assert sha(BINARIES["bitmap"]) == bitmap_oracle["binary_sha256"]
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, name
    manifests = {row["case_id"]: row for manifest in MANIFESTS
                 for row in json.loads(manifest.read_text()) if row["case_id"] in QUICK | AA}
    assert set(manifests) == QUICK | AA
    for row in manifests.values():
        assert sha(RUST / row["file"]) == row["fixture_sha256"]

    jobs = [(manifests[case], "fused", mode, workers) for case in sorted(QUICK)
            for mode in ("two-pass", "tagged-two-pass", "tagged-fused")
            for workers in (1, 4)]
    jobs.extend((manifests[case], "owner", "owner-ahash", workers)
                for case in sorted(QUICK) for workers in (1, 4))
    jobs.extend((manifests[case], "serial", "cf32-ahash-checked", 1)
                for case in sorted(QUICK))
    jobs.extend((manifests[case], "bitmap", mode, workers)
                for case in sorted(AA) for mode in ("sort", "bitmap-adaptive")
                for workers in (1, 4))
    assert len(jobs) == 30
    random.Random(20260930).shuffle(jobs)
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())
    observations = []
    with tempfile.TemporaryDirectory(prefix="radical-fused-bitmap-quick-") as temp_name:
        temp = Path(temp_name)
        for index, (row, kind, mode, workers) in enumerate(jobs, 1):
            trace = temp / f"trace-{index}.json"
            command = build_command(kind, mode, workers, row, trace)
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            result = json.loads(process.stdout.strip().splitlines()[-1])
            assert result["fixture_sha256"] == row["fixture_sha256"]
            assert all(result[name] > 0 for name in
                       ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"))
            if kind == "fused":
                assert result["endpoint_plan_requested"] == mode
                assert result["endpoint_plan_effective"] == mode
            if kind == "bitmap":
                assert result["aa_order"] == mode
            result.update({"case_id": row["case_id"], "kind": kind,
                           "version": mode, "requested_rules": row["rules"],
                           "min_frequency": row["min_frequency"],
                           "outer_call_seconds": outer_seconds,
                           "binary_sha256": sha(BINARIES[kind]),
                           "cpu_affinity": budgets[workers], "cpu_budget": workers,
                           "mean_occupied_cores": result["call_cpu_seconds"] / result["call_seconds"],
                           "command": command})
            observations.append((result, json.loads(trace.read_text())))
            print(f"{index}/30", row["case_id"], kind, mode, workers,
                  f"{result['call_seconds']:.6f}s", flush=True)
        reference = {}
        for result, trace in observations:
            reference.setdefault(result["case_id"], (result["fingerprint"], trace))
        for result, trace in observations:
            fingerprint, expected = reference[result["case_id"]]
            assert result["fingerprint"] == fingerprint
            assert trace == expected, (result["case_id"], result["kind"], result["version"])
            result["full_trace_match"] = True

    sidecar = {
        "status": "passed", "rows": 30, "repetitions": 1, "seed": 20260930,
        "manifest_sha256": {str(path.relative_to(ROOT)): sha(path) for path in MANIFESTS},
        "binary_sha256": {kind: sha(path) for kind, path in BINARIES.items()},
        "new_source_sha256": sources,
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "shared_source_provenance_sha256": sha(OUT / "shared-source-provenance.json"),
        "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds",
        "semantic_check": "full merge-rule and final-token trace plus SHA-256 fingerprint",
    }
    with (OUT / "quick.jsonl").open("x") as output:
        for result, _ in observations:
            output.write(json.dumps(result) + "\n")
    with (OUT / "quick.jsonl.environment.json").open("x") as output:
        json.dump(sidecar, output, indent=2)
        output.write("\n")
    print("verified 30 exact training traces")


if __name__ == "__main__":
    main()

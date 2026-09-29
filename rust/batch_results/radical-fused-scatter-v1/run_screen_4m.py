"""Authorized 4 MiB n=1 diagnostic of only the seven new exact modes."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
MANIFEST = RUST / "ablation_results/fixtures.json"
BINARIES = {
    "owned_fused": RUST / "target/reruns/radical-fused-scatter-v1/owned_fused",
    "owned_scatter": RUST / "target/reruns/radical-fused-scatter-v1/owned_scatter",
    "owned_direct": RUST / "target/reruns/radical-fused-scatter-v1/owned_direct",
}
CONFIGS = {
    "fused_separate": ("owned_fused", ["--commit-mode", "separate"]),
    "fused_owner": ("owned_fused", ["--commit-mode", "owner-fused"]),
    "fused_overlap": ("owned_fused", ["--commit-mode", "overlap"]),
    "scatter_owner": ("owned_scatter", ["--fill-mode", "owner"]),
    "scatter_scatter": ("owned_scatter", ["--fill-mode", "scatter"]),
    "direct_combined": ("owned_direct", ["--reduce-mode", "combined"]),
    "direct_old": ("owned_direct", ["--reduce-mode", "direct-old"]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_fingerprints():
    old = RUST / "batch_results/radical-layout-combo-v1/screen.jsonl"
    reference = {}
    for line in old.read_text().splitlines():
        row = json.loads(line)
        prior = reference.setdefault(row["case_id"], row["fingerprint"])
        if prior != row["fingerprint"]:
            raise AssertionError((row["case_id"], "old fingerprint disagreement"))
    return reference


def main():
    rows = [row for row in json.loads(MANIFEST.read_text())
            if row["case_id"] in ("en-4m-continuous", "zh-4m-continuous")]
    if len(rows) != 2:
        raise AssertionError("missing 4 MiB continuous fixtures")
    expected = expected_fingerprints()
    source_hashes = json.loads((OUT / "new-source-hashes.json").read_text())
    for name, digest in source_hashes.items():
        if sha(ROOT / name) != digest:
            raise AssertionError((name, "source changed after freeze"))
    quick_env = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    binaries_hashes = {name: sha(path) for name, path in BINARIES.items()}
    for name, digest in binaries_hashes.items():
        if digest != quick_env["binary_sha256"][next(
                version for version in CONFIGS if CONFIGS[version][0] == name)]:
            raise AssertionError((name, "binary changed after quick"))
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    if not all(set(cpus) <= original_affinity for cpus in budgets.values()):
        raise RuntimeError(("required CPU affinity unavailable", sorted(original_affinity)))
    jobs = [(row, version, workers) for row in rows
            for version in CONFIGS for workers in (1, 4)]
    random.Random(20260930).shuffle(jobs)
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha(MANIFEST),
        "new_source_sha256": source_hashes,
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "binary_sha256": binaries_hashes,
        "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
        "seed": 20260930, "repeats": 1, "rows": len(jobs),
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds (CLOCK_PROCESS_CPUTIME_ID)",
        "semantic_check": "prior full-training SHA-256 fingerprint of every rule and final tokens",
        "old_fingerprint_source": "rust/batch_results/radical-layout-combo-v1/screen.jsonl",
    }
    with (OUT / "screen-4m.jsonl").open("x") as output, \
            (OUT / "screen-4m.jsonl.environment.json").open("x") as meta:
        meta.write(json.dumps(sidecar, indent=2) + "\n")
        for index, (row, version, workers) in enumerate(jobs, 1):
            fixture = RUST / row["file"]
            if sha(fixture) != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], "fixture SHA mismatch"))
            binary_name, extra = CONFIGS[version]
            binary = BINARIES[binary_name]
            command = [str(binary), "--input", str(fixture), "--workers", str(workers),
                       "--chunk-size", "4096", "--rules", str(row["rules"]),
                       "--min-frequency", str(row["min_frequency"]),
                       "--heap-policy", "lazy", *extra]
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            if observed["fixture_sha256"] != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], version, "fixture mismatch"))
            if observed["fingerprint"] != expected[row["case_id"]]:
                raise AssertionError((row["case_id"], version, "fingerprint mismatch"))
            for metric in ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"):
                if metric not in observed:
                    raise AssertionError((version, metric, "missing"))
            observed.update({
                "case_id": row["case_id"], "version": version,
                "requested_rules": row["rules"], "min_frequency": row["min_frequency"],
                "outer_call_seconds": outer_seconds, "binary_sha256": binaries_hashes[binary_name],
                "cpu_affinity": budgets[workers], "cpu_budget": workers,
                "mean_occupied_cores": observed["call_cpu_seconds"] / observed["call_seconds"],
                "command": command, "full_training_fingerprint_match": True,
            })
            output.write(json.dumps(observed) + "\n")
            output.flush()
            print(f"{index}/{len(jobs)}", row["case_id"], version, workers,
                  f"{observed['call_seconds']:.4f}s",
                  f"{observed['mean_occupied_cores']:.2f} cores",
                  f"{observed['train_vm_hwm_mib']:.2f}MiB", flush=True)
    print(f"verified {len(jobs)} exact training fingerprints")


if __name__ == "__main__":
    main()

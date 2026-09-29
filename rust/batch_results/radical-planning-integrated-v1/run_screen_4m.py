"""Authorized two-repeat 4 MiB screen of the integrated candidate and controls."""

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
NATIVE = RUST / "target/reruns/radical-fixed-budget-v1/ablation"
COMBO = RUST / "target/reruns/radical-layout-combo-v1/combo"
INTEGRATED = RUST / "target/reruns/radical-planning-integrated-v1/owned_fused_direct"
CONFIGS = (("combo_lazy", 1), ("combo_lazy", 4),
           ("integrated_control", 1), ("integrated_control", 4),
           ("integrated_candidate", 1), ("integrated_candidate", 4),
           ("best_direct_scalar", 1))


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


def command_for(row, version, workers):
    fixture = RUST / row["file"]
    common = ["--input", str(fixture), "--workers", str(workers),
              "--rules", str(row["rules"]),
              "--min-frequency", str(row["min_frequency"])]
    if version == "best_direct_scalar":
        native_variant = ("combined_filtered_halfword" if row["case_id"].startswith("en-")
                          else "combined_filtered")
        return NATIVE, [str(NATIVE), *common, "--variant", native_variant,
                        "--bounds", "checked"]
    if version == "combo_lazy":
        return COMBO, [str(COMBO), *common, "--chunk-size", "4096",
                       "--heap-policy", "lazy"]
    extra = (["--commit-mode", "separate", "--reduce-mode", "combined"]
             if version == "integrated_control" else
             ["--commit-mode", "owner-fused", "--reduce-mode", "direct-old"])
    return INTEGRATED, [str(INTEGRATED), *common, "--chunk-size", "4096",
                        "--heap-policy", "lazy", *extra]


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
    binary_hashes = {"native": sha(NATIVE), "combo": sha(COMBO),
                     "integrated": sha(INTEGRATED)}
    if binary_hashes["integrated"] != quick_env["binary_sha256"]["integrated_candidate"]:
        raise AssertionError("integrated binary changed after quick")
    if binary_hashes["combo"] != quick_env["binary_sha256"]["combo_lazy"]:
        raise AssertionError("combo binary changed after quick")
    if binary_hashes["native"] != quick_env["native_reference_binary_sha256"]:
        raise AssertionError("native binary changed after quick")
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    if not all(set(cpus) <= original_affinity for cpus in budgets.values()):
        raise RuntimeError(("required CPU affinity unavailable", sorted(original_affinity)))
    first = [(row, version, workers) for row in rows for version, workers in CONFIGS]
    random.Random(20260930).shuffle(first)
    jobs = [(0, *job) for job in first]
    jobs.extend((1, *job) for job in reversed(first))
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST),
        "binary_sha256": binary_hashes, "new_source_sha256": source_hashes,
        "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
        "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
        "seed": 20260930, "repeats": 2,
        "schedule": "seeded shuffle then exact reverse",
        "training_memory_metric": "train_vm_hwm_mib",
        "process_cpu_metric": "call_cpu_seconds for integrated binary; frozen combo lacks it",
        "semantic_check": "prior full-training SHA-256 fingerprint of every rule and final tokens",
    }
    with (OUT / "screen-4m.jsonl").open("x") as output, \
            (OUT / "screen-4m.jsonl.environment.json").open("x") as meta:
        meta.write(json.dumps(sidecar, indent=2) + "\n")
        for index, (repeat, row, version, workers) in enumerate(jobs, 1):
            fixture = RUST / row["file"]
            if sha(fixture) != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], "fixture SHA mismatch"))
            binary, command = command_for(row, version, workers)
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            if observed["fixture_sha256"] != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], version, repeat, "fixture mismatch"))
            if observed["fingerprint"] != expected[row["case_id"]]:
                raise AssertionError((row["case_id"], version, repeat, "fingerprint mismatch"))
            if "train_vm_hwm_mib" not in observed:
                raise AssertionError((version, "training VmHWM missing"))
            if version.startswith("integrated_") and "call_cpu_seconds" not in observed:
                raise AssertionError((version, "process CPU metric missing"))
            observed.update({
                "case_id": row["case_id"], "version": version, "repetition": repeat,
                "requested_rules": row["rules"], "min_frequency": row["min_frequency"],
                "outer_call_seconds": outer_seconds,
                "binary_sha256": binary_hashes["native" if binary == NATIVE else
                                               "combo" if binary == COMBO else "integrated"],
                "cpu_affinity": budgets[workers], "cpu_budget": workers,
                "mean_occupied_cores": (
                    observed["call_cpu_seconds"] / observed["call_seconds"]
                    if version.startswith("integrated_") else None),
                "command": command, "full_training_fingerprint_match": True,
            })
            output.write(json.dumps(observed) + "\n")
            output.flush()
            print(f"{index}/{len(jobs)}", row["case_id"], version, workers, repeat,
                  f"{observed['call_seconds']:.4f}s",
                  f"{observed['train_vm_hwm_mib']:.2f}MiB", flush=True)
    print(f"verified {len(jobs)} exact training fingerprints")


if __name__ == "__main__":
    main()

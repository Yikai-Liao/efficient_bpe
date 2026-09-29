"""Single-run 4 MiB W4 screen of frozen grouped and owner-shard binaries."""

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
    "owned_lazy": RUST / "target/reruns/radical-owner-boxed-v1/owned",
    "counts_lazy": RUST / "target/reruns/radical-owned-extensions-v1/owned_counts",
    "grouped_lazy": RUST / "target/reruns/radical-owner-routing-v1/grouped",
    "shards_s4": RUST / "target/reruns/radical-owner-routing-v1/shards",
    "shards_s8": RUST / "target/reruns/radical-owner-routing-v1/shards",
    "shards_s16": RUST / "target/reruns/radical-owner-routing-v1/shards",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_fingerprints():
    prior = RUST / "batch_results/radical-owned-4m-n2-v1/screen.jsonl"
    rows = [json.loads(line) for line in prior.read_text().splitlines()]
    fingerprints = {}
    for row in rows:
        old = fingerprints.setdefault(row["case_id"], row["fingerprint"])
        if old != row["fingerprint"]:
            raise AssertionError((row["case_id"], "prior fingerprints disagree"))
    return fingerprints


def main():
    cases = [row for row in json.loads(MANIFEST.read_text())
             if row["case_id"] in ("en-4m-continuous", "zh-4m-continuous")]
    if len(cases) != 2:
        raise ValueError("both continuous 4 MiB cases are required")
    expected = prior_fingerprints()
    old_checks = json.loads((RUST / "batch_results/radical-owner-boxed-v1/checks.json").read_text())
    counts_checks = json.loads((RUST / "batch_results/radical-owned-extensions-v1/checks.json").read_text())
    expected_hashes = {
        "owned_lazy": old_checks["binary_sha256"]["owned_lazy"],
        "counts_lazy": counts_checks["binary_sha256"]["counts_lazy"],
        "grouped_lazy": (OUT / "grouped.sha256").read_text().split()[0],
        **{f"shards_s{s}": (OUT / "shards.sha256").read_text().split()[0]
           for s in (4, 8, 16)},
    }
    if {name: digest(binary) for name, binary in BINARIES.items()} != expected_hashes:
        raise RuntimeError("a frozen binary hash changed")
    original_affinity = set(os.sched_getaffinity(0))
    budget = [0, 1, 2, 5]
    if not set(budget) <= original_affinity:
        raise RuntimeError("fixed four-CPU budget is unavailable")
    jobs = [(row, name) for row in cases for name in BINARIES]
    random.Random(20260930).shuffle(jobs)
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": digest(MANIFEST),
        "binary_sha256": expected_hashes,
        "source_snapshots": [
            "rust/batch_results/radical-owner-boxed-v1/native-source-snapshot.tar.gz",
            "rust/batch_results/radical-owned-extensions-v1/new-source-snapshot.tar.gz",
            "rust/batch_results/radical-owner-routing-v1/new-source-snapshot.tar.gz",
        ],
        "initial_affinity": sorted(original_affinity),
        "cpu_budget": budget, "seed": 20260930, "repeats": 1,
        "training_memory_metric": "train_vm_hwm_mib",
        "semantic_check": "SHA-256 of every rule and the final token sequence",
    }
    with (OUT / "smoke-4m.jsonl").open("x") as output, \
            (OUT / "smoke-4m.jsonl.environment.json").open("x") as meta:
        meta.write(json.dumps(sidecar, indent=2) + "\n")
        for index, (row, name) in enumerate(jobs, 1):
            fixture = RUST / row["file"]
            if digest(fixture) != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], "fixture hash mismatch"))
            binary = BINARIES[name]
            command = [str(binary), "--input", str(fixture), "--workers", "4",
                       "--chunk-size", "4096", "--rules", str(row["rules"]),
                       "--min-frequency", str(row["min_frequency"]),
                       "--heap-policy", "lazy"]
            if name.startswith("shards_"):
                command.extend(["--owner-shards", name.removeprefix("shards_s")])
            try:
                os.sched_setaffinity(0, budget)
                started = time.perf_counter()
                completed = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(completed.stdout.strip().splitlines()[-1])
            if observed["fixture_sha256"] != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], name, "fixture mismatch"))
            if observed["fingerprint"] != expected[row["case_id"]]:
                raise AssertionError((row["case_id"], name, "fingerprint mismatch"))
            if name.startswith("shards_") and observed["owner_shards"] != int(name.removeprefix("shards_s")):
                raise AssertionError((row["case_id"], name, "owner shard mismatch"))
            if "train_vm_hwm_mib" not in observed:
                raise AssertionError((name, "training-time VmHWM missing"))
            observed.update({"case_id": row["case_id"], "variant_label": name,
                             "requested_rules": row["rules"],
                             "min_frequency": row["min_frequency"],
                             "outer_call_seconds": outer_seconds,
                             "binary_sha256": expected_hashes[name],
                             "cpu_affinity": budget, "cpu_budget": 4,
                             "command": command, "full_training_fingerprint_match": True})
            output.write(json.dumps(observed) + "\n")
            output.flush()
            print(f"{index}/{len(jobs)}", row["case_id"], name,
                  f"{observed['call_seconds']:.4f}s",
                  f"{observed['train_vm_hwm_mib']:.2f}MiB", flush=True)
    print("verified 12 complete 4 MiB training fingerprints")


if __name__ == "__main__":
    main()

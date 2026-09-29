"""Interleaved, fixed-budget 4 MiB n=2 screen of frozen pair-owned binaries."""

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
OWNED = RUST / "target/reruns/radical-owner-boxed-v1/owned"
PROBE = RUST / "target/reruns/radical-owned-extensions-v1/owned_probe"
COUNTS = RUST / "target/reruns/radical-owned-extensions-v1/owned_counts"
BINARIES = {"native": NATIVE, "owned": OWNED, "probe": PROBE, "counts": COUNTS}
CONFIGS = [(label, workers) for label in
           ("owned_lazy", "probe_off", "probe_budgeted", "counts_lazy")
           for workers in (1, 4)]
CONFIGS.extend((("direct_scalar", 1), ("native_pair_owned", 4)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes():
    older = json.loads((RUST / "batch_results/radical-owner-boxed-v1/checks.json").read_text())
    newer = json.loads((RUST / "batch_results/radical-owned-extensions-v1/checks.json").read_text())
    return {"native": older["binary_sha256"]["native"],
            "owned": older["binary_sha256"]["owned_lazy"],
            "probe": newer["binary_sha256"]["probe_off"],
            "counts": newer["binary_sha256"]["counts_lazy"]}


def expected_fingerprints():
    rows = [json.loads(line) for line in
            (RUST / "batch_results/radical-owner-boxed-v1/smoke-4m.jsonl").read_text().splitlines()]
    by_case = {}
    for row in rows:
        old = by_case.setdefault(row["case_id"], row["fingerprint"])
        if old != row["fingerprint"]:
            raise AssertionError((row["case_id"], "prior fingerprint disagreement"))
    return by_case


def command_for(row, label, workers):
    fixture = RUST / row["file"]
    common = ["--input", str(fixture), "--workers", str(workers),
              "--rules", str(row["rules"]),
              "--min-frequency", str(row["min_frequency"])]
    if label in ("direct_scalar", "native_pair_owned"):
        variant = (("combined_filtered_halfword" if row["case_id"].startswith("en-")
                    else "combined_filtered") if label == "direct_scalar"
                   else "parallel_pair_owned")
        return NATIVE, [str(NATIVE), *common, "--variant", variant, "--bounds", "checked"]
    if label == "owned_lazy":
        binary, extra = OWNED, ["--heap-policy", "lazy"]
    elif label.startswith("probe_"):
        binary = PROBE
        extra = ["--heap-policy", "lazy", "--spatial-probe", label.removeprefix("probe_")]
    else:
        binary, extra = COUNTS, ["--heap-policy", "lazy"]
    return binary, [str(binary), *common, "--chunk-size", "4096", *extra]


def main():
    manifest_rows = json.loads(MANIFEST.read_text())
    cases = [row for row in manifest_rows
             if row["case_id"] in ("en-4m-continuous", "zh-4m-continuous")]
    if len(cases) != 2:
        raise ValueError("both continuous 4 MiB cases are required")
    expected = expected_fingerprints()
    hashes = frozen_hashes()
    if {kind: digest(path) for kind, path in BINARIES.items()} != hashes:
        raise RuntimeError("binary hash changed since frozen validation")
    original_affinity = set(os.sched_getaffinity(0))
    if 5 not in original_affinity or len(original_affinity) < 4:
        raise RuntimeError("CPU 5 and at least four permitted CPUs are required")
    cpu_order = [5, *sorted(original_affinity - {5})]
    budgets = {1: [5], 4: sorted(cpu_order[:4])}
    round_zero = [(row, label, workers) for row in cases for label, workers in CONFIGS]
    random.Random(20260930).shuffle(round_zero)
    jobs = [(0, *job) for job in round_zero]
    jobs.extend((1, *job) for job in reversed(round_zero))
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": digest(MANIFEST),
        "binary_sha256": hashes,
        "source_snapshots": [
            "rust/batch_results/radical-owner-boxed-v1/native-source-snapshot.tar.gz",
            "rust/batch_results/radical-owned-extensions-v1/new-source-snapshot.tar.gz",
        ],
        "initial_affinity": sorted(original_affinity),
        "cpu_budget": budgets, "repeats": 2, "seed": 20260930,
        "schedule": "first seeded shuffle, second round exact reverse",
        "training_memory_metric": "train_vm_hwm_mib",
        "semantic_check": "SHA-256 of every rule and the final token sequence",
    }
    results = []
    with (OUT / "screen.jsonl").open("x") as output, \
            (OUT / "screen.jsonl.environment.json").open("x") as meta:
        meta.write(json.dumps(sidecar, indent=2) + "\n")
        for index, (repeat, row, label, workers) in enumerate(jobs, 1):
            fixture = RUST / row["file"]
            if digest(fixture) != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], "fixture SHA mismatch"))
            binary, command = command_for(row, label, workers)
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                completed = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(completed.stdout.strip().splitlines()[-1])
            if observed["fingerprint"] != expected[row["case_id"]]:
                raise AssertionError((row["case_id"], label, repeat, "fingerprint mismatch"))
            if observed["fixture_sha256"] != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], label, repeat, "fixture mismatch"))
            if "train_vm_hwm_mib" not in observed:
                raise AssertionError((label, "training-time VmHWM missing"))
            observed.update({
                "case_id": row["case_id"], "variant_label": label,
                "repetition": repeat, "requested_rules": row["rules"],
                "min_frequency": row["min_frequency"],
                "outer_call_seconds": outer_seconds,
                "binary_sha256": hashes[next(kind for kind, path in BINARIES.items() if path == binary)],
                "cpu_affinity": budgets[workers], "cpu_budget": workers,
                "command": command, "full_training_fingerprint_match": True,
            })
            output.write(json.dumps(observed) + "\n")
            output.flush()
            results.append(observed)
            print(f"{index}/{len(jobs)}", row["case_id"], label, workers,
                  repeat, f"{observed['call_seconds']:.4f}s",
                  f"{observed['train_vm_hwm_mib']:.2f}MiB", flush=True)
    assert len(results) == 40
    for row in cases:
        for label, workers in CONFIGS:
            group = [item for item in results if item["case_id"] == row["case_id"]
                     and item["variant_label"] == label and item["workers"] == workers]
            if len(group) != 2 or {item["repetition"] for item in group} != {0, 1}:
                raise AssertionError((row["case_id"], label, workers, "missing repeat"))
    print("verified 40 calls, 20 configurations, two reverse-order rounds")


if __name__ == "__main__":
    main()

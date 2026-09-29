"""One-shot 4 MiB screen with one/four process CPU budgets and full fingerprints."""

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
VARIANTS = (("combined_filtered", 1), ("combined_filtered_halfword", 1),
            ("parallel_pair_owned", 1), ("parallel_pair_owned", 4),
            ("owned_lazy", 1), ("owned_lazy", 4))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cases = [row for row in json.loads(MANIFEST.read_text())
             if row["case_id"] in ("en-4m-continuous", "zh-4m-continuous")]
    if len(cases) != 2:
        raise ValueError("both continuous 4 MiB cases are required")
    original_affinity = set(os.sched_getaffinity(0))
    if 5 not in original_affinity or len(original_affinity) < 4:
        raise RuntimeError("expected CPU 5 and at least four permitted CPUs")
    order = [5, *sorted(original_affinity - {5})]
    budgets = {1: [5], 4: sorted(order[:4])}
    jobs = [(row, variant, workers) for row in cases
            for variant, workers in VARIANTS]
    random.Random(20260930).shuffle(jobs)
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": digest(MANIFEST),
        "binary_sha256": {"native": digest(NATIVE), "owned": digest(OWNED)},
        "initial_affinity": sorted(original_affinity),
        "cpu_budget": budgets, "seed": 20260930, "repeats": 1,
        "training_memory_metric": "train_vm_hwm_mib",
        "source_snapshot": "rust/batch_results/radical-owner-boxed-v1/native-source-snapshot.tar.gz",
        "source_snapshot_sha256": digest(OUT / "native-source-snapshot.tar.gz"),
    }
    observed_rows = []
    with (OUT / "smoke-4m.jsonl").open("x") as output, \
            (OUT / "smoke-4m.jsonl.environment.json").open("x") as meta:
        meta.write(json.dumps(sidecar, indent=2) + "\n")
        for row, variant, workers in jobs:
            fixture = RUST / row["file"]
            if digest(fixture) != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], "fixture SHA mismatch"))
            command = ([str(OWNED), "--input", str(fixture),
                        "--workers", str(workers), "--chunk-size", "4096",
                        "--rules", str(row["rules"]),
                        "--min-frequency", str(row["min_frequency"]),
                        "--heap-policy", "lazy"] if variant == "owned_lazy" else
                       [str(NATIVE), "--input", str(fixture), "--variant", variant,
                        "--workers", str(workers), "--rules", str(row["rules"]),
                        "--min-frequency", str(row["min_frequency"]),
                        "--bounds", "checked"])
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                completed = subprocess.run(command, text=True, capture_output=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            if result["fixture_sha256"] != row["fixture_sha256"]:
                raise AssertionError((row["case_id"], variant, "fixture mismatch"))
            if variant == "owned_lazy" and result.get("heap_policy") != "lazy":
                raise AssertionError("wrong owned heap policy")
            if "train_vm_hwm_mib" not in result:
                raise AssertionError("training-time VmHWM missing")
            result.update({"case_id": row["case_id"], "variant_label": variant,
                           "requested_rules": row["rules"], "min_frequency": row["min_frequency"],
                           "outer_call_seconds": outer_seconds,
                           "binary_sha256": digest(OWNED if variant == "owned_lazy" else NATIVE),
                           "cpu_affinity": budgets[workers], "cpu_budget": workers,
                           "command": command})
            observed_rows.append(result)
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(row["case_id"], variant, workers,
                  f"{result['call_seconds']:.6f}s",
                  f"{result['train_vm_hwm_mib']:.2f}MiB", flush=True)
    for row in cases:
        group = [item for item in observed_rows if item["case_id"] == row["case_id"]]
        if len(group) != len(VARIANTS) or len({item["fingerprint"] for item in group}) != 1:
            raise AssertionError((row["case_id"], "fingerprint mismatch or missing result"))
    print("verified", len(observed_rows), "4 MiB calls with two stable fingerprints")


if __name__ == "__main__":
    main()

"""One-shot exact-trace screen for the three frozen radical binaries.

Run from the repository root with the task Python environment. The script
uses prebuilt binaries and never invokes Cargo.
"""

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
NATIVE = RUST / "target/reruns/pair-owned-spatial-v1/ablation"
BINARIES = {v: RUST / f"target/reruns/radical-unified-v1/{v}" for v in ("v1", "v2", "v3")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    rows = json.loads(MANIFEST.read_text())
    native_rows = [json.loads(line) for line in
                   (RUST / "batch_results/pair-owned-spatial-v1/common-quick.jsonl").read_text().splitlines()]
    references = {row["case_id"]: row for row in native_rows
                  if row["variant"] == "combined_filtered"}
    original_affinity = sorted(os.sched_getaffinity(0))
    jobs = [(v, row, workers) for row in rows for v in BINARIES for workers in (1, 4)]
    random.Random(20260930).shuffle(jobs)
    output_path = OUT / "quick.jsonl"
    sidecar_path = OUT / "quick.jsonl.environment.json"
    native_hashes = json.loads((RUST / "batch_results/radical-unified-v1/native-source-hashes.json").read_text())
    sidecar = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha(MANIFEST),
        "native_reference_binary": str(NATIVE.relative_to(ROOT)),
        "native_reference_binary_sha256": sha(NATIVE),
        "radical_binary_sha256": {v: sha(b) for v, b in BINARIES.items()},
        "source_snapshot": "rust/batch_results/radical-unified-v1/native-source-snapshot.tar.gz",
        "native_source_sha256": native_hashes,
        "cpu_affinity": original_affinity,
        "seed": 20260930,
        "repeats": 1,
        "fingerprint": "complete rule trace plus final token sequence",
        "memory_metric": "vm_hwm_mib",
    }
    with tempfile.TemporaryDirectory(prefix="radical-quick-") as tmp:
        tmp = Path(tmp)
        reference_traces = {}
        for row in rows:
            fixture = RUST / row["file"]
            assert sha(fixture) == row["fixture_sha256"]
            trace = tmp / f"reference-{row['case_id']}.json"
            observed = run([str(NATIVE), "--input", str(fixture),
                            "--variant", "combined_filtered", "--workers", "1",
                            "--rules", str(row["rules"]),
                            "--min-frequency", str(row["min_frequency"]),
                            "--bounds", "checked", "--trace", str(trace)])
            assert observed["fingerprint"] == references[row["case_id"]]["fingerprint"]
            reference_traces[row["case_id"]] = json.loads(trace.read_text())
        with output_path.open("x") as output, sidecar_path.open("x") as meta:
            meta.write(json.dumps(sidecar, indent=2) + "\n")
            for version, row, workers in jobs:
                fixture = RUST / row["file"]
                trace = tmp / f"{version}-{row['case_id']}-{workers}.json"
                command = [str(BINARIES[version]), "--input", str(fixture),
                           "--workers", str(workers), "--chunk-size", "4096",
                           "--rules", str(row["rules"]),
                           "--min-frequency", str(row["min_frequency"]),
                           "--trace", str(trace)]
                started = time.perf_counter()
                observed = run(command)
                outer_seconds = time.perf_counter() - started
                if observed["fixture_sha256"] != row["fixture_sha256"]:
                    raise AssertionError((version, row["case_id"], "fixture SHA mismatch"))
                if observed["fingerprint"] != references[row["case_id"]]["fingerprint"]:
                    raise AssertionError((version, row["case_id"], workers, "fingerprint mismatch"))
                if json.loads(trace.read_text()) != reference_traces[row["case_id"]]:
                    raise AssertionError((version, row["case_id"], workers, "full trace mismatch"))
                observed.update({"case_id": row["case_id"], "version": version,
                                 "requested_rules": row["rules"],
                                 "min_frequency": row["min_frequency"],
                                 "outer_call_seconds": outer_seconds,
                                 "binary_sha256": sha(BINARIES[version]),
                                 "cpu_affinity": original_affinity,
                                 "command": command, "full_trace_match": True})
                output.write(json.dumps(observed) + "\n")
                output.flush()
                print(version, row["case_id"], workers,
                      f"{observed['call_seconds']:.6f}s",
                      f"{observed['vm_hwm_mib']:.2f}MiB", flush=True)
    print(f"wrote {len(jobs)} verified rows to {output_path}")


if __name__ == "__main__":
    main()

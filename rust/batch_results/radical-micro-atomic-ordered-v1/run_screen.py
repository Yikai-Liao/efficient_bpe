"""Authorized 56-call fixed-budget screen for three exact BPE mechanisms."""

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
QUICK = RUST / "batch_results/quick-fixtures-262144-512.json"
AB_MANIFEST = RUST / "ablation_results/fixtures.json"
BINS = {
    "micro": RUST / "target/reruns/radical-region-tasks-gate-v1/radical-owned-region-tasks",
    "atomic": RUST / "target/reruns/radical-atomic-old-gate-v1/radical-owned-atomic-old",
    "ordered": RUST / "target/reruns/radical-ordered-posting-gate-v1/radical-owned-ordered-posting",
    "serial": RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash",
}
GATES = {"micro": "radical-region-tasks-gate-v1",
         "atomic": "radical-atomic-old-gate-v1",
         "ordered": "radical-ordered-posting-gate-v1",
         "serial": "radical-serial-integer-gate-v1"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_for(job, row, trace):
    family, mode, workers, _ = job
    common = ["--input", str(RUST / row["file"]), "--workers", str(workers),
              "--rules", str(row["rules"]), "--min-frequency",
              str(row["min_frequency"]), "--trace", str(trace)]
    binary = BINS[family]
    if family == "serial":
        return [str(binary), *common, "--backend", "combined_filtered",
                "--integer-hash", "ahash", "--bounds", "checked"]
    common.extend(("--chunk-size", "4096", "--heap-policy", "lazy",
                   "--integer-hash", "ahash"))
    if family == "micro":
        region_mode, factor = mode
        return [str(binary), *common, "--endpoint-plan", "tagged-fused",
                "--region-mode", region_mode, "--regions-per-worker", str(factor)]
    if family == "atomic":
        return [str(binary), *common, "--old-reduce", mode]
    if family == "ordered":
        return [str(binary), *common, "--endpoint-plan", "tagged-fused",
                "--region-mode", "region", "--regions-per-worker", "1",
                "--posting-order", mode]
    raise ValueError(family)


def label(job):
    family, mode, workers, case_id = job
    if family == "micro":
        region_mode, factor = mode
        return f"micro_{region_mode}_k{factor}"
    if family == "atomic":
        return f"atomic_{mode}"
    if family == "ordered":
        return f"ordered_{mode}"
    return "serial_cf32_ahash_checked"


def main():
    hashes = {family: sha(binary) for family, binary in BINS.items()}
    for family, gate in GATES.items():
        checks = json.loads((RUST / "batch_results" / gate / "differential.json").read_text()) if family != "serial" else json.loads((RUST / "batch_results" / gate / "checks.json").read_text())
        assert checks["status"] == "passed" and checks["binary_sha256"] == hashes[family]
    assert json.loads((RUST / "batch_results/radical-region-tasks-gate-v1/differential.json").read_text())["k4_standard_fulltrace_matches"] == 80
    assert json.loads((RUST / "batch_results/radical-atomic-old-gate-v1/differential.json").read_text())["standard_fulltrace_matches"] == 80
    assert json.loads((RUST / "batch_results/radical-ordered-posting-gate-v1/differential.json").read_text())["standard_fulltrace_matches"] == 80
    source_hashes = json.loads((OUT / "new-source-hashes.json").read_text())
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items())
    fixtures = {row["case_id"]: row for row in json.loads(QUICK.read_text())}
    assert set(fixtures) == {"quick-en-continuous-262144", "quick-zh-continuous-262144"}
    ab = next(row for row in json.loads(AB_MANIFEST.read_text())
              if row["case_id"] == "single-piece-ab-65536")
    fixtures[ab["case_id"]] = ab
    assert all(sha(RUST / row["file"]) == row["fixture_sha256"] for row in fixtures.values())
    old_combo = [json.loads(line) for line in
                 (RUST / "batch_results/radical-endpoint-bitmap-combo-quick-v1/quick.jsonl").read_text().splitlines()]
    old_ab = {row["fingerprint"] for row in old_combo if row["case_id"] == ab["case_id"]}
    assert len(old_ab) == 1
    ab_prior_fingerprint = next(iter(old_ab))

    natural = sorted(case for case in fixtures if case != ab["case_id"])
    n2_jobs = []
    for case in natural:
        n2_jobs.extend(("micro", ("region", factor), workers, case)
                       for factor in (1, 4) for workers in (1, 4))
        n2_jobs.extend(("micro", ("snapshot", factor), 4, case)
                       for factor in (1, 4))
        n2_jobs.extend(("atomic", mode, workers, case)
                       for mode in ("owner", "producer-atomic") for workers in (1, 4))
        n2_jobs.append(("serial", None, 1, case))
    n2_jobs.extend(("ordered", mode, 4, ab["case_id"])
                   for mode in ("region", "global"))
    assert len(n2_jobs) == 24
    n1_jobs = [("ordered", mode, workers, case) for case in natural
               for mode in ("region", "global") for workers in (1, 4)]
    assert len(n1_jobs) == 8
    rng = random.Random(20260930)
    first_n2 = n2_jobs.copy()
    rng.shuffle(first_n2)
    first = first_n2 + n1_jobs
    rng.shuffle(first)
    sequence = [(job, 1 if job in n2_jobs else 0) for job in first]
    sequence.extend((job, 2) for job in reversed(first_n2))
    assert len(sequence) == 56
    assert sum(rep == 2 for _, rep in sequence) == 24
    original_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= original_affinity for cpus in budgets.values())

    observations = []
    with tempfile.TemporaryDirectory(prefix="micro-atomic-ordered-screen-") as temp_name:
        temp = Path(temp_name)
        for index, (job, rep) in enumerate(sequence, 1):
            family, mode, workers, case = job
            row = fixtures[case]
            trace = temp / f"trace-{index}.json"
            command = command_for(job, row, trace)
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, original_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            assert observed["fixture_sha256"] == row["fixture_sha256"]
            assert all(observed[field] > 0 for field in
                       ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"))
            if family == "micro":
                region_mode, factor = mode
                assert observed["region_mode_effective"] == region_mode
                assert observed["regions_per_worker_requested"] == factor
                assert observed["region_count_effective"] == workers * factor
            elif family == "atomic":
                assert observed["old_reduce_effective"] == mode
            elif family == "ordered":
                assert observed["posting_order_effective"] == mode
            observed.update({"case_id": case, "mode_label": label(job),
                             "repeat": rep, "requested_rules": row["rules"],
                             "min_frequency": row["min_frequency"],
                             "outer_call_seconds": outer_seconds,
                             "binary_sha256": hashes[family],
                             "cpu_affinity": budgets[workers], "cpu_budget": workers,
                             "mean_occupied_cores": observed["call_cpu_seconds"] / observed["call_seconds"],
                             "command": command})
            observations.append((observed, json.loads(trace.read_text())))
            print(f"{index}/56", case, label(job), f"W{workers}", f"rep{rep}",
                  f"{observed['call_seconds']:.6f}s", flush=True)

        references = {observed["case_id"]: (observed["fingerprint"], trace)
                      for observed, trace in observations if observed["mode_label"] == "serial_cf32_ahash_checked"}
        assert len(references) == 2
        ab_reference = next((observed["fingerprint"], trace) for observed, trace in observations
                            if observed["case_id"] == ab["case_id"] and observed["mode_label"] == "ordered_region")
        assert ab_reference[0] == ab_prior_fingerprint
        references[ab["case_id"]] = ab_reference
        for observed, trace in observations:
            fingerprint, expected = references[observed["case_id"]]
            assert observed["fingerprint"] == fingerprint and trace == expected
            observed["full_trace_match"] = True
        assert any(observed["atomic_old_calls"] > 0 for observed, _ in observations
                   if observed["mode_label"] == "atomic_producer-atomic")
        assert all(observed["aa_sort_elided_batches"] > 0 and observed["aa_sort_seconds"] == 0
                   for observed, _ in observations if observed["case_id"] == ab["case_id"]
                   and observed["mode_label"] == "ordered_global")

    sidecar = {"status": "passed", "measured_calls": 56,
               "n2_jobs": 24, "n1_jobs": 8, "seed": 20260930,
               "n2_second_pass_reverses_first_pass_relative_order": True,
               "manifest_sha256": {str(QUICK.relative_to(ROOT)): sha(QUICK),
                                   str(AB_MANIFEST.relative_to(ROOT)): sha(AB_MANIFEST)},
               "binary_sha256": hashes, "source_sha256": source_hashes,
               "source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
               "shared_source_provenance_sha256": sha(OUT / "shared-source-provenance.json"),
               "ab_prior_fingerprint": ab_prior_fingerprint,
               "ab_prior_archive": "rust/batch_results/radical-endpoint-bitmap-combo-quick-v1/quick.jsonl",
               "initial_affinity": sorted(original_affinity), "cpu_budget": budgets,
               "release_profile": {"debug": 1, "lto": "thin", "codegen_units": 1},
               "training_memory_metric": "train_vm_hwm_mib",
               "process_cpu_metric": "call_cpu_seconds",
               "semantic_check": "complete merge and final-token trace plus SHA-256 fingerprint"}
    with (OUT / "screen.jsonl").open("x") as output:
        for observed, _ in observations:
            output.write(json.dumps(observed) + "\n")
    with (OUT / "screen.jsonl.environment.json").open("x") as output:
        json.dump(sidecar, output, indent=2)
        output.write("\n")
    print("verified 56 exact training traces")


if __name__ == "__main__":
    main()

"""Authorized 26-call fixed-budget screen; config and hashes come from frozen sources."""

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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(job):
    family, mode, _, _ = job
    return f"{family}_{mode}"


def command_for(job, fixture, trace, config):
    family, mode, workers, _ = job
    binary = ROOT / config["families"][family]["binary"]
    common = ["--input", str(RUST / fixture["file"]), "--workers", str(workers),
              "--rules", str(fixture["rules"]), "--min-frequency",
              str(fixture["min_frequency"]), "--trace", str(trace)]
    family_config = config["families"][family]
    if family == "serial":
        return [str(binary), *common, *family_config["fixed_args"]]
    return [str(binary), *common, *family_config["fixed_args"],
            *family_config["modes"][mode]["args"]]


def main():
    config_path = OUT / "frozen-config.json"
    config = json.loads(config_path.read_text())
    assert set(config["families"]) == {"counts", "serial"}
    assert set(config["families"]["counts"]["modes"]) == {"chain", "control", "experiment"}
    hashes = {}
    for family, settings in config["families"].items():
        binary = ROOT / settings["binary"]
        hashes[family] = sha(binary)
        assert hashes[family] == settings["binary_sha256"]
        for gate_path in settings.get("gates", []):
            gate = json.loads((ROOT / gate_path).read_text())
            assert gate["status"] == "passed"
            assert gate["binary_sha256"] == hashes[family]
    source_hashes = json.loads((OUT / "new-source-hashes.json").read_text())
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items())
    fixtures = {row["case_id"]: row for row in json.loads(QUICK.read_text())}
    assert set(fixtures) == {"quick-en-continuous-262144", "quick-zh-continuous-262144"}
    ab = next(row for row in json.loads(AB_MANIFEST.read_text())
              if row["case_id"] == "single-piece-ab-65536")
    fixtures[ab["case_id"]] = ab
    assert all(sha(RUST / row["file"]) == row["fixture_sha256"] for row in fixtures.values())

    natural = sorted(case for case in fixtures if case != ab["case_id"])
    n2_jobs = []
    for case in natural:
        n2_jobs.extend(("counts", mode, 4, case)
                       for mode in ("chain", "control", "experiment"))
        n2_jobs.append(("counts", "experiment", 1, case))
        n2_jobs.append(("serial", "reference", 1, case))
    n2_jobs.extend(("counts", mode, 4, ab["case_id"])
                   for mode in ("chain", "control", "experiment"))
    assert len(n2_jobs) == 13
    rng = random.Random(20260930)
    first = n2_jobs.copy()
    rng.shuffle(first)
    sequence = [(job, 1) for job in first]
    sequence.extend((job, 2) for job in reversed(first))
    assert len(sequence) == 26
    initial_affinity = set(os.sched_getaffinity(0))
    budgets = {1: [5], 4: [0, 1, 2, 5]}
    assert all(set(cpus) <= initial_affinity for cpus in budgets.values())
    observations = []
    with tempfile.TemporaryDirectory(prefix="adaptive-replay-screen-") as temp_name:
        temp = Path(temp_name)
        for index, (job, rep) in enumerate(sequence, 1):
            family, mode, workers, case = job
            fixture = fixtures[case]
            trace = temp / f"trace-{index}.json"
            command = command_for(job, fixture, trace, config)
            try:
                os.sched_setaffinity(0, budgets[workers])
                started = time.perf_counter()
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                outer_seconds = time.perf_counter() - started
            finally:
                os.sched_setaffinity(0, initial_affinity)
            observed = json.loads(process.stdout.strip().splitlines()[-1])
            assert observed["fixture_sha256"] == fixture["fixture_sha256"]
            assert all(observed[field] > 0 for field in
                       ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"))
            for field, value in config["families"][family].get("modes", {}).get(mode, {}).get(
                    "expected_fields", {}).items():
                assert observed[field] == value, (family, mode, field, observed[field])
            for field, value in config["families"][family].get("modes", {}).get(mode, {}).get(
                    "expected_fields_by_workers", {}).get(str(workers), {}).items():
                assert observed[field] == value, (family, mode, workers, field, observed[field])
            observed.update({"case_id": case, "mode_label": label(job),
                             "repeat": rep, "requested_rules": fixture["rules"],
                             "min_frequency": fixture["min_frequency"],
                             "outer_call_seconds": outer_seconds,
                             "binary_sha256": hashes[family],
                             "cpu_affinity": budgets[workers], "cpu_budget": workers,
                             "mean_occupied_cores": observed["call_cpu_seconds"] / observed["call_seconds"],
                             "command": command})
            observations.append((observed, json.loads(trace.read_text())))
            print(f"{index}/26", case, label(job), f"W{workers}", f"rep{rep}",
                  f"{observed['call_seconds']:.6f}s", flush=True)
        references = {observed["case_id"]: (observed["fingerprint"], trace)
                      for observed, trace in observations if observed["mode_label"] == "serial_reference"}
        assert len(references) == 2
        ab_reference = next((observed["fingerprint"], trace)
                            for observed, trace in observations
                            if observed["case_id"] == ab["case_id"]
                            and observed["mode_label"] == "counts_chain")
        # AB has no direct serial run in this 26-call window; compare both modes
        # against the same-window control plus the older frozen fulltrace fingerprint.
        old_rows = [json.loads(line) for line in
                    (RUST / "batch_results/radical-micro-atomic-ordered-v1/screen.jsonl").read_text().splitlines()]
        old_ab = {row["fingerprint"] for row in old_rows
                  if row["case_id"] == ab["case_id"]}
        assert old_ab == {ab_reference[0]}
        references[ab["case_id"]] = ab_reference
        for observed, trace in observations:
            fingerprint, expected = references[observed["case_id"]]
            assert observed["fingerprint"] == fingerprint and trace == expected
            observed["full_trace_match"] = True
    sidecar = {"status": "passed", "measured_calls": 26,
               "n2_jobs": 13, "seed": 20260930,
               "second_pass_reverses_first_pass_relative_order": True,
               "manifest_sha256": {str(QUICK.relative_to(ROOT)): sha(QUICK),
                                   str(AB_MANIFEST.relative_to(ROOT)): sha(AB_MANIFEST)},
               "binary_sha256": hashes, "source_sha256": source_hashes,
               "source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
               "shared_source_provenance_sha256": sha(OUT / "shared-source-provenance.json"),
               "config_sha256": sha(config_path),
               "initial_affinity": sorted(initial_affinity), "cpu_budget": budgets,
               "training_memory_metric": "train_vm_hwm_mib",
               "process_cpu_metric": "call_cpu_seconds",
               "semantic_check": "complete merge and final-token trace plus SHA-256 fingerprint"}
    with (OUT / "screen.jsonl").open("x") as output:
        for observed, _ in observations:
            output.write(json.dumps(observed) + "\n")
    with (OUT / "screen.jsonl.environment.json").open("x") as output:
        json.dump(sidecar, output, indent=2)
        output.write("\n")
    print("verified 26 exact training traces")


if __name__ == "__main__":
    main()

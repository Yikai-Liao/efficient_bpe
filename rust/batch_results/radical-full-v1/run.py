"""Sequential full matrix with durable progress and exact output verification."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def progress(status, completed, total, **details):
    value = {"status": status, "completed_measurements": completed,
             "planned_measurements": total, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             **details}
    temporary = OUT / "progress.next.json"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(OUT / "progress.json")


def observe(command, cpus, timeout):
    invocation = ["taskset", "-c", ",".join(map(str, cpus)), *command]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    process = subprocess.Popen(invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, cwd=ROOT)
    peak_swap_kib = 0
    while True:
        try:
            status = Path(f"/proc/{process.pid}/status").read_text()
            for line in status.splitlines():
                if line.startswith("VmSwap:"):
                    peak_swap_kib = max(peak_swap_kib, int(line.split()[1]))
        except FileNotFoundError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if time.perf_counter() - started > timeout:
                process.kill()
                stdout, stderr = process.communicate()
                raise RuntimeError(("call timeout", command, stderr, stdout))
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if process.returncode:
        raise RuntimeError((process.returncode, command, stderr, stdout))
    row = json.loads(stdout.strip().splitlines()[-1])
    row.update(command=invocation, outer_call_seconds=time.perf_counter() - started,
               sampled_peak_vm_swap_kib=peak_swap_kib,
               process_major_faults=after.ru_majflt - before.ru_majflt)
    return row


def command_for(config, fixtures, job, trace=None):
    case, mode, workers = job
    fixture, settings = fixtures[case], config["modes"][mode]
    command = [str(ROOT / settings["binary"]), "--input", str(RUST / fixture["file"]),
               "--workers", str(workers), "--rules", str(fixture["rules"]),
               "--min-frequency", str(fixture["min_frequency"]), *settings["args"]]
    if trace is not None:
        command += ["--trace", str(trace)]
    return command


def verify_row(row, config, fixtures, job, references):
    case, mode, workers = job
    assert row["fixture_sha256"] == fixtures[case]["fixture_sha256"]
    assert row["fingerprint"] == references[case]["fingerprint"], (job, "complete result fingerprint")
    assert row["rules"] == references[case]["actual_rules"]
    assert row["workers"] == workers
    assert all(row[field] > 0 for field in ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib"))
    for field, expected in config["modes"][mode].get("expected_fields", {}).items():
        assert row[field] == expected, (job, field, row[field], expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path, fixture_path = OUT / "config.json", OUT / "fixtures.json"
    config = json.loads(config_path.read_text())
    fixtures = {row["case_id"]: row for row in json.loads(fixture_path.read_text())}
    assert len(fixtures) == 8 and config["repeats"] == 5
    for row in fixtures.values():
        assert sha(RUST / row["file"]) == row["fixture_sha256"]
    for mode in config["modes"].values():
        assert sha(ROOT / mode["binary"]) == mode["binary_sha256"]
        for path in mode["gates"]:
            assert json.loads((ROOT / path).read_text())["status"] == "passed"
    affinity = set(os.sched_getaffinity(0))
    assert all(set(cpus) <= affinity for cpus in config["cpu_budget"].values())
    cells = []
    for case, fixture in fixtures.items():
        selected_workers = config["natural_workers"] if fixture["category"] == "natural" else config["synthetic_workers"]
        for mode, settings in config["modes"].items():
            cells.extend((case, mode, workers) for workers in settings["workers"]
                         if workers in selected_workers)
    assert len(cells) == config["warmup_rows"] == 200
    rng = random.Random(config["seed"])
    schedule = []
    for repetition in range(1, 6):
        if repetition % 2:
            block = cells.copy()
            rng.shuffle(block)
        else:
            block = list(reversed(block))
        schedule.extend((repetition, *job) for job in block)
    assert len(schedule) == config["measurement_rows"] == 1000
    runtime_files = [OUT / name for name in ("prepare.py", "run.py", "config.json", "fixtures.json")]
    source_files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "rust/src",
         "rust/experiments/radical/owned_integer_hash", "rust/experiments/radical/owned_adaptive_cuts",
         "rust/experiments/radical/owned_replay_counts", "rust/experiments/radical/serial_integer_hash",
         "rust/Cargo.toml", "rust/Cargo.lock", "rust/experiments/aa_parity.rs"], cwd=ROOT,
        text=True).splitlines()
    source_hashes = {path: sha(ROOT / path) for path in source_files}
    runtime_hashes = {str(path.relative_to(ROOT)): sha(path) for path in runtime_files}
    env_path = OUT / "environment.json"
    if args.resume:
        env = json.loads(env_path.read_text())
        assert env["workspace_source_sha256"] == source_hashes
        assert env["runner_sha256"] == runtime_hashes
        assert env["config_sha256"] == sha(config_path)
    else:
        env = {"git_head_during_measurement": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "config_sha256": sha(config_path), "fixture_manifest_sha256": sha(fixture_path),
               "runner_sha256": runtime_hashes, "workspace_source_sha256": source_hashes,
               "compiled_binary_provenance": {
                   mode: {**settings,
                          "source_archive_sha256": sha(ROOT / settings["source_archive"]),
                          "shared_provenance_sha256": sha(ROOT / settings["shared_provenance"])}
                   for mode, settings in config["modes"].items()},
               "cpu_budget": config["cpu_budget"], "initial_affinity": sorted(affinity),
               "lscpu": subprocess.check_output(["lscpu"], text=True),
               "uname": subprocess.check_output(["uname", "-a"], text=True),
               "meminfo_at_start": Path("/proc/meminfo").read_text(),
               "measurement_rows": 1000, "warmup_rows": 200, "reference_rows": 8,
               "timing": "call_seconds includes training, initialization, thread pool and finalization; excludes JSON input and trace/fingerprint output. outer_call_seconds includes whole child process.",
               "memory": "Process VmHWM immediately after training, before output; includes prior input parsing high-water. VmSwap sampled every ~200ms; major faults are per process, including input.",
               "semantic_check": "Every measurement hashes all rules/frequencies/final tokens; every cell warmup also compares its complete serialized trace with direct CF32."}
        write_json(env_path, env)

    references_path = OUT / "references.json"
    if references_path.exists():
        assert args.resume
        references = json.loads(references_path.read_text())
    else:
        references = {}
        for case, fixture in fixtures.items():
            job = (case, config["reference_mode"], 1)
            trace = OUT / "references" / f"{case}.json"
            if case == config["reference_case_already_run"]:
                row = json.loads((OUT / "validation/en-16m-serial-reference.json").read_text())
            else:
                row = observe(command_for(config, fixtures, job, trace), [5], config["per_call_timeout_seconds"])
            assert row["fixture_sha256"] == fixture["fixture_sha256"]
            full = json.loads(trace.read_text())
            compressed = RUST / "fixtures/full/references" / f"{case}.json.gz"
            compressed.parent.mkdir(parents=True, exist_ok=True)
            with trace.open("rb") as src, gzip.open(compressed, "wb", compresslevel=1) as dst:
                shutil.copyfileobj(src, dst)
            references[case] = {"fingerprint": row["fingerprint"], "actual_rules": row["rules"],
                                "raw_trace_sha256": sha(trace), "compressed_trace_file": str(compressed.relative_to(ROOT)),
                                "compressed_trace_sha256": sha(compressed), "final_token_count": len(full["final"]),
                                "merge_rows": len(full["merges"]), "reference_observation": row}
            trace.unlink()
            print("reference", case, row["rules"], f"{row['call_seconds']:.3f}s", flush=True)
        write_json(references_path, references)

    measured_path, warmup_path = OUT / "measurements.jsonl", OUT / "warmups.jsonl"
    previous = [json.loads(line) for line in measured_path.read_text().splitlines()] if args.resume else []
    warmed = [json.loads(line) for line in warmup_path.read_text().splitlines()] if args.resume else []
    completed = {(row["repeat"], row["case_id"], row["mode"], row["workers"]) for row in previous}
    warm_keys = {(row["case_id"], row["mode"], row["workers"]) for row in warmed}
    assert len(completed) == len(previous) and len(warm_keys) == len(warmed)
    for row in previous + warmed:
        verify_row(row, config, fixtures, (row["case_id"], row["mode"], row["workers"]), references)
    started = time.perf_counter()
    with measured_path.open("a" if args.resume else "x") as measured_output, \
            warmup_path.open("a" if args.resume else "x") as warm_output, \
            tempfile.TemporaryDirectory(prefix="bpe-full-trace-") as temp_name:
        for repeat, case, mode, workers in schedule:
            key = (repeat, case, mode, workers)
            if key in completed:
                continue
            job = (case, mode, workers)
            progress("running", len(completed), 1000, current={"case": case, "mode": mode, "workers": workers, "repeat": repeat})
            settings = config["modes"][mode]
            cpus = config["cpu_budget"][str(workers)]
            if job not in warm_keys:
                trace = Path(temp_name) / "warmup.json"
                warm = observe(command_for(config, fixtures, job, trace), cpus, config["per_call_timeout_seconds"])
                verify_row(warm, config, fixtures, job, references)
                same_bytes = sha(trace) == references[case]["raw_trace_sha256"]
                if not same_bytes:
                    # Crates can use different serde_json feature sets and
                    # object key order. Compare the complete semantic objects.
                    with gzip.open(ROOT / references[case]["compressed_trace_file"], "rt") as source:
                        expected_trace = json.load(source)
                    actual_trace = json.loads(trace.read_text())
                    assert actual_trace == expected_trace, (job, "complete rules and final tokens")
                    warm["trace_key_order_observed"] = list(actual_trace)
                    warm["trace_key_order_reference"] = list(expected_trace)
                    del actual_trace, expected_trace
                warm["serialized_trace_bytes_match"] = same_bytes
                warm.update(case_id=case, mode=mode, workers=workers, role="warmup", complete_trace_match=True)
                warm_output.write(json.dumps(warm) + "\n")
                warm_output.flush()
                warm_keys.add(job)
                trace.unlink()
            row = observe(command_for(config, fixtures, job), cpus, config["per_call_timeout_seconds"])
            verify_row(row, config, fixtures, job, references)
            row.update(case_id=case, mode=mode, workers=workers, repeat=repeat,
                       cpu_affinity=cpus, cpu_budget=workers,
                       binary_sha256=settings["binary_sha256"], fixture_sha256=fixtures[case]["fixture_sha256"],
                       input_bytes=fixtures[case]["input_bytes"], category=fixtures[case]["category"],
                       requested_rules=fixtures[case]["rules"], min_frequency=fixtures[case]["min_frequency"],
                       full_training_fingerprint_match=True,
                       mean_occupied_cores=row["call_cpu_seconds"] / row["call_seconds"])
            measured_output.write(json.dumps(row) + "\n")
            measured_output.flush()
            completed.add(key)
            if len(completed) % 10 == 0 or len(completed) == 1:
                print(f"{len(completed)}/1000", case, mode, f"W{workers}", f"rep{repeat}",
                      f"{row['call_seconds']:.3f}s", f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    assert len(completed) == 1000 and len(warm_keys) == 200
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items())
    assert all(sha(ROOT / path) == digest for path, digest in runtime_hashes.items())
    progress("passed", 1000, 1000, warmups=200, references=8,
             elapsed_execution_seconds=time.perf_counter() - started)
    print("passed: 1000 measured results and 200 complete warmup traces", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure = {"status": "failed", "error": repr(error),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (OUT / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise

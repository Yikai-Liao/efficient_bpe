"""Complete a bounded core comparison without resuming the full matrix."""

import json
import os
from pathlib import Path
import random
import subprocess
import time

from run import ROOT, RUST, OUT, command_for, observe, sha, verify_row


def main():
    destination = OUT / "focused"
    destination.mkdir(exist_ok=True)
    config = json.loads((OUT / "config.json").read_text())
    fixtures = {r["case_id"]: r for r in json.loads((OUT / "fixtures.json").read_text())}
    references = json.loads((OUT / "references.json").read_text())
    original_environment = json.loads((OUT / "environment.json").read_text())
    modes = ["serial_cf32_checked", "serial_cf16_unchecked", "owner_ahash",
             "cuts_adaptive", "birth_chain", "birth_replay_inline"]
    cells = [(case, mode, 1 if mode.startswith("serial") else 4)
             for case in ("en-16m-continuous", "zh-16m-continuous") for mode in modes]
    for case in {c for c, _, _ in cells}:
        assert sha(RUST / fixtures[case]["file"]) == fixtures[case]["fixture_sha256"]
    for mode in modes:
        settings = config["modes"][mode]
        assert sha(ROOT / settings["binary"]) == settings["binary_sha256"]
        for gate in settings["gates"]:
            assert json.loads((ROOT / gate).read_text())["status"] == "passed"
    for path, digest in original_environment["workspace_source_sha256"].items():
        assert sha(ROOT / path) == digest
    assert all(set(config["cpu_budget"][str(w)]) <= os.sched_getaffinity(0)
               for _, _, w in cells)
    selected = set(cells)
    prior = [json.loads(line) for line in (OUT / "measurements.jsonl").read_text().splitlines()]
    reused = [r for r in prior if (r["case_id"], r["mode"], r["workers"]) in selected]
    assert all(r["repeat"] == 1 for r in reused)
    for r in reused:
        verify_row(r, config, fixtures, (r["case_id"], r["mode"], r["workers"]), references)
        r["measurement_window"] = "original_partial_matrix"
    rows = reused.copy()
    completed = {(r["repeat"], r["case_id"], r["mode"], r["workers"]) for r in rows}
    rng = random.Random(20260930)
    schedule = []
    for repeat in (1, 2, 3):
        block = cells.copy()
        rng.shuffle(block)
        schedule.extend((repeat, *cell) for cell in block
                        if (repeat, *cell) not in completed)
    environment = {
        "git_head_during_measurement": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_environment_sha256": sha(OUT / "environment.json"),
        "config_sha256": sha(OUT / "config.json"), "runner_sha256": sha(Path(__file__)),
        "cells": cells, "repeats_per_cell": 3, "reused_rows": len(reused),
        "new_calls": len(schedule), "cpu_budget": config["cpu_budget"],
        "verification": "Frozen prevalidated binaries; every call verifies the complete training fingerprint against the independent serial reference. Existing full trace checks are retained; no new trace serialization or per-cell warmup.",
        "timing": original_environment["timing"],
        "memory": original_environment["memory"],
    }
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    started = time.perf_counter()
    with (destination / "measurements.jsonl").open("x") as output:
        for row in reused:
            output.write(json.dumps(row) + "\n")
        output.flush()
        print(f"reused={len(reused)} new_calls={len(schedule)} total=36", flush=True)
        for index, (repeat, case, mode, workers) in enumerate(schedule, 1):
            job = (case, mode, workers)
            cpus = config["cpu_budget"][str(workers)]
            row = observe(command_for(config, fixtures, job), cpus, 120)
            verify_row(row, config, fixtures, job, references)
            row.update(case_id=case, mode=mode, workers=workers, repeat=repeat,
                       cpu_affinity=cpus, cpu_budget=workers,
                       binary_sha256=config["modes"][mode]["binary_sha256"],
                       input_bytes=fixtures[case]["input_bytes"], category="natural",
                       requested_rules=32000, min_frequency=2,
                       full_training_fingerprint_match=True,
                       measurement_window="focused_completion",
                       mean_occupied_cores=row["call_cpu_seconds"] / row["call_seconds"])
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
            elapsed = time.perf_counter() - started
            (destination / "progress.json").write_text(json.dumps({
                "status": "running", "new_calls_done": index,
                "new_calls_total": len(schedule), "elapsed_seconds": elapsed,
            }, indent=2) + "\n")
            print(f"{index}/{len(schedule)} {case} {mode} W{workers} "
                  f"rep{repeat} {row['call_seconds']:.3f}s elapsed={elapsed:.1f}s", flush=True)
    assert len(rows) == 36
    assert len({(r["repeat"], r["case_id"], r["mode"], r["workers"]) for r in rows}) == 36
    assert all(r["full_training_fingerprint_match"] for r in rows)
    for path, digest in original_environment["workspace_source_sha256"].items():
        assert sha(ROOT / path) == digest
    (destination / "progress.json").write_text(json.dumps({
        "status": "passed", "reused_rows": len(reused), "new_calls": len(schedule),
        "measured_rows": 36, "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

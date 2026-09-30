"""Validate the completed matrix without repeating performance calls."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config = json.loads((OUT / "config.json").read_text())
    env = json.loads((OUT / "environment.json").read_text())
    assert sha(OUT / "config.json") == env["config_sha256"]
    assert sha(OUT / "fixtures.json") == env["fixture_manifest_sha256"]
    assert all(sha(ROOT / path) == digest for path, digest in env["runner_sha256"].items())
    assert all(sha(ROOT / path) == digest for path, digest in env["workspace_source_sha256"].items())
    for mode, details in env["compiled_binary_provenance"].items():
        assert config["modes"][mode]["binary_sha256"] == sha(ROOT / details["binary"])
        assert sha(ROOT / details["source_archive"]) == details["source_archive_sha256"]
        assert sha(ROOT / details["shared_provenance"]) == details["shared_provenance_sha256"]
    fixtures = {row["case_id"]: row for row in json.loads((OUT / "fixtures.json").read_text())}
    for fixture in fixtures.values():
        assert sha(ROOT / "rust" / fixture["file"]) == fixture["fixture_sha256"]
    references = json.loads((OUT / "references.json").read_text())
    for row in references.values():
        assert sha(ROOT / row["compressed_trace_file"]) == row["compressed_trace_sha256"]
    rows = [json.loads(line) for line in (OUT / "measurements.jsonl").read_text().splitlines()]
    warm = [json.loads(line) for line in (OUT / "warmups.jsonl").read_text().splitlines()]
    assert len(rows) == 1000 and len(warm) == 200
    assert len({(row["case_id"], row["mode"], row["workers"], row["repeat"]) for row in rows}) == 1000
    cells = {(row["case_id"], row["mode"], row["workers"]) for row in warm}
    assert len(cells) == 200
    for case, mode, workers in cells:
        observed = [row for row in rows if (row["case_id"], row["mode"], row["workers"]) == (case, mode, workers)]
        assert sorted(row["repeat"] for row in observed) == [1, 2, 3, 4, 5]
    for row in rows + warm:
        assert row["fingerprint"] == references[row["case_id"]]["fingerprint"]
        assert row["fixture_sha256"] == fixtures[row["case_id"]]["fixture_sha256"]
    assert all(row["full_training_fingerprint_match"] for row in rows)
    assert all(row["complete_trace_match"] for row in warm)
    assert json.loads((OUT / "progress.json").read_text())["status"] == "passed"
    report = {"status": "passed", "measurement_rows": 1000, "warmup_full_trace_matches": 200,
              "reference_cases": 8, "completed_cells": 200,
              "measurement_sha256": sha(OUT / "measurements.jsonl"),
              "warmup_sha256": sha(OUT / "warmups.jsonl"),
              "environment_sha256": sha(OUT / "environment.json"),
              "max_sampled_vm_swap_kib": max(row["sampled_peak_vm_swap_kib"] for row in rows),
              "sum_process_major_faults": sum(row["process_major_faults"] for row in rows)}
    with (OUT / "audit.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

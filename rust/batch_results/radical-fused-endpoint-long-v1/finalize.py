"""Verify the 20-call follow-up matrix and its frozen source/binary provenance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
QUICK = RUST / "batch_results/radical-fused-bitmap-quick-v1"
FUSED = RUST / "target/reruns/radical-fused-endpoint-gate-v1/radical-owned-fused-endpoint"
SERIAL = RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sidecar = json.loads((OUT / "screen.jsonl.environment.json").read_text())
    assert sidecar["rows"] == 20 and sidecar["repetitions"] == 2
    assert sidecar["binary_sha256"] == {"fused": sha(FUSED), "serial": sha(SERIAL)}
    assert sidecar["manifest_sha256"] == sha(ROOT / sidecar["manifest"])
    assert sidecar["quick_checks_sha256"] == sha(QUICK / "checks.json")
    assert sidecar["new_source_snapshot_sha256"] == sha(QUICK / "new-source-snapshot.tar.gz")
    assert sidecar["shared_source_provenance_sha256"] == sha(QUICK / "shared-source-provenance.json")
    for path, expected in sidecar["new_source_sha256"].items():
        assert sha(ROOT / path) == expected, path
    rows = [json.loads(line) for line in (OUT / "screen.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    actual = {(row["case_id"], row["version"], row["workers"], row["repetition"])
              for row in rows}
    cases = ("en-4m-continuous", "zh-4m-continuous")
    expected = {(case, version, workers, repetition) for case in cases
                for version, workers in (("two-pass", 1), ("two-pass", 4),
                                         ("tagged-fused", 1), ("tagged-fused", 4),
                                         ("serial-cf32-ahash-checked", 1))
                for repetition in (0, 1)}
    assert len(actual) == len(rows) and actual == expected
    first = [(row["case_id"], row["version"], row["workers"]) for row in rows[:10]]
    second = [(row["case_id"], row["version"], row["workers"]) for row in rows[10:]]
    assert second == first[::-1]
    for row in rows:
        assert row["full_trace_match"] and row["full_training_fingerprint_match"]
        assert row["cpu_affinity"] == ([5] if row["workers"] == 1 else [0, 1, 2, 5])
        assert row["cpu_budget"] == row["workers"]
        assert row["call_seconds"] > 0 and row["call_cpu_seconds"] > 0
        assert row["train_vm_hwm_mib"] > 0
        binary = SERIAL if row["version"].startswith("serial-") else FUSED
        assert row["binary_sha256"] == sha(binary)
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["rows"] == 20 and summary["groups"] == 10
    report = {"status": "passed", "training_calls": 20, "full_trace_matches": 20,
              "groups": 10, "repetitions_per_group": 2,
              "schedule": "seeded shuffle then exact reverse",
              "base_commit": json.loads((QUICK / "shared-source-provenance.json").read_text())["git_base_commit"],
              "binary_sha256": sidecar["binary_sha256"]}
    with (OUT / "checks.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "binary_sha256"}))


if __name__ == "__main__":
    main()

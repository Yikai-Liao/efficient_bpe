"""Verify the three-mode quick matrix and frozen source provenance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-reuse-accumulator-gate-v1/radical-owned-reuse-accumulator"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = json.loads((OUT / "new-source-hashes.json").read_text())
    for path, expected in sources.items():
        assert sha(ROOT / path) == expected, path
    shared = json.loads((OUT / "shared-source-provenance.json").read_text())
    assert shared["all_shared_files_match_base_commit_byte_for_byte"]
    for path, expected in shared["shared_files_sha256"].items():
        assert sha(ROOT / path) == expected, path
    oracle = json.loads((RUST / "batch_results/radical-reuse-accumulator-gate-v1/differential.json").read_text())
    assert oracle["status"] == "passed" and oracle["fulltrace_matches"] == 240
    assert oracle["binary_sha256"] == sha(BIN)
    sidecar = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    assert sidecar["rows"] == 12 and sidecar["binary_sha256"] == sha(BIN)
    assert sidecar["new_source_sha256"] == sources
    assert sidecar["new_source_snapshot_sha256"] == sha(OUT / "new-source-snapshot.tar.gz")
    assert sidecar["shared_source_provenance_sha256"] == sha(OUT / "shared-source-provenance.json")
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    expected = {(case, mode, workers)
                for case in ("quick-en-continuous-262144", "quick-zh-continuous-262144")
                for mode in ("staged", "fused-fresh", "fused-reuse")
                for workers in (1, 4)}
    assert {(row["case_id"], row["version"], row["workers"]) for row in rows} == expected
    assert all(row["full_trace_match"] and row["cpu_affinity"] ==
               ([5] if row["workers"] == 1 else [0, 1, 2, 5]) for row in rows)
    assert all(row["call_seconds"] > 0 and row["call_cpu_seconds"] > 0
               and row["train_vm_hwm_mib"] > 0 for row in rows)
    report = {"status": "passed", "quick_rows": 12, "full_trace_matches": 12,
              "oracle_matches": 240, "git_base_commit": shared["git_base_commit"],
              "source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
              "binary_sha256": sha(BIN)}
    with (OUT / "checks.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

"""Verify the archived oracle, provenance, and fixed-budget quick matrix."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BINARIES = {
    "fused": RUST / "target/reruns/radical-fused-endpoint-gate-v1/radical-owned-fused-endpoint",
    "bitmap": RUST / "target/reruns/radical-aa-bitmap-gate-v1/radical-owned-aa-bitmap",
    "owner": RUST / "target/reruns/radical-local-hash-v1/owned_integer_hash",
    "serial": RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash",
}


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
    sidecar = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    assert sidecar["rows"] == 30 and sidecar["repetitions"] == 1
    assert sidecar["new_source_sha256"] == sources
    assert sidecar["new_source_snapshot_sha256"] == sha(OUT / "new-source-snapshot.tar.gz")
    assert sidecar["shared_source_provenance_sha256"] == sha(OUT / "shared-source-provenance.json")
    assert sidecar["binary_sha256"] == {kind: sha(path) for kind, path in BINARIES.items()}
    fused = json.loads((RUST / "batch_results/radical-fused-endpoint-gate-v1/differential.json").read_text())
    bitmap = json.loads((RUST / "batch_results/radical-aa-bitmap-gate-v1/differential.json").read_text())
    assert fused["status"] == bitmap["status"] == "passed"
    assert (fused["standard_fulltrace_matches"], fused["custom_fulltrace_matches"]) == (240, 18)
    assert (bitmap["standard_fulltrace_matches"], bitmap["directed_fulltrace_matches"]) == (160, 6)
    assert fused["binary_sha256"] == sha(BINARIES["fused"])
    assert bitmap["binary_sha256"] == sha(BINARIES["bitmap"])
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    assert len(rows) == 30
    actual = {(row["case_id"], row["kind"], row["version"], row["workers"]) for row in rows}
    assert len(actual) == len(rows)
    quick = ("quick-en-continuous-262144", "quick-zh-continuous-262144")
    aa = ("single-run-a-65536", "single-piece-ab-65536", "quick-en-continuous-262144")
    expected = {(case, "fused", mode, workers) for case in quick
                for mode in ("two-pass", "tagged-two-pass", "tagged-fused")
                for workers in (1, 4)}
    expected |= {(case, "owner", "owner-ahash", workers) for case in quick
                 for workers in (1, 4)}
    expected |= {(case, "serial", "cf32-ahash-checked", 1) for case in quick}
    expected |= {(case, "bitmap", mode, workers) for case in aa
                 for mode in ("sort", "bitmap-adaptive") for workers in (1, 4)}
    assert actual == expected
    for row in rows:
        assert row["full_trace_match"]
        assert row["cpu_affinity"] == ([5] if row["workers"] == 1 else [0, 1, 2, 5])
        assert row["cpu_budget"] == row["workers"]
        assert row["binary_sha256"] == sidecar["binary_sha256"][row["kind"]]
        assert row["call_seconds"] > 0 and row["call_cpu_seconds"] > 0
        assert row["train_vm_hwm_mib"] > 0
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["rows"] == len(summary["observations"]) == 30
    negative = [row for row in rows if row["case_id"] == "quick-en-continuous-262144"
                and row["kind"] == "bitmap" and row["version"] == "bitmap-adaptive"]
    assert len(negative) == 2 and all(row["aa_bitmap_rounds"] == 0 for row in negative)
    report = {"status": "passed", "quick_rows": 30, "full_trace_matches": 30,
              "fused_oracle_matches": 258, "bitmap_oracle_matches": 166,
              "natural_en_bitmap_dense_rounds": 0,
              "git_base_commit": shared["git_base_commit"],
              "new_source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
              "binary_sha256": sidecar["binary_sha256"]}
    with (OUT / "checks.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("binary_sha256",)}))


if __name__ == "__main__":
    main()

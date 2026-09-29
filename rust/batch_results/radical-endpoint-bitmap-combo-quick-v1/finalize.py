"""Verify combo quick matrix, fulltrace gate, and source provenance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-endpoint-bitmap-combo-gate-v1/radical-owned-endpoint-bitmap-combo"


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
    oracle = json.loads((RUST / "batch_results/radical-endpoint-bitmap-combo-gate-v1/differential.json").read_text())
    assert oracle["status"] == "passed"
    assert (oracle["standard_fulltrace_matches"], oracle["directed_fulltrace_matches"]) == (480, 5)
    assert oracle["binary_sha256"] == sha(BIN)
    sidecar = json.loads((OUT / "quick.jsonl.environment.json").read_text())
    assert sidecar["rows"] == 16 and sidecar["binary_sha256"] == sha(BIN)
    assert sidecar["new_source_sha256"] == sources
    assert sidecar["new_source_snapshot_sha256"] == sha(OUT / "new-source-snapshot.tar.gz")
    assert sidecar["shared_source_provenance_sha256"] == sha(OUT / "shared-source-provenance.json")
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    ab = {(endpoint, aa, workers)
          for endpoint in ("two-pass", "tagged-fused")
          for aa in ("sort", "bitmap-adaptive") for workers in (1, 4)}
    observed_ab = {(row["endpoint_plan_effective"], row["aa_order"], row["workers"])
                   for row in rows if row["case_id"] == "single-piece-ab-65536"}
    assert observed_ab == ab
    for case in ("quick-en-continuous-262144", "quick-zh-continuous-262144"):
        actual = {(row["aa_order"], row["workers"]) for row in rows if row["case_id"] == case}
        assert actual == {(aa, workers) for aa in ("sort", "bitmap-adaptive")
                          for workers in (1, 4)}
    assert all(row["full_trace_match"] and row["cpu_affinity"] ==
               ([5] if row["workers"] == 1 else [0, 1, 2, 5]) for row in rows)
    assert all(row["call_seconds"] > 0 and row["call_cpu_seconds"] > 0
               and row["train_vm_hwm_mib"] > 0 for row in rows)
    natural_adaptive = [row for row in rows if row["case_id"].startswith("quick-")
                        and row["aa_order"] == "bitmap-adaptive"]
    assert len(natural_adaptive) == 4
    assert all(row["aa_bitmap_rounds"] == 0 for row in natural_adaptive)
    ab_combo = [row for row in rows if row["case_id"] == "single-piece-ab-65536"
                and row["endpoint_plan_effective"] == "tagged-fused"
                and row["aa_order"] == "bitmap-adaptive"]
    assert len(ab_combo) == 2
    assert all(row["fused_non_aa_batches"] > 0 and row["aa_bitmap_rounds"] > 0
               and row["aa_bitmap_fallback_rounds"] > 0 for row in ab_combo)
    report = {"status": "passed", "quick_rows": 16, "full_trace_matches": 16,
              "oracle_matches": 485, "natural_adaptive_dense_rounds": 0,
              "ab_combo_non_aa_and_bitmap_both_executed": True,
              "git_base_commit": shared["git_base_commit"],
              "source_snapshot_sha256": sha(OUT / "new-source-snapshot.tar.gz"),
              "binary_sha256": sha(BIN)}
    with (OUT / "checks.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

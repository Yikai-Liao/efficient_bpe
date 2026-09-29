"""Summarize lazy certificate work and memory without new timing."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
METRICS = (
    "batch_rounds", "batch_rules", "max_batch_width", "sketch_negative_admissions",
    "sketch_positive_stops", "sketch_initial_visits", "sketch_birth_visits",
    "sketch_peak_capacity_scaled_bytes", "lazy_mask_builds", "lazy_mask_hits",
    "lazy_mask_visits", "lazy_mask_stale_visits", "lazy_parallel_builds",
    "lazy_max_single_scan", "lazy_mask_seconds",
    "lazy_cache_peak_capacity_scaled_bytes", "lazy_cache_final_capacity_scaled_bytes",
    "lazy_cache_key_value_bytes", "select_seconds", "plan_seconds", "apply_seconds",
    "posting_visits", "stale_visits", "generated_birth_records", "stored_born_postings",
)


def main():
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    assert len(rows) == 14 and all(row["full_trace_match"] for row in rows)
    by_cell = {(row["case_id"], row["workers"], row["version"]): row for row in rows}
    assert len(by_cell) == 14
    cells = []
    comparisons = []
    for row in sorted(rows, key=lambda r: (r["case_id"], r["workers"], r["version"])):
        cells.append({
            "case_id": row["case_id"], "workers": row["workers"],
            "version": row["version"], "call_seconds": row["call_seconds"],
            "call_cpu_seconds": row["call_cpu_seconds"],
            "mean_occupied_cores": row["mean_occupied_cores"],
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "metrics": {key: row[key] for key in METRICS},
        })
    for case_id in sorted({row["case_id"] for row in rows}):
        for workers in (1, 4):
            type_only = by_cell[(case_id, workers, "type")]
            eager = by_cell[(case_id, workers, "birth_neighbor64")]
            lazy = by_cell[(case_id, workers, "lazy_default")]
            comparisons.append({
                "case_id": case_id, "workers": workers,
                "type_call_seconds": type_only["call_seconds"],
                "eager_call_seconds": eager["call_seconds"],
                "lazy_call_seconds": lazy["call_seconds"],
                "type_batch_rounds": type_only["batch_rounds"],
                "eager_batch_rounds": eager["batch_rounds"],
                "lazy_batch_rounds": lazy["batch_rounds"],
                "eager_sketch_visits": (eager["sketch_initial_visits"]
                                        + eager["sketch_birth_visits"]),
                "lazy_mask_visits": lazy["lazy_mask_visits"],
                "lazy_mask_builds": lazy["lazy_mask_builds"],
                "lazy_mask_hits": lazy["lazy_mask_hits"],
                "lazy_parallel_builds": lazy["lazy_parallel_builds"],
                "lazy_max_single_scan": lazy["lazy_max_single_scan"],
                "eager_capacity_proxy_bytes": eager["sketch_peak_capacity_scaled_bytes"],
                "lazy_cache_capacity_proxy_bytes": lazy["lazy_cache_peak_capacity_scaled_bytes"],
                "type_train_hwm_mib": type_only["train_vm_hwm_mib"],
                "eager_train_hwm_mib": eager["train_vm_hwm_mib"],
                "lazy_train_hwm_mib": lazy["train_vm_hwm_mib"],
            })
    report = {
        "rows": 14, "cases": sorted({row["case_id"] for row in rows}),
        "workers": [1, 4], "repeats": 1,
        "full_trace_and_fingerprint_match_same_binary_type_w1_reference": True,
        "default_parallel_builds_on_these_fixtures": sum(
            row["lazy_parallel_builds"] for row in rows if row["version"] == "lazy_default"),
        "capacity_proxy_note": "HashMap capacity multiplied by entry payload; not measured allocation bytes or simultaneous RSS.",
        "cells": cells, "comparisons": comparisons,
    }
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": 14, "cases": 2}))


if __name__ == "__main__":
    main()

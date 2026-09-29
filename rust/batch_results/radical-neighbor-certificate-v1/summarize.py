"""Summarize certificate effects while allowing batch-dependent work counts."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
METRICS = (
    "batch_rounds", "batch_rules", "max_batch_width", "sketch_negative_admissions",
    "sketch_positive_stops", "sketch_selected_popcount", "sketch_final_popcount",
    "entry_tuple_bytes", "sketch_payload_bytes", "sketch_peak_capacity_scaled_bytes",
    "sketch_initial_visits", "sketch_birth_visits", "posting_visits", "stale_visits",
    "generated_birth_records", "stored_born_postings", "plan_seconds", "select_seconds",
)


def main():
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and all(row["full_trace_match"] for row in rows)
    by_cell = {(row["case_id"], row["workers"], row["version"]): row for row in rows}
    assert len(by_cell) == 8
    cells = []
    comparisons = []
    for case_id in sorted({row["case_id"] for row in rows}):
        for workers in (1, 4):
            type_only = by_cell[(case_id, workers, "type")]
            sketch = by_cell[(case_id, workers, "birth_neighbor64")]
            comparisons.append({
                "case_id": case_id, "workers": workers,
                "type_call_seconds": type_only["call_seconds"],
                "sketch_call_seconds": sketch["call_seconds"],
                "sketch_over_type_call_ratio": sketch["call_seconds"] / type_only["call_seconds"],
                "type_batch_rounds": type_only["batch_rounds"],
                "sketch_batch_rounds": sketch["batch_rounds"],
                "sketch_negative_admissions": sketch["sketch_negative_admissions"],
                "extra_sketch_visits": (sketch["sketch_initial_visits"]
                                        + sketch["sketch_birth_visits"]),
                "type_train_hwm_mib": type_only["train_vm_hwm_mib"],
                "sketch_train_hwm_mib": sketch["train_vm_hwm_mib"],
            })
    for row in sorted(rows, key=lambda r: (r["case_id"], r["workers"], r["version"])):
        cells.append({
            "case_id": row["case_id"], "workers": row["workers"],
            "version": row["version"], "call_seconds": row["call_seconds"],
            "call_cpu_seconds": row["call_cpu_seconds"],
            "mean_occupied_cores": row["mean_occupied_cores"],
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "metrics": {key: row[key] for key in METRICS},
        })
    report = {
        "rows": 8, "cases": sorted({row["case_id"] for row in rows}),
        "workers": [1, 4], "repeats": 1,
        "full_trace_and_fingerprint_match_native_reference": True,
        "batch_dependent_work_counts_may_differ": True,
        "cells": cells, "comparisons": comparisons,
    }
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": 8, "cases": 2, "workers": 2}))


if __name__ == "__main__":
    main()

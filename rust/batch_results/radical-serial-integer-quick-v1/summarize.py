"""Summarize the completed one-repeat 22-call exact quick screen."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    rows = [json.loads(line) for line in (OUT / "screen.jsonl").read_text().splitlines()]
    assert len(rows) == 22 and all(row["full_trace_match"] for row in rows)
    by_cell = {(row["case_id"], row["version"], row["workers"]): row for row in rows}
    assert len(by_cell) == len(rows)
    cells = []
    comparisons = []
    for row in sorted(rows, key=lambda r: (r["case_id"], r["version"], r["workers"])):
        cells.append({
            "case_id": row["case_id"], "version": row["version"],
            "workers": row["workers"], "call_seconds": row["call_seconds"],
            "call_cpu_seconds": row["call_cpu_seconds"],
            "mean_occupied_cores": row["mean_occupied_cores"],
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "merge_seconds": row.get("merge_seconds"),
            "init_seconds": row.get("init_seconds"),
            "plan_seconds": row.get("plan_seconds"),
            "backend_buffer_bytes": row.get("backend_buffer_bytes"),
            "initial_occurrence_bytes": row.get("initial_occurrence_bytes"),
            "position_visits": row.get("position_visits"),
            "posting_visits": row.get("posting_visits"),
        })
    for case_id in sorted({row["case_id"] for row in rows}):
        native = by_cell[(case_id, "native_direct", 1)]
        for backend in ("cf32", "cf16"):
            for bounds in ("checked", "unchecked"):
                old = by_cell[(case_id, f"{backend}_std_{bounds}", 1)]
                new = by_cell[(case_id, f"{backend}_ahash_{bounds}", 1)]
                comparisons.append({
                    "case_id": case_id, "backend": backend, "bounds": bounds,
                    "std_call_seconds": old["call_seconds"],
                    "ahash_call_seconds": new["call_seconds"],
                    "std_over_ahash_ratio": old["call_seconds"] / new["call_seconds"],
                    "native_over_ahash_ratio": native["call_seconds"] / new["call_seconds"],
                    "std_train_hwm_mib": old["train_vm_hwm_mib"],
                    "ahash_train_hwm_mib": new["train_vm_hwm_mib"],
                })
    report = {
        "rows": len(rows), "cases": sorted({row["case_id"] for row in rows}),
        "serial_modes": 8, "owner_modes": 2, "native_modes": 1,
        "repeats": 1, "all_complete_traces_and_fingerprints_match_native": True,
        "cpu_metric": "call_cpu_seconds/call_seconds is average occupied cores, not useful-compute utilization",
        "cells": cells, "comparisons": comparisons,
    }
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "serial_modes": 8}))


if __name__ == "__main__":
    main()

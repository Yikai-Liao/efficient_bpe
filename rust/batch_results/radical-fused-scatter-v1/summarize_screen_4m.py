"""Summarize the authorized one-repeat 4 MiB diagnostic."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
WORK = ("actual_merges", "batch_rounds", "batch_rules", "generated_birth_records",
        "stored_born_postings", "posting_visits", "stale_visits", "final_live_edges")
PHASES = ("init_seconds", "plan_seconds", "frequency_reduce_seconds",
          "update_stage_seconds", "owner_update_branch_seconds",
          "birth_group_fill_seconds", "scatter_setup_seconds", "scatter_fill_seconds")


def main():
    rows = [json.loads(line) for line in (OUT / "screen-4m.jsonl").read_text().splitlines()]
    if len(rows) != 28:
        raise AssertionError("expected exactly 28 diagnostic runs")
    if not all(row["full_training_fingerprint_match"] for row in rows):
        raise AssertionError("fingerprint mismatch")
    mismatches = []
    for case_id in {row["case_id"] for row in rows}:
        for workers in (1, 4):
            group = [row for row in rows if row["case_id"] == case_id
                     and row["workers"] == workers]
            for key in WORK:
                values = {row[key] for row in group}
                if len(values) != 1:
                    mismatches.append([case_id, workers, key, sorted(values)])
    cells = []
    for row in sorted(rows, key=lambda r: (r["case_id"], r["workers"], r["version"])):
        cells.append({
            "case_id": row["case_id"], "workers": row["workers"],
            "version": row["version"], "call_seconds": row["call_seconds"],
            "call_cpu_seconds": row["call_cpu_seconds"],
            "mean_occupied_cores": row["mean_occupied_cores"],
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "phases": {key: row[key] for key in PHASES if key in row},
            "update_dispatches": row.get("update_dispatches"),
            "update_completion_barriers": row.get("update_completion_barriers"),
            "peak_reduce_temporary_keys_sum": row.get("peak_reduce_temporary_keys_sum"),
            "peak_reduce_temporary_capacity_sum": row.get("peak_reduce_temporary_capacity_sum"),
            "scatter_heavy_keys": row.get("scatter_heavy_keys"),
            "scatter_heavy_positions": row.get("scatter_heavy_positions"),
            "scatter_tasks": row.get("scatter_tasks"),
            "scatter_extra_lookups": row.get("scatter_extra_lookups"),
        })
    report = {
        "rows": len(rows), "cases": sorted({row["case_id"] for row in rows}),
        "workers": [1, 4], "repeats": 1,
        "fingerprint": "all match prior complete-trace native reference",
        "deterministic_work_fields": list(WORK),
        "deterministic_work_mismatches": mismatches,
        "process_cpu_explanation": "CPU seconds / call seconds is average occupied cores, including stalls; not useful-computation utilization.",
        "phase_explanation": "Some per-mode phase metrics are nested inside update or plan; do not sum them with parent phases.",
        "cells": cells,
    }
    with (OUT / "screen-4m-summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "work_mismatches": len(mismatches)}))


if __name__ == "__main__":
    main()

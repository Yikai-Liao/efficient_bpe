"""Summarize the fixed-budget quick screen without rerunning training."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
WORK = ("actual_merges", "batch_rounds", "batch_rules", "generated_birth_records",
        "stored_born_postings", "posting_visits", "stale_visits", "final_live_edges")
PHASES = ("init_seconds", "plan_seconds", "apply_seconds", "frequency_reduce_seconds",
          "birth_decode_seconds", "birth_group_fill_seconds", "birth_append_seconds",
          "owner_update_branch_seconds", "update_stage_seconds", "scatter_setup_seconds",
          "scatter_fill_seconds", "owner_fill_work_seconds_sum", "owner_fill_work_seconds_max",
          "owner_reduce_work_seconds_sum", "owner_reduce_work_seconds_max")


def main():
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    by_cell = {(row["case_id"], row["workers"], row["version"]): row for row in rows}
    cells = []
    mismatches = []
    for row in sorted(rows, key=lambda r: (r["case_id"], r["workers"], r["version"])):
        baseline = by_cell[(row["case_id"], row["workers"], "combo_lazy")]
        for key in WORK:
            if key in baseline and key in row and baseline[key] != row[key]:
                mismatches.append([row["case_id"], row["workers"], row["version"],
                                   key, baseline[key], row[key]])
        cpu = row.get("call_cpu_seconds")
        cell = {
            "case_id": row["case_id"], "workers": row["workers"],
            "version": row["version"], "call_seconds": row["call_seconds"],
            "relative_to_combo_same_workers": row["call_seconds"] / baseline["call_seconds"],
            "call_cpu_seconds": cpu,
            "cpu_seconds_per_wall_second": cpu / row["call_seconds"] if cpu is not None else None,
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "phases": {key: row[key] for key in PHASES if key in row},
            "update_dispatches": row.get("update_dispatches"),
            "update_completion_barriers": row.get("update_completion_barriers"),
            "scatter_heavy_positions": row.get("scatter_heavy_positions"),
            "scatter_tasks": row.get("scatter_tasks"),
        }
        cells.append(cell)
    report = {
        "rows": len(rows), "cases": sorted({row["case_id"] for row in rows}),
        "workers": [1, 4], "repeats": 1,
        "trace_and_fingerprint": "all checked against same native reference",
        "deterministic_work_fields": list(WORK),
        "deterministic_work_mismatches": mismatches,
        "process_cpu_explanation": "CPU seconds / call seconds is average occupied cores; it includes stalls and is not useful-computation utilization. Frozen combo has no process-CPU field.",
        "phase_explanation": "Phase metrics vary by implementation; some subphases are nested and must not be summed with their parent.",
        "cells": cells,
    }
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "work_mismatches": len(mismatches)}))


if __name__ == "__main__":
    main()

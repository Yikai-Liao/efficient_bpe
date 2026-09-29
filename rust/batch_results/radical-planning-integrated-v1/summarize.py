"""Summarize the fixed-budget planning experiment without rerunning training."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
WORK = ("actual_merges", "batch_rounds", "batch_rules", "generated_birth_records",
        "stored_born_postings", "posting_visits", "stale_visits", "final_live_edges")


def main():
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    by_cell = {(row["case_id"], row["workers"], row["version"]): row for row in rows}
    mismatches = []
    cells = []
    for row in sorted(rows, key=lambda r: (r["case_id"], r["workers"], r["version"])):
        baseline = by_cell[(row["case_id"], row["workers"], "combo_lazy")]
        for key in WORK:
            if key in baseline and key in row and baseline[key] != row[key]:
                mismatches.append([row["case_id"], row["workers"], row["version"],
                                   key, baseline[key], row[key]])
        cpu = row.get("call_cpu_seconds")
        cells.append({
            "case_id": row["case_id"], "workers": row["workers"],
            "version": row["version"], "call_seconds": row["call_seconds"],
            "relative_to_combo_same_workers": row["call_seconds"] / baseline["call_seconds"],
            "call_cpu_seconds": cpu,
            "mean_occupied_cores": cpu / row["call_seconds"] if cpu is not None else None,
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "plan_seconds": row["plan_seconds"],
            "update_stage_seconds": row.get("update_stage_seconds"),
            "selected_table_slots_peak": row.get("selected_table_slots_peak"),
            "route_cache_hits": row.get("route_cache_hits"),
            "route_cache_misses": row.get("route_cache_misses"),
            "route_cache_hit_rate": (
                row["route_cache_hits"] /
                (row["route_cache_hits"] + row["route_cache_misses"])
                if row.get("route_cache_hits", 0) + row.get("route_cache_misses", 0) else None),
            "route_cache_bytes_initialized": row.get("route_cache_bytes_initialized"),
            "peak_route_cache_slot_bytes": row.get("peak_route_cache_slot_bytes"),
        })
    report = {
        "rows": len(rows), "cases": sorted({row["case_id"] for row in rows}),
        "workers": [1, 4], "repeats": 1,
        "trace_and_fingerprint": "all checked against same native reference",
        "deterministic_work_fields": list(WORK),
        "deterministic_work_mismatches": mismatches,
        "process_cpu_explanation": "CPU seconds / call seconds is average occupied cores, including stalls; not useful-computation utilization. Frozen combo has no CPU field.",
        "cells": cells,
    }
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "work_mismatches": len(mismatches)}))


if __name__ == "__main__":
    main()

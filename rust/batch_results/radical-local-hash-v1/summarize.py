"""Summarize local-scratch and integer-hash quick screens without new timing."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
WORK = ("actual_merges", "batch_rounds", "batch_rules", "generated_birth_records",
        "stored_born_postings", "posting_visits", "stale_visits", "final_live_edges")


def main():
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    modes = json.loads((OUT / "modes.json").read_text())["quick"]
    assert len(rows) == 2 * 2 * len(modes) == 20
    by_cell = {(row["case_id"], row["workers"], row["version"]): row for row in rows}
    assert len(by_cell) == len(rows)
    mismatches = []
    cells = []
    comparisons = []
    for case_id in sorted({row["case_id"] for row in rows}):
        for workers in (1, 4):
            group = [row for row in rows if row["case_id"] == case_id
                     and row["workers"] == workers]
            for key in WORK:
                values = {row[key] for row in group}
                if len(values) != 1:
                    mismatches.append([case_id, workers, key, sorted(values)])
            for family, control, experiment in (
                ("scratch_placement", "context_borrowed", "context_local"),
                ("integer_hash", "integer_std", "integer_ahash"),
            ):
                old = by_cell[(case_id, workers, control)]
                new = by_cell[(case_id, workers, experiment)]
                comparisons.append({
                    "case_id": case_id, "workers": workers, "family": family,
                    "control": control, "experiment": experiment,
                    "control_call_seconds": old["call_seconds"],
                    "experiment_call_seconds": new["call_seconds"],
                    "control_over_experiment_speed_ratio": old["call_seconds"] / new["call_seconds"],
                    "control_plan_seconds": old["plan_seconds"],
                    "experiment_plan_seconds": new["plan_seconds"],
                    "control_cpu_seconds": old["call_cpu_seconds"],
                    "experiment_cpu_seconds": new["call_cpu_seconds"],
                    "control_hwm_mib": old["train_vm_hwm_mib"],
                    "experiment_hwm_mib": new["train_vm_hwm_mib"],
                })
    for row in sorted(rows, key=lambda r: (r["case_id"], r["workers"], r["version"])):
        cells.append({
            "case_id": row["case_id"], "workers": row["workers"],
            "version": row["version"], "call_seconds": row["call_seconds"],
            "call_cpu_seconds": row["call_cpu_seconds"],
            "mean_occupied_cores": row["mean_occupied_cores"],
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "plan_seconds": row["plan_seconds"],
            "context_hits": row.get("context_hits"),
            "context_lookups": row.get("context_lookups"),
            "effective_planner": row.get("effective_planner"),
            "scratch_placement": row.get("scratch_placement"),
            "integer_hash": row.get("integer_hash"),
        })
    report = {
        "rows": len(rows), "cases": sorted({row["case_id"] for row in rows}),
        "workers": [1, 4], "repeats": 1,
        "all_complete_traces_and_fingerprints_match_native_reference": True,
        "deterministic_work_fields": list(WORK),
        "deterministic_work_mismatches": mismatches,
        "cpu_metric": "call_cpu_seconds/call_seconds is average occupied cores, not useful-compute utilization",
        "cells": cells, "comparisons": comparisons,
    }
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "work_mismatches": len(mismatches)}))


if __name__ == "__main__":
    main()

"""Summarize the authorized one-repeat controlled 4 MiB diagnostic."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
WORK = ("actual_merges", "batch_rounds", "batch_rules", "generated_birth_records",
        "stored_born_postings", "posting_visits", "stale_visits", "final_live_edges")
METRIC_PREFIXES = ("corpus_", "input_corpus_", "endpoint_corpus_", "conversion_",
                   "route_cache_", "peak_route_cache_", "context_", "cache_")
METRIC_NAMES = ("plan_seconds", "corpus_allocation", "effective_planner")


def main():
    rows = [json.loads(line) for line in (OUT / "screen-4m.jsonl").read_text().splitlines()]
    modes = json.loads((OUT / "modes.json").read_text())["screen_4m"]
    if len(rows) != 2 * 2 * len(modes):
        raise AssertionError("incomplete diagnostic")
    if not all(row["full_training_fingerprint_match"] for row in rows):
        raise AssertionError("fingerprint mismatch")
    by_cell = {(row["case_id"], row["version"], row["workers"]): row for row in rows}
    work_mismatches = []
    comparisons = []
    cells = []
    for case_id in sorted({row["case_id"] for row in rows}):
        for workers in (1, 4):
            group = [row for row in rows if row["case_id"] == case_id
                     and row["workers"] == workers]
            for key in WORK:
                values = {row[key] for row in group}
                if len(values) != 1:
                    work_mismatches.append([case_id, workers, key, sorted(values)])
        for version in modes:
            one = by_cell[(case_id, version, 1)]
            four = by_cell[(case_id, version, 4)]
            t1, t4 = one["call_seconds"], four["call_seconds"]
            c1, c4 = one["call_cpu_seconds"], four["call_cpu_seconds"]
            u1, u4, inflation = c1 / t1, c4 / t4, c4 / c1
            speedup = t1 / t4
            if abs(speedup - u4 / (inflation * u1)) > 1e-12:
                raise AssertionError("CPU accounting identity failed")
            comparisons.append({
                "case_id": case_id, "version": version,
                "w1_call_seconds": t1, "w4_call_seconds": t4,
                "w1_cpu_seconds": c1, "w4_cpu_seconds": c4,
                "self_w1_to_w4_speedup": speedup,
                "u1_cpu_per_wall": u1, "u4_cpu_per_wall": u4,
                "cpu_time_ratio_c4_over_c1": inflation,
                "w1_train_vm_hwm_mib": one["train_vm_hwm_mib"],
                "w4_train_vm_hwm_mib": four["train_vm_hwm_mib"],
                "identity": "S=T1/T4=U4/(I*U1)",
            })
    for row in sorted(rows, key=lambda r: (r["case_id"], r["workers"], r["version"])):
        cells.append({
            "case_id": row["case_id"], "workers": row["workers"],
            "version": row["version"], "call_seconds": row["call_seconds"],
            "call_cpu_seconds": row["call_cpu_seconds"],
            "mean_occupied_cores": row["mean_occupied_cores"],
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "metrics": {key: value for key, value in row.items()
                        if key.startswith(METRIC_PREFIXES) or key in METRIC_NAMES},
        })
    report = {
        "rows": len(rows), "cases": sorted({row["case_id"] for row in rows}),
        "workers": [1, 4], "repeats": 1,
        "fingerprints_match_prior_reference": True,
        "deterministic_work_fields": list(WORK),
        "deterministic_work_mismatches": work_mismatches,
        "cpu_identity_explanation": "U1=C1/T1, U4=C4/T4, I=C4/C1, S=T1/T4=U4/(I*U1). This is accounting, not causal attribution; process CPU includes stalls and spin.",
        "cells": cells, "comparisons": comparisons,
    }
    with (OUT / "screen-4m-summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "work_mismatches": len(work_mismatches)}))


if __name__ == "__main__":
    main()

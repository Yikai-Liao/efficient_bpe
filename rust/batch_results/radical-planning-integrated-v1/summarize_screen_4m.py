"""Summarize the two-repeat 4 MiB integrated screen using one aggregation rule."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
WORK = ("actual_merges", "batch_rounds", "batch_rules", "generated_birth_records",
        "stored_born_postings", "posting_visits", "stale_visits", "final_live_edges")


def main():
    rows = [json.loads(line) for line in (OUT / "screen-4m.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        if not row["full_training_fingerprint_match"]:
            raise AssertionError("fingerprint mismatch")
        groups[(row["case_id"], row["version"], row["workers"])].append(row)
    if len(rows) != 28 or len(groups) != 14 or not all(len(group) == 2 for group in groups.values()):
        raise AssertionError("incomplete 4 MiB screen")
    cells = []
    for (case_id, version, workers), group in sorted(groups.items()):
        call = [row["call_seconds"] for row in group]
        cpu = ([row["call_cpu_seconds"] for row in group]
               if version.startswith("integrated_") else None)
        cells.append({
            "case_id": case_id, "version": version, "workers": workers,
            "call_seconds_median": median(call), "call_seconds_min": min(call),
            "call_seconds_max": max(call),
            "call_cpu_seconds_median": median(cpu) if cpu is not None else None,
            "train_vm_hwm_mib_min": min(row["train_vm_hwm_mib"] for row in group),
            "train_vm_hwm_mib_max": max(row["train_vm_hwm_mib"] for row in group),
            "update_stage_seconds_median": (
                median(row["update_stage_seconds"] for row in group)
                if version.startswith("integrated_") else None),
        })
    lookup = {(cell["case_id"], cell["version"], cell["workers"]): cell for cell in cells}
    comparisons = []
    for case_id in sorted({row["case_id"] for row in rows}):
        native = lookup[(case_id, "best_direct_scalar", 1)]
        combo = lookup[(case_id, "combo_lazy", 4)]
        for version in ("integrated_control", "integrated_candidate"):
            one = lookup[(case_id, version, 1)]
            four = lookup[(case_id, version, 4)]
            t1, t4 = one["call_seconds_median"], four["call_seconds_median"]
            c1, c4 = one["call_cpu_seconds_median"], four["call_cpu_seconds_median"]
            u1, u4 = c1 / t1, c4 / t4
            inflation = c4 / c1
            speedup = t1 / t4
            if abs(speedup - u4 / (inflation * u1)) > 1e-12:
                raise AssertionError("CPU decomposition identity failed")
            comparisons.append({
                "case_id": case_id, "version": version,
                "self_w1_to_w4_speedup": speedup,
                "vs_best_direct_scalar_w1": native["call_seconds_median"] / t4,
                "vs_frozen_combo_w4": combo["call_seconds_median"] / t4,
                "u1_cpu_per_wall": u1, "u4_cpu_per_wall": u4,
                "cpu_work_inflation_c4_over_c1": inflation,
                "identity": "S=T1/T4=U4/(I*U1)",
            })
    work_mismatches = []
    for case_id in {row["case_id"] for row in rows}:
        for workers in (1, 4):
            comparable = [row for row in rows if row["case_id"] == case_id
                          and row["workers"] == workers
                          and row["version"] != "best_direct_scalar"]
            for key in WORK:
                values = {row[key] for row in comparable}
                if len(values) != 1:
                    work_mismatches.append([case_id, workers, key, sorted(values)])
    report = {
        "rows": len(rows), "configurations": len(groups), "repeats": 2,
        "schedule": "seeded shuffle then exact reverse",
        "fingerprints_match": True,
        "deterministic_work_fields": list(WORK),
        "deterministic_work_mismatches": work_mismatches,
        "aggregation": "median of two call wall values and independently median of two process CPU values; min/max retain observed dispersion",
        "cpu_identity_explanation": "U1=C1/T1, U4=C4/T4, I=C4/C1, S=T1/T4=U4/(I*U1). This is accounting, not a causal attribution. CPU time includes stalls, scheduling and spin.",
        "cells": cells, "comparisons": comparisons,
    }
    with (OUT / "screen-4m-summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "work_mismatches": len(work_mismatches)}))


if __name__ == "__main__":
    main()

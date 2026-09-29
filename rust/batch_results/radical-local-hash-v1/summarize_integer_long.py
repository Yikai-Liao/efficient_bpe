"""Summarize the authorized two-repeat 4 MiB integer-hash follow-up."""

import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
WORK = ("actual_merges", "batch_rounds", "batch_rules", "generated_birth_records",
        "stored_born_postings", "posting_visits", "stale_visits", "final_live_edges")


def main():
    rows = [json.loads(line) for line in (OUT / "integer-long.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    assert all(row["full_training_fingerprint_match"] for row in rows)
    assert len({(row["case_id"], row["version"], row["workers"], row["repetition"])
                for row in rows}) == 20
    cells = []
    mismatches = []
    cases = sorted({row["case_id"] for row in rows})
    for case_id in cases:
        for version, workers in (("integer_std", 1), ("integer_std", 4),
                                 ("integer_ahash", 1), ("integer_ahash", 4),
                                 ("best_direct_scalar", 1)):
            group = [row for row in rows if (row["case_id"], row["version"], row["workers"])
                     == (case_id, version, workers)]
            assert len(group) == 2
            wall = [row["call_seconds"] for row in group]
            cpu = [row["call_cpu_seconds"] for row in group]
            hwm = [row["train_vm_hwm_mib"] for row in group]
            cells.append({
                "case_id": case_id, "version": version, "workers": workers,
                "wall_median_seconds": median(wall), "wall_min_seconds": min(wall),
                "wall_max_seconds": max(wall), "cpu_median_seconds": median(cpu),
                "mean_occupied_cores_from_medians": median(cpu) / median(wall),
                "train_vm_hwm_median_mib": median(hwm),
                "train_vm_hwm_min_mib": min(hwm), "train_vm_hwm_max_mib": max(hwm),
                "plan_median_seconds": (median(row["plan_seconds"] for row in group)
                                        if version != "best_direct_scalar" else None),
            })
        for repetition in (0, 1):
            group = [row for row in rows if row["case_id"] == case_id
                     and row["repetition"] == repetition and row["version"].startswith("integer_")]
            for key in WORK:
                values = {row[key] for row in group}
                if len(values) != 1:
                    mismatches.append([case_id, repetition, key, sorted(values)])
    lookup = {(row["case_id"], row["version"], row["workers"]): row for row in cells}
    comparisons = []
    for case_id in cases:
        reference = lookup[(case_id, "best_direct_scalar", 1)]
        for workers in (1, 4):
            std = lookup[(case_id, "integer_std", workers)]
            ahash = lookup[(case_id, "integer_ahash", workers)]
            comparisons.append({
                "case_id": case_id, "workers": workers,
                "std_over_ahash_wall_ratio": std["wall_median_seconds"] / ahash["wall_median_seconds"],
                "ahash_over_std_wall_ratio": ahash["wall_median_seconds"] / std["wall_median_seconds"],
                "ahash_vs_direct_scalar_speed_ratio": (
                    reference["wall_median_seconds"] / ahash["wall_median_seconds"]),
            })
        for version in ("integer_std", "integer_ahash"):
            one = lookup[(case_id, version, 1)]
            four = lookup[(case_id, version, 4)]
            t1, t4 = one["wall_median_seconds"], four["wall_median_seconds"]
            c1, c4 = one["cpu_median_seconds"], four["cpu_median_seconds"]
            speedup = t1 / t4
            inflation = c4 / c1
            u1, u4 = c1 / t1, c4 / t4
            assert abs(speedup - u4 / (inflation * u1)) < 1e-12
            comparisons.append({
                "case_id": case_id, "version": version, "workers": "1_to_4",
                "self_speedup_s": speedup, "cpu_ratio_i_c4_over_c1": inflation,
                "u1_occupied_cores": u1, "u4_occupied_cores": u4,
                "identity": "S=U4/(I*U1)",
            })
    report = {
        "rows": len(rows), "cases": cases, "repeats": 2,
        "schedule": "seeded shuffle then exact reverse",
        "fingerprints_match_prior_reference": True,
        "deterministic_work_fields": list(WORK),
        "deterministic_work_mismatches": mismatches,
        "median_policy": "Median wall and median process CPU calculated separately; CPU identity uses those same medians.",
        "cells": cells, "comparisons": comparisons,
    }
    with (OUT / "integer-long-summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "work_mismatches": len(mismatches)}))


if __name__ == "__main__":
    main()

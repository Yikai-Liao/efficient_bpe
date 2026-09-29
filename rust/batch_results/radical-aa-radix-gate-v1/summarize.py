"""Summarize the authorized AA-sort quick screen without new training."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and all(row["full_trace_match"] for row in rows)
    by_cell = {(row["case_id"], row["workers"], row["version"]): row for row in rows}
    assert len(by_cell) == 12
    cells = []
    comparisons = []
    for row in sorted(rows, key=lambda r: (r["case_id"], r["workers"], r["version"])):
        cells.append({
            "case_id": row["case_id"], "workers": row["workers"],
            "sort": row["version"], "call_seconds": row["call_seconds"],
            "call_cpu_seconds": row["call_cpu_seconds"],
            "train_vm_hwm_mib": row["train_vm_hwm_mib"],
            "aa_sort_seconds": row["aa_sort_seconds"],
            "aa_sort_share_of_call": row["aa_sort_share_of_call"],
            "aa_sort_positions": row["aa_sort_positions"],
            "aa_radix_distribution_passes": row["aa_radix_distribution_passes"],
            "aa_radix_swaps": row["aa_radix_swaps"],
            "aa_radix_fallback_calls": row["aa_radix_fallback_calls"],
            "aa_radix_max_stack_payload_bytes": row["aa_radix_max_stack_payload_bytes"],
        })
    for case_id in sorted({row["case_id"] for row in rows}):
        for workers in (1, 4):
            std = by_cell[(case_id, workers, "std")]
            radix = by_cell[(case_id, workers, "radix")]
            assert std["aa_sort_positions"] == radix["aa_sort_positions"]
            comparisons.append({
                "case_id": case_id, "workers": workers,
                "std_call_seconds": std["call_seconds"],
                "radix_call_seconds": radix["call_seconds"],
                "radix_over_std_call_ratio": radix["call_seconds"] / std["call_seconds"],
                "std_aa_sort_seconds": std["aa_sort_seconds"],
                "radix_aa_sort_seconds": radix["aa_sort_seconds"],
                "radix_over_std_aa_sort_ratio": (
                    radix["aa_sort_seconds"] / std["aa_sort_seconds"]
                    if std["aa_sort_seconds"] else None),
                "aa_sort_positions": std["aa_sort_positions"],
            })
    report = {
        "rows": 12, "cases": sorted({row["case_id"] for row in rows}),
        "workers": [1, 4], "repeats": 1, "integer_hash": "ahash",
        "full_trace_and_fingerprint_match_same_binary_std_w1_reference": True,
        "std_sort_implementation": "Rayon pool.install(par_sort_unstable)",
        "radix_sort_implementation": "serial in-place u32 radix",
        "cells": cells, "comparisons": comparisons,
    }
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": 12, "cases": 3}))


if __name__ == "__main__":
    main()

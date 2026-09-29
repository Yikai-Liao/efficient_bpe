"""Summarize only the archived calls, preserving each raw repetition."""

import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
METRICS = (
    "call_seconds", "call_cpu_seconds", "train_vm_hwm_mib", "plan_seconds",
    "region_non_aa_posting_visits", "region_non_aa_valid_merges",
    "region_sum_visit_makespan_lower_bound", "region_sum_merge_makespan_lower_bound",
    "region_peak_route_header_capacity_bytes", "region_count_effective",
    "region_partition_worker_seconds", "region_partition_searches",
    "snapshot_boundary_queries", "snapshot_deferred_stores",
    "old_route_records_before_flush", "atomic_old_calls", "atomic_retire_markers",
    "atomic_flush_worker_seconds_sum", "preflush_route_capacity_peak",
    "fused_commit_seconds", "frequency_reduce_seconds", "birth_group_fill_seconds",
    "aa_sort_seconds", "aa_sort_elided_batches", "aa_sort_elided_positions",
    "ordered_birth_reversal_segments", "ordered_birth_reversal_positions",
)


def main():
    rows = [json.loads(line) for line in (OUT / "screen.jsonl").read_text().splitlines()]
    assert len(rows) == 56 and all(row["full_trace_match"] for row in rows)
    groups = {}
    for row in rows:
        key = (row["case_id"], row["mode_label"], row["workers"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (case, mode, workers), group in sorted(groups.items()):
        group.sort(key=lambda row: row["repeat"])
        assert len(group) == (1 if case != "single-piece-ab-65536" and mode.startswith("ordered_") else 2)
        record = {"case_id": case, "mode": mode, "workers": workers,
                  "repeats": [row["repeat"] for row in group],
                  "raw_call_seconds": [row["call_seconds"] for row in group],
                  "raw_call_cpu_seconds": [row["call_cpu_seconds"] for row in group],
                  "raw_train_vm_hwm_mib": [row["train_vm_hwm_mib"] for row in group],
                  "fingerprint": group[0]["fingerprint"]}
        for metric in METRICS:
            if all(metric in row for row in group):
                record[f"median_{metric}"] = median(row[metric] for row in group)
        record["median_occupied_cores"] = record["median_call_cpu_seconds"] / record["median_call_seconds"]
        if "median_region_non_aa_posting_visits" in record:
            lower = record["median_region_sum_visit_makespan_lower_bound"]
            record["static_non_aa_visit_ratio"] = (record["median_region_non_aa_posting_visits"] / lower if lower else None)
        summary.append(record)
    lookup = {(record["case_id"], record["mode"], record["workers"]): record for record in summary}
    for record in summary:
        case, mode, workers = record["case_id"], record["mode"], record["workers"]
        if case == "single-piece-ab-65536":
            continue
        serial = lookup[(case, "serial_cf32_ahash_checked", 1)]
        if workers == 4:
            record["fair_serial_over_w4"] = serial["median_call_seconds"] / record["median_call_seconds"]
            one = lookup.get((case, mode, 1))
            if one:
                record["self_w1_over_w4"] = one["median_call_seconds"] / record["median_call_seconds"]
                record["cpu_inflation_c4_over_c1"] = record["median_call_cpu_seconds"] / one["median_call_cpu_seconds"]
    report = {"status": "passed", "rows": len(rows), "groups": summary,
              "aggregation": "separate medians of wall, process CPU, HWM and each metric; n=2 median is arithmetic midpoint",
              "interpretation": "n=2 gives a range, not a confidence interval; static load ratio assumes equal visit cost"}
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"groups": len(summary), "rows": len(rows)}))


if __name__ == "__main__":
    main()

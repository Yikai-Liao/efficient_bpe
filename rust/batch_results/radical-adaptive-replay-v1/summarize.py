"""Archive medians, raw repetitions, and explicitly separated speedup ratios."""

import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent


def summarize(directory=OUT):
    rows = [json.loads(line) for line in (directory / "screen.jsonl").read_text().splitlines()]
    assert all(row["full_trace_match"] for row in rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["case_id"], row["mode_label"], row["workers"]), []).append(row)
    summary = []
    for (case, mode, workers), group in sorted(groups.items()):
        group.sort(key=lambda row: row["repeat"])
        assert len(group) == 2
        record = {"case_id": case, "mode": mode, "workers": workers,
                  "raw_call_seconds": [row["call_seconds"] for row in group],
                  "raw_call_cpu_seconds": [row["call_cpu_seconds"] for row in group],
                  "raw_train_vm_hwm_mib": [row["train_vm_hwm_mib"] for row in group],
                  "fingerprint": group[0]["fingerprint"]}
        for field in group[0]:
            if (field.endswith("seconds") or field.startswith(("cut_", "replay_", "region_"))
                    or field in ("train_vm_hwm_mib", "posting_visits", "actual_merges",
                                 "stored_born_postings", "generated_birth_records")):
                if all(isinstance(row.get(field), (int, float)) and
                       not isinstance(row[field], bool) for row in group):
                    record["median_" + field] = median(row[field] for row in group)
        record["median_occupied_cores"] = (record["median_call_cpu_seconds"] /
                                           record["median_call_seconds"])
        summary.append(record)
    lookup = {(row["case_id"], row["mode"], row["workers"]): row for row in summary}
    for row in summary:
        if row["workers"] != 4:
            continue
        case, mode = row["case_id"], row["mode"]
        serial = lookup.get((case, "serial_reference", 1))
        one = lookup.get((case, mode, 1))
        if serial:
            row["fair_serial_over_w4"] = serial["median_call_seconds"] / row["median_call_seconds"]
        if one:
            row["self_w1_over_w4"] = one["median_call_seconds"] / row["median_call_seconds"]
            row["cpu_inflation_c4_over_c1"] = row["median_call_cpu_seconds"] / one["median_call_cpu_seconds"]
    report = {"status": "passed", "rows": len(rows), "groups": summary,
              "aggregation": "Separate metric medians; n=2 median is arithmetic midpoint.",
              "interpretation": "Raw ranges are not confidence intervals. Cross-family times do not isolate a policy."}
    with (directory / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "groups": len(summary)}))


if __name__ == "__main__":
    summarize()

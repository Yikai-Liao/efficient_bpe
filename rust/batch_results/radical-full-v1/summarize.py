"""Summarize the completed full matrix, with raw repetitions and clear baselines."""

import json
from pathlib import Path
from statistics import median, mean, pstdev

OUT = Path(__file__).resolve().parent


def main():
    rows = [json.loads(line) for line in (OUT / "measurements.jsonl").read_text().splitlines()]
    warmups = [json.loads(line) for line in (OUT / "warmups.jsonl").read_text().splitlines()]
    fixtures = {row["case_id"]: row for row in json.loads((OUT / "fixtures.json").read_text())}
    references = json.loads((OUT / "references.json").read_text())
    assert len(rows) == 1000 and len(warmups) == 200
    assert json.loads((OUT / "progress.json").read_text())["status"] == "passed"
    groups = {}
    for row in rows:
        assert row["full_training_fingerprint_match"]
        assert row["fingerprint"] == references[row["case_id"]]["fingerprint"]
        groups.setdefault((row["case_id"], row["mode"], row["workers"]), []).append(row)
    assert len(groups) == 200
    summary = []
    for (case, mode, workers), group in sorted(groups.items()):
        group.sort(key=lambda row: row["repeat"])
        assert [row["repeat"] for row in group] == [1, 2, 3, 4, 5]
        fixture = fixtures[case]
        times = [row["call_seconds"] for row in group]
        record = {"case_id": case, "category": fixture["category"], "mode": mode,
                  "workers": workers, "actual_rules": group[0]["rules"],
                  "input_bytes": fixture["input_bytes"], "corpus_positions": fixture["corpus_positions"],
                  "fingerprint": group[0]["fingerprint"],
                  "raw_call_seconds": times,
                  "raw_call_cpu_seconds": [row["call_cpu_seconds"] for row in group],
                  "raw_train_vm_hwm_mib": [row["train_vm_hwm_mib"] for row in group],
                  "min_call_seconds": min(times), "max_call_seconds": max(times),
                  "relative_mad": median(abs(value - median(times)) for value in times) / median(times),
                  "coefficient_of_variation": pstdev(times) / mean(times),
                  "max_sampled_swap_kib": max(row["sampled_peak_vm_swap_kib"] for row in group),
                  "sum_process_major_faults": sum(row["process_major_faults"] for row in group)}
        for field in group[0]:
            if (field.endswith("seconds") or field.startswith(("cut_", "replay_", "region_"))
                    or field in ("train_vm_hwm_mib", "posting_visits", "actual_merges",
                                 "stored_born_postings", "generated_birth_records")):
                if all(isinstance(row.get(field), (int, float)) and
                       not isinstance(row[field], bool) for row in group):
                    record["median_" + field] = median(row[field] for row in group)
        record["median_occupied_cores"] = record["median_call_cpu_seconds"] / record["median_call_seconds"]
        record["source_mib_per_second"] = fixture["input_bytes"] / (1 << 20) / record["median_call_seconds"]
        record["initial_symbols_per_second"] = (fixture["corpus_positions"] - 2) / record["median_call_seconds"]
        summary.append(record)
    lookup = {(row["case_id"], row["mode"], row["workers"]): row for row in summary}
    fastest_serial = {
        case: min((row for row in summary if row["case_id"] == case and row["mode"].startswith("serial_")),
                  key=lambda row: row["median_call_seconds"])
        for case in fixtures}
    fastest_w4 = {
        case: min((row for row in summary if row["case_id"] == case and row["workers"] == 4),
                  key=lambda row: row["median_call_seconds"])
        for case in fixtures}
    for row in summary:
        case, mode, workers = row["case_id"], row["mode"], row["workers"]
        serial = fastest_serial[case]
        row["fastest_serial_mode"] = serial["mode"]
        row["fastest_serial_over_this"] = serial["median_call_seconds"] / row["median_call_seconds"]
        matched = lookup[case, "serial_cf32_checked", 1]
        row["cf32_checked_over_this"] = matched["median_call_seconds"] / row["median_call_seconds"]
        if workers > 1:
            one = lookup[case, mode, 1]
            row["self_w1_over_wp"] = one["median_call_seconds"] / row["median_call_seconds"]
            row["cpu_inflation_cp_over_c1"] = row["median_call_cpu_seconds"] / one["median_call_cpu_seconds"]
            row["raw_same_block_w1_over_wp"] = [a / b for a, b in zip(one["raw_call_seconds"], row["raw_call_seconds"])]
        # Pareto dominance only within one case and worker count.
        row["dominated_in_wall_and_hwm"] = any(
            other["case_id"] == case and other["workers"] == workers and other["mode"] != mode
            and other["median_call_seconds"] <= row["median_call_seconds"]
            and other["median_train_vm_hwm_mib"] <= row["median_train_vm_hwm_mib"]
            and (other["median_call_seconds"] < row["median_call_seconds"]
                 or other["median_train_vm_hwm_mib"] < row["median_train_vm_hwm_mib"])
            for other in summary)
    report = {"status": "passed", "measurement_rows": 1000, "warmup_rows": 200,
              "groups": summary,
              "fastest_serial_by_case": {case: row["mode"] for case, row in fastest_serial.items()},
              "fastest_w4_by_case": {case: row["mode"] for case, row in fastest_w4.items()},
              "aggregation": "Five raw repetitions and separate medians. Ratios of medians, with block ratios also retained. Fastest serial is selected among four serial controls in this window; rankings are observed, not significance claims."}
    with (OUT / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"groups": len(summary), "measurement_rows": len(rows),
                      "fastest_w4_by_case": report["fastest_w4_by_case"]}))


if __name__ == "__main__":
    main()

"""Summarize n=2 4 MiB results using per-field medians."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
FIELDS = ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib", "plan_seconds",
          "apply_seconds", "frequency_reduce_seconds", "birth_group_fill_seconds",
          "init_seconds", "select_seconds", "fused_non_aa_batches",
          "fused_non_aa_merges", "non_aa_start_bytes_peak_proxy",
          "peak_task_starts", "decoder_zero_rereads")


def main():
    rows = [json.loads(line) for line in (OUT / "screen.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"], row["version"], row["workers"]].append(row)
    summary = []
    for (case, version, workers), records in sorted(groups.items()):
        assert len(records) == 2 and {record["repetition"] for record in records} == {0, 1}
        item = {"case_id": case, "version": version, "workers": workers,
                "repetitions": 2, "call_seconds_range": [min(record["call_seconds"] for record in records),
                                                        max(record["call_seconds"] for record in records)]}
        item.update({field: median(record[field] for record in records)
                     for field in FIELDS if all(field in record for record in records)})
        item["mean_occupied_cores_from_medians"] = item["call_cpu_seconds"] / item["call_seconds"]
        summary.append(item)
    with (OUT / "summary.json").open("x") as output:
        json.dump({"rows": len(rows), "groups": len(summary),
                   "aggregation": "each field's two observations are medianed independently",
                   "observations": summary}, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(rows), "groups": len(summary)}))


if __name__ == "__main__":
    main()

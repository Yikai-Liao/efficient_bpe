"""Build a compact, lossless-indexed summary from the frozen quick raw rows."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
FIELDS = ("call_seconds", "call_cpu_seconds", "train_vm_hwm_mib", "plan_seconds",
          "apply_seconds", "frequency_reduce_seconds", "birth_group_fill_seconds",
          "fused_non_aa_batches", "fused_non_aa_merges", "decoder_zero_rereads",
          "non_aa_start_positions_peak", "non_aa_start_bytes_peak_proxy",
          "peak_task_starts", "aa_bitmap_rounds", "aa_bitmap_fallback_rounds",
          "aa_bitmap_peak_bytes", "aa_bitmap_valid_edges", "aa_bitmap_chunks",
          "peak_aa_plan_capacity_bytes", "aa_bitmap_init_seconds",
          "aa_bitmap_scatter_seconds", "aa_bitmap_summary_seconds",
          "aa_bitmap_prefix_seconds", "aa_bitmap_route_seconds",
          "aa_bitmap_apply_seconds")


def main():
    rows = [json.loads(line) for line in (OUT / "quick.jsonl").read_text().splitlines()]
    summary = [{"case_id": row["case_id"], "kind": row["kind"],
                "version": row["version"], "workers": row["workers"],
                **{name: row[name] for name in FIELDS if name in row}}
               for row in sorted(rows, key=lambda row: (row["case_id"], row["kind"],
                                                     row["version"], row["workers"]))]
    with (OUT / "summary.json").open("x") as output:
        json.dump({"rows": len(summary), "observations": summary}, output, indent=2)
        output.write("\n")
    print(json.dumps({"rows": len(summary)}))


if __name__ == "__main__":
    main()

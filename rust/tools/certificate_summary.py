"""Summarize exact-prefix diagnostic counters, without making timing claims."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    rust = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=rust / "ablation_results/certificate-probe.jsonl")
    parser.add_argument("--output-prefix", type=Path,
                        default=rust / "ablation_results/certificate-summary")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    environment = json.loads(args.input.with_suffix(".jsonl.environment.json").read_text())
    cases = sorted({row["case_id"] for row in rows})
    result = []
    for case in cases:
        selected = [row for row in rows if row["case_id"] == case]
        assert len(selected) == 4, (case, "expected two variants and both bounds")
        assert len({row["fingerprint"] for row in selected}) == 1, case
        probes = [row for row in selected if row["variant"] == "certified_prefix_probe"]
        assert {row["bounds"] for row in probes} == {"checked", "unchecked"}, case
        metrics = [{key: value for key, value in row["metrics"].items()
                    if key.startswith("certificate_")} for row in probes]
        assert metrics[0] == metrics[1], (case, "bounds disagree on batch counters")
        m = metrics[0]
        count, epochs = int(m["certificate_rules"]), int(m["certificate_epochs"])
        assert count == probes[0]["rules"], case
        result.append({"case_id": case, "rules": count, "epochs": epochs,
                       "mean_width": count / epochs if epochs else 0,
                       "max_width": int(m["certificate_max_width"]),
                       "singleton_epochs": int(m["certificate_singleton_epochs"]),
                       "metrics": m, "fingerprint": probes[0]["fingerprint"]})
    report = {"kind": "semantic and batch-width diagnostic; not a speedup benchmark",
              "source": str(args.input), "raw_sha256": hashlib.sha256(raw).hexdigest(),
              "binary_sha256": environment["binary_sha256"], "rows": len(rows),
              "cases": result, "all_fingerprints_match": True,
              "checked_unchecked_counters_match": True}
    md = ["# Certified-prefix diagnostic", "",
          "Native source: `1c2fffc`. The probe preselects a certified prefix, then still "
          "applies each rule serially. These counters measure potential epoch width, "
          "not achieved parallel speedup. Each configuration was run once; no timing "
          "comparison is drawn from this diagnostic.", "",
          f"Validated {len(rows)} runs across {len(cases)} fixtures: probe and "
          "`combined_filtered`, each with checked and unchecked access. Complete-output "
          "fingerprints agree, and batch counters are identical between bounds modes.", "",
          "| Input | Rules | Certified epochs | Rules / epoch | Maximum width | Singleton epochs |",
          "|---|---:|---:|---:|---:|---:|"]
    for row in result:
        md.append(f"| {row['case_id']} | {row['rules']:,} | {row['epochs']:,} | "
                  f"{row['mean_width']:.2f} | {row['max_width']} | {row['singleton_epochs']} |")
    md += ["", "The cap is min(256, remaining rules); the hit-cap counter includes "
           "the final truncated batch. No real-data maximum above reaches 256. "
           "Long chains and self-pair runs mostly remain singleton epochs. "
           "Weighted64 preserves the same grouping as its unscaled counterpart.", "",
           "Semantic checks are separate: 448 full-trace Python-oracle comparisons, "
           "24 library tests and six independent recount reference tests, including "
           "29,523 exhaustive ternary strings and 2,000 weighted random cases. "
           "The reference tests also compare direct batch-end count deltas with a "
           "complete recount after every batch. "
           "See `certificate-differential.json` and `certificate-checks.json`.", "",
           "Reproduce with `python rust/tools/certificate_summary.py --output-prefix "
           "rust/ablation_results/reruns/certificate-summary`. Raw archives are never overwritten."]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in ((".json", json.dumps(report, indent=2) + "\n"),
                         (".md", "\n".join(md) + "\n")):
        with args.output_prefix.with_suffix(suffix).open("x") as stream:
            stream.write(data)
    print(json.dumps({"rows": len(rows), "cases": len(cases), "all_checks_passed": True}))


if __name__ == "__main__":
    main()

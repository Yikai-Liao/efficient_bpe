"""Bounded full-trace checks for the promoted CLI and its default settings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ablation_differential import ROOT, RUST, make_cases, naive, prepared_wire


def observe(binary, fixture, rules, minimum, *extra):
    result = subprocess.run([str(binary), "--input", str(fixture), "--rules", str(rules),
                             "--min-frequency", str(minimum), *map(str, extra)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=RUST / "target/release/ebpe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-fixtures", action="store_true",
                        help="Also check the two existing 16 MiB reference fingerprints")
    args = parser.parse_args()
    count = 0
    with tempfile.TemporaryDirectory(prefix="bpe-primary-check-") as name:
        temp = Path(name)
        for case, prepared, minimum, rules in make_cases(8):
            fixture = temp / "input.json"
            fixture.write_text(json.dumps(prepared_wire(prepared)))
            merges, final = naive(prepared, rules, minimum)
            expected = {"merges": [list(r) for r in merges], "final": final}
            for workers in (1, 4):
                trace = temp / "trace.json"
                row = observe(args.binary, fixture, rules, minimum,
                              "--workers", workers, "--trace", trace)
                assert json.loads(trace.read_text()) == expected, (case, workers)
                assert row["integer_hash"] == "ahash" and row["heap_policy"] == "lazy"
                count += 1
        # Exercise actual CLI defaults and preservation of CRLF/Unicode by the
        # public preparation command, rather than a helper-derived fixture.
        source = temp / "text.txt"
        text = "ab\r\n界🙂 ab"
        source.write_bytes(text.encode("utf-8"))
        fixture = temp / "text.json"
        subprocess.run([sys.executable, str(RUST / "tools/prepare_text.py"), str(source),
                        "--output", str(fixture)], check=True, capture_output=True)
        prepared = json.loads(fixture.read_text())
        symbols = prepared["text_metadata"]["symbols"]
        assert "".join(symbols[token] for token in prepared["corpus"][1:-1]) == text
        expected_merges, expected_final = naive(
            tuple(prepared[key] for key in ("corpus", "initial_lengths", "pivots", "weights")),
            16, 1)
        trace = temp / "default-trace.json"
        row = observe(args.binary, fixture, 16, 1, "--trace", trace)
        assert 1 <= row["workers"] <= min(4, len(os.sched_getaffinity(0)))
        assert row["integer_hash"] == "ahash" and row["heap_policy"] == "lazy"
        assert json.loads(trace.read_text()) == {
            "merges": [list(r) for r in expected_merges], "final": expected_final}
        count += 1
    large_rows = []
    if args.full_fixtures:
        archive = RUST / "batch_results/radical-full-v1"
        fixtures = {r["case_id"]: r for r in json.loads((archive / "fixtures.json").read_text())}
        refs = json.loads((archive / "references.json").read_text())
        for case in ("en-16m-continuous", "zh-16m-continuous"):
            row = observe(args.binary, RUST / fixtures[case]["file"], 32000, 2, "--workers", 4)
            assert row["fixture_sha256"] == fixtures[case]["fixture_sha256"]
            assert row["fingerprint"] == refs[case]["fingerprint"]
            assert row["rules"] == refs[case]["actual_rules"] == 32000
            row["case_id"] = case
            large_rows.append(row)
            print(case, "complete fingerprint matched", flush=True)
    report = {"status": "passed", "naive_complete_trace_runs": count,
              "default_cli_and_text_preparation": "passed", "large_fixture_checks": large_rows,
              "large_checks_are_correctness_only": True,
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "large_fixture_checks"}))


if __name__ == "__main__":
    main()

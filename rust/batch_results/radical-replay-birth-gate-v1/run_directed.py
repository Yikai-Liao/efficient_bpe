"""Independent traces and physical/semantic accounting for birth replay."""

from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rust/tools"))
from ablation_differential import naive, prepared_wire  # noqa: E402


def weighted_pieces(pieces):
    alphabet = sorted({character for word, _ in pieces for character in word})
    ids = {character: index + 1 for index, character in enumerate(alphabet)}
    corpus = array("I", [0])
    pivots, weights = [], []
    for word, weight in pieces:
        pivots.append(len(corpus))
        weights.append(weight)
        corpus.extend(ids[character] for character in word)
        corpus.append(0)
    return corpus, [1] * (len(alphabet) + 1), pivots, weights


def main():
    config = json.loads((OUT / "oracle-config.json").read_text())
    binary = ROOT / config["binary"]
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == config["binary_sha256"]
    cases = [
        ("aa-cross-cuts", [("a" * 513, 3)], 2, 12, 4),
        ("adjacent-fresh-fresh", [("abcd" * 257, 2)], 2, 15, 4),
        ("weighted-filtered-births", [("a" * 64, 1), ("abcd" * 32, 7), ("abcx", 1)], 7, 20, 4),
        ("large-u64-weights", [("ab" * 128, 1 << 40), ("abx" * 64, 3 << 40)], 1 << 40, 15, 4),
        ("long-token-cross-cuts", [("a" * 512 + "bc", 3)], 2, 12, 4),
        ("more-workers-than-positions", [("ab", 3)], 1, 6, 16),
        ("single-region-raw-head-check", [("abcdabca" * 16, 2)], 2, 15, 1),
        ("tagged-domain-fallback", [("a" * 64, 5)], 1, 1 << 31, 4),
    ]
    records = []
    with tempfile.TemporaryDirectory(prefix="replay-directed-") as temp_name:
        temp = Path(temp_name)
        for index, (name, pieces, minimum, rules, workers) in enumerate(cases):
            prepared = weighted_pieces(pieces)
            fixture = temp / f"fixture-{index}.json"
            trace = temp / f"trace-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            merges, final = naive(prepared, rules, minimum)
            expected = {"merges": [list(row) for row in merges], "final": final}
            fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
            pair = {}
            for mode in ("chain", "replay"):
                command = [str(binary), "--input", str(fixture), "--workers", str(workers),
                           "--rules", str(rules), "--min-frequency", str(minimum),
                           "--chunk-size", "7", "--heap-policy", "lazy", "--integer-hash", "std",
                           "--endpoint-plan", "tagged-fused", "--region-mode", "region",
                           "--regions-per-worker", "1", "--posting-order", "global",
                           "--birth-fill", mode, "--trace", str(trace)]
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                result = json.loads(process.stdout.strip().splitlines()[-1])
                assert json.loads(trace.read_text()) == expected, (name, mode, "trace")
                assert result["fingerprint"] == fingerprint, (name, mode, "fingerprint")
                effective = "chain" if result["endpoint_domain_fallback"] else mode
                assert result["birth_fill_effective"] == effective
                if effective == "replay":
                    assert all(result[field] == 0 for field in
                               ("replay_birth_nodes_actual", "grouped_birth_nodes",
                                "region_peak_route_born_capacity"))
                    assert result["replay_filled_positions"] == result["stored_born_postings"]
                    assert result["replay_zero_initialized_bytes"] == 4 * result["replay_filled_positions"]
                pair[mode] = result
                records.append({"case": name, "mode": mode, "workers": workers,
                                "full_trace_match": True,
                                **{field: result[field] for field in
                                   ("birth_fill_effective", "endpoint_domain_fallback",
                                    "region_cross_births", "actual_merges", "posting_visits",
                                    "generated_birth_records", "stored_born_postings",
                                    "replay_birth_nodes_actual", "replay_posting_visits",
                                    "replay_filled_positions", "replay_zero_initialized_bytes")}})
            for field in ("actual_merges", "generated_birth_records", "stored_born_postings"):
                assert pair["chain"][field] == pair["replay"][field], (name, field)
            assert (pair["replay"]["posting_visits"] ==
                    pair["chain"]["posting_visits"] + pair["replay"]["replay_posting_visits"])
    assert len(records) == 16
    assert any(row["region_cross_births"] > 0 for row in records)
    assert any(row["replay_filled_positions"] > 0 for row in records)
    assert any(row["endpoint_domain_fallback"] for row in records)
    report = {"status": "passed", "directed_fulltrace_matches": len(records),
              "binary_sha256": config["binary_sha256"], "records": records}
    with (OUT / "directed.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()

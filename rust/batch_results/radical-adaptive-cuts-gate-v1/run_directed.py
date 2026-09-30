"""Directed full-trace and cut-activation checks on the frozen adaptive binary."""

from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-adaptive-cuts-gate-v1/radical-owned-adaptive-cuts"
sys.path.insert(0, str(RUST / "tools"))
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
    expected_sha = json.loads((OUT / "oracle-config.json").read_text())["binary_sha256"]
    assert hashlib.sha256(BIN.read_bytes()).hexdigest() == expected_sha
    cases = [
        ("dominant-ab", weighted_pieces([("ab" * 2048 + "cd" * 32, 1)]), 2, 12, 4, 64),
        ("two-disjoint-hot-pairs", weighted_pieces([("ab" * 1024 + "cd" * 1024, 1)]), 2, 12, 4, 7),
        ("single-region-bypass", weighted_pieces([("abcd" * 128, 2)]), 2, 12, 1, 9),
        ("more-workers-than-positions", weighted_pieces([("ab", 3)]), 1, 6, 16, 1),
        ("weighted-aa-and-non-aa", weighted_pieces([("a" * 128, 3), ("abcd" * 64, 2)]), 2, 15, 4, 5),
        ("long-token-cross-cuts", weighted_pieces([("a" * 512 + "bc", 3)]), 2, 9, 4, 17),
        ("tagged-domain-fallback", weighted_pieces([("a" * 64, 5)]), 1, 1 << 31, 4, 4),
    ]
    records = []
    with tempfile.TemporaryDirectory(prefix="adaptive-directed-") as temp_name:
        temp = Path(temp_name)
        for index, (name, prepared, minimum, rules, workers, chunk) in enumerate(cases):
            fixture = temp / f"fixture-{index}.json"
            trace = temp / f"trace-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            merges, final = naive(prepared, rules, minimum)
            expected = {"merges": [list(row) for row in merges], "final": final}
            fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
            for mode in ("fixed", "adaptive"):
                command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
                           "--chunk-size", str(chunk), "--rules", str(rules),
                           "--min-frequency", str(minimum), "--heap-policy", "lazy",
                           "--integer-hash", "std", "--endpoint-plan", "tagged-fused",
                           "--region-mode", "region", "--regions-per-worker", "1",
                           "--posting-order", "global", "--cut-policy", mode,
                           "--trace", str(trace)]
                process = subprocess.run(command, capture_output=True, text=True)
                if process.returncode:
                    raise RuntimeError((name, mode, process.stderr, process.stdout))
                result = json.loads(process.stdout.strip().splitlines()[-1])
                assert json.loads(trace.read_text()) == expected, (name, mode, "trace")
                assert result["fingerprint"] == fingerprint, (name, mode, "fingerprint")
                if mode == "adaptive" and not result["endpoint_domain_fallback"]:
                    assert result["cut_selected_max_visits_sum"] <= result["cut_fixed_max_visits_sum"]
                records.append({"case": name, "mode": mode, "workers": workers,
                                "endpoint_domain_fallback": result["endpoint_domain_fallback"],
                                **{field: result[field] for field in (
                                    "cut_non_aa_batches", "cut_adaptive_chosen_batches",
                                    "cut_longest_candidates", "cut_sampled_candidates",
                                    "cut_budget_fallbacks", "cut_small_work_fallbacks",
                                    "cut_single_region_bypasses", "cut_fixed_max_visits_sum",
                                    "cut_selected_max_visits_sum", "region_count_effective")}})
    experiments = [record for record in records if record["mode"] == "adaptive"]
    assert len(records) == 14
    assert any(record["cut_longest_candidates"] + record["cut_sampled_candidates"] > 0
               for record in experiments)
    assert any(record["cut_single_region_bypasses"] > 0 for record in experiments)
    assert any(record["endpoint_domain_fallback"] for record in experiments)
    report = {"status": "passed", "directed_fulltrace_matches": len(records),
              "binary_sha256": expected_sha, "directed": records}
    with (OUT / "directed.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "passed", "directed_fulltrace_matches": len(records)}))


if __name__ == "__main__":
    main()

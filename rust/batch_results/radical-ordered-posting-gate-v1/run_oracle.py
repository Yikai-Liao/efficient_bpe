"""Full-trace oracle for region-ordered versus globally sorted postings."""

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
BIN = RUST / "target/reruns/radical-ordered-posting-gate-v1/radical-owned-ordered-posting"
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402


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


def expected_for(prepared, rules, minimum):
    merges, final = naive(prepared, rules, minimum)
    return ({"merges": [list(row) for row in merges], "final": final},
            hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest())


def check(case, fixture, expected, fingerprint, minimum, rules, order, mode,
          hasher, workers, factor, chunk, trace):
    command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", str(chunk), "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", "lazy",
               "--integer-hash", hasher, "--endpoint-plan", "tagged-fused",
               "--region-mode", mode, "--regions-per-worker", str(factor),
               "--posting-order", order, "--trace", str(trace)]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError((case, order, mode, workers, process.stderr, process.stdout))
    result = json.loads(process.stdout.strip().splitlines()[-1])
    if json.loads(trace.read_text()) != expected or result["fingerprint"] != fingerprint:
        raise AssertionError((case, order, mode, hasher, workers, "full trace differs"))
    if not result["endpoint_domain_fallback"]:
        assert result["posting_order_effective"] == order
    return result


def main():
    assert BIN.is_file()
    standard = 0
    directed = []
    cases = make_cases(8)
    with tempfile.TemporaryDirectory(prefix="ordered-posting-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for index, (case, prepared, minimum, rules) in enumerate(cases):
            fixture = temp / f"standard-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint = expected_for(prepared, rules, minimum)
            for order in ("region", "global"):
                for workers in (1, 4):
                    check(case, fixture, expected, fingerprint, minimum, rules,
                          order, "region", "ahash", workers, 1, 4096, trace)
                    standard += 1
        specials = [
            ("adjacent-fresh", weighted_pieces([("abcd" * 24, 3)]), 1, 9, 4, 3),
            ("aa-to-non-aa", weighted_pieces([("a" * 64, 3), ("abcd" * 16, 2)]), 1, 15, 4, 5),
            ("long-multicut", weighted_pieces([("a" * 512 + "bc", 3)]), 1, 9, 4, 17),
            ("remote-endpoint", weighted_pieces([("ab" * 129, 1)]), 1, 2, 4, 3),
            ("domain-fallback", weighted_pieces([("a" * 64, 5)]), 1, 1 << 31, 4, 4),
        ]
        for index, (case, prepared, minimum, rules, workers, chunk) in enumerate(specials):
            fixture = temp / f"directed-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint = expected_for(prepared, rules, minimum)
            for order in ("region", "global"):
                result = check(case, fixture, expected, fingerprint, minimum, rules,
                               order, "snapshot", "std", workers, 4, chunk, trace)
                directed.append({"case": case, "order": order,
                                 "posting_order_effective": result["posting_order_effective"],
                                 "aa_sort_elided_batches": result["aa_sort_elided_batches"],
                                 "aa_sort_elided_positions": result["aa_sort_elided_positions"],
                                 "ordered_birth_reversal_segments": result["ordered_birth_reversal_segments"],
                                 "ordered_birth_reversal_positions": result["ordered_birth_reversal_positions"],
                                 "snapshot_deferred_stores": result["snapshot_deferred_stores"],
                                 "endpoint_domain_fallback": result["endpoint_domain_fallback"]})
                if case == "domain-fallback":
                    assert result["endpoint_domain_fallback"]
                    assert result["posting_order_effective"] == "region"
                if order == "global" and case == "aa-to-non-aa":
                    assert result["aa_sort_elided_batches"] > 0
                    assert result["ordered_birth_reversal_segments"] > 0
                if case == "remote-endpoint":
                    assert result["snapshot_deferred_stores"] > 0
        # Global ordering is intentionally unavailable to a dynamic planner.
        fixture = temp / "invalid-global-dynamic.json"
        fixture.write_text(json.dumps(prepared_wire(specials[0][1]), separators=(",", ":")))
        invalid = [str(BIN), "--input", str(fixture), "--workers", "4",
                   "--endpoint-plan", "tagged-fused", "--region-mode", "dynamic",
                   "--posting-order", "global"]
        rejected = subprocess.run(invalid, capture_output=True, text=True)
        assert rejected.returncode != 0
    report = {"status": "passed", "standard_cases": len(cases),
              "standard_fulltrace_matches": standard,
              "directed_fulltrace_matches": len(directed), "directed": directed,
              "invalid_global_dynamic_rejections": 1,
              "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
              "oracle": "Python naive full recount; complete merge trace and final tokens"}
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "directed"}))


if __name__ == "__main__":
    main()

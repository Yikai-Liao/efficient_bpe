"""Complete-trace oracle for dynamic versus physical-region fused planning."""

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
BIN = RUST / "target/reruns/radical-region-fused-gate-v1/radical-owned-region-fused"
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
    trace = {"merges": [list(row) for row in merges], "final": final}
    fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
    lengths = list(prepared[1])
    for a, b, _ in merges:
        lengths.append(lengths[a] + lengths[b])
    return trace, fingerprint, max(lengths)


def check(case, fixture, expected, fingerprint, minimum, rules, mode, hasher,
          workers, chunk, trace):
    command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", str(chunk), "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", "lazy",
               "--integer-hash", hasher, "--endpoint-plan", "tagged-fused",
               "--region-mode", mode, "--trace", str(trace)]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError((case, mode, hasher, workers, process.stderr, process.stdout))
    result = json.loads(process.stdout.strip().splitlines()[-1])
    if json.loads(trace.read_text()) != expected or result["fingerprint"] != fingerprint:
        raise AssertionError((case, mode, hasher, workers, "full trace differs"))
    return result


def main():
    if not BIN.is_file():
        raise FileNotFoundError(BIN)
    compared = 0
    directed = []
    with tempfile.TemporaryDirectory(prefix="region-fused-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for index, (case, prepared, minimum, rules) in enumerate(make_cases(8)):
            fixture = temp / f"standard-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint, _ = expected_for(prepared, rules, minimum)
            for mode in ("dynamic", "region"):
                for hasher in ("std", "ahash"):
                    for workers in (1, 4):
                        check(case, fixture, expected, fingerprint, minimum, rules,
                              mode, hasher, workers, 4096, trace)
                        compared += 1

        special = [
            ("long-cross-cut", weighted_pieces([("a" * 512 + "bc", 3)]), 1, 9, 4, 17),
            ("weighted-empty-cuts", weighted_pieces([("a" * 9, 7), ("bc", 2)]), 1, 12, 16, 3),
            ("empty-all-cuts", (array("I", [0]), [1], [], []), 1, 12, 16, 3),
            ("domain-fallback", weighted_pieces([("a" * 64, 5)]), 1, 1 << 31, 4, 4),
        ]
        for index, (case, prepared, minimum, rules, workers, chunk) in enumerate(special):
            fixture = temp / f"special-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint, max_length = expected_for(prepared, rules, minimum)
            for mode in ("dynamic", "region"):
                result = check(case, fixture, expected, fingerprint, minimum, rules,
                               mode, "ahash", workers, chunk, trace)
                directed.append({"case": case, "mode": mode, "workers": workers,
                                 "max_token_length": max_length,
                                 "region_cross_births": result["region_cross_births"],
                                 "effective_region_mode": result["region_mode_effective"],
                                 "effective_endpoint_plan": result["endpoint_plan_effective"],
                                 "domain_fallback": result["endpoint_domain_fallback"]})
                if case == "long-cross-cut" and mode == "region":
                    assert max_length >= 256 and result["region_cross_births"] > 0
                if case == "domain-fallback":
                    assert result["endpoint_domain_fallback"]
                    assert result["region_mode_effective"] == "dynamic"
                    assert result["endpoint_plan_effective"] == "two-pass"
    report = {"status": "passed", "standard_cases": 20,
              "standard_fulltrace_matches": compared,
              "directed_fulltrace_matches": len(directed),
              "directed": directed,
              "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
              "oracle": "Python naive full recount; complete merge trace and final tokens"}
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "directed"}))


if __name__ == "__main__":
    main()

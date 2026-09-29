"""Full-trace gate for k=4 microregions and explicit frozen k=1 controls."""

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
BIN = RUST / "target/reruns/radical-region-tasks-gate-v1/radical-owned-region-tasks"
OLD = RUST / "target/reruns/radical-region-snapshot-gate-v1/radical-owned-region-snapshot"
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
          workers, factor, chunk, trace, *, old=False):
    binary = OLD if old else BIN
    command = [str(binary), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", str(chunk), "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", "lazy",
               "--integer-hash", hasher, "--endpoint-plan", "tagged-fused",
               "--region-mode", mode, "--trace", str(trace)]
    if not old:
        command.extend(("--regions-per-worker", str(factor)))
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError((case, mode, workers, factor, old, process.stderr, process.stdout))
    result = json.loads(process.stdout.strip().splitlines()[-1])
    observed = json.loads(trace.read_text())
    if observed != expected or result["fingerprint"] != fingerprint:
        raise AssertionError((case, mode, hasher, workers, factor, old, "full trace differs"))
    if not old and result["region_mode_effective"] == mode:
        assert result["region_count_effective"] == min(workers * factor, len(json.loads(fixture.read_text())["corpus"]))
    return result, observed


def main():
    assert BIN.is_file() and OLD.is_file()
    standard = 0
    old_controls = 0
    directed = []
    cases = make_cases(8)
    with tempfile.TemporaryDirectory(prefix="region-tasks-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for index, (case, prepared, minimum, rules) in enumerate(cases):
            fixture = temp / f"standard-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint, _ = expected_for(prepared, rules, minimum)
            for mode in ("region", "snapshot"):
                for workers in (1, 4):
                    check(case, fixture, expected, fingerprint, minimum, rules,
                          mode, "ahash", workers, 4, 4096, trace)
                    standard += 1
            if case in {"overlap-aaa", "weighted64", "random-000", "single-run-long"}:
                for mode in ("region", "snapshot"):
                    current, current_trace = check(case, fixture, expected, fingerprint,
                                                   minimum, rules, mode, "ahash", 4, 1, 4096, trace)
                    frozen, frozen_trace = check(case, fixture, expected, fingerprint,
                                                 minimum, rules, mode, "ahash", 4, 1, 4096, trace,
                                                 old=True)
                    assert current["fingerprint"] == frozen["fingerprint"]
                    assert current_trace == frozen_trace
                    old_controls += 1

        specials = [
            ("remote-endpoint", weighted_pieces([("ab" * 129, 1)]), 1, 2, 4, 3),
            ("long-multicut", weighted_pieces([("a" * 512 + "bc", 3)]), 1, 9, 4, 17),
            ("weighted-wn", weighted_pieces([("a" * 9, 7), ("bc", 2)]), 1, 12, 16, 3),
            ("empty-wn", (array("I", [0]), [1], [], []), 1, 12, 16, 3),
            ("aa-to-non-aa", weighted_pieces([("a" * 64, 3), ("abcd" * 16, 2)]), 1, 15, 4, 5),
            ("domain-fallback", weighted_pieces([("a" * 64, 5)]), 1, 1 << 31, 4, 4),
        ]
        for index, (case, prepared, minimum, rules, workers, chunk) in enumerate(specials):
            fixture = temp / f"directed-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint, max_length = expected_for(prepared, rules, minimum)
            for mode in ("region", "snapshot"):
                result, _ = check(case, fixture, expected, fingerprint, minimum, rules,
                                  mode, "std", workers, 4, chunk, trace)
                row = {"case": case, "mode": mode, "workers": workers,
                       "factor": 4, "positions": len(prepared[0]),
                       "effective_regions": result["region_count_effective"],
                       "max_token_length": max_length,
                       "cross_births": result["region_cross_births"],
                       "snapshot_boundary_queries": result["snapshot_boundary_queries"],
                       "snapshot_deferred_stores": result["snapshot_deferred_stores"],
                       "effective_mode": result["region_mode_effective"],
                       "endpoint_domain_fallback": result["endpoint_domain_fallback"]}
                directed.append(row)
                if case == "remote-endpoint" and mode == "snapshot":
                    assert result["snapshot_boundary_queries"] > 0
                    assert result["snapshot_deferred_stores"] > 0
                if case == "long-multicut":
                    assert max_length >= 256 and result["region_cross_births"] > 0
                if case.endswith("-wn"):
                    assert workers > len(prepared[0])
                    assert result["region_count_effective"] == len(prepared[0])
                if case == "domain-fallback":
                    assert result["endpoint_domain_fallback"]
                    assert result["region_mode_effective"] == "dynamic"
                    assert result["endpoint_plan_effective"] == "two-pass"
    report = {"status": "passed", "standard_cases": len(cases),
              "k4_standard_fulltrace_matches": standard,
              "k1_frozen_binary_control_pairs": old_controls,
              "directed_fulltrace_matches": len(directed), "directed": directed,
              "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
              "frozen_control_binary_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
              "oracle": "Python naive full recount plus explicit frozen k=1 binary comparison"}
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "directed"}))


if __name__ == "__main__":
    main()

"""Full-trace oracle for staged, direct combined, and pending owner commits."""

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
BIN = RUST / "target/reruns/radical-pending-entry-gate-v1/radical-owned-pending-entry"
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402

MODES = ("staged", "fused-direct-combined", "fused-direct-pending")


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
    return trace, fingerprint


def check(case, fixture, expected, fingerprint, minimum, rules, mode, hasher,
          workers, chunk, trace):
    command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", str(chunk), "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", "lazy",
               "--integer-hash", hasher, "--owner-commit", mode,
               "--trace", str(trace)]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError((case, mode, hasher, workers, process.stderr, process.stdout))
    result = json.loads(process.stdout.strip().splitlines()[-1])
    if json.loads(trace.read_text()) != expected or result["fingerprint"] != fingerprint:
        raise AssertionError((case, mode, hasher, workers, "full trace differs"))
    assert result["owner_commit"] == mode
    return result


def main():
    if not BIN.is_file():
        raise FileNotFoundError(BIN)
    compared = 0
    directed = []
    with tempfile.TemporaryDirectory(prefix="pending-entry-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for index, (case, prepared, minimum, rules) in enumerate(make_cases(8)):
            fixture = temp / f"standard-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint = expected_for(prepared, rules, minimum)
            for mode in MODES:
                for hasher in ("std", "ahash"):
                    for workers in (1, 4):
                        check(case, fixture, expected, fingerprint, minimum, rules,
                              mode, hasher, workers, 4096, trace)
                        compared += 1

        specials = [
            ("many-ineligible-fresh", weighted_pieces([
                ("ab" + chr(0x1000 + i), 1) for i in range(256)]), 2, 1),
            ("weighted-mixed-fresh", weighted_pieces([
                ("abc", 3), ("abc", 5), ("abd", 2)]), 4, 2),
        ]
        for index, (case, prepared, minimum, rules) in enumerate(specials):
            fixture = temp / f"special-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint = expected_for(prepared, rules, minimum)
            for mode in MODES:
                for hasher in ("std", "ahash"):
                    for workers in (1, 4):
                        result = check(case, fixture, expected, fingerprint, minimum,
                                       rules, mode, hasher, workers, 17, trace)
                        record = {"case": case, "mode": mode, "hash": hasher,
                                  "workers": workers, "fresh_unique_keys": result["fresh_unique_keys"],
                                  "fresh_eligible_keys": result["fresh_eligible_keys"],
                                  "fresh_route_visits": result["fresh_route_visits"],
                                  "owner_capacity_before_peak": result["owner_capacity_before_peak"],
                                  "owner_capacity_after_reduce_peak": result["owner_capacity_after_reduce_peak"],
                                  "owner_capacity_after_retire_peak": result["owner_capacity_after_retire_peak"],
                                  "fresh_scratch_capacity_peak": result["fresh_scratch_capacity_peak"]}
                        directed.append(record)
                        if case == "many-ineligible-fresh" and mode != "staged":
                            assert result["fresh_unique_keys"] >= 256
                            assert result["fresh_eligible_keys"] == 0
                        if case == "weighted-mixed-fresh" and mode != "staged":
                            assert result["fresh_unique_keys"] >= 2
                            assert result["fresh_eligible_keys"] >= 1
    report = {"status": "passed", "standard_cases": 20,
              "standard_fulltrace_matches": compared,
              "directed_fulltrace_matches": len(directed),
              "directed": directed,
              "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
              "oracle": "Python naive full recount; complete merge trace and final tokens",
              "collision_test": "Rust lib test pending_counts_survive_collisions_and_foreign_only_birth"}
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "directed"}))


if __name__ == "__main__":
    main()

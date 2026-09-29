"""Complete-trace oracle for atomic versus word-cached AA bitmap scatter."""

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
BIN = RUST / "target/reruns/radical-aa-bitmap-cache-gate-v1/radical-owned-aa-bitmap-cache"
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402


def expected_for(prepared, rules, minimum):
    merges, final = naive(prepared, rules, minimum)
    trace = {"merges": [list(row) for row in merges], "final": final}
    fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
    lengths = list(prepared[1])
    for a, b, _ in merges:
        lengths.append(lengths[a] + lengths[b])
    return trace, fingerprint, max(lengths)


def weighted_aa():
    corpus = array("I", [0])
    pivots, weights = [], []
    for length, weight in ((4095, 2), (4096, 5), (4097, 3)):
        pivots.append(len(corpus))
        weights.append(weight)
        corpus.extend([1] * length)
        corpus.append(0)
    return corpus, [1, 1], pivots, weights


def check(case, fixture, expected, fingerprint, minimum, rules, mode, hasher, workers, trace):
    command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", "4096", "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", "lazy",
               "--integer-hash", hasher, "--aa-order", "bitmap-adaptive",
               "--bitmap-scatter", mode, "--trace", str(trace)]
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
    with tempfile.TemporaryDirectory(prefix="aa-word-cache-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for index, (case, prepared, minimum, rules) in enumerate(make_cases(8)):
            fixture = temp / f"standard-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint, _ = expected_for(prepared, rules, minimum)
            for mode in ("atomic", "word-cache"):
                for hasher in ("std", "ahash"):
                    for workers in (1, 4):
                        check(case, fixture, expected, fingerprint, minimum, rules,
                              mode, hasher, workers, trace)
                        compared += 1
        prepared = weighted_aa()
        fixture = temp / "weighted-4095-4096-4097.json"
        fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
        expected, fingerprint, max_length = expected_for(prepared, 16, 2)
        for mode in ("atomic", "word-cache"):
            result = check("weighted-aa", fixture, expected, fingerprint, 2, 16,
                           mode, "ahash", 4, trace)
            directed.append({"mode": mode, "max_token_length": max_length,
                             "bitmap_rounds": result["aa_bitmap_rounds"],
                             "fallback_rounds": result["aa_bitmap_fallback_rounds"],
                             "atomic_or_calls": result["aa_bitmap_atomic_or_calls"]})
        assert all(row["max_token_length"] > 255 and row["bitmap_rounds"] > 0
                   and row["fallback_rounds"] > 0 for row in directed)
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

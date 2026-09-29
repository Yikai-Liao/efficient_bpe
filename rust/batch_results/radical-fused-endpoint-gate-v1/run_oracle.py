"""Complete-trace oracle for the three endpoint planning modes."""

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
BIN = RUST / "target/reruns/radical-fused-endpoint-gate-v1/radical-owned-fused-endpoint"
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
    expected_merges, expected_final = naive(prepared, rules, minimum)
    expected = {"merges": [list(row) for row in expected_merges], "final": expected_final}
    fingerprint = hashlib.sha256(json.dumps([expected_merges, expected_final]).encode()).hexdigest()
    return expected, fingerprint


def check(case_id, expected, fingerprint, minimum, rules, mode, hasher, workers,
          chunk, trace, fixture):
    command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", str(chunk), "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", "lazy",
               "--integer-hash", hasher, "--endpoint-plan", mode,
               "--trace", str(trace)]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError((case_id, mode, hasher, workers, chunk, run.stderr, run.stdout))
    observed = json.loads(run.stdout.strip().splitlines()[-1])
    if json.loads(trace.read_text()) != expected or observed["fingerprint"] != fingerprint:
        raise AssertionError((case_id, mode, hasher, workers, chunk, "full trace differs"))
    return observed


def main():
    if not BIN.is_file():
        raise FileNotFoundError(BIN)
    compared = 0
    custom_compared = 0
    custom_metrics = []
    with tempfile.TemporaryDirectory(prefix="fused-endpoint-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for case_index, (case_id, prepared, minimum, rules) in enumerate(make_cases(8)):
            fixture = temp / f"standard-{case_index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint = expected_for(prepared, rules, minimum)
            for mode in ("two-pass", "tagged-two-pass", "tagged-fused"):
                for hasher in ("std", "ahash"):
                    for workers in (1, 4):
                        check(case_id, expected, fingerprint, minimum, rules, mode, hasher,
                              workers, 4096, trace, fixture)
                        compared += 1

        custom = [
            ("abab-16384", weighted_pieces([("ab" * 8192, 2)]), 2, 18),
            ("mixed-adjacent", weighted_pieces([("abcd" * 2048, 2),
                                                  ("cdab" * 1024, 3)]), 2, 18),
            ("weighted-boundary", weighted_pieces([("abcd" * 512, 2),
                                                     ("cdab" * 512, 5),
                                                     ("aabbaa" * 512, 3)]), 2, 18),
        ]
        for case_index, (case_id, prepared, minimum, rules) in enumerate(custom):
            fixture = temp / f"custom-{case_index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint = expected_for(prepared, rules, minimum)
            for chunk in (3, 7):
                for repetition in range(3):
                    result = check(case_id, expected, fingerprint, minimum, rules, "tagged-fused",
                                   "ahash", 4, chunk, trace, fixture)
                    custom_metrics.append({"case": case_id, "chunk": chunk,
                                           "repetition": repetition,
                                           "batch_rounds": result["batch_rounds"],
                                           "fused_batches": result.get("fused_non_aa_batches"),
                                           "zero_rereads": result.get("decoder_zero_rereads")})
                    custom_compared += 1
    report = {
        "status": "passed", "standard_cases": 20, "random_cases": 8,
        "standard_fulltrace_matches": compared,
        "custom_cases": [item[0] for item in custom],
        "custom_fulltrace_matches": custom_compared,
        "custom_metrics": custom_metrics,
        "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
        "oracle": "Python naive full recount; complete rule trace and final tokens",
    }
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "custom_metrics"}))


if __name__ == "__main__":
    main()

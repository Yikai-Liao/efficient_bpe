"""Complete-trace oracle for sorted and adaptive-bitmap AA handling."""

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
BIN = RUST / "target/reruns/radical-aa-bitmap-gate-v1/radical-owned-aa-bitmap"
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


def check(case_id, expected, fingerprint, minimum, rules, mode, hasher, workers,
          fixture, trace):
    command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", "4096", "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", "lazy",
               "--integer-hash", hasher, "--aa-order", mode, "--trace", str(trace)]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError((case_id, mode, hasher, workers, run.stderr, run.stdout))
    result = json.loads(run.stdout.strip().splitlines()[-1])
    if json.loads(trace.read_text()) != expected or result["fingerprint"] != fingerprint:
        raise AssertionError((case_id, mode, hasher, workers, "full trace differs"))
    return result


def main():
    if not BIN.is_file():
        raise FileNotFoundError(BIN)
    compared = 0
    directed = []
    with tempfile.TemporaryDirectory(prefix="aa-bitmap-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for case_index, (case_id, prepared, minimum, rules) in enumerate(make_cases(8)):
            fixture = temp / f"standard-{case_index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint, _ = expected_for(prepared, rules, minimum)
            for mode in ("sort", "bitmap-adaptive"):
                for hasher in ("std", "ahash"):
                    for workers in (1, 4):
                        check(case_id, expected, fingerprint, minimum, rules,
                              mode, hasher, workers, fixture, trace)
                        compared += 1

        special = [
            ("weighted-aa-4095-4096-4097",
             weighted_pieces([("a" * 4095, 2), ("a" * 4096, 5),
                              ("a" * 4097, 3)]), 2, 16, (1, 4)),
            ("dense-aa-4m", weighted_pieces([("a" * 4_194_304, 2)]),
             2, 12, (4,)),
        ]
        for case_index, (case_id, prepared, minimum, rules, workers_list) in enumerate(special):
            fixture = temp / f"special-{case_index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint, max_length = expected_for(prepared, rules, minimum)
            for mode in ("sort", "bitmap-adaptive"):
                for workers in workers_list:
                    result = check(case_id, expected, fingerprint, minimum, rules,
                                   mode, "ahash", workers, fixture, trace)
                    directed.append({"case": case_id, "mode": mode, "workers": workers,
                                     "max_token_length": max_length,
                                     "aa_bitmap_rounds": result["aa_bitmap_rounds"],
                                     "aa_bitmap_fallback_rounds": result["aa_bitmap_fallback_rounds"],
                                     "aa_bitmap_peak_bytes": result["aa_bitmap_peak_bytes"],
                                     "peak_aa_plan_capacity_bytes": result["peak_aa_plan_capacity_bytes"]})
        weighted = [row for row in directed if row["case"] == "weighted-aa-4095-4096-4097"
                    and row["mode"] == "bitmap-adaptive"]
        if not all(row["aa_bitmap_rounds"] > 0 and row["aa_bitmap_fallback_rounds"] > 0
                   and row["max_token_length"] > 255 for row in weighted):
            raise AssertionError(("weighted AA case missed required bitmap/fallback/long-token", weighted))
        dense = [row for row in directed if row["case"] == "dense-aa-4m"
                 and row["mode"] == "bitmap-adaptive"]
        if not dense or dense[0]["aa_bitmap_rounds"] == 0:
            raise AssertionError("dense 4MiB case did not use bitmap")
    report = {"status": "passed", "standard_cases": 20,
              "standard_fulltrace_matches": compared,
              "directed_fulltrace_matches": len(directed),
              "directed": directed,
              "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
              "oracle": "Python naive full recount; complete rule trace and final tokens"}
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "directed"}))


if __name__ == "__main__":
    main()

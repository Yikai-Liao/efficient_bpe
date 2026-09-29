"""Complete-trace oracle for endpoint × AA ordering combinations."""

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
BIN = RUST / "target/reruns/radical-endpoint-bitmap-combo-gate-v1/radical-owned-endpoint-bitmap-combo"
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


def check(case, fixture, expected, fingerprint, minimum, rules, endpoint, aa,
          hasher, workers, chunk, trace):
    command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", str(chunk), "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", "lazy",
               "--integer-hash", hasher, "--endpoint-plan", endpoint,
               "--aa-order", aa, "--trace", str(trace)]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError((case, endpoint, aa, hasher, workers,
                            process.stderr, process.stdout))
    result = json.loads(process.stdout.strip().splitlines()[-1])
    if json.loads(trace.read_text()) != expected or result["fingerprint"] != fingerprint:
        raise AssertionError((case, endpoint, aa, hasher, workers, "full trace differs"))
    return result


def main():
    if not BIN.is_file():
        raise FileNotFoundError(BIN)
    compared = 0
    directed = []
    with tempfile.TemporaryDirectory(prefix="endpoint-bitmap-combo-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for index, (case, prepared, minimum, rules) in enumerate(make_cases(8)):
            fixture = temp / f"standard-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint, _ = expected_for(prepared, rules, minimum)
            for endpoint in ("two-pass", "tagged-two-pass", "tagged-fused"):
                for aa in ("sort", "bitmap-adaptive"):
                    for hasher in ("std", "ahash"):
                        for workers in (1, 4):
                            check(case, fixture, expected, fingerprint, minimum, rules,
                                  endpoint, aa, hasher, workers, 4096, trace)
                            compared += 1

        prepared = weighted_pieces([("a" * 4096, 3), ("ababcdcd" * 16, 7)])
        fixture = temp / "mixed-epochs.json"
        fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
        expected, fingerprint, max_length = expected_for(prepared, 24, 1)
        for endpoint, aa, workers in (("tagged-fused", "bitmap-adaptive", 1),
                                      ("tagged-fused", "bitmap-adaptive", 4),
                                      ("tagged-fused", "sort", 4),
                                      ("two-pass", "sort", 4)):
            result = check("mixed-epochs", fixture, expected, fingerprint, 1, 24,
                           endpoint, aa, "ahash", workers, 7, trace)
            directed.append({"case": "mixed-epochs", "endpoint": endpoint,
                             "aa": aa, "workers": workers,
                             "max_token_length": max_length,
                             "fused_non_aa_batches": result["fused_non_aa_batches"],
                             "aa_bitmap_rounds": result["aa_bitmap_rounds"],
                             "aa_bitmap_fallback_rounds": result["aa_bitmap_fallback_rounds"]})
            if endpoint == "tagged-fused" and aa == "bitmap-adaptive":
                assert max_length > 255 and result["fused_non_aa_batches"] > 0
                assert result["aa_bitmap_rounds"] > 0
                assert result["aa_bitmap_fallback_rounds"] > 0

        prepared = weighted_pieces([("a" * 64, 5)])
        fixture = temp / "domain-fallback.json"
        fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
        domain_limit = 1 << 31
        expected, fingerprint, _ = expected_for(prepared, domain_limit, 1)
        result = check("domain-fallback", fixture, expected, fingerprint, 1,
                       domain_limit, "tagged-fused", "bitmap-adaptive", "ahash", 4,
                       4, trace)
        assert result["endpoint_domain_fallback"]
        assert result["endpoint_plan_effective"] == "two-pass"
        assert result["aa_bitmap_rounds"] > 0
        directed.append({"case": "domain-fallback", "effective": result["endpoint_plan_effective"],
                         "aa_bitmap_rounds": result["aa_bitmap_rounds"]})
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

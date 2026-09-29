"""Full-trace oracle for owner versus producer-atomic old-key reduction."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-atomic-old-gate-v1/radical-owned-atomic-old"
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402


def expected_for(prepared, rules, minimum):
    merges, final = naive(prepared, rules, minimum)
    return ({"merges": [list(row) for row in merges], "final": final},
            hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest())


def check(case, fixture, expected, fingerprint, minimum, rules, mode, hasher,
          workers, heap, chunk, trace):
    command = [str(BIN), "--input", str(fixture), "--workers", str(workers),
               "--chunk-size", str(chunk), "--rules", str(rules),
               "--min-frequency", str(minimum), "--heap-policy", heap,
               "--integer-hash", hasher, "--old-reduce", mode,
               "--trace", str(trace)]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError((case, mode, hasher, workers, heap, process.stderr, process.stdout))
    result = json.loads(process.stdout.strip().splitlines()[-1])
    if json.loads(trace.read_text()) != expected or result["fingerprint"] != fingerprint:
        raise AssertionError((case, mode, hasher, workers, heap, "full trace differs"))
    assert result["old_reduce_requested"] == mode
    assert result["old_reduce_effective"] == ("owner" if heap == "eager" else mode)
    return result


def main():
    assert BIN.is_file()
    standard = 0
    directed = []
    cases = make_cases(8)
    with tempfile.TemporaryDirectory(prefix="atomic-old-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for index, (case, prepared, minimum, rules) in enumerate(cases):
            fixture = temp / f"standard-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint = expected_for(prepared, rules, minimum)
            for mode in ("owner", "producer-atomic"):
                for workers in (1, 4):
                    check(case, fixture, expected, fingerprint, minimum, rules,
                          mode, "ahash", workers, "lazy", 4096, trace)
                    standard += 1
        # std HashMap, weighted u64 subtraction, and eager-mode fallback are
        # separately covered without repeating the whole standard matrix.
        chosen = {name: (prep, minimum, rules) for name, prep, minimum, rules in cases
                  if name in ("weighted64", "large-u64-weight")}
        assert len(chosen) == 2
        for case, (prepared, minimum, rules) in chosen.items():
            fixture = temp / f"directed-{case}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected, fingerprint = expected_for(prepared, rules, minimum)
            for mode in ("owner", "producer-atomic"):
                result = check(case, fixture, expected, fingerprint, minimum, rules,
                               mode, "std", 4, "lazy", 3, trace)
                directed.append({"case": case, "mode": mode, "hasher": "std",
                                 "heap": "lazy", "atomic_old_calls": result["atomic_old_calls"],
                                 "atomic_retire_markers": result["atomic_retire_markers"],
                                 "old_route_records_before_flush": result["old_route_records_before_flush"]})
                if mode == "producer-atomic":
                    assert result["atomic_old_calls"] > 0
        case = "weighted64-eager-fallback"
        prepared, minimum, rules = chosen["weighted64"]
        fixture = temp / "eager-fallback.json"
        fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
        expected, fingerprint = expected_for(prepared, rules, minimum)
        result = check(case, fixture, expected, fingerprint, minimum, rules,
                       "producer-atomic", "ahash", 4, "eager", 3, trace)
        assert result["old_reduce_effective"] == "owner"
        assert result["atomic_old_calls"] == 0
        directed.append({"case": case, "requested": "producer-atomic",
                         "effective": result["old_reduce_effective"],
                         "atomic_old_calls": result["atomic_old_calls"]})
    report = {"status": "passed", "standard_cases": len(cases),
              "standard_fulltrace_matches": standard,
              "directed_fulltrace_matches": len(directed), "directed": directed,
              "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
              "oracle": "Python naive full recount; complete merge trace and final tokens"}
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "directed"}))


if __name__ == "__main__":
    main()

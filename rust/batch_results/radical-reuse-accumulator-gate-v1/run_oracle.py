"""Full-trace Python oracle for staged, fresh-map, and reused-map commit."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-reuse-accumulator-gate-v1/radical-owned-reuse-accumulator"
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402


def main():
    if not BIN.is_file():
        raise FileNotFoundError(BIN)
    compared = 0
    with tempfile.TemporaryDirectory(prefix="reuse-accumulator-oracle-") as temp_name:
        temp = Path(temp_name)
        trace = temp / "trace.json"
        for index, (case, prepared, minimum, rules) in enumerate(make_cases(8)):
            fixture = temp / f"case-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            merges, final = naive(prepared, rules, minimum)
            expected = {"merges": [list(row) for row in merges], "final": final}
            fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
            for mode in ("staged", "fused-fresh", "fused-reuse"):
                for hasher in ("std", "ahash"):
                    for workers in (1, 4):
                        command = [str(BIN), "--input", str(fixture),
                                   "--workers", str(workers), "--chunk-size", "4096",
                                   "--rules", str(rules), "--min-frequency", str(minimum),
                                   "--heap-policy", "lazy", "--integer-hash", hasher,
                                   "--owner-commit", mode, "--trace", str(trace)]
                        process = subprocess.run(command, capture_output=True, text=True)
                        if process.returncode:
                            raise RuntimeError((case, mode, hasher, workers,
                                                process.stderr, process.stdout))
                        observed = json.loads(process.stdout.strip().splitlines()[-1])
                        if json.loads(trace.read_text()) != expected or observed["fingerprint"] != fingerprint:
                            raise AssertionError((case, mode, hasher, workers, "full trace differs"))
                        if observed["owner_commit"] != mode:
                            raise AssertionError((case, mode, observed["owner_commit"]))
                        compared += 1
    report = {"status": "passed", "standard_cases": 20, "random_cases": 8,
              "fulltrace_matches": compared,
              "modes": ["staged", "fused-fresh", "fused-reuse"],
              "hashes": ["std", "ahash"], "workers": [1, 4],
              "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
              "oracle": "Python naive full recount; complete rule trace and final tokens"}
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

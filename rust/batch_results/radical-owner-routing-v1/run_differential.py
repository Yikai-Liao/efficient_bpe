"""Compare grouped births and logical owner shards with the full-recount oracle."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402

VARIANTS = {
    "grouped_lazy": (RUST / "target/reruns/radical-owner-routing-v1/grouped",
                     ["--heap-policy", "lazy"]),
    "grouped_eager": (RUST / "target/reruns/radical-owner-routing-v1/grouped",
                      ["--heap-policy", "eager"]),
    "shards_w": (RUST / "target/reruns/radical-owner-routing-v1/shards",
                 ["--heap-policy", "lazy"]),
    "shards_2w": (RUST / "target/reruns/radical-owner-routing-v1/shards",
                  ["--heap-policy", "lazy"]),
    "shards_4w": (RUST / "target/reruns/radical-owner-routing-v1/shards",
                  ["--heap-policy", "lazy"]),
}


def main():
    for name, (binary, _) in VARIANTS.items():
        if not binary.is_file():
            raise FileNotFoundError((name, binary))
    compared = 0
    with tempfile.TemporaryDirectory(prefix="radical-owner-routing-differential-") as temp:
        temp = Path(temp)
        for case_index, (case_id, prepared, minimum, max_merges) in enumerate(make_cases(8)):
            fixture = temp / f"case-{case_index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            merges, final = naive(prepared, max_merges, minimum)
            expected = {"merges": [list(row) for row in merges], "final": final}
            fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
            for name, (binary, extra) in VARIANTS.items():
                for workers in (1, 4):
                    trace = temp / "trace.json"
                    command = [str(binary), "--input", str(fixture),
                               "--workers", str(workers), "--chunk-size", "4096",
                               "--rules", str(max_merges), "--min-frequency", str(minimum),
                               "--trace", str(trace), *extra]
                    if name.startswith("shards_"):
                        multiplier = {"shards_w": 1, "shards_2w": 2, "shards_4w": 4}[name]
                        command.extend(["--owner-shards", str(workers * multiplier)])
                    result = subprocess.run(command, text=True, capture_output=True)
                    if result.returncode:
                        raise RuntimeError((case_id, name, workers,
                                            result.stdout, result.stderr))
                    observed = json.loads(result.stdout.strip().splitlines()[-1])
                    if json.loads(trace.read_text()) != expected:
                        raise AssertionError((case_id, name, workers, "trace mismatch"))
                    if observed["fingerprint"] != fingerprint:
                        raise AssertionError((case_id, name, workers, "fingerprint mismatch"))
                    if name.startswith("shards_") and observed["owner_shards"] != workers * multiplier:
                        raise AssertionError((case_id, name, workers, "owner shard mismatch"))
                    compared += 1
    report = {
        "status": "passed", "cases": 20, "random_cases": 8,
        "variants": list(VARIANTS), "workers": [1, 4],
        "bounds": "checked (CLI fixed)", "compared_runs": compared,
        "full_rule_trace_and_final_tokens_match": True,
        "binary_sha256": {name: hashlib.sha256(binary.read_bytes()).hexdigest()
                          for name, (binary, _) in VARIANTS.items()},
        "oracle": "Python naive full recount", "seed": 20260930,
    }
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

"""Complete-trace Python oracle for local scratch and integer hashing."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN_DIR = RUST / "target/reruns/radical-local-hash-v1"
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    modes = json.loads((OUT / "modes.json").read_text())["oracle"]
    for config in modes.values():
        if not (BIN_DIR / config["binary"]).is_file():
            raise FileNotFoundError(BIN_DIR / config["binary"])
    compared = 0
    with tempfile.TemporaryDirectory(prefix="radical-local-hash-oracle-") as temp_name:
        temp = Path(temp_name)
        for index, (case_id, prepared, minimum, max_merges) in enumerate(make_cases(8)):
            fixture = temp / f"case-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            merges, final = naive(prepared, max_merges, minimum)
            expected = {"merges": [list(row) for row in merges], "final": final}
            fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
            for version, config in modes.items():
                binary = BIN_DIR / config["binary"]
                for workers in (1, 4):
                    trace = temp / "trace.json"
                    command = [str(binary), "--input", str(fixture),
                               "--workers", str(workers), "--chunk-size", "4096",
                               "--rules", str(max_merges), "--min-frequency", str(minimum),
                               "--heap-policy", "lazy", "--trace", str(trace),
                               *config["args"]]
                    process = subprocess.run(command, text=True, capture_output=True)
                    if process.returncode:
                        raise RuntimeError((case_id, version, workers,
                                            process.stdout, process.stderr))
                    observed = json.loads(process.stdout.strip().splitlines()[-1])
                    if json.loads(trace.read_text()) != expected:
                        raise AssertionError((case_id, version, workers, "trace mismatch"))
                    if observed["fingerprint"] != fingerprint:
                        raise AssertionError((case_id, version, workers, "fingerprint mismatch"))
                    compared += 1
    report = {
        "status": "passed", "cases": 20, "random_cases": 8,
        "versions": list(modes), "workers": [1, 4],
        "compared_runs": compared, "full_rule_trace_and_final_tokens_match": True,
        "binary_sha256": {name: sha(BIN_DIR / name)
                          for name in {config["binary"] for config in modes.values()}},
        "modes_sha256": sha(OUT / "modes.json"),
        "oracle": "Python naive full recount", "seed": 20260930,
    }
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

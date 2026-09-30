"""Reusable small-input full-trace gate; mode configuration is filled after freeze.

Example config structure:
{
  "binary": "rust/target/reruns/.../binary",
  "binary_sha256": "...",
  "fixed_args": ["--chunk-size", "4096", "--integer-hash", "ahash"],
  "modes": [
    {"name": "control", "args": ["--mode", "control"]},
    {"name": "experiment", "args": ["--mode", "experiment"]}
  ]
}
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    binary = ROOT / config["binary"]
    assert binary.is_file() and sha(binary) == config["binary_sha256"]
    modes = config["modes"]
    assert len(modes) == 2 and {mode["name"] for mode in modes} == {"control", "experiment"}
    cases = make_cases(8)
    assert len(cases) == 20
    records = []
    with tempfile.TemporaryDirectory(prefix="adaptive-replay-oracle-") as temp_name:
        temp = Path(temp_name)
        for index, (case, prepared, minimum, rules) in enumerate(cases):
            fixture = temp / f"fixture-{index}.json"
            trace = temp / f"trace-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            merges, final = naive(prepared, rules, minimum)
            expected = {"merges": [list(row) for row in merges], "final": final}
            fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
            for mode in modes:
                for workers in (1, 4):
                    command = [str(binary), "--input", str(fixture), "--workers", str(workers),
                               "--rules", str(rules), "--min-frequency", str(minimum),
                               "--trace", str(trace), *config["fixed_args"], *mode["args"]]
                    process = subprocess.run(command, capture_output=True, text=True)
                    if process.returncode:
                        raise RuntimeError((case, mode["name"], workers, process.stderr,
                                            process.stdout))
                    result = json.loads(process.stdout.strip().splitlines()[-1])
                    if json.loads(trace.read_text()) != expected or result["fingerprint"] != fingerprint:
                        raise AssertionError((case, mode["name"], workers, "complete trace differs"))
                    for field, value in mode.get("expected_fields", {}).items():
                        assert result[field] == value, (case, mode["name"], workers, field, result[field])
                    for field, value in mode.get("expected_fields_by_workers", {}).get(
                            str(workers), {}).items():
                        assert result[field] == value, (case, mode["name"], workers, field, result[field])
                    records.append({"case": case, "mode": mode["name"], "workers": workers,
                                    "fingerprint": result["fingerprint"],
                                    "full_trace_match": True})
    assert len(records) == 80
    report = {"status": "passed", "standard_cases": len(cases),
              "standard_fulltrace_matches": len(records),
              "per_mode_worker_matches": 20,
              "binary_sha256": config["binary_sha256"],
              "config_sha256": sha(args.config),
              "oracle": "Python naive full recount; complete merge trace and final tokens",
              "records": records}
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()

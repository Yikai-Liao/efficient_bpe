"""Complete-trace oracle for serial CF32/CF16 integer-hash controls."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUST = ROOT / "rust"
OUT = Path(__file__).resolve().parent
BIN = RUST / "target/reruns/radical-serial-integer-gate-v1/serial_integer_hash"
sys.path.insert(0, str(RUST / "tools"))
from ablation_differential import make_cases, naive, prepared_wire  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    modes = json.loads((OUT / "modes.json").read_text())["oracle"]
    assert BIN.is_file() and len(modes) == 8
    compared = 0
    rejections = []
    with tempfile.TemporaryDirectory(prefix="radical-serial-integer-oracle-") as temp_name:
        temp = Path(temp_name)
        for index, (case_id, prepared, minimum, max_merges) in enumerate(make_cases(8)):
            fixture = temp / f"case-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            merges, final = naive(prepared, max_merges, minimum)
            expected = {"merges": [list(rule) for rule in merges], "final": final}
            fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
            for version, config in modes.items():
                trace = temp / "trace.json"
                command = [str(BIN), "--input", str(fixture), "--workers", "1",
                           "--rules", str(max_merges), "--min-frequency", str(minimum),
                           "--trace", str(trace), *config["args"]]
                process = subprocess.run(command, text=True, capture_output=True)
                if process.returncode:
                    if (case_id == "initial-alphabet-65536"
                            and version.startswith("cf16_")
                            and "halfword initial alphabet must fit u16" in process.stderr):
                        rejections.append({"case_id": case_id, "version": version,
                                           "reason": process.stderr.strip()})
                        continue
                    raise RuntimeError((case_id, version, process.stdout, process.stderr))
                observed = json.loads(process.stdout.strip().splitlines()[-1])
                assert json.loads(trace.read_text()) == expected, (case_id, version, "trace")
                assert observed["fingerprint"] == fingerprint, (case_id, version, "fingerprint")
                compared += 1
    report = {
        "status": "passed", "cases": 20, "random_cases": 8,
        "modes": list(modes), "workers": [1],
        "compared_runs": compared, "expected_rejections": rejections,
        "full_rule_trace_and_final_tokens_match": True,
        "binary_sha256": sha(BIN), "modes_sha256": sha(OUT / "modes.json"),
        "oracle": "Python naive full recount", "seed": 20260930,
    }
    assert compared + len(rejections) == 20 * 8
    assert len(rejections) == 4
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"matches": compared, "expected_rejections": len(rejections)}))


if __name__ == "__main__":
    main()

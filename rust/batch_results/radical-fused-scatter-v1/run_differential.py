"""Complete-trace oracle for fused, scatter, and direct-reduce ablations."""

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

BINARIES = {
    "fused_separate": ("owned_fused", ["--commit-mode", "separate"]),
    "fused_owner": ("owned_fused", ["--commit-mode", "owner-fused"]),
    "fused_overlap": ("owned_fused", ["--commit-mode", "overlap"]),
    "scatter_owner": ("owned_scatter", ["--fill-mode", "owner"]),
    "scatter_scatter": ("owned_scatter", ["--fill-mode", "scatter"]),
    "direct_combined": ("owned_direct", ["--reduce-mode", "combined"]),
    "direct_old": ("owned_direct", ["--reduce-mode", "direct-old"]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    binary_dir = RUST / "target/reruns/radical-fused-scatter-v1"
    for binary_name, _ in BINARIES.values():
        if not (binary_dir / binary_name).is_file():
            raise FileNotFoundError(binary_dir / binary_name)
    compared = 0
    with tempfile.TemporaryDirectory(prefix="radical-fused-scatter-oracle-") as temp_name:
        temp = Path(temp_name)
        for index, (case_id, prepared, minimum, max_merges) in enumerate(make_cases(8)):
            fixture = temp / f"case-{index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            merges, final = naive(prepared, max_merges, minimum)
            expected = {"merges": [list(row) for row in merges], "final": final}
            fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
            for version, (binary_name, extra_args) in BINARIES.items():
                binary = binary_dir / binary_name
                for workers in (1, 4):
                    trace = temp / "trace.json"
                    command = [str(binary), "--input", str(fixture),
                               "--workers", str(workers), "--chunk-size", "4096",
                               "--rules", str(max_merges), "--min-frequency", str(minimum),
                               "--heap-policy", "lazy", "--trace", str(trace), *extra_args]
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
        "versions": list(BINARIES), "workers": [1, 4],
        "compared_runs": compared, "full_rule_trace_and_final_tokens_match": True,
        "binary_sha256": {name: sha(binary_dir / name)
                          for name in {binary for binary, _ in BINARIES.values()}},
        "oracle": "Python naive full recount", "seed": 20260930,
    }
    with (OUT / "differential.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

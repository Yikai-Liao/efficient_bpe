"""Run exact-batch scaling and memory ablations under a whole-process CPU budget.

Uses existing pinned fixtures; does not download or modify other workspaces.
Every stage runs sequentially, and a new output directory is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "rust"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    binary = RUST / "target/release/ablation"
    available = set(json.loads(subprocess.check_output([str(binary), "--list-variants"], text=True)))
    stages = [
        ("small", RUST / "ablation_results/fixtures.json",
         ["en-4m-continuous", "zh-4m-continuous", "en-1m--paragraph", "zh-4m--regex"],
         ["combined_filtered", "combined_filtered_halfword", "parallel_certified"],
         "1,4" if args.pilot else "1,2,4,6"),
        ("large", RUST / "ablation_results/large-fixtures.json",
         ["en-16m-continuous", "zh-16m-continuous"],
         ["combined_filtered", "combined_filtered_halfword", "parallel_certified"],
         "1,4" if args.pilot else "1,2,4,6"),
        ("single-round", RUST / "ablation_results/fixtures.json",
         ["en-4m-continuous", "zh-4m-continuous"],
         ["parallel_certified_single"], "1,4"),
        ("relaxed", RUST / "ablation_results/fixtures.json",
         ["en-4m-continuous", "zh-4m-continuous"],
         ["parallel_batch_relaxed"], "1,4"),
    ]
    for _, manifest, cases, variants, _ in stages:
        if not set(variants) <= available:
            raise RuntimeError(f"binary missing variants: {set(variants) - available}")
        rows = {r["case_id"]: r for r in json.loads(manifest.read_text())}
        for case in cases:
            row = rows[case]
            body = (RUST / row["file"]).read_bytes()
            if hashlib.sha256(body).hexdigest() != row["fixture_sha256"]:
                raise RuntimeError(f"changed fixture: {case}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    for name, manifest, cases, variants, workers in stages:
        command = [sys.executable, str(RUST / "tools/ablation_benchmark.py"),
                   "--profile", "all", "--manifest", str(manifest),
                   "--cases", ",".join(cases), "--variants", ",".join(variants),
                   "--workers", workers, "--bounds", "checked",
                   "--parallel-core-budget", "workers", "--repeats", str(args.repeats),
                   "--output", str(out / f"{name}.jsonl")]
        commands.append({"name": name, "command": command})
    (out / "commands.json").write_text(json.dumps({
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "repeats": args.repeats, "pilot": args.pilot,
        "protocol": "Entire child process uses p CPUs including the coordinator; p=1 and scalar both use CPU 5.",
        "stages": commands,
    }, indent=2) + "\n")
    for stage in commands:
        print(f"Starting {stage['name']}", flush=True)
        with (out / f"{stage['name']}.progress.log").open("x") as log:
            subprocess.run(stage["command"], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {stage['name']}", flush=True)


if __name__ == "__main__":
    main()

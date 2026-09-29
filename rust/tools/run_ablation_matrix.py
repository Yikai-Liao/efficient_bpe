"""Reproduce the native matrix sequentially, without competing CPU jobs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

RUST = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = RUST / "target/release/ablation"
    variants = json.loads(subprocess.check_output([binary, "--list-variants"], text=True))
    scalar = [v for v in variants if not v.startswith("parallel_")]
    parallel = [v for v in variants if v.startswith("parallel_")]
    fixtures = json.loads((RUST / "ablation_results/fixtures.json").read_text())
    continuous = ["en-4m-continuous", "zh-4m-continuous"]
    other_cases = [f["case_id"] for f in fixtures if f["case_id"] not in continuous]
    continuous_scalar = ["archived", "packed", "arena_counted", "combined_filtered",
                         "combined_filtered_h3", "combined_filtered_halfword"]
    unchecked_parallel = ["parallel_occurrence_snapshot", "parallel_occurrence_adaptive",
                          "parallel_occurrence_adaptive_256", "parallel_occurrence_adaptive_4096"]
    common = [sys.executable, str(RUST / "tools/ablation_benchmark.py"),
              "--repeats", str(args.repeats)]
    matrices = [
        ("scalar", ["--profile", "all", "--cases", ",".join(other_cases),
                    "--variants", ",".join(scalar), "--bounds", "both"]),
        ("continuous-scalar", ["--profile", "all", "--cases", ",".join(continuous),
                               "--variants", ",".join(continuous_scalar), "--bounds", "both"]),
        ("parallel", ["--profile", "multicore", "--variants", ",".join(parallel),
                      "--workers", "1,2,4,6", "--bounds", "checked"]),
        ("parallel-unchecked", ["--profile", "multicore",
                                 "--cases", ",".join(continuous + ["en-1m--paragraph"]),
                                 "--variants", ",".join(unchecked_parallel),
                                 "--workers", "1,4", "--bounds", "unchecked"]),
    ]
    commands = [(name, common + extra + ["--output", str(out / f"{name}.jsonl")])
                for name, extra in matrices]
    commands.append(("topology", [sys.executable, str(RUST / "tools/ablation_micro.py"),
                                  "--lengths", "32,63,64,128,255,256,1024,8192,65535,65536",
                                  "--patterns", "random,balanced,chain", "--bounds", "both",
                                  "--repeats", str(args.repeats),
                                  "--output", str(out / "topology.jsonl")]))
    with (out / "commands.json").open("x") as stream:
        json.dump(commands, stream, indent=2)
        stream.write("\n")
    for name, command in commands:
        print(json.dumps({"starting": name, "command": command}), flush=True)
        with (out / f"{name}.progress.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"completed": name}), flush=True)


if __name__ == "__main__":
    main()

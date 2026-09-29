"""Plan and sequentially run the second scalar/grouped-parallel matrix.

The required grouped/shrink variants are deferred patches, not yet measured.
This runner refuses to start before variant and fixture preflight succeeds.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
CAPACITY_CASES = (
    "en-1m--regex", "zh-1m--regex", "de-1m--regex", "ja-1m--regex",
    "en-1m--paragraph", "en-4m--regex", "zh-4m--regex",
    "en-4m-continuous", "zh-4m-continuous",
)
CAPACITY_VARIANTS = (
    "packed", "packed_shrink", "combined_filtered",
    "combined_filtered_shrink", "combined_filtered_halfword",
)
GROUPED_VARIANTS = (
    "parallel_occurrence_snapshot", "parallel_occurrence_adaptive",
    "parallel_occurrence_grouped", "parallel_occurrence_grouped_adaptive",
)
LARGE_VARIANTS = (
    "combined_filtered", "combined_filtered_shrink",
    "parallel_occurrence_adaptive", "parallel_occurrence_grouped",
    "parallel_occurrence_grouped_adaptive",
)
WORKERS = "1,2,4,6"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected fixture manifest list: {path}")
    return rows


def validate_cases(rust_dir, manifest_path, requested):
    rows = read_manifest(manifest_path)
    by_id = {row["case_id"]: row for row in rows}
    missing = sorted(set(requested) - by_id.keys())
    if missing:
        raise ValueError(f"manifest {manifest_path} is missing cases: {missing}")
    for case_id in requested:
        row = by_id[case_id]
        fixture = rust_dir / row["file"]
        if not fixture.is_file():
            raise FileNotFoundError(f"fixture not generated: {fixture}")
        if sha256(fixture) != row["fixture_sha256"]:
            raise RuntimeError(f"fixture hash mismatch: {fixture}")
    return rows


def stage(name, profile, cases, variants, bounds, workers, manifest=None):
    args = ["--profile", profile, "--cases", ",".join(cases),
            "--variants", ",".join(variants), "--bounds", bounds]
    if workers is not None:
        args.extend(["--workers", workers])
    if manifest is not None:
        args.extend(["--manifest", str(manifest)])
    return {"name": name, "profile": profile, "cases": list(cases),
            "variants": list(variants), "bounds": bounds,
            "workers": workers.split(",") if workers else ["1"],
            "manifest": str(manifest) if manifest else None,
            "benchmark_args": args}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for JSONL files, logs, and commands.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--repo", type=Path, default=DEFAULT_ROOT,
                        help="repository root")
    parser.add_argument("--binary", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    root = args.repo.resolve()
    rust_dir = root / "rust"
    tools_dir = rust_dir / "tools"
    binary = args.binary.resolve() if args.binary else rust_dir / "target/release/ablation"
    small_manifest = rust_dir / "ablation_results/fixtures.json"
    large_manifest = rust_dir / "ablation_results/large-fixtures.json"
    if not binary.is_file():
        raise FileNotFoundError(f"release ablation binary is required: {binary}")
    if not small_manifest.is_file():
        raise FileNotFoundError(small_manifest)
    if not large_manifest.is_file():
        raise FileNotFoundError(f"large fixtures are not generated yet: {large_manifest}")

    available = set(json.loads(subprocess.check_output(
        [str(binary), "--list-variants"], text=True)))
    planned_variants = set(CAPACITY_VARIANTS + GROUPED_VARIANTS + LARGE_VARIANTS)
    missing_variants = sorted(planned_variants - available)
    if missing_variants:
        raise RuntimeError(f"release binary does not register planned variants: {missing_variants}")

    validate_cases(rust_dir, small_manifest, CAPACITY_CASES)
    grouped_cases = ("en-4m-continuous", "zh-4m-continuous", "en-4m--regex",
                     "zh-4m--regex", "en-1m--paragraph", "single-run-a-65536",
                     "single-piece-ab-65536")
    validate_cases(rust_dir, small_manifest, grouped_cases)
    unchecked_cases = ("en-4m-continuous", "zh-4m-continuous", "en-1m--paragraph")
    validate_cases(rust_dir, small_manifest, unchecked_cases)
    large_cases = ("en-16m-continuous", "zh-16m-continuous")
    validate_cases(rust_dir, large_manifest, large_cases)

    stages = [
        stage("capacity", "all", CAPACITY_CASES, CAPACITY_VARIANTS,
              "checked", "1"),
        stage("grouped", "multicore", grouped_cases, GROUPED_VARIANTS,
              "checked", WORKERS),
        stage("grouped-unchecked", "multicore", unchecked_cases, GROUPED_VARIANTS,
              "unchecked", "1,4"),
        stage("large", "all", large_cases, LARGE_VARIANTS,
              "checked", WORKERS, large_manifest),
    ]

    out = args.output_dir.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    benchmark = tools_dir / "ablation_benchmark.py"
    commands = []
    for item in stages:
        command = [sys.executable, str(benchmark), *item["benchmark_args"],
                   "--binary", str(binary), "--parallel-core-budget", "workers",
                   "--repeats", str(args.repeats), "--output",
                   str(out / f"{item['name']}.jsonl")]
        commands.append({**item, "command": command})
    command_manifest = {
        "schema_version": 1,
        "repo": str(root), "binary": str(binary), "binary_sha256": sha256(binary),
        "repeats": args.repeats,
        "execution_order": [item["name"] for item in stages],
        "stages": commands,
        "expected_rows": {
            "capacity": len(CAPACITY_CASES) * len(CAPACITY_VARIANTS) * args.repeats,
            "grouped": len(grouped_cases) * len(GROUPED_VARIANTS) * 4 * args.repeats,
            "grouped-unchecked": len(unchecked_cases) * len(GROUPED_VARIANTS) * 2 * args.repeats,
            "large": len(large_cases) * args.repeats * (2 + 3 * 4),
        },
        "fixture_manifest_sha256": {
            str(small_manifest): sha256(small_manifest),
            str(large_manifest): sha256(large_manifest),
        },
    }
    with (out / "commands.json").open("x", encoding="utf-8") as stream:
        json.dump(command_manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    for item in commands:
        name = item["name"]
        command = item["command"]
        print(json.dumps({"starting": name, "command": command}), flush=True)
        with (out / f"{name}.progress.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"completed": name}), flush=True)


if __name__ == "__main__":
    main()

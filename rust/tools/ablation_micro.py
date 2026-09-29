"""Run isolated boundary-topology microbenchmarks from the Rust micro CLI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "rust"
DEFAULT_VARIANTS = (
    "u8_only", "bytespans", "full_clear", "endpoints", "lean", "linked12",
    "linked16", "bitmap_u32", "halfword", "h3", "h25",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv(text, convert=str):
    return tuple(dict.fromkeys(convert(item.strip()) for item in text.split(",")
                               if item.strip()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=RUST / "target/release/boundary_micro")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--lengths", default="63,64,255,256,65535,65536")
    parser.add_argument("--positions", default="131072", help="comma-separated corpus sizes")
    parser.add_argument("--patterns", default="random,balanced,chain")
    parser.add_argument("--bounds", default="checked")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--scalar-cpu", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSONL path; existing paths are never overwritten")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    variants = csv(args.variants)
    lengths = csv(args.lengths, int)
    positions = csv(args.positions, int)
    patterns = csv(args.patterns)
    bounds = csv(args.bounds)
    if not variants or not lengths or not positions or not patterns or not bounds:
        parser.error("variants, lengths, positions, patterns, and bounds must be non-empty")
    if any(value < 1 for value in lengths + positions):
        parser.error("lengths and positions must be positive")
    if any(value not in ("checked", "unchecked", "both") for value in bounds):
        parser.error("bounds must contain checked, unchecked, or both")
    if not args.binary.is_file():
        raise FileNotFoundError(f"build boundary micro CLI first: {args.binary}")

    affinity = set(os.sched_getaffinity(0))
    if args.scalar_cpu not in affinity:
        raise RuntimeError(f"scalar CPU {args.scalar_cpu} is not in affinity {sorted(affinity)}")
    bound_values = tuple(bound for value in bounds for bound in
                         (("checked", "unchecked") if value == "both" else (value,)))
    jobs = [(repeat, variant, length, count, pattern, bound)
            for repeat in range(args.repeats)
            for variant in variants
            for length in lengths
            for count in positions
            for pattern in patterns
            for bound in bound_values]
    random.Random(args.seed).shuffle(jobs)
    expected_skips = sum(variant == "u8_only" and length > 255
                         for _, variant, length, _, _, _ in jobs)

    source_files = [*sorted((RUST / "src").rglob("*.rs")), Path(__file__)]
    environment = {
        "binary": str(args.binary.resolve()), "binary_sha256": digest(args.binary),
        "sources_sha256": {str(path.relative_to(ROOT)): digest(path)
                           for path in source_files if path.is_file()},
        "rustc": subprocess.run(["rustc", "-Vv"], text=True, capture_output=True,
                                 check=True).stdout,
        "cpu_model": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "affinity": sorted(affinity), "scalar_cpu": args.scalar_cpu,
        "lengths": lengths, "positions": positions, "patterns": patterns,
        "variants": variants, "bounds": bounds, "repeats": args.repeats,
        "expected_explicit_skips": expected_skips,
        "seed": args.seed,
        "measurement_scope": "boundary topology only; not complete BPE training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment_path = args.output.with_suffix(args.output.suffix + ".environment.json")
    env = dict(os.environ, RUST_BACKTRACE="1")
    checksums = {}
    skipped_jobs = 0
    with args.output.open("x", encoding="utf-8", buffering=1) as output, \
            environment_path.open("x", encoding="utf-8") as metadata:
        metadata.write(json.dumps(environment, indent=2) + "\n")
        original_affinity = set(os.sched_getaffinity(0))
        try:
            for index, (repeat, variant, length, count, pattern, bound) in enumerate(jobs, 1):
                actual_positions = max(1, count // length) * (length + 1) + 1
                if variant == "u8_only" and length > 255:
                    skipped_jobs += 1
                    row = {"variant": variant, "length": length,
                           "positions": actual_positions, "requested_positions": count, "pattern": pattern,
                           "bounds": bound, "repetition": repeat,
                           "skipped": True,
                           "skip_reason": "u8_only supports length <=255"}
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps({"completed": index, "total": len(jobs),
                                      "variant": variant, "length": length,
                                      "positions": count, "pattern": pattern,
                                      "bounds": bound, "skipped": True}), flush=True)
                    continue
                os.sched_setaffinity(0, {args.scalar_cpu})
                command = [str(args.binary.resolve()), "--variant", variant,
                           "--length", str(length), "--positions", str(count),
                           "--pattern", pattern, "--bounds", bound]
                started = time.perf_counter()
                run = subprocess.run(command, text=True, capture_output=True,
                                     timeout=3600, env=env)
                outer_seconds = time.perf_counter() - started
                if run.returncode:
                    raise RuntimeError((command, run.returncode, run.stdout, run.stderr))
                try:
                    result = json.loads(run.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError) as exc:
                    raise RuntimeError((command, run.stdout, run.stderr)) from exc
                expected_identity = {"variant": variant, "length": length,
                                     "positions": actual_positions, "pattern": pattern,
                                     "bounds": bound}
                actual_identity = {key: result.get(key) for key in expected_identity}
                if actual_identity != expected_identity:
                    raise AssertionError(("micro CLI result identity mismatch",
                                          actual_identity, expected_identity))
                for required in ("operations", "checksum", "seconds",
                                 "buffer_bytes", "capacity_bytes", "vm_hwm_mib"):
                    if required not in result:
                        raise AssertionError(f"micro CLI omitted {required}: {result}")
                checksum_key = (length, count, pattern)
                previous = checksums.setdefault(checksum_key, result["checksum"])
                if result["checksum"] != previous:
                    raise AssertionError(
                        f"checksum disagreement for {checksum_key}: "
                        f"{variant}/{bound}={result['checksum']} expected {previous}"
                    )
                result.update({"variant": variant, "length": length,
                               "positions": actual_positions, "requested_positions": count, "pattern": pattern,
                               "bounds": bound, "repetition": repeat,
                               "train_seconds": result["seconds"],
                               "skipped": False,
                               "outer_call_seconds": outer_seconds,
                               "cpu_affinity": [args.scalar_cpu], "command": command})
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(json.dumps({"completed": index, "total": len(jobs),
                                  "variant": variant, "length": length,
                                  "positions": count, "pattern": pattern,
                                  "bounds": bound}), flush=True)
        finally:
            os.sched_setaffinity(0, original_affinity)
    print(json.dumps({"rows": len(jobs), "skipped": skipped_jobs,
                      "output": str(args.output),
                      "environment": str(environment_path)}))


if __name__ == "__main__":
    main()

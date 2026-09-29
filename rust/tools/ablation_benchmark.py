"""Run immutable, interleaved Rust ablation jobs in isolated processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "rust"
MANIFEST = RUST / "ablation_results/fixtures.json"
PARALLEL = {"parallel_broadcast", "parallel_owner", "parallel_occurrence",
            "parallel_occurrence_snapshot", "parallel_occurrence_adaptive",
            "parallel_occurrence_adaptive_256", "parallel_occurrence_adaptive_4096",
            "parallel_serial", "parallel_occurrence_grouped",
            "parallel_occurrence_grouped_adaptive", "parallel_certified",
            "parallel_certified_single", "parallel_batch_relaxed", "parallel_pair_owned",
            "parallel_pair_owned_compact", "parallel_pair_owned_single",
            "parallel_pair_owned_pipeline",
            "parallel_pair_owned_spatial",
            "parallel_pair_owned_spatial_extra",
            "parallel_sparse_owner", "parallel_sparse_owner_all"}
BASE_VARIANTS = (
    "full_clear", "endpoints", "lean", "packed", "unfused_endpoints",
    "unfused_halfword", "unfused_h3", "separate_counted", "linked12", "linked16",
    "bitmap_u32", "halfword", "h3", "h25", "filtered", "arena",
    "arena_counted", "filtered_h3", "combined", "combined_filtered",
    "combined_filtered_h3", "combined_filtered_halfword",
    "bucket", "bucket_normalized",
)
PARALLEL_VARIANTS = ("parallel_broadcast", "parallel_owner", "parallel_occurrence",
                     "parallel_occurrence_snapshot", "parallel_occurrence_adaptive",
                     "parallel_occurrence_adaptive_256",
                     "parallel_occurrence_adaptive_4096", "parallel_serial")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rust_sources():
    files = [RUST / "Cargo.toml", RUST / "Cargo.lock"]
    files.extend(sorted((RUST / "src").rglob("*.rs")))
    # Hash the Python experimental implementations and their dependencies so
    # each cross-language matrix documents the exact historical comparison.
    for folder in (ROOT / "benchmarks/bpe_core_comparison/python_rewrite",
                   ROOT / "benchmarks/bpe_core_comparison/evolution"):
        files.extend(sorted(folder.glob("*.py")))
    files.extend([Path(__file__), RUST / "tools/ablation_fixtures.py",
                  RUST / "tools/ablation_differential.py"])
    files.extend(sorted((RUST / "tools").glob("*ablation*.py")))
    unique = sorted(set(files))
    return {str(path.relative_to(ROOT)): digest(path) for path in unique if path.is_file()}


def cpu_name():
    try:
        text = Path("/proc/cpuinfo").read_text(errors="replace")
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        return match.group(1).strip() if match else platform.processor()
    except OSError:
        return platform.processor()


def numa_nodes():
    try:
        return Path("/sys/devices/system/node/online").read_text().strip()
    except OSError:
        return None


def selected_cases(rows, profile):
    by_id = {row["case_id"]: row for row in rows}
    if profile == "all":
        return rows
    if profile == "legacy":
        return [r for r in rows if r["case_id"].startswith((
            "en-1m--", "zh-1m--", "de-1m--", "ja-1m--", "en-4m--", "zh-4m--",
            "random-131072--", "runs-131072--", "chain-8000--", "chain-16000--"))]
    if profile == "sweetspot":
        wanted = {
            "en-1m--regex", "zh-1m--regex", "de-1m--regex", "ja-1m--regex",
            "en-1m--paragraph", "en-4m--regex", "zh-4m--regex",
            "weighted64-en-1m-regex", "chain-2000", "chain-4000", "chain-8000--regex",
            "mixed-long-short-rare", "rare-pair-pressure-512",
        }
        missing = wanted - by_id.keys()
        if missing:
            raise ValueError(f"fixture manifest missing sweetspot cases: {sorted(missing)}")
        return [by_id[case_id] for case_id in sorted(wanted)]
    if profile == "multicore":
        wanted = {
            "en-4m-continuous", "zh-4m-continuous", "en-4m--regex",
            "zh-4m--regex", "en-1m--paragraph", "single-run-a-65536",
            "single-piece-ab-65536",
        }
        missing = wanted - by_id.keys()
        if missing:
            raise ValueError(f"fixture manifest missing multicore cases: {sorted(missing)}")
        return [by_id[case_id] for case_id in sorted(wanted)]
    if profile == "stress":
        prefixes = ("chain-", "weighted64-", "mixed-long-short-rare",
                    "rare-pair-pressure-", "single-run-", "single-piece-ab-")
        return [r for r in rows if r["case_id"].startswith(prefixes)]
    raise ValueError(profile)


def default_variants(profile):
    if profile == "multicore":
        return ("packed", *PARALLEL_VARIANTS)
    if profile == "all":
        return (*BASE_VARIANTS, *PARALLEL_VARIANTS)
    return BASE_VARIANTS


def parse_csv(text, convert=str):
    return tuple(dict.fromkeys(convert(part.strip()) for part in text.split(",")
                               if part.strip()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=RUST / "target/release/ablation")
    parser.add_argument("--profile", choices=("legacy", "sweetspot", "multicore",
                                               "stress", "all"), default="sweetspot")
    parser.add_argument("--variants", default=None,
                        help="comma-separated variant override")
    parser.add_argument("--workers", default="1,2,4",
                        help="worker counts for parallel variants")
    parser.add_argument("--bounds", default="checked",
                        help="checked, unchecked, or both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--scalar-cpu", type=int, default=5)
    parser.add_argument("--parallel-core-budget", choices=("all", "workers"), default="workers",
                        help="workers (default) limits the entire process, including its coordinator, to p CPUs; all explicitly restores the original unrestricted worker-count experiment")
    parser.add_argument("--rules", type=int, default=None,
                        help="override rules for every fixture")
    parser.add_argument("--min-frequency", type=int, default=None,
                        help="override each fixture's threshold")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--cases", default=None,
                        help="comma-separated exact fixture IDs for a narrow pilot")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSONL path; existing paths are never overwritten")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    original_affinity = set(os.sched_getaffinity(0))
    if not original_affinity:
        raise RuntimeError("empty CPU affinity")
    scalar_cpu = args.scalar_cpu
    if scalar_cpu not in original_affinity:
        raise RuntimeError(f"scalar CPU {scalar_cpu} is not in affinity {sorted(original_affinity)}")
    variants = parse_csv(args.variants, str) if args.variants else default_variants(args.profile)
    workers = parse_csv(args.workers, int)
    bounds = parse_csv(args.bounds, str)
    if not variants or not workers or any(w < 1 for w in workers):
        parser.error("variants and positive worker counts are required")
    if args.parallel_core_budget == "workers" and max(workers) > len(original_affinity):
        parser.error("worker-count CPU budget exceeds available CPU affinity")
    cpu_order = [scalar_cpu] + sorted(original_affinity - {scalar_cpu})
    if not bounds or any(value not in ("checked", "unchecked", "both") for value in bounds):
        parser.error("bounds must be checked, unchecked, or both")
    if not args.binary.is_file():
        raise FileNotFoundError(f"build the ablation CLI first: {args.binary}")

    manifest_bytes = args.manifest.read_bytes()
    manifest_rows = json.loads(manifest_bytes)
    cases = selected_cases(manifest_rows, args.profile)
    if args.cases:
        requested_cases = parse_csv(args.cases, str)
        by_id = {row["case_id"]: row for row in manifest_rows}
        missing = set(requested_cases) - by_id.keys()
        if missing:
            raise ValueError(f"manifest missing requested cases: {sorted(missing)}")
        cases = [by_id[case_id] for case_id in requested_cases]
    if not cases:
        raise ValueError("selected benchmark profile contains no fixtures")
    for case in cases:
        fixture = RUST / case["file"]
        if digest(fixture) != case["fixture_sha256"]:
            raise RuntimeError(f"fixture hash mismatch: {fixture}")

    jobs = []
    for repetition in range(args.repeats):
        for case in cases:
            rules = args.rules if args.rules is not None else case["rules"]
            minimum = (args.min_frequency if args.min_frequency is not None
                       else case.get("min_frequency", 2))
            for variant in variants:
                counts = workers if variant in PARALLEL else (1,)
                for worker_count in counts:
                    # Preserve the requested bound mode for every variant;
                    # whether a mode changes generated code is variant-specific.
                    variant_bounds = tuple(v for value in bounds
                                           for v in (("checked", "unchecked")
                                                     if value == "both" else (value,)))
                    for bound in variant_bounds:
                        jobs.append((repetition, case, variant, worker_count,
                                     bound, rules, minimum))
    random.Random(args.seed).shuffle(jobs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment_path = args.output.with_suffix(args.output.suffix + ".environment.json")
    env = dict(os.environ, PYTHONHASHSEED="0", RUST_BACKTRACE="1")
    rustc = subprocess.run(["rustc", "-Vv"], text=True, capture_output=True, check=True).stdout
    cargo = subprocess.run(["cargo", "-V"], text=True, capture_output=True, check=True).stdout.strip()
    environment = {
        "binary": str(args.binary.resolve()), "binary_sha256": digest(args.binary),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_bytes(manifest_bytes),
        "profile": args.profile, "variants": variants, "workers": workers,
        "bounds": bounds, "repeats": args.repeats, "seed": args.seed,
        "rustc": rustc, "cargo": cargo, "python": sys.version,
        "cpu_name": cpu_name(), "logical_cpu_count": os.cpu_count(),
        "initial_affinity": sorted(original_affinity), "numa_nodes_online": numa_nodes(),
        "scalar_cpu": scalar_cpu, "parallel_affinity": sorted(original_affinity),
        "parallel_core_budget_policy": args.parallel_core_budget,
        "profiling_enabled": False,
        "sources_sha256": rust_sources(),
        "fixture_sha256": {row["case_id"]: row["fixture_sha256"] for row in cases},
        "environment": {key: env.get(key) for key in
                        ("RUSTFLAGS", "RUSTC_WRAPPER", "CARGO_PROFILE_RELEASE_LTO",
                         "CARGO_PROFILE_RELEASE_CODEGEN_UNITS")},
    }
    seen = {}
    try:
        with args.output.open("x", encoding="utf-8", buffering=1) as output, \
                environment_path.open("x", encoding="utf-8") as metadata:
            metadata.write(json.dumps(environment, indent=2, ensure_ascii=False) + "\n")
            for index, (repetition, case, variant, worker_count, bound,
                        rules, minimum) in enumerate(jobs, 1):
                is_parallel = variant in PARALLEL
                affinity = original_affinity if is_parallel else {scalar_cpu}
                if is_parallel and args.parallel_core_budget == "workers":
                    affinity = set(cpu_order[:worker_count])
                os.sched_setaffinity(0, affinity)
                fixture = RUST / case["file"]
                cmd = [str(args.binary.resolve()), "--input", str(fixture),
                       "--variant", variant, "--workers", str(worker_count),
                       "--rules", str(rules), "--min-frequency", str(minimum),
                       "--bounds", bound]
                started = time.perf_counter()
                run = subprocess.run(cmd, text=True, capture_output=True, timeout=3600,
                                     env=env)
                outer_seconds = time.perf_counter() - started
                if run.returncode:
                    raise RuntimeError((cmd, run.returncode, run.stdout, run.stderr))
                try:
                    result = json.loads(run.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError) as exc:
                    raise RuntimeError((cmd, run.stdout, run.stderr)) from exc
                if result.get("profiling_enabled") is not False:
                    raise AssertionError("profiling-enabled binary cannot be timed")
                if result.get("fixture_sha256") != case["fixture_sha256"]:
                    raise AssertionError(("fixture hash mismatch from binary", case["case_id"], result))
                key = (case["case_id"], rules, minimum)
                seen.setdefault(key, set()).add(result["fingerprint"])
                if len(seen[key]) != 1:
                    raise AssertionError(("semantic fingerprint mismatch", key, seen[key]))
                result.update({
                    "case_id": case["case_id"], "dataset": case.get("dataset"),
                    "split": case.get("split"), "variant": variant,
                    "workers": worker_count, "bounds": bound,
                    "repetition": repetition, "requested_rules": rules,
                    "min_frequency": minimum, "weight_scale": case.get("weight_scale", 1),
                    "input_sha256": case.get("input_sha256"),
                    "fixture_sha256": case["fixture_sha256"],
                    "outer_call_seconds": outer_seconds,
                    "cpu_affinity": sorted(affinity), "command": cmd,
                    "cpu_budget": len(affinity),
                })
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(json.dumps({"completed": index, "total": len(jobs),
                                  "case_id": case["case_id"], "variant": variant,
                                  "workers": worker_count, "bounds": bound,
                                  "train_cpu_seconds": result.get("train_cpu_seconds"),
                                  "call_seconds": result.get("call_seconds")}), flush=True)
    finally:
        os.sched_setaffinity(0, original_affinity)
    print(json.dumps({"rows": len(jobs), "semantic_groups": len(seen),
                      "all_fingerprints_stable": all(len(values) == 1
                                                      for values in seen.values()),
                      "output": str(args.output), "environment": str(environment_path)}))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()

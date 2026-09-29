"""Quick batch screen; use --smoke for a short multicore check and --full last.

Uses existing pinned fixtures; does not download or modify other workspaces.
Every stage runs sequentially, and a new output directory is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "rust"


def quick_manifest(byte_limit, rules):
    from ablation_fixtures import prepare, wire_bytes, write_immutable
    prior = {r["case_id"]: r for r in json.loads((RUST / "ablation_results/fixtures.json").read_text())}
    rows = []
    for lang in ("en", "zh"):
        source = ROOT / f"benchmarks/bpe_core_comparison/data/{lang}-4m.txt"
        raw = source.read_bytes()
        parent = prior[f"{lang}-4m-continuous"]
        if hashlib.sha256(raw).hexdigest() != parent["input_sha256"]:
            raise RuntimeError(f"source snapshot changed: {source}")
        # The source is valid UTF-8; only a partial final scalar can be dropped.
        text = raw[:byte_limit].decode("utf-8", errors="ignore")
        sample = text.encode("utf-8")
        prep = prepare([text])
        body = wire_bytes(prep)
        case = f"quick-{lang}-continuous-{byte_limit}"
        fixture = RUST / "fixtures/batch-quick" / f"{case}.json"
        write_immutable(fixture, body)
        rows.append({"case_id": case, "dataset": case, "split": "continuous-single-piece",
                     "source": str(source.relative_to(ROOT)), "source_sha256": parent["input_sha256"],
                     "source_byte_offset": 0, "input_bytes": len(sample),
                     "input_sha256": hashlib.sha256(sample).hexdigest(),
                     "file": str(fixture.relative_to(RUST)),
                     "fixture_sha256": hashlib.sha256(body).hexdigest(),
                     "corpus_positions": len(prep[0]), "initial_alphabet": len(prep[1]) - 1,
                     "stored_piece_count": 1, "piece_weight": 1,
                     "rules": rules, "min_frequency": 2})
    manifest = RUST / f"batch_results/quick-fixtures-{byte_limit}-{rules}.json"
    write_immutable(manifest, (json.dumps(rows, indent=2) + "\n").encode())
    return manifest, [r["case_id"] for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="only two 4 MiB continuous inputs, 1/4 CPUs, one repeat")
    mode.add_argument("--pilot", action="store_true", help="larger 4/16 MiB diagnostic, one repeat by default")
    mode.add_argument("--full", action="store_true", help="explicit final matrix, five repeats by default")
    parser.add_argument("--quick-bytes", type=int, default=262144)
    parser.add_argument("--quick-rules", type=int, default=512)
    parser.add_argument("--variants", help="comma-separated variants for a focused quick screen")
    parser.add_argument("--bounds", choices=("checked", "unchecked", "both"), default="checked",
                        help="access mode; use unchecked for a focused bounds-check ablation")
    args = parser.parse_args()
    quick = not (args.pilot or args.full or args.smoke)
    if args.repeats is None:
        args.repeats = 5 if args.full else 1
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.quick_bytes < 4 or args.quick_rules < 1:
        parser.error("quick bytes must be >= 4 and quick rules must be positive")
    if args.variants and not (quick or args.smoke):
        parser.error("--variants is for focused quick or smoke screens")
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
    if quick:
        manifest, cases = quick_manifest(args.quick_bytes, args.quick_rules)
        variants = [v.strip() for v in args.variants.split(",")] if args.variants else [
            "combined_filtered", "parallel_certified", "parallel_batch_relaxed"]
        exact = [v for v in variants if v != "parallel_batch_relaxed"]
        relaxed = [v for v in variants if v == "parallel_batch_relaxed"]
        stages = []
        if exact:
            stages.append(("quick-exact", manifest, cases, exact, "1,4"))
        if relaxed:
            stages.append(("quick-relaxed", manifest, cases, relaxed, "1,4"))
    elif args.smoke:
        variants = [v.strip() for v in args.variants.split(",")] if args.variants else ["combined_filtered", "parallel_certified"]
        stages = []
        for name, chosen in [("smoke-exact", [v for v in variants if v != "parallel_batch_relaxed"]),
                             ("smoke-relaxed", [v for v in variants if v == "parallel_batch_relaxed"])]:
            if chosen:
                stages.append((name, RUST / "ablation_results/fixtures.json",
                               ["en-4m-continuous", "zh-4m-continuous"], chosen, "1,4"))
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
                   "--workers", workers, "--bounds", args.bounds,
                   "--parallel-core-budget", "workers", "--repeats", str(args.repeats),
                   "--output", str(out / f"{name}.jsonl")]
        commands.append({"name": name, "command": command})
    (out / "commands.json").write_text(json.dumps({
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "repeats": args.repeats, "mode": "quick" if quick else "smoke" if args.smoke else "pilot" if args.pilot else "full",
        "purpose": "Screen changes cheaply; do not infer server-scale speedup from small one-shot samples." if quick else "Scaling diagnostic" if not args.full else "Final comparison",
        "protocol": "Entire child process uses p CPUs including the coordinator; p=1 and scalar both use CPU 5.",
        "stages": commands,
    }, indent=2) + "\n")
    started = time.perf_counter()
    total_rows = 0
    summary = []
    for stage in commands:
        print(f"Starting {stage['name']}", flush=True)
        with (out / f"{stage['name']}.progress.log").open("x") as log:
            subprocess.run(stage["command"], stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (out / f"{stage['name']}.jsonl").read_text().splitlines()]
        total_rows += len(rows)
        by_case = {}
        for row in rows:
            by_case.setdefault(row["case_id"], set()).add(row["fingerprint"])
        if any(len(fps) != 1 for fps in by_case.values()):
            raise RuntimeError(f"output mismatch within {stage['name']}")
        groups = {}
        for row in rows:
            key = (row["case_id"], row["variant"], row["workers"], row["bounds"])
            groups.setdefault(key, []).append(row)
        for (case, variant, workers, bounds), samples in sorted(groups.items()):
            seconds = median(row["call_seconds"] for row in samples)
            rss = median(row["vm_hwm_mib"] for row in samples)
            summary.append({"case_id": case, "variant": variant, "workers": workers,
                            "bounds": bounds, "samples": len(samples),
                            "median_call_seconds": seconds, "median_vm_hwm_mib": rss})
            print(f"  {case} {variant}/{workers}/{bounds}: {seconds:.4f}s, "
                  f"{rss:.1f} MiB (n={len(samples)})", flush=True)
        print(f"Completed {stage['name']}", flush=True)
    elapsed = time.perf_counter() - started
    completion = {"runs": total_rows, "matrix_seconds": elapsed, "semantic_checks_passed": True}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()

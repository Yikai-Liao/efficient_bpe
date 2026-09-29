"""Full-trace oracle checks for every native ablation and worker count."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RUST = ROOT / "rust"
sys.path.insert(0, str(ROOT / "benchmarks/bpe_core_comparison/python_rewrite"))
from common_fused import naive, prepare  # noqa: E402

DEFAULT_VARIANTS = (
    "full_clear", "endpoints", "lean", "packed", "unfused_endpoints",
    "unfused_halfword", "unfused_h3", "separate_counted", "linked12", "linked16",
    "bitmap_u32", "halfword", "h3", "h25", "filtered", "arena",
    "arena_counted", "filtered_h3", "combined", "combined_filtered",
    "certified_prefix_probe",
    "parallel_certified", "parallel_certified_single", "parallel_pair_owned",
    "parallel_pair_owned_compact", "parallel_pair_owned_single", "parallel_pair_owned_pipeline", "parallel_sparse_owner",
    "parallel_sparse_owner_all",
    "combined_filtered_h3", "combined_filtered_halfword",
    "bucket", "bucket_normalized", "parallel_broadcast", "parallel_owner",
    "parallel_occurrence", "parallel_occurrence_snapshot",
    "parallel_occurrence_adaptive", "parallel_occurrence_adaptive_256",
    "parallel_occurrence_adaptive_4096", "parallel_serial",
)
PARALLEL_VARIANTS = {"parallel_broadcast", "parallel_owner", "parallel_occurrence",
                     "parallel_occurrence_snapshot", "parallel_occurrence_adaptive",
                     "parallel_occurrence_adaptive_256",
                     "parallel_occurrence_adaptive_4096", "parallel_serial", "parallel_occurrence_grouped",
                     "parallel_occurrence_grouped_adaptive", "parallel_certified",
                     "parallel_certified_single", "parallel_pair_owned",
                     "parallel_pair_owned_compact", "parallel_pair_owned_single",
                     "parallel_pair_owned_pipeline",
                     "parallel_pair_owned_spatial",
                     "parallel_sparse_owner", "parallel_sparse_owner_all"}
COMPACT_VARIANTS = {"halfword", "h3", "h25", "filtered_h3", "unfused_halfword", "unfused_h3",
                    "combined_filtered_h3",
                    "combined_filtered_halfword"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepared_wire(prepared):
    return dict(zip(("corpus", "initial_lengths", "pivots", "weights"),
                    (list(part) for part in prepared)))


def make_cases(random_cases):
    cases = []
    def add(name, pieces, *, deduplicate=True, weight_scale=1, min_frequency=1,
            max_merges=40):
        prep = prepare(pieces, deduplicate)
        if weight_scale != 1:
            prep = prep[:3] + ([weight * weight_scale for weight in prep[3]],)
            min_frequency *= weight_scale
        cases.append((name, prep, min_frequency, max_merges))

    add("empty", [])
    add("one-symbol", ["a"])
    add("overlap-aaa", ["aaa", "aaaa", "aaaaa"] * 3,
        min_frequency=2, max_merges=24)
    add("piece-boundaries", ["ab", "c", "ab", "c", "abc", "a", "bc"],
        min_frequency=2, max_merges=32)
    add("continuous-whitespace-punctuation", ["  a,\n界 a!  "] * 3,
        min_frequency=2, max_merges=32)
    add("weighted64", ["ababab", "ababab", "abx", "abx", "ab"],
        weight_scale=64, min_frequency=2, max_merges=32)
    add("large-u64-weight", ["ababab", "ababab", "abx", "abx", "ab"],
        weight_scale=1 << 40, min_frequency=1, max_merges=32)
    add("single-run-long", ["a" * 4096] * 2,
        min_frequency=2, max_merges=24)
    add("single-piece-ab-long", ["ab" * 2048] * 2,
        min_frequency=2, max_merges=32)
    add("mixed-long-short-rare", ["a" * 257 + "b", "a" * 7 + "c",
                                   "ab", "ac", "zz", "rare-x"],
        min_frequency=2, max_merges=32)
    add("multiple-weight-groups", ["abcdabcd", "abcdabcd", "abcabc", "xyxy",
                                   "xyxy", "xyxy", "abab", "q"],
        min_frequency=2, max_merges=40)

    rng = random.Random(20260930)
    alphabet = "abcd界文🙂"
    for index in range(random_cases):
        pieces = ["".join(rng.choices(alphabet, k=rng.randint(1, 32)))
                  for _ in range(rng.randint(1, 8))]
        if index % 2 == 0:
            pieces.extend(pieces[:rng.randint(1, min(3, len(pieces)))])
        add(f"random-{index:03d}", pieces,
            deduplicate=bool(index % 2), min_frequency=1 + index % 4,
            max_merges=24)

    # A compact backend limitation case: a valid dense u16+ alphabet just past
    # 65535. Other variants still run with zero rules and are checked exactly.
    symbols = "".join(chr(0x1000 + i) for i in range(65536))
    cases.append(("initial-alphabet-65536", prepare([symbols]), 1, 0))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=RUST / "target/release/ablation")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--workers", default="1,2,4",
                        help="worker counts for parallel variants")
    parser.add_argument("--bounds", default="checked,unchecked")
    parser.add_argument("--random-cases", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=RUST / "ablation_results/differential.json")
    args = parser.parse_args()
    variants = tuple(dict.fromkeys(x.strip() for x in args.variants.split(",") if x.strip()))
    bounds = tuple(dict.fromkeys(x.strip() for x in args.bounds.split(",") if x.strip()))
    workers = tuple(dict.fromkeys(int(x) for x in args.workers.split(",") if x.strip()))
    if not variants or not bounds or any(x not in ("checked", "unchecked") for x in bounds):
        parser.error("provide variants and bounds from checked,unchecked")
    if any(n < 1 for n in workers):
        parser.error("worker counts must be positive")

    cases = make_cases(args.random_cases)
    compared = skipped = 0
    skip_rows = []
    fingerprints = {}
    with tempfile.TemporaryDirectory(prefix="bpe-rust-ablation-diff-") as temp:
        root = Path(temp)
        for case_index, (case_id, prepared, minimum, max_merges) in enumerate(cases):
            fixture = root / f"case-{case_index}.json"
            fixture.write_text(json.dumps(prepared_wire(prepared), separators=(",", ":")))
            expected_merges, expected_final = naive(prepared, max_merges, minimum)
            expected = {"merges": [list(row) for row in expected_merges],
                        "final": expected_final}
            fingerprint = sha256(json.dumps([expected_merges, expected_final]).encode())
            for variant in variants:
                counts = workers if variant in PARALLEL_VARIANTS else (1,)
                # The Rust backends now dispatch to const-generic checked and
                # unchecked implementations for every layout. Keep both paths
                # in differential coverage instead of assuming only endpoints
                # have a meaningful unchecked implementation.
                variant_bounds = bounds
                for worker_count in counts:
                    for bound in variant_bounds:
                        trace = root / "trace.json"
                        cmd = [str(args.binary), "--input", str(fixture),
                               "--variant", variant, "--workers", str(worker_count),
                               "--rules", str(max_merges), "--min-frequency", str(minimum),
                               "--bounds", bound, "--trace", str(trace)]
                        run = subprocess.run(cmd, text=True, capture_output=True)
                        if run.returncode:
                            diagnostic = run.stderr + "\n" + run.stdout
                            supported_limit = (
                                case_id == "initial-alphabet-65536"
                                and variant in COMPACT_VARIANTS
                                and any(word in diagnostic.lower() for word in
                                        ("alphabet", "u16", "16-bit", "65535",
                                         "compact", "unsupported"))
                            )
                            if supported_limit:
                                skip_rows.append({"case_id": case_id, "variant": variant,
                                                  "workers": worker_count, "bounds": bound,
                                                  "reason": diagnostic[-1000:]})
                                skipped += 1
                                continue
                            raise RuntimeError((cmd, run.returncode, diagnostic[-4000:]))
                        try:
                            result = json.loads(run.stdout.strip().splitlines()[-1])
                            observed = json.loads(trace.read_text())
                        except (IndexError, json.JSONDecodeError) as exc:
                            raise RuntimeError((cmd, run.stdout, run.stderr)) from exc
                        if observed != expected:
                            raise AssertionError((case_id, variant, worker_count, bound,
                                                  "trace mismatch", observed, expected))
                        if result.get("fingerprint") != fingerprint:
                            raise AssertionError((case_id, variant, worker_count, bound,
                                                  "fingerprint mismatch", result))
                        key = (case_id, minimum, max_merges)
                        fingerprints.setdefault(key, set()).add(result["fingerprint"])
                        compared += 1
    if any(len(values) != 1 for values in fingerprints.values()):
        raise AssertionError("semantic fingerprints differ across variants")
    report = {
        "cases": len(cases), "random_cases": args.random_cases,
        "compared_runs": compared, "explicit_compact_limit_skips": skipped,
        "variants": variants, "workers_for_parallel": workers, "bounds": bounds,
        "oracle": "full-recount Python naive; complete merge trace and final tokens",
        "all_compared_traces_match": True,
        "all_compared_fingerprints_match": True,
        "seed": 20260930,
        "binary_sha256": sha256(args.binary.read_bytes()),
        "skips": skip_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(body)
    print(json.dumps({key: value for key, value in report.items() if key != "skips"}))


if __name__ == "__main__":
    main()

"""Summarize immutable Rust ablation JSONL archives and their provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

TIME_FIELDS = (
    "train_seconds", "init_seconds", "merge_seconds", "call_cpu_seconds",
    "call_seconds", "outer_call_seconds", "vm_hwm_mib",
)
PARALLEL = {"parallel_broadcast", "parallel_owner", "parallel_occurrence",
            "parallel_occurrence_snapshot", "parallel_occurrence_adaptive",
            "parallel_occurrence_adaptive_256", "parallel_occurrence_adaptive_4096",
            "parallel_serial", "parallel_occurrence_grouped",
            "parallel_occurrence_grouped_adaptive"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_inputs(values):
    paths = []
    for value in values:
        paths.extend(Path(part).expanduser().resolve() for part in value.split(",") if part)
    if not paths:
        raise ValueError("at least one raw JSONL input is required")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate input path")
    return paths


def median_or_none(values):
    return statistics.median(values) if values else None


def numeric_summary(values):
    values = [value for value in values if isinstance(value, (int, float))
              and not isinstance(value, bool)]
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(values), "min": min(values),
            "median": median_or_none(values), "max": max(values)}


def semantic_key(row):
    return (row.get("case_id"), row.get("requested_rules"), row.get("min_frequency"))


def read_archives(paths, allow_pilot):
    all_rows = []
    provenance = []
    expected_by_file = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        env_path = path.with_suffix(path.suffix + ".environment.json")
        if not env_path.is_file():
            raise ValueError(f"missing environment metadata sidecar: {env_path}")
        env_bytes = env_path.read_bytes()
        metadata = json.loads(env_bytes)
        expected = metadata.get("repeats")
        if not isinstance(expected, int) or expected < 1:
            raise ValueError(f"invalid/missing expected repeats in {env_path}")
        if expected == 1 and not allow_pilot:
            raise ValueError(f"{path} is a one-repeat pilot; pass --allow-pilot n1 explicitly")
        file_rows = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                required = ("case_id", "variant", "bounds", "workers", "fingerprint")
                missing = [key for key in required if key not in row]
                if missing:
                    raise ValueError(f"missing {missing} at {path}:{line_number}")
                if row.get("profiling_enabled") is not False:
                    raise ValueError(f"profiling-enabled or unspecified row: {path}:{line_number}")
                file_rows.append(row)
        if not file_rows:
            raise ValueError(f"empty JSONL archive: {path}")

        # Check the number of samples per requested group within each archive,
        # before combining independently repeated files.
        per_group = {}
        for row in file_rows:
            key = (row["case_id"], row["variant"], row["bounds"], row["workers"],
                   row.get("requested_rules"), row.get("min_frequency"))
            per_group.setdefault(key, []).append(row)
        wrong = {key: len(rows) for key, rows in per_group.items() if len(rows) != expected}
        if wrong:
            raise ValueError(f"sample counts disagree with repeats={expected} in {path}: {wrong}")
        duplicate_repetitions = {}
        for key, samples in per_group.items():
            repetitions = [row.get("repetition") for row in samples]
            if any(value is not None for value in repetitions):
                if len(set(repetitions)) != len(repetitions):
                    duplicate_repetitions[key] = repetitions
        if duplicate_repetitions:
            raise ValueError(f"duplicate repetition IDs in {path}: {duplicate_repetitions}")

        expected_binary = metadata.get("binary_sha256")
        declared_fixtures = metadata.get("fixture_sha256") or {}
        for row in file_rows:
            if row.get("binary_sha256") not in (None, expected_binary):
                raise ValueError(f"row binary hash disagrees with metadata in {path}")
            case_hash = declared_fixtures.get(row["case_id"])
            if case_hash is not None and row.get("fixture_sha256") != case_hash:
                raise ValueError(f"row fixture hash disagrees with metadata in {path}")
        provenance.append({
            "path": str(path), "raw_sha256": sha256(path),
            "environment_path": str(env_path),
            "environment_sha256": hashlib.sha256(env_bytes).hexdigest(),
            "expected_repeats": expected, "rows": len(file_rows),
            "binary": metadata.get("binary"), "binary_sha256": expected_binary,
            "sources_sha256": metadata.get("sources_sha256"),
            "manifest": metadata.get("manifest"),
            "manifest_sha256": metadata.get("manifest_sha256"),
            "fixture_sha256": metadata.get("fixture_sha256"),
            "cpu_name": metadata.get("cpu_name"),
            "logical_cpu_count": metadata.get("logical_cpu_count"),
            "initial_affinity": metadata.get("initial_affinity"),
            "numa_nodes_online": metadata.get("numa_nodes_online"),
            "profile": metadata.get("profile"),
        })
        expected_by_file[str(path)] = expected
        for row in file_rows:
            row["_source_file"] = str(path)
        all_rows.extend(file_rows)
    binaries = {entry["binary_sha256"] for entry in provenance}
    if len(binaries) != 1:
        raise ValueError("do not combine different binaries in one performance summary")
    return all_rows, provenance, expected_by_file


def aggregate(rows):
    settings_by_case = {}
    for row in rows:
        settings_by_case.setdefault(row.get("case_id"), set()).add(
            (row.get("requested_rules"), row.get("min_frequency"))
        )
    mixed_settings = {case: sorted(settings, key=repr)
                      for case, settings in settings_by_case.items() if len(settings) > 1}
    if mixed_settings:
        raise ValueError(f"one case has mixed requested_rules/min_frequency values: {mixed_settings}")

    semantic = {}
    for row in rows:
        key = semantic_key(row)
        semantic.setdefault(key, set()).add(row["fingerprint"])
    inconsistent = {str(key): sorted(values) for key, values in semantic.items()
                    if len(values) != 1}
    if inconsistent:
        raise ValueError(f"semantic fingerprints disagree across files: {inconsistent}")

    grouped = {}
    for row in rows:
        key = (row["case_id"], row["variant"], row["bounds"], int(row["workers"]))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (case_id, variant, bounds, workers), samples in sorted(grouped.items()):
        fields = {name: numeric_summary([row.get(name) for row in samples])
                  for name in TIME_FIELDS}
        metric_names = sorted({
            name for row in samples for name, value in (row.get("metrics") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        })
        metrics = {}
        for name in metric_names:
            metrics[name] = numeric_summary([
                (row.get("metrics") or {}).get(name) for row in samples
            ])
        actual_workers = []
        for row in samples:
            value = (row.get("metrics") or {}).get("workers_actual")
            if value is None:
                value = row.get("workers_actual")
            if value is not None:
                actual_workers.append(value)
        cpu_med = fields["call_cpu_seconds"]["median"]
        wall_med = fields["call_seconds"]["median"]
        summaries.append({
            "case_id": case_id, "variant": variant, "bounds": bounds,
            "workers_requested": workers, "workers_actual": numeric_summary(actual_workers),
            "n": len(samples), "fingerprint": samples[0]["fingerprint"],
            "times": fields, "cpu_wall_utilization_median":
                (cpu_med / wall_med if cpu_med is not None and wall_med else None),
            "metrics": metrics,
            "available_metric_names": sorted({name for row in samples
                                              for name in row.get("metrics", {})}),
        })
    return summaries


def dominates(left, right):
    ltime = left["times"]["train_seconds"]["median"]
    rtime = right["times"]["train_seconds"]["median"]
    lmem = left["times"]["vm_hwm_mib"]["median"]
    rmem = right["times"]["vm_hwm_mib"]["median"]
    if None in (ltime, rtime, lmem, rmem):
        return False
    return ltime <= rtime and lmem <= rmem and (ltime < rtime or lmem < rmem)


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def add_comparisons(summaries):
    by_key = {(row["case_id"], row["variant"], row["bounds"],
               row["workers_requested"]): row for row in summaries}
    pareto = {"all": [], "scalar": [], "parallel": []}
    cases = sorted({row["case_id"] for row in summaries})
    for case in cases:
        bounds_values = sorted({row["bounds"] for row in summaries if row["case_id"] == case})
        workers_values = sorted({row["workers_requested"] for row in summaries if row["case_id"] == case})
        for bound in bounds_values:
            for workers in workers_values:
                rows = [r for r in summaries if r["case_id"] == case
                        and r["bounds"] == bound and r["workers_requested"] == workers]
                for label, subset in (("all", rows),
                                      ("scalar", [r for r in rows if r["variant"] not in PARALLEL]),
                                      ("parallel", [r for r in rows if r["variant"] in PARALLEL])):
                    front = [row for row in subset
                             if not any(other is not row and dominates(other, row)
                                        for other in subset)]
                    pareto[label].append({"case_id": case, "bounds": bound,
                                          "workers_requested": workers,
                                          "members": sorted(r["variant"] for r in front)})

    comparisons = []
    for row in summaries:
        case, variant, bound = row["case_id"], row["variant"], row["bounds"]
        workers = row["workers_requested"]
        packed = by_key.get((case, "packed", bound, 1))
        fallback = False
        if packed is None and bound != "checked":
            packed = by_key.get((case, "packed", "checked", 1))
            fallback = packed is not None
        train = row["times"]["train_seconds"]["median"]
        wall = row["times"]["call_seconds"]["median"]
        packed_train = packed["times"]["train_seconds"]["median"] if packed else None
        packed_wall = packed["times"]["call_seconds"]["median"] if packed else None

        parallel_w1 = None
        scaling = direct = efficiency = None
        parallel_w1_fallback = False
        actual_w = row["workers_actual"]["median"]
        if variant in PARALLEL:
            parallel_w1 = by_key.get((case, variant, bound, 1))
            if parallel_w1 is None and bound != "checked":
                parallel_w1 = by_key.get((case, variant, "checked", 1))
                parallel_w1_fallback = parallel_w1 is not None
            p1_wall = parallel_w1["times"]["call_seconds"]["median"] if parallel_w1 else None
            p1_train = parallel_w1["times"]["train_seconds"]["median"] if parallel_w1 else None
            scaling = {"callwall_w1_over_w": ratio(p1_wall, wall),
                       "train_w1_over_w": ratio(p1_train, train)}
            efficiency = ratio(p1_wall, wall * actual_w if wall and actual_w else None)
            direct = {"callwall_packed_w1_over_parallel": ratio(packed_wall, wall),
                      "train_packed_w1_over_parallel": ratio(packed_train, train)}
        comparisons.append({
            "case_id": case, "variant": variant, "bounds": bound,
            "workers_requested": workers, "workers_actual_median": actual_w,
            "packed_reference_bounds": packed["bounds"] if packed else None,
            "packed_reference_fallback_to_checked": fallback,
            "speedup_vs_packed": {
                "callwall": ratio(packed_wall, wall), "train": ratio(packed_train, train),
            },
            "parallel_strong_scaling_speedup": scaling,
            "parallel_w1_reference_bounds": parallel_w1["bounds"] if parallel_w1 else None,
            "parallel_w1_fallback_to_checked": parallel_w1_fallback,
            "parallel_efficiency_callwall_per_actual_worker": efficiency,
            "parallel_vs_direct_packed_acceleration": direct,
        })
    return pareto, comparisons


def markdown(report):
    lines = ["# Rust Ablation Summary", "",
             f"Inputs: {len(report['provenance'])}; raw rows: {report['raw_row_count']};",
             f"groups: {len(report['groups'])}.", "",
             "All times and memory below are per `(case, variant, bounds, workers)` group. "
             "Speedups compare medians; different cases are never pooled into a single mean.", "",
             "## Group medians", "",
             "| Case | Variant | Bounds | Workers req/actual | n | Train s (med [min,max]) | "
             "Init s (med [min,max]) | Merge s (med [min,max]) | Call CPU s (med [min,max]) | "
             "Call wall s (med [min,max]) | VmHWM MiB (med [min,max]) | CPU/wall |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        t = group["times"]
        def val(name):
            summary = t[name]
            value = summary["median"]
            if value is None:
                return "—"
            return f"{value:.5g} [{summary['min']:.5g},{summary['max']:.5g}]"
        actual = group["workers_actual"]["median"]
        actual = "—" if actual is None else f"{actual:g}"
        util = group["cpu_wall_utilization_median"]
        util = "—" if util is None else f"{util:.3f}"
        lines.append(f"| {group['case_id']} | {group['variant']} | {group['bounds']} | "
                     f"{group['workers_requested']}/{actual} | {group['n']} | "
                     f"{val('train_seconds')} | {val('init_seconds')} | {val('merge_seconds')} | "
                     f"{val('call_cpu_seconds')} | {val('call_seconds')} | "
                     f"{val('vm_hwm_mib')} | {util} |")
    lines.extend(["", "## Per-case Pareto fronts", "",
                  "Fronts minimize median `train_seconds` and median `vm_hwm_mib`; "
                  "they are stratified by case, bounds, and requested workers.", ""])
    for cohort in ("scalar", "parallel", "all"):
        lines.extend([f"### {cohort}", ""])
        for row in report["pareto"][cohort]:
            lines.append(f"- `{row['case_id']}` bounds={row['bounds']} workers="
                         f"{row['workers_requested']}: {', '.join(row['members']) or 'none'}")
        lines.append("")
    lines.extend(["## Speed and parallel comparisons", "",
                  "Packed comparisons use matching bounds when present, otherwise checked is "
                  "explicitly marked. Parallel strong scaling compares the same parallel variant "
                  "at one requested worker with the target worker count; direct packed comparison "
                  "is reported separately.", "",
                  "| Case | Variant | Bounds | Workers req/actual | Packed bounds | Train x packed | "
                  "Wall x packed | Parallel x direct packed (train/wall) | "
                  "Same-variant parallel wall speedup | Efficiency |",
                  "|---|---|---|---:|---|---:|---:|---:|---:|---:|"])
    for row in report["comparisons"]:
        def fmt(value):
            return "—" if value is None else f"{value:.4g}"
        packed_bounds = row["packed_reference_bounds"] or "—"
        if row["packed_reference_fallback_to_checked"]:
            packed_bounds += " (fallback)"
        scaling = row["parallel_strong_scaling_speedup"] or {}
        direct = row["parallel_vs_direct_packed_acceleration"] or {}
        lines.append(f"| {row['case_id']} | {row['variant']} | {row['bounds']} | "
                     f"{row['workers_requested']}/{fmt(row['workers_actual_median'])} | "
                     f"{packed_bounds} | {fmt((row['speedup_vs_packed'] or {}).get('train'))} | "
                     f"{fmt((row['speedup_vs_packed'] or {}).get('callwall'))} | "
                     f"{fmt(direct.get('train_packed_w1_over_parallel'))}/"
                     f"{fmt(direct.get('callwall_packed_w1_over_parallel'))} | "
                     f"{fmt(scaling.get('callwall_w1_over_w'))} | "
                     f"{fmt(row['parallel_efficiency_callwall_per_actual_worker'])} |")
    lines.extend(["", "## Provenance", "",
                  "Each input's raw-file hash, environment sidecar hash, binary/source hashes, "
                  "fixture hashes, and host metadata are preserved in the JSON report.", ""])
    for source in report["provenance"]:
        lines.append(f"- `{source['path']}`: raw SHA-256 `{source['raw_sha256']}`, "
                     f"environment SHA-256 `{source['environment_sha256']}`, "
                     f"expected repeats {source['expected_repeats']}, binary "
                     f"`{source['binary_sha256']}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="comma-separated paths and/or multiple raw JSONL paths")
    parser.add_argument("--output-prefix", type=Path, required=True,
                        help="new prefix producing .json and .md; existing files are not overwritten")
    parser.add_argument("--allow-pilot", choices=("n1",), default=None,
                        help="explicitly permit archives declaring only one repeat")
    args = parser.parse_args()
    try:
        paths = parse_inputs(args.inputs)
        rows, provenance, expected = read_archives(paths, args.allow_pilot == "n1")
        groups = aggregate(rows)
        pareto_fronts, comparisons = add_comparisons(groups)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    output_prefix = args.output_prefix
    json_path = output_prefix.with_suffix(".json")
    markdown_path = output_prefix.with_suffix(".md")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "input_paths": [str(path) for path in paths],
        "raw_row_count": len(rows), "expected_repeats_by_file": expected,
        "provenance": provenance, "groups": groups,
        "pareto": pareto_fronts, "comparisons": comparisons,
        "fingerprints_consistent_by_case_and_training_request": True,
        "method": {
            "summary": "min/median/max over independent process rows",
            "pareto_objectives": ["median train_seconds", "median vm_hwm_mib"],
            "pareto_strata": ["case_id", "bounds", "workers_requested"],
            "no_cross_case_mean": True,
            "metrics_summary": "all numeric metrics present in input rows, including queue, round, byte, plan, reduce, sync, and worker-wait counters",
            "cpu_wall_utilization": "median call_cpu_seconds / median call_seconds",
            "parallel_efficiency": "parallel callwall at 1 worker / callwall at w / actual_workers",
            "source_hash_validation": "provenance preserved per archive; not compared to current source tree",
        },
    }
    json_body = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    markdown_body = markdown(report)
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"summary output already exists: {json_path} or {markdown_path}")
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(json_body)
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(markdown_body)
    print(json.dumps({"rows": len(rows), "groups": len(groups),
                      "json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()

"""Validate and summarize the boundary-topology JSONL without rerunning Rust.

The microbenchmark times one precomputed merge trace. Its layout buffers and
operation contracts are not a measurement of complete BPE training.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import median


RUST = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = RUST / "ablation_results/formal/topology.jsonl"
DEFAULT_JSON = RUST / "ablation_results/topology-summary.json"
DEFAULT_MARKDOWN = RUST / "ablation_results/topology-summary.md"
KEYS = ("variant", "length", "pattern", "bounds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    values = sorted(values)
    return {"min": values[0], "median": median(values), "max": values[-1]}


def read_rows(path):
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if raw.strip():
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at line {line_number}") from exc
    require(rows, "input is empty")
    return rows, digest.hexdigest()


def summarize(rows, digest, input_path):
    grouped = defaultdict(list)
    checksums = defaultdict(set)
    contracts = defaultdict(set)
    requested = set()
    skips = Counter()
    for row in rows:
        key = tuple(row[field] for field in KEYS)
        grouped[key].append(row)
        length = row["length"]
        request = row["requested_positions"]
        requested.add(request)
        blocks = max(1, request // length)
        expected_positions = blocks * (length + 1) + 1
        expected_operations = blocks * (length - 1)
        require(row["positions"] == expected_positions,
                f"actual position count differs from sentinels formula: {key}")
        require(row["positions"] >= 3, f"invalid position count: {key}")
        if row["skipped"]:
            skips[row["skip_reason"]] += 1
            require(row["variant"] == "u8_only" and length > 255,
                    f"unexpected skip: {key}")
            continue
        require(row["operations"] == expected_operations,
                f"operation count mismatch: {key}")
        require(math.isfinite(row["seconds"]) and row["seconds"] > 0,
                f"invalid timing: {key}")
        require(row["buffer_bytes"] > 0 and row["capacity_bytes"] >= row["buffer_bytes"],
                f"invalid buffer/capacity: {key}")
        if row["variant"] in {"bytespans", "u8_only"}:
            require(row["buffer_bytes"] == row["positions"]
                    and row["capacity_bytes"] == row["positions"],
                    f"one-byte component buffer changed: {key}")
        if row["variant"] in {"full_clear", "lean"}:
            require(row["buffer_bytes"] == 4 * row["positions"],
                    f"four-byte endpoint buffer changed: {key}")
        require(row["train_seconds"] == row["seconds"],
                f"time alias mismatch: {key}")
        checksums[(length, row["pattern"])].add(row["checksum"])
        contracts[row["variant"]].add((row["operation_contract"], row["component_only"]))
    require(len(requested) == 1, "grouping requires one requested_positions value")
    require(all(len(values) == 1 for values in checksums.values()),
            "checksum differs across variants, bounds, or repetitions")
    require(all(len(values) == 1 for values in contracts.values()),
            "operation contract differs within a variant")
    require(contracts["bytespans"] ==
            {("prev+next+next+checked_merge_no_id", True)},
            "ByteSpans contract changed")
    require(contracts["u8_only"] ==
            {("prev+next+next+merge_known_no_id", True)},
            "u8_only contract changed")
    require(all(value == {("inspect_pair+merge_known", False)}
                for variant, value in contracts.items()
                if variant not in {"bytespans", "u8_only"}),
            "ID-bearing contract changed")
    dimensions = [{row[field] for row in rows} for field in KEYS]
    require(set(grouped) == set(product(*dimensions)),
            "missing variant × length × pattern × bounds group")

    entries = []
    repeat_counts = Counter()
    for key, records in sorted(grouped.items()):
        reps = sorted(row["repetition"] for row in records)
        require(reps == list(range(len(reps))), f"missing/duplicate repetitions: {key}")
        repeat_counts[len(reps)] += 1
        variant, length, pattern, bounds = key
        common = dict(zip(KEYS, key))
        common["repetitions"] = len(reps)
        common["positions"] = records[0]["positions"]
        if all(row["skipped"] for row in records):
            reasons = {row["skip_reason"] for row in records}
            require(len(reasons) == 1, f"mixed skip reasons: {key}")
            entries.append({**common, "skipped": True, "skip_reason": reasons.pop()})
            continue
        require(not any(row["skipped"] for row in records), f"mixed skipped group: {key}")
        stable = ("positions", "operations", "checksum", "buffer_bytes",
                  "capacity_bytes", "component_only", "operation_contract")
        for field in stable:
            require(len({row[field] for row in records}) == 1,
                    f"{field} changes across repetitions: {key}")
        first = records[0]
        seconds = stats([row["seconds"] for row in records])
        entries.append({
            **common, "skipped": False, "operations": first["operations"],
            "checksum": first["checksum"], "seconds": seconds,
            "ns_per_operation": {name: value * 1e9 / first["operations"]
                                 for name, value in seconds.items()},
            "buffer_bytes": first["buffer_bytes"],
            "capacity_bytes": first["capacity_bytes"],
            "buffer_bytes_per_position": first["buffer_bytes"] / first["positions"],
            "capacity_bytes_per_position": first["capacity_bytes"] / first["positions"],
            "vm_hwm_mib": stats([row["vm_hwm_mib"] for row in records]),
            "component_only": first["component_only"],
            "operation_contract": first["operation_contract"],
        })
    require(len(repeat_counts) == 1 and next(iter(repeat_counts)) == 5,
            "expected exactly five repetitions in every group")
    require(sum(len(records) for records in grouped.values()) == len(rows),
            "rows were lost during grouping")
    requested_value = next(iter(requested))
    lengths = sorted({row["length"] for row in rows})
    return {
        "source": str(input_path), "source_sha256": digest,
        "measurement_scope": "boundary inspection and merge on a precomputed trace; "
                             "excludes training histogram, heap, occurrence index, "
                             "fixture construction, and final traversal",
        "requested_positions": requested_value,
        "integrity": {
            "rows": len(rows), "groups": len(entries),
            "valid_rows": len(rows) - sum(skips.values()),
            "skipped_rows": sum(skips.values()), "groups_by_repetition_count": dict(repeat_counts),
            "checksum_cases_verified": len(checksums),
            "checksum_agreement": True, "position_formula_verified": True,
            "operation_formula_verified": True,
            "buffer_capacity_order_verified": True,
            "skip_reasons": dict(sorted(skips.items())),
        },
        "contracts": {variant: {"operation_contract": next(iter(values))[0],
                                "component_only": next(iter(values))[1]}
                      for variant, values in sorted(contracts.items())},
        "lengths": [
            {"length": length,
             "blocks": max(1, requested_value // length),
             "positions": max(1, requested_value // length) * (length + 1) + 1,
             "operations": max(1, requested_value // length) * (length - 1)}
            for length in lengths
        ],
        "groups": entries,
    }


def markdown(summary):
    entries = summary["groups"]
    valid = [row for row in entries if not row["skipped"]]
    by_key = {(row["variant"], row["length"], row["pattern"], row["bounds"]): row
              for row in valid}
    variants = sorted(summary["contracts"])
    lengths = sorted({row["length"] for row in entries})
    position_map = {row["length"]: row["positions"] for row in entries}
    integrity = summary["integrity"]
    out = [
        "# Boundary-topology microbenchmark summary", "",
        f"Source: `formal/topology.jsonl` (SHA-256 `{summary['source_sha256']}`). "
        "Reproduce with `python3 rust/tools/ablation_micro_summarize.py` from the repository root.",
        "",
        f"Validated {integrity['rows']} rows in {integrity['groups']} groups; each "
        "variant × length × pattern × bounds group has five repetitions. "
        f"There are {integrity['valid_rows']} timed rows and "
        f"{integrity['skipped_rows']} explicit skips. All "
        f"{integrity['checksum_cases_verified']} length × pattern cases agree on checksum "
        "across every available variant, bound mode, and repetition. Actual positions "
        "and operation counts match the fixture formula below; buffer capacity is never "
        "below logical buffer size.",
        "",
        f"The fixture uses `blocks = max(1, floor({summary['requested_positions']} / length))`, "
        "`positions = blocks × (length + 1) + 1`, and "
        "`operations = blocks × (length − 1)`. The extra positions are the initial "
        "sentinel and one separator per block. Timings below are the internal `seconds` "
        "field, with min/median/max over five isolated runs; process startup is excluded.",
        "",
        "This microbenchmark replays a precomputed trace and times boundary inspection "
        "and merge only. It excludes BPE frequency counting, candidate selection, "
        "occurrence indexing, fixture creation, and final traversal. It does not rank "
        "complete trainers. The two 1-byte components have a different operation "
        "contract from the ID-bearing backends, and their timings should not be used "
        "as direct full-trainer speed comparisons.",
        "",
        "Logical/capacity figures in the next table use length 32; exact byte counts "
        "for every group are in the JSON. They count backend vector buffers only, "
        "excluding Vec headers, allocator overhead, and the prepared trace.",
        "", "| Variant | Logical bytes / position | Capacity bytes / position | Timed contract |",
        "|---|---:|---:|---|",
    ]
    for variant in variants:
        row = next(r for r in valid if r["variant"] == variant and r["length"] == 32)
        contract = summary["contracts"][variant]
        suffix = " (component only)" if contract["component_only"] else ""
        out.append(f"| `{variant}` | {row['buffer_bytes_per_position']:.4f} | "
                   f"{row['capacity_bytes_per_position']:.4f} | "
                   f"`{contract['operation_contract']}`{suffix} |")
    out += [
        "", "## Length boundaries", "",
        "`ByteSpans` remains exactly 1 logical and capacity byte per original "
        "position at length 255, 256, 65,535, and 65,536. Its long spans use "
        "multiple endpoint tag bytes within that one-byte-per-position array; "
        "these figures exclude token IDs. The `bounds` switch has no effect on "
        "either component-only backend, whose calls remain checked. `u8_only` "
        "explicitly skips all five tested "
        "lengths above 255 (`u8_only supports length <=255`: 150 rows).",
        "",
        "| Length | Actual positions | Operations | ByteSpans bytes | u8_only |",
        "|---:|---:|---:|---:|---|",
    ]
    for length in lengths:
        row = by_key[("bytespans", length, "chain", "checked")]
        u8 = next(r for r in entries if r["variant"] == "u8_only"
                  and r["length"] == length and r["pattern"] == "chain"
                  and r["bounds"] == "checked")
        status = "skipped" if u8["skipped"] else "measured"
        out.append(f"| {length:,} | {position_map[length]:,} | {row['operations']:,} | "
                   f"{row['buffer_bytes']:,} | {status} |")
    out += [
        "", "## Chain clearing cost", "",
        "On the chain trace, `full_clear` fills the entire merged interior on each "
        "operation. `lean` updates two or three endpoint cells. Both use the same "
        "four-byte ID array, and operation counts remain around 127k–131k across "
        "lengths. For a chain of length L, full clearing does Θ(L²) writes per block "
        "and lean does Θ(L); with this approximately fixed-size corpus, that means "
        "Θ(NL) versus Θ(N) boundary writes. The measured medians reflect this "
        "difference; they are not a prediction of whole-trainer speed.",
        "", "| Length | Full clear checked (ms) | Lean checked (ms) | Ratio |",
        "|---:|---:|---:|---:|",
    ]
    for length in lengths:
        full = by_key[("full_clear", length, "chain", "checked")]["seconds"]["median"]
        lean = by_key[("lean", length, "chain", "checked")]["seconds"]["median"]
        out.append(f"| {length:,} | {full*1000:.3f} | {lean*1000:.3f} | {full/lean:.1f}× |")
    out += [
        "", "Each cell's min/median/max, operation count, buffer and capacity bytes, "
        "and process peak RSS are in `topology-summary.json`. The `vm_hwm_mib` field "
        "is process peak RSS, not backend buffer size; fixture and runtime allocations "
        "also contribute. The selected layouts and timed contracts are defined in "
        "[`micro.rs`](../src/ablation/micro.rs), "
        "[`backends.rs`](../src/ablation/backends.rs), and "
        "[`spans.rs`](../src/ablation/spans.rs).", "",
    ]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    rows, digest = read_rows(args.input)
    summary = summarize(rows, digest, args.input)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(summary), encoding="utf-8")
    print(f"{len(rows)} rows, {len(summary['groups'])} groups, "
          f"{summary['integrity']['skipped_rows']} skips; "
          f"wrote {args.json} and {args.markdown}")


if __name__ == "__main__":
    main()

"""Plot same-binary parallel scaling from ablation_summarize.py summary.json.

Only median call wall time is divided to obtain speedup. Whiskers on the
absolute-time panels are the five-run min/max ranges, not confidence bounds.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MAIN_CASES = (
    ("en-4m-continuous", "English 4 MiB · continuous"),
    ("zh-4m-continuous", "Chinese 4 MiB · continuous"),
    ("en-1m--paragraph", "English 1 MiB · paragraphs"),
)
LARGE_CASES = (
    ("en-16m-continuous", "English 16 MiB · continuous"),
    ("zh-16m-continuous", "Chinese 16 MiB · continuous"),
)
# The full names are the native trainer variant names; the labels remain short.
VARIANTS = (
    ("parallel_occurrence_snapshot", "Snapshot", "#756bb1"),
    ("parallel_occurrence_adaptive", "Adaptive", "#e69f00"),
    ("parallel_occurrence_grouped", "Grouped", "#0072b2"),
    ("parallel_occurrence_grouped_adaptive", "Grouped adaptive", "#009e73"),
)


def value(group, field="call_seconds", stat="median"):
    return (group.get("times", {}).get(field) or {}).get(stat)


def actual_workers(group):
    actual = group.get("workers_actual")
    if isinstance(actual, dict):
        return actual.get("median")
    return actual if isinstance(actual, (int, float)) else None


def select_bounds(groups):
    variants = {name for name, _, _ in VARIANTS}
    cases = {case for case, _ in MAIN_CASES}
    counts = {}
    for group in groups:
        if group.get("variant") in variants and group.get("case_id") in cases:
            bounds = group.get("bounds")
            counts[bounds] = counts.get(bounds, 0) + 1
    if not counts:
        raise ValueError("no requested parallel variants found")
    # One bound mode for the entire graphic; never combine checked and unchecked.
    return max(counts, key=lambda key: (counts[key], key == "checked"))


def check_single_binary(report):
    provenance = report.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("summary must include provenance with one binary hash")
    hashes = {item.get("binary_sha256") for item in provenance}
    if len(hashes) != 1 or None in hashes or "" in hashes:
        raise ValueError("refusing to mix archives from different or unknown binaries")
    return next(iter(hashes))


def index_groups(groups, bounds):
    result = {}
    for group in groups:
        if group.get("bounds") != bounds:
            continue
        key = (group.get("case_id"), group.get("variant"), group.get("workers_requested"))
        if key in result:
            raise ValueError(f"duplicate summary group: {key}")
        result[key] = group
    return result


def best_scalar(groups, case_id, bounds):
    candidates = (
        (value(group), group.get("variant"))
        for group in groups
        if group.get("case_id") == case_id
        and group.get("bounds") == bounds
        and group.get("workers_requested") == 1
        and not group.get("variant", "").startswith("parallel_")
    )
    return min(((seconds, name) for seconds, name in candidates
                if seconds is not None and seconds > 0), default=None)


def plot_panel(top, bottom, case_id, title, by_key, groups, bounds, annotate_actual):
    present = False
    requested = set()
    actual_notes = []
    for variant, label, color in VARIANTS:
        series = sorted(
            (workers, group) for (case, name, workers), group in by_key.items()
            if case == case_id and name == variant and isinstance(workers, int)
        )
        one = by_key.get((case_id, variant, 1))
        one_time = value(one) if one else None
        if one_time is None or one_time <= 0:
            continue  # Same-algorithm 1-worker denominator is mandatory.
        valid = [(workers, group) for workers, group in series
                 if value(group) is not None and value(group) > 0]
        if not valid:
            continue
        present = True
        xs = [workers for workers, _ in valid]
        mids = [value(group) for _, group in valid]
        requested.update(xs)
        top.plot(xs, [one_time / seconds for seconds in mids], marker="o",
                 markersize=5, linewidth=1.7, color=color, label=label)
        lows = [max(0.0, mid - (value(group, stat="min") or mid))
                for mid, (_, group) in zip(mids, valid)]
        highs = [max(0.0, (value(group, stat="max") or mid) - mid)
                 for mid, (_, group) in zip(mids, valid)]
        bottom.errorbar(xs, mids, yerr=[lows, highs], marker="o", markersize=5,
                        linewidth=1.7, capsize=2, color=color)
        if annotate_actual:
            for workers, group in valid:
                actual = actual_workers(group)
                if actual is not None and actual != workers:
                    actual_notes.append(f"{label} {workers}→{actual:g}")
    if not present:
        top.text(0.5, 0.5, "No comparable one-worker series", ha="center",
                 va="center", transform=top.transAxes, color="#666666")
        bottom.text(0.5, 0.5, "No matching rows", ha="center", va="center",
                    transform=bottom.transAxes, color="#666666")
    top.axhline(1.0, color="#777777", linewidth=0.9, linestyle="--")
    if requested:
        ideal = sorted(requested)
        top.plot(ideal, ideal, color="#bbbbbb", linestyle=":", linewidth=0.9,
                 zorder=0)
        top.set_xticks(ideal)
        bottom.set_xticks(ideal)
    scalar = best_scalar(groups, case_id, bounds)
    if scalar:
        bottom.axhline(scalar[0], color="#444444", linestyle="--", linewidth=1.1)
        bottom.text(0.98, scalar[0], f" best direct scalar ({scalar[1]})",
                    ha="right", va="bottom", color="#444444", fontsize=7,
                    transform=bottom.get_yaxis_transform())
    else:
        bottom.text(0.98, 0.97, "No same-binary direct scalar row",
                    ha="right", va="top", color="#666666", fontsize=7,
                    transform=bottom.transAxes)
    top.set_title(title, fontsize=11)
    top.set_ylabel("Same-variant speedup (×)")
    bottom.set_ylabel("Call wall time (s)")
    bottom.set_xlabel("Requested workers")
    if actual_notes:
        note = "Actual workers: " + ", ".join(actual_notes)
        bottom.text(0.02, 0.02, note, transform=bottom.transAxes,
                    ha="left", va="bottom", fontsize=6.5, color="#555555")
    for ax in (top, bottom):
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)


def make_figure(cases, by_key, groups, bounds, title):
    columns = len(cases)
    fig, axes = plt.subplots(2, columns, figsize=(5.4 * columns, 8.0),
                             squeeze=False, sharex="col")
    for j, (case_id, label) in enumerate(cases):
        plot_panel(axes[0, j], axes[1, j], case_id, label, by_key, groups,
                   bounds, annotate_actual=True)
    handles = [Line2D([0], [0], marker="o", color=color, label=label,
                      linewidth=1.7) for variant, label, color in VARIANTS
               if any((case, variant, 1) in by_key for case, _ in cases)]
    handles.append(Line2D([0], [0], color="#444444", linestyle="--",
                          label="Best direct scalar (same binary/bounds)"))
    fig.legend(handles=handles, loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 0.955), fontsize=9)
    fig.suptitle(f"{title} · {bounds} bounds", y=0.995, fontsize=14)
    fig.text(0.5, 0.012,
             "Speedup = median one-worker call / median p-worker call for the same algorithm. "
             "Time whiskers = observed min–max over five runs, not confidence intervals. "
             "Grey dotted line = ideal linear speedup.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.905), h_pad=1.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="single-binary ablation_summarize.py summary.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    binary_hash = check_single_binary(report)
    groups = report.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("summary has no groups")
    bounds = select_bounds(groups)
    by_key = index_groups(groups, bounds)
    if not any((case, variant, 1) in by_key for case, _ in MAIN_CASES
               for variant, _, _ in VARIANTS):
        raise ValueError("main cases have no one-worker parallel reference")
    parallel_names = {name for name, _, _ in VARIANTS}
    large = all(any(key[0] == case and key[1] in parallel_names and key[2] == 1
                    for key in by_key) for case, _ in LARGE_CASES)
    stems = ["parallel_speedup_main"] + (["parallel_speedup_16m"] if large else [])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    paths = [output_dir / f"{stem}.{ext}" for stem in stems
             for ext in ("svg", "png")]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    figures = [("parallel_speedup_main",
                make_figure(MAIN_CASES, by_key, groups, bounds,
                            "Parallel scaling: 4 MiB continuous and 1 MiB paragraph"))]
    if large:
        figures.append(("parallel_speedup_16m",
                        make_figure(LARGE_CASES, by_key, groups, bounds,
                                    "Parallel scaling: 16 MiB continuous")))
    try:
        for stem, fig in figures:
            fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
            fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    finally:
        for _, fig in figures:
            plt.close(fig)
    print(json.dumps({"input": str(args.input.resolve()),
                      "binary_sha256": binary_hash, "bounds": bounds,
                      "files": [str(path.resolve()) for path in paths]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

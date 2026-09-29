"""Render publication-friendly ablation plots from a summary JSON artifact."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCALAR_CASES = (
    ("en-1m--regex", "English 1 MiB, regex"),
    ("zh-1m--regex", "Chinese 1 MiB, regex"),
    ("en-1m--paragraph", "English 1 MiB, paragraph"),
    ("zh-4m--regex", "Chinese 4 MiB, regex"),
)
SCALAR_VARIANTS = (
    "packed", "linked12", "halfword", "h3", "arena_counted",
    "combined_filtered", "combined_filtered_halfword",
)
PARALLEL_CASES = (
    ("en-4m-continuous", "English 4 MiB, one continuous piece"),
    ("zh-4m-continuous", "Chinese 4 MiB, one continuous piece"),
    ("en-1m--paragraph", "English 1 MiB, paragraph pieces"),
)
PARALLEL_VARIANTS = (
    "parallel_broadcast", "parallel_owner", "parallel_occurrence",
    "parallel_occurrence_snapshot", "parallel_occurrence_adaptive",
)
COLORS = {
    "packed": "#3b4cc0", "linked12": "#1f9e89", "halfword": "#73d055",
    "h3": "#d8a600", "arena_counted": "#e76f51",
    "combined_filtered": "#9b5de5", "combined_filtered_halfword": "#f15bb5",
    "parallel_broadcast": "#3b4cc0", "parallel_owner": "#1f9e89",
    "parallel_occurrence": "#e76f51", "parallel_occurrence_snapshot": "#9b5de5",
    "parallel_occurrence_adaptive": "#d8a600",
}


LABELS = {"combined_filtered": "combined+filter",
          "combined_filtered_halfword": "combined+filter+u16",
          "arena_counted": "arena+filter"}


def median(group, field):
    return (group.get("times", {}).get(field) or {}).get("median")


def asymmetric_error(group, field):
    values = group.get("times", {}).get(field) or {}
    mid, low, high = values.get("median"), values.get("min"), values.get("max")
    if None in (mid, low, high):
        return None
    return [[max(0.0, mid - low)], [max(0.0, high - mid)]]


def save_both(fig, output_dir, stem):
    for suffix in (".svg", ".png"):
        path = output_dir / f"{stem}{suffix}"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")


def plot_scalar(groups, pareto, output_dir):
    selected = [g for g in groups if g.get("bounds") == "checked"
                and g.get("workers_requested") == 1]
    by_key = {(g["case_id"], g["variant"]): g for g in selected}
    fronts = {(r["case_id"], variant)
              for r in pareto.get("scalar", [])
              if r.get("bounds") == "checked" and r.get("workers_requested") == 1
              for variant in r.get("members", [])}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    for ax, (case, title) in zip(axes.flat, SCALAR_CASES):
        plotted = 0
        for variant in SCALAR_VARIANTS:
            group = by_key.get((case, variant))
            if not group:
                continue
            x = median(group, "train_seconds")
            y = median(group, "vm_hwm_mib")
            if x is None or y is None:
                continue
            xerr = asymmetric_error(group, "train_seconds")
            yerr = asymmetric_error(group, "vm_hwm_mib")
            is_pareto = (case, variant) in fronts
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="*" if is_pareto else "o",
                        color=COLORS[variant], markersize=9 if is_pareto else 6,
                        capsize=2, alpha=0.9, zorder=3)
            offset = (5, -12) if variant in ("halfword", "combined_filtered_halfword") else (5, 5)
            if case == "zh-1m--regex" and variant == "arena_counted":
                offset = (5, -14)
            elif case == "zh-1m--regex" and variant == "combined_filtered_halfword":
                offset = (5, -8)
            elif case == "en-1m--regex" and variant == "combined_filtered":
                offset = (5, -7)
            ax.annotate(LABELS.get(variant, variant), (x, y), xytext=offset, textcoords="offset points",
                        fontsize=8, color=COLORS[variant])
            plotted += 1
        ax.set_title(title)
        ax.set_xlabel("Training time (s, median; whiskers min–max)")
        ax.set_ylabel("VmHWM (MiB, median; whiskers min–max)")
        ax.grid(True, alpha=0.25)
        if not plotted:
            ax.text(0.5, 0.5, "No matching summary rows", ha="center", va="center",
                    transform=ax.transAxes, color="#666666")
    legend = [Line2D([0], [0], marker="*", color="none", markerfacecolor="#333333",
                     markeredgecolor="#333333", label="Nondominated in this case"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777",
                     markeredgecolor="#777777", label="Dominated or not on frontier")]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Scalar time–memory tradeoffs by workload", y=1.02, fontsize=15)
    fig.text(0.5, -0.01,
             "Lower-left is better. Stars mark the observed Pareto front among all scalar variants "
             "in that case; selected points show median and full min–max run range.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    save_both(fig, output_dir, "scalar_pareto")
    plt.close(fig)


def plot_parallel(groups, output_dir):
    selected = [g for g in groups if g.get("bounds") == "checked"]
    by_key = {(g["case_id"], g["variant"], g["workers_requested"]): g
              for g in selected}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=False)
    worker_values = (1, 2, 4, 6)
    for ax, (case, title) in zip(axes, PARALLEL_CASES):
        available = []
        for variant in PARALLEL_VARIANTS:
            points = []
            for workers in worker_values:
                group = by_key.get((case, variant, workers))
                value = median(group, "call_seconds") if group else None
                if value is not None:
                    points.append((workers, value, group))
            if not points:
                continue
            available.append((variant, points))
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            high, low = [], []
            for _, value, group in points:
                summary = group["times"]["call_seconds"]
                high.append(max(0.0, (value if summary["max"] is None
                                      else summary["max"]) - value))
                low.append(max(0.0, value - (value if summary["min"] is None
                                             else summary["min"])))
            ax.errorbar(x, y, yerr=[low, high], marker="o", markersize=4,
                        linewidth=1.5, capsize=2, color=COLORS[variant],
                        label=variant.removeprefix("parallel_").replace("occurrence_", ""))

        packed = by_key.get((case, "packed", 1))
        packed_wall = median(packed, "call_seconds") if packed else None
        scalar_rows = [g for g in selected if g.get("case_id") == case
                       and g.get("workers_requested") == 1
                       and not g.get("variant", "").startswith("parallel_")]
        scalar_wall = min((median(g, "call_seconds") for g in scalar_rows
                           if median(g, "call_seconds") is not None), default=None)
        if packed_wall is not None:
            ax.axhline(packed_wall, color="#444444", linestyle="--", linewidth=1,
                       label="direct packed, 1 worker")
        if scalar_wall is not None:
            ax.axhline(scalar_wall, color="#777777", linestyle=":", linewidth=1,
                       label="best scalar, 1 worker")
        ax.set_xticks(worker_values)
        ax.set_xlabel("Requested workers")
        ax.set_ylabel("Call wall time (s, median; whiskers min–max)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if not available:
            ax.text(0.5, 0.5, "No matching parallel rows", ha="center", va="center",
                    transform=ax.transAxes, color="#666666")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=7, fontsize=9, frameon=False)
    fig.suptitle("Parallel scaling by workload", y=1.05, fontsize=15)
    fig.text(0.5, -0.02,
             "Points are median call wall time; whiskers span min–max runs. Dashed references "
             "show direct packed and best scalar one-worker call times when available.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    save_both(fig, output_dir, "parallel_scaling")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True,
                        help="machine JSON emitted by ablation_summarize.py")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new or empty directory for standalone SVG and PNG files")
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    if not args.output_dir.is_dir():
        raise NotADirectoryError(args.output_dir)
    groups = report.get("groups", [])
    if not groups:
        raise ValueError("summary contains no groups")
    plot_scalar(groups, report.get("pareto", {}), args.output_dir)
    plot_parallel(groups, args.output_dir)
    print(json.dumps({"summary": str(args.summary.resolve()),
                      "output_dir": str(args.output_dir.resolve()),
                      "files": ["scalar_pareto.svg", "scalar_pareto.png",
                                "parallel_scaling.svg", "parallel_scaling.png"]}))


if __name__ == "__main__":
    main()

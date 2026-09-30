"""Export standalone scientific figures from the finished, audited matrix."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CASES = [f"{language}-16m-continuous" for language in ("en", "zh", "de", "ja")]
LABELS = {
    "owner_ahash": "Prior owner aHash",
    "cuts_fixed": "Fixed cuts",
    "cuts_adaptive": "Adaptive cuts",
    "birth_chain": "Birth chain",
    "birth_replay_vec": "Replay / Vec counts",
    "birth_replay_inline": "Replay / inline counts",
}
COLORS = dict(zip(LABELS, ["#0072B2", "#999999", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]))


def main():
    report = json.loads((OUT / "summary.json").read_text())
    assert report["status"] == "passed" and report["measurement_rows"] == 1000
    rows = report["groups"]
    lookup = {(row["case_id"], row["mode"], row["workers"]): row for row in rows}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for case, axis in zip(CASES, axes.flat):
        for mode, label in LABELS.items():
            values = [lookup[case, mode, workers] for workers in (1, 2, 4, 6)]
            one = values[0]["median_call_seconds"]
            axis.plot([1, 2, 4, 6], [one / value["median_call_seconds"] for value in values],
                      marker="o", linewidth=1.8, color=COLORS[mode], label=label)
        axis.plot([1, 6], [1, 6], linestyle=":", color="#444444", linewidth=1, label="Linear reference")
        axis.set(title=case[:2].upper(), xlabel="Workers / whole-process CPU budget",
                 ylabel="Same implementation W1 / Wp", xticks=[1, 2, 4, 6], ylim=(0.7, 6.2))
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("Exact BPE training: 16 MiB continuous text, 32,000 merges\nFive repetitions per point; medians of complete training calls")
    for suffix in ("svg", "png", "pdf"):
        figure.savefig(OUT / f"scaling.{suffix}", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for case, axis in zip(CASES, axes.flat):
        for mode, label in LABELS.items():
            row = lookup[case, mode, 4]
            x, y = row["median_train_vm_hwm_mib"], row["median_call_seconds"]
            axis.errorbar(x, y, xerr=[[x - min(row["raw_train_vm_hwm_mib"])],
                                     [max(row["raw_train_vm_hwm_mib"]) - x]],
                          yerr=[[y - row["min_call_seconds"]], [row["max_call_seconds"] - y]],
                          fmt="o", color=COLORS[mode], capsize=2, label=label)
        serial_mode = report["fastest_serial_by_case"][case]
        serial = lookup[case, serial_mode, 1]
        axis.scatter(serial["median_train_vm_hwm_mib"], serial["median_call_seconds"],
                     marker="*", s=110, color="#111111", label="Fastest direct serial (W1)")
        axis.set(title=case[:2].upper(), xlabel="Process high-water memory after training (MiB)",
                 ylabel="Complete training call (seconds)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("Speed and memory with a budget of at most four CPUs\nParallel variants W4; fastest serial W1. Whiskers show raw min/max, not confidence intervals.")
    for suffix in ("svg", "png", "pdf"):
        figure.savefig(OUT / f"speed-memory.{suffix}", dpi=160)
    plt.close(figure)
    print("Exported scaling and speed-memory figures as SVG, PNG and PDF")


if __name__ == "__main__":
    main()

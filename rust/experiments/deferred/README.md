# Deferred local ablations

These patches have not been applied or benchmarked. They isolate worker-side grouping of new occurrence lists and shrinking incoming endpoint Vec capacity. They remain useful local ablations, but do not by themselves address the sequential greedy decision and per-rule barrier structure. The next architecture is being selected using [the broader algorithm review](../../PARALLEL_RETHINK.md).

`grouped.patch` adds two modes and reduction counters; `audit.md` records a read-only review, not executed validation. `shrink.patch` adds `packed_shrink` and `combined_filtered_shrink`. Apply from the repository root only after checking against the current source, then run full Rust checks and the independent differential oracle. None of these names is currently part of the measured 33-variant matrix.

`rust/tools/run_grouped_matrix.py` prepares a separate five-repeat, 1,045-run matrix and preflights the needed variants and manifests. It has not been executed. `rust/tools/ablation_large_fixtures.py` can prepare separately pinned true 16 MiB continuous inputs from the existing read-only source snapshots. It also has not been run. Do not combine a future binary's samples with the first formal matrix when calculating a median.

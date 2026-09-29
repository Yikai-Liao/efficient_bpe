# Spatial extra-only probe: first fixed-budget screen

`parallel_pair_owned_spatial_extra` scans only the additional candidate prefix in its two-sided spatial probe. It retains the spatial v1 rule order and exact certification. Targeted Rust tests passed 2/2; strict Clippy and release build passed; the Python full-recount oracle matched complete traces and final tokens in 80/80 runs (20 cases including 8 seeded random cases, workers 1/4, checked/unchecked).

The same frozen binary ran all variants below on the 256 KiB continuous English and Chinese fixtures with 512 rules. The process CPU budget was fixed: W1 and scalar used CPU 5, while W4 used CPUs 0, 1, 2, and 5, including the coordinator. Each number is a single checked run. `VmHWM` is the native process high-water memory value.

| Case | Variant | W1 call (s) | W4 call (s) | W1 / W4 VmHWM (MiB) |
|---|---|---:|---:|---:|
| EN | combined_filtered | 0.07381 | — | 8.51 / — |
| EN | combined_filtered_halfword | 0.08544 | — | 7.92 / — |
| EN | pair_owned | 0.13020 | 0.10541 | 9.75 / 11.61 |
| EN | spatial v1 | 0.15512 | 0.10043 | 9.94 / 11.96 |
| EN | spatial extra-only | 0.15151 | 0.10820 | 9.98 / 11.52 |
| ZH | combined_filtered | 0.03013 | — | 6.09 / — |
| ZH | combined_filtered_halfword | 0.05632 | — | 6.04 / — |
| ZH | pair_owned | 0.06397 | 0.04339 | 7.37 / 10.01 |
| ZH | spatial v1 | 0.07929 | 0.04390 | 7.14 / 10.29 |
| ZH | spatial extra-only | 0.06450 | 0.05369 | 7.27 / 10.25 |

Extra-only reduced probe visits by 36.6% on EN (515,737 → 327,003) and 40.9% on ZH (54,249 → 32,087), while each case kept the same epoch count and total selected width as spatial v1 (EN 55 epochs, ZH 81, both 512 rules). W1 call time fell in both cases, but W4 time rose in this one-shot screen. The additional probe work and barriers still leave both spatial variants above the best direct scalar on these small fixtures. This is a narrow diagnostic, not a stable ranking or a 4 MiB result.

The frozen binary is at ignored `rust/target/reruns/pair-owned-spatial-extra-v1/ablation`, SHA-256 `4f98e2549a6414734b09071796b514643e23c8174e8cf76f792d4e4a89c08d24`. `native-source-snapshot.tar.gz` and `native-source-hashes.json` match every Rust source hash in the benchmark sidecar. Raw runs are in `common-quick.jsonl`; its environment sidecar records commands, source and fixture hashes, CPU affinity, and tool versions. `checks.json` records validation and comparison fields. No large binary is staged for Git.

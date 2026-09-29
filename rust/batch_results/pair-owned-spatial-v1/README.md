# Spatial-prefix exact batch: first screen

The registered `parallel_pair_owned_spatial` variant passed two focused Rust tests and strict Clippy, then 80/80 full-trace Python oracle runs (20 cases, workers 1/4, checked/unchecked). Its frozen release binary is at `rust/target/reruns/pair-owned-spatial-v1/ablation`; `native-source-snapshot.tar.gz` and `native-source-hashes.json` record the exact native source, and the benchmark sidecar hashes match the snapshot.

The following checked 256 KiB measurements use **one binary**, one run per cell, and complete `call_seconds`. The scalar variants are pinned to one CPU; worker variants inherit the original six-CPU process affinity, as recorded in the sidecar. `VmHWM` comes from the native process.

| Case | Variant | Workers | Call (s) | VmHWM (MiB) |
|---|---|---:|---:|---:|
| EN | combined_filtered | 1 | 0.1037 | 8.51 |
| EN | combined_filtered_halfword | 1 | 0.0963 | 7.92 |
| EN | pair_owned | 1 / 4 | 0.1918 / 0.0831 | 9.75 / 11.61 |
| EN | pair_owned_pipeline | 1 / 4 | 0.1427 / 0.0855 | 9.94 / 11.41 |
| EN | pair_owned_spatial | 1 / 4 | 0.1938 / 0.0908 | 9.80 / 11.63 |
| ZH | combined_filtered | 1 | 0.0365 | 6.09 |
| ZH | combined_filtered_halfword | 1 | 0.0399 | 6.04 |
| ZH | pair_owned | 1 / 4 | 0.1030 / 0.0585 | 7.37 / 10.01 |
| ZH | pair_owned_pipeline | 1 / 4 | 0.0807 / 0.0698 | 7.66 / 10.04 |
| ZH | pair_owned_spatial | 1 / 4 | 0.1201 / 0.0696 | 7.14 / 9.99 |

Spatial probing widens some exact certified batches: EN widened 15 epochs and ZH 9; the sum of selected widths reached 512 rules in both cases. At four workers the probe visited 515,737 EN historical positions (149,945 stale) and 54,249 ZH (5,586 stale); probe wall time was 0.0086/0.0046 seconds. In this one-shot quick screen, the additional metadata and probing cost outweighed fewer batch barriers. Spatial W4 was slower than pair_owned W4 by 9.2% EN and 18.9% ZH. No larger timing claim follows from one run.

Artifacts: `differential.json`, `common-quick.jsonl`, its environment sidecar, `checks.json`, source snapshot/hashes, and `binary.sha256`. The large native binary is ignored by Git.

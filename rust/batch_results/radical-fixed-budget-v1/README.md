# Exact batch and posting ownership: fixed-budget screen

This archive compares radical v2, exact certified-batch v4, and v2's promptly freed boxed postings against a freshly built native reference. All use the same 256 KiB continuous EN/ZH fixtures (512 rules, `min_frequency=2`), checked bounds, `chunk_size=4096`, `thin` LTO and one codegen unit. The **entire process**, including its coordinator, is pinned to CPU 5 for W1 and CPUs 0, 1, 2, 5 for W4. Each cell is one run; these are diagnostics, not a stable ranking.

V4 passed 5/5 library tests and boxed passed 6/6; both passed strict Clippy and release builds. The Python full-recount oracle matched complete traces and final tokens for v4 and boxed in **80/80** runs (20 cases, W1/W4, checked CLI). All 12 quick runs of v2/v4/boxed also matched the complete trace, final tokens, fixture SHA-256 and fingerprint of the native direct scalar reference.

`call_seconds` encloses the full training call after input JSON parsing and dropping source bytes. `train_vm_hwm_mib` is sampled just after training returns, before fingerprint/trace formatting. It is still a process high-water mark, including startup and input parsing, not a net allocator measurement. Final `vm_hwm_mib` is retained in raw output; boxed's trace formatting can raise that later peak by several MiB, so memory comparisons here use the training-time field.

| 256 KiB case | Variant | W1 seconds / MiB | W4 seconds / MiB |
|---|---|---:|---:|
| EN | Best direct scalar (`combined_filtered`) | 0.07679 / 8.32 | — |
| EN | Native pair_owned | 0.15710 / 9.98 | 0.07011 / 11.45 |
| EN | Native pipeline | 0.14059 / 10.09 | 0.06300 / 11.49 |
| EN | Radical v2 | 0.08791 / 8.36 | 0.10253 / 8.94 |
| EN | Radical v4 batch | 0.09402 / 8.61 | 0.10089 / 8.89 |
| EN | Boxed v2 | 0.07989 / 7.63 | 0.10766 / 8.42 |
| ZH | Best direct scalar (`combined_filtered_halfword`) | 0.03283 / 5.83 | — |
| ZH | Native pair_owned | 0.06375 / 7.07 | 0.04148 / 9.51 |
| ZH | Native pipeline | 0.06090 / 7.27 | 0.04684 / 10.01 |
| ZH | Radical v2 | 0.04813 / 6.49 | 0.05150 / 6.59 |
| ZH | Radical v4 batch | 0.04064 / 6.69 | 0.04221 / 6.97 |
| ZH | Boxed v2 | 0.05014 / 5.62 | 0.05148 / 5.37 |

V4 applied 512 exact rules in 69 EN and 87 ZH certified batches, with maximum widths 22 and 26. It avoids some intermediate newly born edges: EN generated births fell from v2's 288,489 to 284,407, stored births 274,060 → 269,984, and historical posting visits 203,150 → 202,877. ZH generated births fell 51,692 → 51,001, stored 41,649 → 40,975, and visits 29,674 → 29,670. Those are deterministic algorithm savings, but too small to offset coordination and central reduction costs on these quick fixtures. EN W4 v4 reports 0.0285 s planning, 0.0091 s central change combination, 0.0139 s frequency reduction, and 0.0197 s initialization within a 0.1009 s call; ZH W4 reports 0.0096/0.0030/0.0035/0.0161 s. Stage times describe different work and must not be multiplied into speedups.

Boxed v2 preserves v2's posting visits and generated/stored births exactly, isolating the storage change. EN W4's end-of-call allocated posting records are 276,545 (peak 329,035), versus v2's append arena length 534,072 and capacity 1,040,048; ZH is 63,848 (peak 81,279) versus arena length 121,309 and capacity 159,320. Training VmHWM fell 8.94 → 8.42 MiB EN and 6.59 → 5.37 MiB ZH. The difference between record-capacity savings and process RSS reflects other allocations and allocator behavior; a larger input would be needed to judge its scale benefit.

## Minimal 4 MiB native calibration

The same frozen native binary was run once per cell at the same fixed CPU budgets to correct earlier all-affinity pipeline measurements. Both languages' variants produced stable fingerprints.

| Continuous case | Variant | W1 seconds / MiB | W4 seconds / MiB |
|---|---|---:|---:|
| EN 4 MiB | Best direct scalar (`combined_filtered`) | 1.6903 / 95.56 | — |
| EN 4 MiB | pair_owned | 2.7333 / 111.74 | 1.0967 / 137.18 |
| EN 4 MiB | pipeline | 2.7260 / 110.82 | 1.1231 / 135.87 |
| ZH 4 MiB | Best direct scalar (`combined_filtered`) | 1.0570 / 99.18 | — |
| ZH 4 MiB | pair_owned | 1.7229 / 122.70 | 0.6250 / 119.15 |
| ZH 4 MiB | pipeline | 1.6587 / 122.99 | 0.7366 / 123.09 |

Pair_owned W4 is 1.54× EN and 1.69× ZH faster than the best direct scalar in these one-shot measurements. Pipeline is 2.4% slower than pair_owned EN and 17.8% slower ZH. The prior all-affinity screen suggested a pipeline improvement; these fixed-budget results do not support that claim. No radical 4 MiB timing was run because none of its W4 quick calls beat the fastest reference for that fixture (native pipeline on EN, direct scalar on ZH).

## Provenance

The frozen binaries are in ignored `rust/target/reruns/radical-fixed-budget-v1/{ablation,v2,v4,boxed}`. `native-source-snapshot.tar.gz` and `native-source-hashes.json` capture the root and experiment crate source at build time; SHA files identify each binary. Raw runs and affinity/source/fixture sidecars are `native-quick.jsonl`, `quick.jsonl`, and `native-4m-calibration.jsonl` with matching `.environment.json` files. `run_differential.py` and `run_quick.py` reproduce the focused checks from those binaries; `differential.json` and `checks.json` summarize validation. The shared Cargo target directory was `rust/target`. Large binaries are not staged for Git.

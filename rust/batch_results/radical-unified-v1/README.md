# Radical posting-arena prototypes: first common quick screen

This archive compares the three independent Rust crates under the same 256 KiB continuous English and Chinese fixtures, 512 rules, `min_frequency=2`, checked bounds, `chunk_size=4096`, and release settings (`thin` LTO, one codegen unit, debug info). All three library suites passed 5/5 tests and all three release CLIs built. Every one of the 12 quick runs matched the **complete merge trace and final tokens** of the same-fixture native `combined_filtered` reference; fixture SHA-256 and canonical fingerprint also matched.

**CPU-budget limitation:** this first screen left all six allowed CPUs available to each radical process, including its coordinator, while native scalar was pinned to one CPU and native four-worker runs were also allowed all six in the preceding archive. The radical worker count therefore was not an enforced process CPU budget. These timings cannot establish fair 1→4 CPU scaling or an equal-budget speedup over the scalar reference. A subsequent fixed-affinity rerun must pin W1 to one CPU and W4 to four CPUs. The raw timings below are preserved as measured.

The crates measure `call_seconds` around the complete training call after JSON parsing and dropping the input byte buffer, like the native CLI. The direct scalar references come from the separate, one-binary native [common quick screen](../pair-owned-spatial-v1/README.md); they use identical fixture hashes and their binaries/sources are archived there. Native scalar jobs were pinned to one CPU; radical processes inherited the original six-CPU affinity and use a Rayon pool of one or four workers. Each number below is one run, subject to the CPU-budget limitation above. `VmHWM` is native process high-water memory in MiB.

| Case | Algorithm | Workers | Call (s) | VmHWM (MiB) |
|---|---|---:|---:|---:|
| EN | Best direct scalar (`combined_filtered_halfword`) | 1 | 0.0963 | 7.92 |
| EN | Radical v1 | 1 / 4 | 0.1176 / 0.1199 | 9.48 / 10.22 |
| EN | Radical v2 | 1 / 4 | 0.0847 / 0.1117 | 8.41 / 9.32 |
| EN | Radical v3 | 1 / 4 | 0.1061 / 0.1095 | 8.11 / 9.26 |
| ZH | Best direct scalar (`combined_filtered`) | 1 | 0.0365 | 6.09 |
| ZH | Radical v1 | 1 / 4 | 0.0451 / 0.0551 | 6.99 / 6.96 |
| ZH | Radical v2 | 1 / 4 | 0.0506 / 0.0534 | 7.30 / 6.67 |
| ZH | Radical v3 | 1 / 4 | 0.0600 / 0.0737 | 10.70 / 12.77 |

None of these 256 KiB runs shows a lower W4 wall time than W1, even with the permissive affinity. V2's English one-worker call is numerically below the direct scalar in this sample, but the CPU budgets differ, so this is not an equal-budget speedup. The native pair_owned four-worker runs took 0.0831 s EN and 0.0585 s ZH in the common quick screen; all three radical four-worker calls were slower on both fixtures. No 4 MiB or 16 MiB timing was run for these prototypes.

The phase counters identify the main costs to investigate. V1/V2 English four-worker `apply_seconds` is about 0.041 s of a 0.120/0.112 s call; posting visits are 203,150, including 58,805 stale visits. V3 cuts that apply phase to 0.027 s, but its birth count/prefix/scatter phases add about 0.041 s and frequency reduction about 0.015 s; its total remains 0.109 s. On Chinese, V3's 0.025 s initialization and roughly 0.020 s birth pipeline occupy much of its 0.074 s four-worker call. These stage timers describe the algorithm's work and are not independent speedups. The designs still perform a global winner decision per rule; v1/v2 retain central frequency reduction and birth handling, while v3 retains distinct-key coordination.

The frozen binaries are in ignored `rust/target/reruns/radical-unified-v1/{v1,v2,v3}`. `native-source-snapshot.tar.gz` and `native-source-hashes.json` include the root crate and all three experiment crates at build time; `v1.sha256`, `v2.sha256`, and `v3.sha256` identify the binaries. `quick.jsonl.environment.json` records fixture hashes, binary hashes, CPU affinity, and source hashes; `run_quick.py` reproduces the full-trace checks and one-shot matrix from the frozen binaries. Raw numbers are in `quick.jsonl`, with validation and comparisons in `checks.json`.

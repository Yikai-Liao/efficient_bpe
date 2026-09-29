# Frozen pair-owned 4 MiB discriminant, two repeats

This screen reuses the exact frozen binaries and source snapshots from `radical-owner-boxed-v1` and `radical-owned-extensions-v1`. It compares owned lazy, spatial probe off/budgeted, and counted deltas lazy on continuous EN/ZH inputs at 4 MiB and 3000 rules. Every experimental configuration ran at W1 and W4 twice. The same window also ran the best direct scalar and native pair_owned W4 twice per language: **40 complete training calls** in total. The second round reversed the first seeded-shuffle order. No binaries were rebuilt.

The entire process, including its coordinator, was pinned to CPU 5 for W1 and CPUs 0, 1, 2, 5 for W4. Every input's fixture SHA-256 was checked before each call, and every run's fingerprint of the complete rule sequence and final tokens matched the previously validated native reference. The binaries use checked input, release `thin` LTO/one codegen unit, `chunk_size=4096` for the radical variants, and `min_frequency=2`. `call_seconds` excludes JSON parsing but includes the full training call. `train_vm_hwm_mib` is sampled before fingerprint formatting and remains a process high-water mark including startup and parsing.

The table reports the **median of two**, with the observed min–max in brackets. With n=2 these are diagnostics, not confidence intervals or a stable ranking.

| Input | Variant | W1 seconds [range] | W4 seconds [range] | W4 training VmHWM MiB [range] |
|---|---|---:|---:|---:|
| EN | Best direct scalar (`CF16`) | 1.722 [1.673–1.771] | — | — |
| EN | Native pair_owned | — | 1.151 [1.100–1.202] | 137.60 [136.66–138.54] |
| EN | Owned lazy | 2.360 [2.359–2.361] | **0.828 [0.822–0.833]** | 105.79 [104.09–107.50] |
| EN | Probe off | 2.331 [2.263–2.399] | 0.880 [0.855–0.905] | 100.71 [100.44–100.99] |
| EN | Probe budgeted | 2.405 [2.324–2.487] | 0.956 [0.863–1.049] | 107.23 [106.53–107.93] |
| EN | Counted deltas lazy | 2.325 [2.201–2.449] | 0.928 [0.901–0.955] | 102.81 [101.98–103.65] |
| ZH | Best direct scalar (`CF`) | 1.092 [1.043–1.140] | — | — |
| ZH | Native pair_owned | — | 0.701 [0.691–0.710] | 118.97 [117.11–120.83] |
| ZH | Owned lazy | 1.066 [1.043–1.089] | **0.491 [0.485–0.498]** | 114.44 [113.37–115.51] |
| ZH | Probe off | 1.270 [1.082–1.458] | 0.540 [0.477–0.602] | 114.46 [113.86–115.07] |
| ZH | Probe budgeted | 1.100 [1.077–1.123] | 0.532 [0.508–0.557] | 115.53 [115.34–115.72] |
| ZH | Counted deltas lazy | 1.109 [1.091–1.126] | 0.542 [0.482–0.602] | 116.50 [115.59–117.42] |

Using ratios of these medians, owned lazy W4 is 2.08× EN and 2.22× ZH faster than the best direct scalar, and 1.39× EN / 1.43× ZH faster than native pair_owned W4. Its W1-to-W4 ratios are 2.85× EN and 2.17× ZH under the stated total CPU budgets. The two owned-lazy W4 times are close for each language. The probe and counted variants have greater dispersion and no demonstrated advantage over the frozen owned baseline in this screen. EN owned lazy used about 31.8 MiB less training VmHWM than native pair_owned by the median of the two process peaks; ZH used about 4.5 MiB less. These peak-memory differences are allocator- and process-sensitive.

The spatial probe makes a deterministic change despite the mixed timing: EN exact batches fell 251→192 and generated intermediate births fell 5,705,752→5,704,754; ZH batches fell 275→235 and births 1,274,766→1,273,834. EN W4 probed about 106–109 thousand additional historical positions and ZH about 27 thousand, costing roughly 0.010–0.012 s and 0.003 s in the probe timer respectively. Fewer batches did not produce a lower full-call median here. Probe off and counted deltas reproduce the baseline's batch width, births, and historical posting visits exactly. The counted layout's 24-byte pair/delta record therefore isolates representation without changing rewrite work.

`screen.jsonl` contains every call, command, affinity, complete-training fingerprint, stage metrics, and training VmHWM. `screen.jsonl.environment.json` identifies the binaries, source snapshots, fixture manifest, and reverse-order schedule. `summary.json` retains both observations and their medians/ranges; `checks.json` records validation assertions. `run_screen.py` reproduces the run from the frozen ignored binaries. No full matrix or 16 MiB timing was run.

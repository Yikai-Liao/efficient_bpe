# Pair-owned batch and boxed-posting screen

This archive compares two new exact greedy BPE implementations with the frozen native and radical references. `owned_lazy` and `owned_eager` share one v5 kernel; only their owner-heap update policy differs. `batched_boxed` retains v4's certified batches and central reduction, replacing the append arena with promptly released per-key postings. Each result is one run, so the timings are a screen rather than a stable ranking.

The final owned source passed 5/5 library tests, strict Clippy, and release build after the selected-posting lifetime fix. Batched-boxed passed 6/6 library tests, strict Clippy, and release build. The Python full-recount oracle matched every rule and final token for both owned policies and batched-boxed in **120/120** checks (20 cases × W1/W4 × three configurations). All 20 quick calls additionally matched the frozen native scalar's complete trace, fingerprint, and fixture SHA-256. The 12 larger calls matched fingerprints across all variants for each language.

The process, including its coordinator, was pinned to CPU 5 for W1 and CPUs 0, 1, 2, 5 for W4. All binaries used Rust release `thin` LTO and one codegen unit, checked input, 512 rules for the 256 KiB quick inputs and 3000 for the 4 MiB continuous inputs, and `min_frequency=2`. `call_seconds` covers the complete training call after JSON parsing. `train_vm_hwm_mib` is sampled immediately after training, before trace/fingerprint formatting; it remains a process high-water mark that includes startup and input parsing. The final `vm_hwm_mib` is also retained in raw rows.

| 256 KiB input | Variant | W1 seconds / MiB | W4 seconds / MiB |
|---|---|---:|---:|
| EN | Best direct scalar (`combined_filtered`) | 0.07517 / 8.28 | — |
| EN | Native pair_owned | 0.13495 / 9.95 | 0.13603 / 11.24 |
| EN | Native pipeline | 0.12865 / 10.11 | 0.07977 / 11.27 |
| EN | Radical v4 batch | 0.09906 / 8.63 | 0.09482 / 9.04 |
| EN | Batched-boxed | 0.09365 / 8.15 | 0.08121 / 7.95 |
| EN | Owned eager | 0.11345 / 9.22 | 0.06227 / 9.39 |
| EN | Owned lazy | 0.09598 / 8.82 | **0.05589 / 9.31** |
| ZH | Best direct scalar (`combined_filtered_halfword`) | 0.03282 / 5.78 | — |
| ZH | Native pair_owned | 0.06346 / 7.07 | 0.06153 / 9.86 |
| ZH | Native pipeline | 0.06076 / 7.20 | 0.07787 / 10.44 |
| ZH | Radical v4 batch | 0.03796 / 6.64 | 0.05142 / 6.59 |
| ZH | Batched-boxed | 0.04321 / 5.56 | 0.05495 / 5.60 |
| ZH | Owned eager | 0.05085 / 8.34 | 0.03467 / 8.86 |
| ZH | Owned lazy | 0.05262 / 8.28 | **0.03215 / 8.93** |

The EN native pair_owned W4 call was unusually slow in this small sample: 0.136 s versus 0.070 s in the prior fixed-budget quick archive. It should not be used to infer an enormous relative gain. Owned lazy reduced owner-heap work with identical training semantics: for EN W4 it made 2,615 heap pops and peaked at 31,520 candidate slots versus eager's 6,001 pops and 63,040 slots. ZH W4 was 1,609 versus 2,225 pops; both peaked at 23,948 slots. The quick timing difference is consistent with that work reduction but remains one sample.

For each quick input, v4, batched-boxed, owned eager, and owned lazy agree exactly on batch count, maximum width, generated births, stored eligible births, and historical posting visits. EN's values are 69 batches, width 22, 284,407 generated births, 269,984 stored births, and 202,877 visits. ZH's are 87, 26, 51,001, 40,975, and 29,670. Thus the new ownership layout and heap policy did not change the greedy rule order or measured rewrite work. Boxed v2, which has no batch certificate, has the different expected birth/visit counts recorded in the previous archive.

The limited **4 MiB discriminant** used the same binaries and affinities, with six configurations per language and one run per cell:

| Continuous input | Variant | W1 seconds / MiB | W4 seconds / MiB |
|---|---|---:|---:|
| EN | Best direct scalar (`combined_filtered_halfword`) | 1.69855 / 88.19 | — |
| EN | Native pair_owned | 2.66996 / 111.82 | 1.07784 / 136.66 |
| EN | Owned lazy | 2.38488 / 102.92 | **0.90701 / 106.95** |
| ZH | Best direct scalar (`combined_filtered`) | 0.93381 / 100.77 | — |
| ZH | Native pair_owned | 1.48691 / 123.18 | 0.75474 / 120.00 |
| ZH | Owned lazy | 1.17667 / 106.61 | **0.62486 / 113.71** |

Owned lazy W4 was 1.87× EN and 1.49× ZH faster than the best direct scalar in this one-shot screen. Against native pair_owned W4, it was 15.9% EN and 17.2% ZH faster; its training VmHWM was 29.71 MiB EN and 6.29 MiB ZH lower. The owned W1/W4 ratio was 2.63× EN and 1.88× ZH under the stated total CPU budgets. The full-call fingerprints matched in all 12 calls. Owned W1/W4 also agreed on 251 EN and 275 ZH certified batches and on 4,482,863 EN and 806,617 ZH historical posting visits. No 16 MiB input or full matrix was run.

In the 4 MiB W4 calls, EN owned reported 0.120 s initialization, 0.289 s planning, 0.111 s frequency reduction, and 0.247 s birth decoding inside its 0.907 s full call. ZH reported 0.098, 0.108, 0.146, and 0.114 s respectively inside 0.625 s. These phases use different timers and may overlap or nest; their sum is not a separate execution time. The remaining bottlenecks motivate narrower owner routing and posting allocation experiments, not a claim that this version has exhausted parallel speedup.

## Provenance

`differential.json`, `native-quick.jsonl`, `quick.jsonl`, and `smoke-4m.jsonl` contain validation and raw runs. The JSONL sidecars record exact commands, CPU affinities, binary hashes, fixture hashes, and source snapshot. `checks.json` records the assertions. `run_differential.py`, `run_quick.py`, and `run_smoke.py` reproduce the focused checks using the ignored binaries at `rust/target/reruns/radical-fixed-budget-v1/` and `rust/target/reruns/radical-owner-boxed-v1/`. `native-source-snapshot.tar.gz` and `native-source-hashes.json` preserve the Rust sources used to build them; `owned.sha256` and `batched_boxed.sha256` identify the new binaries. The shared Cargo target directory was `rust/target`; large binaries are not staged for Git.

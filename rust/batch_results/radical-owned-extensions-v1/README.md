# Owned batch extensions: spatial probe and counted deltas

This archive screens two independent changes to the exact pair-owned batch trainer. `owned_probe` optionally scans a conflicted candidate's historical positions to prove that its actual matches do not overlap selected matches. `owned_counts` replaces signed `i128` edge-frequency deltas with checked absolute weight plus physical occurrence count and reserves a new eligible posting vector once. The frozen `owned_lazy` binary from `radical-owner-boxed-v1` is the baseline. No source in that earlier archive was changed.

The probe crate passed 8/8 focused tests after a test-only rule-cap correction; the counted-delta crate passed 5/5. Both passed strict Clippy and release builds. A Python full-recount oracle matched **160/160 complete rule traces and final token sequences**: 20 cases, W1/W4, and probe off/budgeted or counts lazy/eager. All 20 quick calls also matched the frozen native direct scalar's full trace, fingerprint, and fixture hash.

Every process, including its coordinator, was pinned to CPU 5 for W1 and CPUs 0, 1, 2, 5 for W4. The two 256 KiB continuous fixtures each trained 512 rules at `min_frequency=2`, checked input, and `chunk_size=4096`. All release profiles use `thin` LTO and one codegen unit. `call_seconds` starts after JSON parsing and includes the complete training call. `train_vm_hwm_mib` is read immediately after training and before fingerprint or trace formatting; it remains a process high-water mark that includes startup and input parsing. Each cell below is **one run** and lasts only about 0.03–0.12 s, so timing differences are exploratory.

| 256 KiB input | Variant | W1 seconds / MiB | W4 seconds / MiB |
|---|---|---:|---:|
| EN | Best direct scalar (`combined_filtered_halfword`) | 0.07733 / 7.82 | — |
| EN | Frozen owned lazy | 0.09528 / 8.88 | 0.06434 / 8.80 |
| EN | Probe off | 0.09847 / 8.92 | 0.04828 / 9.15 |
| EN | Probe budgeted | 0.11449 / 9.02 | 0.05504 / 8.93 |
| EN | Counted deltas, lazy | 0.10322 / 8.45 | 0.05447 / 8.86 |
| EN | Counted deltas, eager | 0.12492 / 9.13 | 0.05582 / 9.78 |
| ZH | Best direct scalar (`combined_filtered_halfword`) | 0.03438 / 5.93 | — |
| ZH | Frozen owned lazy | 0.05615 / 8.38 | 0.02647 / 8.88 |
| ZH | Probe off | 0.05416 / 8.45 | 0.03485 / 9.22 |
| ZH | Probe budgeted | 0.05287 / 8.25 | 0.02698 / 9.19 |
| ZH | Counted deltas, lazy | 0.05034 / 8.36 | 0.03945 / 8.98 |
| ZH | Counted deltas, eager | 0.05562 / 8.41 | 0.03411 / 8.84 |

The deterministic work counters give a clearer result. Probe off and both counted-delta policies reproduce the frozen owned baseline's batch count, maximum width, generated and stored births, and historical posting visits exactly. The budgeted probe lowers EN batch count **69 → 53**, raises maximum width **22 → 32**, and lowers generated intermediate births **284,407 → 284,098**. It lowers ZH batches **87 → 80** and births **51,001 → 50,899**, with width unchanged at 26. Historical posting visits remain **202,877 EN / 29,670 ZH** because the extra probe visits are accounted separately. At W4 the probe scanned 8,998 EN and 873 ZH positions, reported 49/56 conflict stops and 32/12 disjoint proofs, and took 0.00060/0.00009 s within selection. There were no budget stops on these inputs. The reduced batches and births are real algorithmic effects; the short-call timing does not establish a reliable net speedup.

The counted-delta `Delta` occupies 16 bytes and its `(pair, Delta)` record 24 bytes on this compiler, versus the prior `(pair, i128)` record's 32 bytes. The 256 KiB result shows no stable process-RSS gain: map buckets, postings, allocator behavior, and input startup also contribute to VmHWM. The counted variant did not change the rule order, batch structure, or rewrite work on either fixture, including the eager policy's expected extra heap updates. A larger controlled sample would be needed to assess its allocation benefit.

The probe is a sequential certificate on only the candidate being considered. Its extra work is bounded by twice the selected historical posting length per batch, using actual matches and checking both overlap directions. It does not add a corpus-sized mark array or worker-wide dispatch. It also adds coordinator work to the critical path, so fewer batches need not imply a faster run.

`differential.json`, `native-quick.jsonl`, `quick.jsonl`, and their sidecars contain the validation and raw runs. `checks.json` records the assertions and binary paths. `run_differential.py` and `run_quick.py` reproduce the checks with ignored binaries in `rust/target/reruns/`. `new-source-snapshot.tar.gz` and `new-source-hashes.json` preserve the new crates; the frozen baseline's source snapshot is in `rust/batch_results/radical-owner-boxed-v1/`. Large binaries are not staged for Git. No 4 MiB or 16 MiB extension screen was run.

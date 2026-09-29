# Grouped births and logical owner shards

This archive screens two independent exact greedy BPE routing changes against frozen owned and counted-delta baselines. `grouped_lazy` chains final born positions by pair key inside each producer route, then fills the destination posting vector by key. `shards` keeps the old route representation and varies logical pair owners while holding the producer count fixed. Its `S=W`, `S=2W`, and `S=4W` configurations run from the **same binary**. No existing baseline crate was changed.

The final grouped source passed 5/5 debug library tests, including chain-order, reconstructed-final-key, counted-length, AA, weighted, threshold, long-run, and empty-input cases; it passed strict Clippy and release build. Shards passed 7/7 library tests, strict Clippy, and release build. A Python full-recount oracle matched **200/200 complete rule traces and final token sequences** across grouped lazy/eager, shards S=W/2W/4W, W1/W4, and 20 cases. All 24 short quick runs matched the frozen native direct scalar's complete trace, fingerprint, and fixture SHA-256. The 12 later 4 MiB calls matched previously validated complete-training fingerprints and fixture hashes.

The entire process, including its coordinator, was pinned to CPU 5 for W1 and CPUs 0, 1, 2, 5 for W4. Every variant used checked input, `min_frequency=2`, `chunk_size=4096`, and Rust release `thin` LTO/one codegen unit. The 256 KiB continuous EN/ZH inputs trained 512 rules; the 4 MiB inputs trained 3000. `call_seconds` includes the complete training call after JSON parsing. `train_vm_hwm_mib` is read immediately after training, before rule/final-token fingerprint formatting; it remains a process high-water mark including startup and input parsing. **Every timing cell is one run.**

| 256 KiB input | Variant | W1 seconds / MiB | W4 seconds / MiB |
|---|---|---:|---:|
| EN | Best direct scalar (`combined_filtered_halfword`) | 0.07802 / 7.81 | — |
| EN | Frozen owned lazy | 0.11574 / 8.81 | 0.05958 / 9.18 |
| EN | Counted deltas lazy | 0.10072 / 8.61 | 0.07008 / 8.53 |
| EN | Grouped births lazy | 0.09380 / 8.46 | 0.05460 / 9.34 |
| EN | Shards S=W | 0.09322 / 8.95 | 0.06955 / 9.15 |
| EN | Shards S=2W | 0.11854 / 9.07 | 0.09310 / 9.11 |
| EN | Shards S=4W | 0.11642 / 8.61 | 0.05010 / 8.87 |
| ZH | Best direct scalar (`combined_filtered`) | 0.03532 / 6.05 | — |
| ZH | Frozen owned lazy | 0.05205 / 8.33 | 0.03272 / 8.88 |
| ZH | Counted deltas lazy | 0.05127 / 8.36 | 0.03401 / 9.07 |
| ZH | Grouped births lazy | 0.05248 / 8.26 | 0.03366 / 8.55 |
| ZH | Shards S=W | 0.05040 / 8.21 | 0.04348 / 8.75 |
| ZH | Shards S=2W | 0.04890 / 7.88 | 0.02329 / 8.89 |
| ZH | Shards S=4W | 0.04684 / 7.77 | 0.02614 / 8.44 |

The larger W4 discriminant is more useful for judging the work tradeoff:

| 4 MiB input | Variant | Call seconds | Training VmHWM MiB |
|---|---|---:|---:|
| EN | Frozen owned lazy | 0.8542 | 106.74 |
| EN | Counted deltas lazy | 0.9150 | 101.98 |
| EN | Grouped births lazy | **0.7467** | 101.71 |
| EN | Shards S=4 | 1.0156 | 104.30 |
| EN | Shards S=8 | 1.0126 | 103.77 |
| EN | Shards S=16 | 0.8313 | **94.39** |
| ZH | Frozen owned lazy | 0.5861 | 116.14 |
| ZH | Counted deltas lazy | **0.5038** | 116.00 |
| ZH | Grouped births lazy | 0.5258 | **113.52** |
| ZH | Shards S=4 | 0.5191 | 115.83 |
| ZH | Shards S=8 | 0.5925 | 116.53 |
| ZH | Shards S=16 | 0.5588 | 115.82 |

All grouped and shard configurations match the owned baselines' exact batch count, maximum width, generated/stored births, and historical posting visits: EN 251 batches, width 47, 5,705,752 births, 4,482,863 visits; ZH 275, width 50, 1,274,766 births, 806,617 visits. Thus the variations isolate routing and storage work rather than different merge decisions.

On EN 4 MiB, grouped births cut the counted baseline's birth decode timer from 0.243 s to a 0.053 s grouped-fill timer, while planning rose from 0.319 to 0.342 s. The full call fell from 0.915 to 0.747 s. On ZH, birth decode 0.061 s became grouped fill 0.031 s, but planning/reduction offsets left grouped 0.526 s versus counted 0.504 s. These phase timers describe different work; they must not be added as independent speedups. The grouped layout uses 8-byte temporary birth-chain nodes and preserves exact pair-key grouping and physical occurrence counts. Its W4 end-of-call posting capacity equaled the counted baseline's in both fixtures.

With 4 producers, S=4/8/16 increases route bucket headers from 16 to 32 to 64 and frontier head inspections from 1,004 to 2,008 to 4,016 across EN's 251 batches. It reduces the largest initial owner's position count from about 1.36 million to 0.71 million to 0.48 million. EN S16 used 94.39 MiB training VmHWM versus S4's 104.30 MiB, but S4/S8 were slower than the frozen owned baseline in this one-shot screen. ZH shows no monotonic time trend. Sharding a single hot pair cannot split that pair's posting list, and increasing S adds per-owner routing and selection overhead; the measurements do not justify a fixed S choice yet.

`differential.json`, `native-quick.jsonl`, `quick.jsonl`, `smoke-4m.jsonl`, their sidecars, and `checks.json` preserve raw evidence, commands, affinities, binary hashes, and assertions. `run_differential.py`, `run_quick.py`, and `run_smoke.py` reproduce these focused checks using ignored binaries in `rust/target/reruns/`. `new-source-snapshot.tar.gz` captures the new crates at build time. Only `owned_shards/DESIGN.md` received a documentation-only clarification afterward; its pre/post hashes are recorded in `checks.json`, while lib/main/Cargo hashes and binaries are unchanged. No full matrix, larger input, or repeated long-input ranking was run.

# Controlled 4 MiB mechanism screen

This archive tests two controls for earlier exploratory results. `owned_narrow_exact` adds `u32-exact`, which allocates the endpoint corpus with the same exact capacity as `u16`. `owned_route_cache_reset` resets the cache victim phase after each successful batch for both `batch` and `reuse` lifetimes. The previously frozen `owned_context` binary supplies a same-window hash/context comparison. None of these modes is integrated into the main trainer.

The two new crates passed 11/11 and 12/12 debug library tests respectively, strict all-target Clippy, and release builds. The independent Python recount oracle matched **318 complete rule traces and final token sequences**; two `u16` runs correctly rejected an initial alphabet of 65,536 IDs. The 32 long runs matched the prior exact full-training fingerprint, and eight deterministic work counters agreed across all modes. The 4 MiB figures below have **one run per cell**. They diagnose mechanisms; they do not establish a stable speed ranking.

| Input | Mode | W1 s | W4 s | W1→W4 S | CPU ratio I | W4 CPU/wall U4 | HWM W1/W4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| EN | narrow u32 | 1.890 | 0.687 | 2.75 | 1.21 | 3.30 | 84.34 / 95.12 |
| EN | narrow u32 exact | 1.906 | 0.691 | 2.76 | 1.14 | 3.12 | 88.30 / 88.39 |
| EN | narrow u16 | 1.698 | 0.777 | 2.19 | 1.24 | 2.68 | 74.42 / 89.68 |
| EN | cache off | 1.837 | 0.656 | 2.80 | 1.16 | 3.21 | 78.50 / 94.46 |
| EN | cache 4096 batch | 1.705 | 0.712 | 2.39 | 1.25 | 2.98 | 84.14 / 94.53 |
| EN | cache 4096 reuse | 1.722 | 0.633 | 2.72 | 1.17 | 3.17 | 84.36 / 92.92 |
| EN | context hash | 1.906 | 0.685 | 2.78 | 1.14 | 3.17 | 78.47 / 90.79 |
| EN | context planner | 1.563 | 0.725 | 2.15 | 1.48 | 3.16 | 78.69 / 93.12 |
| ZH | narrow u32 | 0.804 | 0.379 | 2.13 | 1.34 | 2.83 | 82.02 / 100.32 |
| ZH | narrow u32 exact | 0.887 | 0.432 | 2.06 | 1.38 | 2.83 | 86.98 / 96.71 |
| ZH | narrow u16 | 0.892 | 0.383 | 2.33 | 1.15 | 2.65 | 83.69 / 92.43 |
| ZH | cache off | 0.821 | 0.439 | 1.87 | 1.33 | 2.48 | 81.85 / 93.87 |
| ZH | cache 4096 batch | 0.888 | 0.365 | 2.44 | 1.20 | 2.88 | 82.03 / 98.34 |
| ZH | cache 4096 reuse | 0.821 | 0.449 | 1.83 | 1.36 | 2.48 | 82.44 / 102.36 |
| ZH | context hash | 0.835 | 0.389 | 2.14 | 1.32 | 2.83 | 82.82 / 95.94 |
| ZH | context planner | 0.974 | 0.514 | 1.90 | 1.48 | 2.77 | 81.75 / 97.59 |

Here `S=T1/T4`, `I=C4/C1`, and `U4=C4/T4`, where `T` is full training-call wall time and `C` is full training-call process CPU time. With `U1=C1/T1`, the identity is `S=U4/(I×U1)`. This is bookkeeping, not a causal estimate of overhead: CPU time includes spin and memory stalls. Four workers ran under CPU affinity `[0,1,2,5]`; one worker ran on CPU 5. The total CPU budget equals the worker count. `train_vm_hwm_mib` is the process high-water mark sampled immediately after training and includes parsing and transient allocations.

The `u16` endpoint array saves exactly half the payload versus `u32-exact`: EN 8,355,986 versus 16,711,972 bytes, and ZH 3,423,512 versus 6,847,024 bytes. The default `u32` control retains input capacity, so it also differs in allocation policy. Exact allocation controls that confound, yet process HWM does not always fall with the narrower array: EN W4 rises from 88.39 MiB (`u32-exact`) to 89.68 MiB (`u16`). The single-run W4 call is also slower on EN (0.777 versus 0.691 s) and approximately equal to default `u32` on ZH (0.383 versus 0.379 s). The narrow layout is a real buffer saving, not a demonstrated end-to-end speed improvement.

With the replacement phase equalized, cache reuse cuts EN W4 cumulative slot initialization from 4,112,384 to 16,384 and slot scanning from 4,112,384 to 831,140; ZH W4 changes from 4,505,600 to 16,384 initialized slots and 4,505,600 to 550,121 scanned. Those totals describe cache operations and initialization payload, **not measured DRAM traffic**. W1 batch/reuse hit and miss counts match exactly. W4 counts differ slightly because dynamic task assignment changes which producer owns a task. EN W4 reuse is faster than batch in this run (0.633 versus 0.712 s), while ZH W4 is slower (0.449 versus 0.365 s); neither direction is a stable conclusion from n=1.

The context planner improves EN W1 in this window (1.563 versus 1.906 s), but W4 is slower than its hash control on both EN (0.725 versus 0.685 s) and ZH (0.514 versus 0.389 s). Its process CPU ratio `I≈1.48` in both languages is higher than the corresponding hash controls (`≈1.14` and `≈1.32`). This points to extra aggregate CPU time under four workers, without establishing whether contention, cache sharing, scheduling, or other work caused it. A separate local-scratch control will test one concrete hypothesis.

To reproduce the source, check out the commit in `shared-source-provenance.json`, then overlay `new-source-snapshot.tar.gz`; the inherited `owned_context` crate is byte-identical to that commit. `new-source-hashes.json` and `shared-source-provenance.json` give per-file hashes. The frozen release binaries are under ignored `rust/target/reruns/radical-controlled-longscreen-v1/`; `checks.json` records their hashes. Run `finalize.py` to recheck current source, binaries, oracle, and screen completeness. Raw observations are in `screen-4m.jsonl`; its environment sidecar records fixture hashes, affinity, seed, and release provenance. `screen-4m-summary.json` retains every cell and CPU-accounting calculation.

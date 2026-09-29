# Sparse dispatch mask isolation (n=3)

This archive compares the frozen `parallel_sparse_owner` implementation with `parallel_sparse_owner_all`. Both use the same sparse-owner kernel; `_all` forces the three dispatch masks to include every worker. The experiment isolates dispatch participation, with no router reset or other algorithm change. All runs use checked bounds, a 4-worker process CPU budget, and three repetitions. These small measurements are diagnostic only; they are not a formal ranking.

## Results

Values below are medians across three runs. `call` is seconds and `VmHWM` is MiB from `/proc/self/status` for the benchmarked process. Every run within a case/variant produced one stable output fingerprint; the variants' fingerprints matched each other.

| Case | Sparse call | All call | Sparse VmHWM | All VmHWM | Sparse → all plan/owner/apply active slots | Sparse → all round messages |
|---|---:|---:|---:|---:|---:|---:|
| EN 256 KiB | 0.2364 | 0.2280 | 11.98 | 12.13 | 2012/2046/2019 → 2048/2048/2048 | 12194 → 12328 |
| ZH 256 KiB | 0.1196 | 0.1423 | 12.57 | 12.00 | 1878/2014/1902 → 2048/2048/2048 | 11700 → 12408 |
| `chain-2000` | 0.1335 | 0.1990 | 4.14 | 4.05 | 1999/4592/3499 → 7996/7996/7996 | 20212 → 48008 |
| `single-run-a-65536` | 0.01249 | 0.02268 | 4.50 | 4.70 | 60/25/60 → 64/64/64 | 442 → 544 |
| `single-piece-ab-65536` | 0.01286 | 0.01519 | 4.76 | 4.64 | 60/25/60 → 64/64/64 | 434 → 536 |

The short quick runs are mixed: forced-all is slightly faster on EN and slower on ZH. The sparse dispatch does reduce activity and message counts, although much less on natural-language quick cases than on `chain-2000`. On the three edge cases it is consistently faster in these n=3 samples, especially `chain-2000`; this is a signal for follow-up, not a ranking claim. `prune_mailbox_locks` is identical within each case between variants (26974, 19956, 3997, 60, 64 respectively), as expected because these are data-mailbox locks, not control dispatch messages. Capacity-related metrics are identical between the paired variants; the small VmHWM differences do not establish a memory effect at n=3.

The earlier table mistakenly used `peak_rss_mib` (a process resource high-water mark inherited across `exec` on this runner), which was almost constant at 19 MiB. The corrected table uses each native process's `vm_hwm_mib` and does not infer equal memory usage from the old field.

**Timing caveat:** the radical prototype agent reported an independent crate build/test during 2026-09-29 17:04–17:09 UTC, overlapping the interval in which these n=3 timings were collected. Treat the measurements as potentially contended and provisional. At root's direction, they are recorded without rerunning now; repeat a small isolated timing later before drawing a performance conclusion.

`round_messages` is the existing aggregate metric and includes the implementation's message categories; it should not be read as a pure wakeup count. `active_*_workers_total` counts phase worker participations. The `_all` variant forces `force_all_dispatch=1`; sparse baseline reports 0. `CoreStats.heap_pops` counts owner-local pair-heap pops only and excludes coordinator Frontier stale pops/rebuilds.

## Validation and provenance

- 20 differential cases including 8 random cases; workers 1 and 4, checked and unchecked: 80 exact full-trace and final-fingerprint comparisons passed, no skips.
- `cargo test` targeted sparse frontier, worker groups (W=1..257), and `sparse_owner` tests passed; `cargo clippy --all-targets -- -D warnings` passed.
- No 4 MiB smoke or full matrix was run. No source changes were made after the release binary was built for this experiment.
- Source revision: `d6c02be108004837a9846fd6215abbcade446056`.
- Release binary SHA-256: `b37079e2ffddd2ffc49487c0252e2a058235aadbfaaeee6d48b4314cf0f36858`.
- Quick fixture manifest SHA-256: `67d02ed75ea85870c62493064c82a92ac6356f37d340a63ae05d5a738ffdfd0c`.
- Differential JSON records all 20 cases, seed, variant, and binary hash. Benchmark JSONL sidecars record exact source hashes, fixtures, CPU affinity/budget, Rust/Python versions, and commands.

Artifacts: `differential.json`, `quick-w4.jsonl` and its environment sidecar, `edges-w4.jsonl` and its environment sidecar, and `checks.json`.

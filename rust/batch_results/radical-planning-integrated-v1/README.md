# Exact BPE planning and integrated update experiments

This archive tests three independent crates derived from the frozen `owned_grouped_inline` trainer. `owned_selected_table` compares the original selected-pair `HashMap` lookup with a fixed-size flat table. `owned_route_cache` compares no cache with 4096 route-cache slots. `owned_fused_direct` combines the previously separate owner-commit and direct old-pair frequency-update changes; its six commit/reduce combinations remain independently selectable. All versions preserve exact greedy rule order.

The final sources passed debug library tests (selected table 11/11, route cache 10/10, integrated 9/9), strict all-target Clippy, and release builds. Ten modes with one and four workers matched a Python naive full-recount oracle for every rule and final token in 20 cases: **400/400 complete-trace comparisons**. The timed runs matched their reference fingerprints. Eight common deterministic work counters, including merge count, batch count, births, and posting visits, agreed for each input and worker count. [checks.json](checks.json), [differential.json](differential.json), [summary.json](summary.json), and [screen-4m-summary.json](screen-4m-summary.json) contain the checks and measured values.

The 256 KiB quick screen used one run per cell, EN and ZH continuous text, 512 rules, CPU 5 for one worker and CPUs 0, 1, 2, 5 for four workers. The frozen combo binary was a same-window control. Every row also matched a native reference complete trace. Times below are full training-call seconds; the quick screen has **n=1**.

| Input | Mode | W1 (s) | W4 (s) |
|---|---|---:|---:|
| EN | frozen combo | 0.0807 | 0.0573 |
| EN | selected hash / flat | 0.0958 / 0.0930 | 0.0661 / 0.0552 |
| EN | route cache off / 4096 | 0.1015 / 0.0956 | 0.0454 / 0.0694 |
| EN | integrated control / candidate | 0.1040 / 0.0921 | 0.0525 / 0.0694 |
| ZH | frozen combo | 0.0403 | 0.0304 |
| ZH | selected hash / flat | 0.0427 / 0.0550 | 0.0329 / 0.0402 |
| ZH | route cache off / 4096 | 0.0439 / 0.0516 | 0.0301 / 0.0326 |
| ZH | integrated control / candidate | 0.0561 / 0.0526 | 0.0280 / 0.0237 |

The flat selected table used at most 64 slots in these quick inputs. EN four-worker `plan_seconds` fell from 0.0281 to 0.0202 s against its same-binary hash control; ZH rose from 0.0083 to 0.0102 s. A 4096-slot route cache had a real 88.28% hit rate on EN four-worker calls and 69.95% on ZH, but its `plan_seconds` rose from 0.0187 to 0.0310 s on EN and from 0.0066 to 0.0124 s on ZH. `route_cache_bytes_initialized` sums 36,175,872 payload bytes on EN and 45,613,056 on ZH with four workers; these are initialization byte counts, not measured DRAM traffic. `peak_route_cache_slot_bytes=524,288` is a per-batch maximum of summed producer slot payload, an upper bound rather than a measured simultaneous allocation peak; the cache drops at finish. A hit count alone therefore does not establish a net benefit. These two routes had no consistent quick-screen gain and were not promoted to the larger run.

The authorized 4 MiB follow-up measured only the integrated binary's `separate+combined` control and `owner-fused+direct-old` candidate, frozen combo, and the best direct scalar native backend (EN halfword, ZH standard). It used 3000 rules, the same fixed CPU budgets, one seeded order and its reverse, **two runs per cell**. The table reports the median and observed min–max of complete training-call wall time.

| Input | Mode | W1 median [min, max] (s) | W4 median [min, max] (s) | W4 training VmHWM range (MiB) |
|---|---|---:|---:|---:|
| EN | best direct scalar | 1.626 [1.624, 1.628] | — | — |
| EN | frozen combo | 2.039 [2.008, 2.070] | 0.789 [0.767, 0.812] | 87.46–93.85 |
| EN | integrated control | 2.100 [1.896, 2.304] | 0.695 [0.636, 0.754] | 94.89–95.08 |
| EN | integrated candidate | 1.979 [1.798, 2.160] | 0.673 [0.671, 0.676] | 88.96–90.65 |
| ZH | best direct scalar | 1.185 [1.091, 1.279] | — | — |
| ZH | frozen combo | 0.833 [0.833, 0.833] | 0.419 [0.359, 0.478] | 96.07–96.68 |
| ZH | integrated control | 1.094 [1.046, 1.142] | 0.416 [0.392, 0.439] | 98.71–99.36 |
| ZH | integrated candidate | 0.926 [0.877, 0.974] | 0.415 [0.399, 0.431] | 96.02–99.50 |

EN candidate's median four-worker call is 1.172× faster than frozen combo and 2.414× faster than best direct scalar; its own W1→W4 ratio is 2.939×. Its 0.671–0.676 s range sits **inside** the integrated control's 0.636–0.754 s range, so these two repeats do not establish that combining both changes stably beats the same-binary control. On ZH, candidate and frozen combo have nearly equal four-worker medians (0.415 and 0.419 s), while candidate W1 is about 11% slower than combo W1 (0.926 versus 0.833 s). Its larger 2.229× self-scaling ratio, compared with combo's approximately 1.99×, partly reflects that slower one-worker baseline. The control's EN self-scaling ratio is 3.022×, but its W1 call is also slower than the candidate's. These are two observations per cell, not a significance or stability test. Frozen combo remains the reliable layout reference; integrated updates remain a promising, unconfirmed candidate for EN.

For the integrated binary, `call_cpu_seconds` measures process CPU time over the same full training call as `call_seconds` wall time. Taking each cell's two wall values and two CPU values to separate medians, define `U1=C1/T1`, `U4=C4/T4`, `I=C4/C1`, and `S=T1/T4=U4/(I*U1)`. This is an accounting identity, not a causal explanation. EN candidate has `U1=0.994`, `U4=3.134`, `I=1.073`, `S=2.939`; ZH candidate has `U1=0.992`, `U4=2.703`, `I=1.223`, `S=2.229`. Process CPU includes stalls, scheduling, allocations, and spin; `I` is a total CPU-time ratio, not extra useful BPE work or proven lock cost. The frozen combo CLI predates this CPU field. `train_vm_hwm_mib` is sampled after training and before fingerprint formatting, but includes process startup and input parsing high water.

The new-source [snapshot](new-source-snapshot.tar.gz) contains only the three independent crates, with hashes in [new-source-hashes.json](new-source-hashes.json). Shared inputs are recorded in [shared-source-provenance.json](shared-source-provenance.json): the 23 files under `rust/src/**`, plus `rust/Cargo.toml`, `rust/Cargo.lock`, and `rust/experiments/aa_parity.rs`, match Git commit `fde3f0ef327be2bdbbc7f3352f0a1983f8a55b78` byte for byte. To reconstruct the measured source, check out that commit and overlay the snapshot. The provenance also records the Rust compiler and Cargo versions. Release binaries are stored under ignored `rust/target/reruns/radical-planning-integrated-v1/`; both environment sidecars record their SHA-256, fixture hashes, process affinity, and invocation details. Reproduction order is debug `cargo test --lib`, strict `cargo clippy --all-targets -- -D warnings`, release build for each crate, then `run_differential.py`, `snapshot_sources.py`, `record_shared_provenance.py`, `run_quick.py`, `summarize.py`, `run_screen_4m.py`, `summarize_screen_4m.py`, and `finalize.py`. The scripts use exclusive output creation and require a fresh archive directory for a repeat.

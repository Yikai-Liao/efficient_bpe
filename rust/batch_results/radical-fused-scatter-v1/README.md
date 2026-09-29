# Exact BPE update-path experiments

This archive compares three independent Rust crates derived from the frozen `owned_grouped_inline` layout. `owned_fused` changes how planned corpus writes and owner updates are committed (`separate`, `owner-fused`, `overlap`). `owned_scatter` changes how large birth lists are filled (`owner`, `scatter`, default threshold 4096). `owned_direct` updates old-pair frequencies in their owner maps directly while still aggregating fresh pairs (`combined`, `direct-old`). Each CLI uses the same greedy rule order, lazy candidate heap, 4096-position planning chunks, and full-training fingerprint.

All three final crates passed debug library tests (9, 11, and 10 tests respectively), strict all-target Clippy, and release builds. Seven modes with one and four workers matched the Python naive full-recount oracle for every rule and final token in 20 cases: **280/280 comparisons**. The small and large timed runs also matched their reference fingerprints. The eight common deterministic work counters, including merge count, batch count, generated births, stored births, and posting visits, agreed across modes for each input and worker count. See [checks.json](checks.json), [differential.json](differential.json), [summary.json](summary.json), and [screen-4m-summary.json](screen-4m-summary.json).

The 256 KiB screen used one run per cell, EN and ZH continuous text, 512 rules, a fixed process affinity of CPU 5 for one worker and CPUs 0, 1, 2, 5 for four workers. It included the frozen combo binary as a same-window control. The seven new modes plus combo gave 32 measured calls. At this size, `scatter_heavy_keys`, heavy positions, and scatter tasks were **zero in both languages**. Those rows only measure the scatter mode's no-work path. EN four-worker calls ranged from 43.1 ms (`direct-old`) to 74.97 ms (`direct` control), with combo at 60.68 ms; ZH ranged from 23.51 ms (combo) to 42.36 ms (`scatter` control). These are single short calls, so the ordering is a screen rather than a stable ranking.

The authorized 4 MiB diagnostic then measured only the seven new modes: EN and ZH continuous text, 3000 rules, one run per cell, both worker counts, 28 calls. It used the same fixed CPU affinities and exact fixture hashes. The table gives full training call seconds, mean occupied cores for the four-worker call, and four-worker training-time process VmHWM. One-worker CPU/wall was approximately one core for all rows.

| Input | Mode | W1 call (s) | W4 call (s) | W4 CPU/wall | W4 VmHWM (MiB) |
|---|---|---:|---:|---:|---:|
| EN | fused separate | 1.835 | 0.794 | 3.13 | 89.36 |
| EN | fused owner | 1.819 | 0.708 | 2.98 | 92.38 |
| EN | fused overlap | 1.851 | 0.741 | 2.80 | 92.48 |
| EN | scatter owner | 1.916 | 0.693 | 3.23 | 91.18 |
| EN | scatter scatter | 1.785 | 0.686 | 3.21 | 90.52 |
| EN | direct combined | 1.969 | 0.753 | 2.91 | 92.85 |
| EN | direct old | 1.754 | 0.711 | 3.08 | 92.78 |
| ZH | fused separate | 0.801 | 0.409 | 2.80 | 95.68 |
| ZH | fused owner | 0.802 | 0.369 | 2.82 | 97.71 |
| ZH | fused overlap | 0.792 | 0.473 | 2.63 | 95.70 |
| ZH | scatter owner | 0.857 | 0.378 | 2.83 | 96.77 |
| ZH | scatter scatter | 0.834 | 0.382 | 2.94 | 95.14 |
| ZH | direct combined | 0.844 | 0.438 | 2.80 | 96.39 |
| ZH | direct old | 0.789 | 0.380 | 2.83 | 98.22 |

The mechanism counters explain more than the one-repeat whole-call order. Fusing owner updates reduced dispatches/completion barriers per batch from three to two: EN four-worker update-stage time fell from 0.254 to 0.176 s, and ZH from 0.142 to 0.129 s. `overlap` used only one completion barrier per batch but took 0.193 s on EN and 0.180 s on ZH in this run. Direct old-key updates reduced four-worker `frequency_reduce_seconds` from 0.148 to 0.104 s on EN and from 0.114 to 0.088 s on ZH. Its peak temporary reducer capacity fell from 14,336 to 7,168 slots in both inputs; temporary key peaks fell from 10,418 to 5,276 on EN and 7,918 to 4,119 on ZH. Fresh-key aggregation remains in this mode.

At 4 MiB, scatter really ran: EN had 87 heavy keys, 765,823 heavy positions, and 343 scatter tasks with four workers; ZH had 3 keys, 19,559 positions, and 9 tasks. Its measured fill subphase was 0.00148 s on EN and 0.000097 s on ZH. Relative to `scatter owner`, full-call time changed from 0.693 to 0.686 s on EN and from 0.378 to 0.382 s on ZH. This diagnostic does not show a material net benefit for scatter. The update and fill fields describe different implementations; `owner_update_branch_seconds` is inside `update_stage_seconds`, and other reported phase fields may likewise be nested. Do not sum them to reconstruct the full call.

`call_cpu_seconds` uses `CLOCK_PROCESS_CPUTIME_ID` around the complete training call, while `call_seconds` is wall time over the same call. Their ratio is average occupied cores, including stalls; it is not a measure of useful arithmetic utilization. `train_vm_hwm_mib` is sampled immediately after training, before fingerprint and trace formatting, but includes startup and input parsing high water. The frozen combo binary predates the CPU field, so its quick-screen CPU ratio is absent. Every 4 MiB result has **n=1**, so differences of a few percent do not establish a final ordering.

Reproduction uses `/root/.cargo/bin/cargo` with `CARGO_TARGET_DIR=/root/code/efficient_bpe/rust/target`, the three manifests under `rust/experiments/radical/`, and the Python environment at `/tmp/efficient-bpe-audit-20260929/env/bin/python`. Run each crate's `cargo test --lib`, `cargo clippy --all-targets -- -D warnings`, then `cargo build --release`. Copy each release binary to the ignored `rust/target/reruns/radical-fused-scatter-v1/` path, then run `run_differential.py`, `snapshot_sources.py`, `run_quick.py`, `summarize.py`, `run_screen_4m.py`, `summarize_screen_4m.py`, and `finalize.py` in that order. These scripts use exclusive output creation except `finalize.py`; use a fresh archive directory to repeat them. [new-source-hashes.json](new-source-hashes.json) and [new-source-snapshot.tar.gz](new-source-snapshot.tar.gz) preserve the measured source, while both environment sidecars record binary hashes, fixture hashes, affinity, and invocation details. The large binaries are ignored by Git.

# Local scratch and integer-hash controls

Two independent crates extend the frozen exact owner trainer. `owned_context_local` compares a borrowed producer `Scratch` header with moving that header into the producer call and back; its buffers remain owned by the same worker. `owned_integer_hash` changes the integer-key map builder between Rust's `RandomState` and `ahash::RandomState` inside the same binary. The main trainer and previous archives were unchanged.

Both crates passed debug library tests (12/12 and 9/9), strict all-target Clippy, and release builds. A Python full-recount oracle matched **240/240 complete rule traces and final token sequences**, including context's zero-budget fallback. The 20 quick runs also matched a native reference trace, and the 20 later 4 MiB runs matched the prior exact full-training fingerprint. Eight deterministic work counters agree across the two integer-hash modes.

The 256 KiB quick screen used one run per cell. `context_borrowed` and `context_local` are the same binary and planner; `integer_std` and `integer_ahash` are the same binary and trainer.

| Input | Mode | W1 call s | W4 call s | W4 HWM MiB |
|---|---|---:|---:|---:|
| EN | context hash, borrowed | 0.0909 | 0.0463 | 8.08 |
| EN | context, borrowed | 0.0766 | 0.0490 | 8.73 |
| EN | context, local | 0.0874 | 0.0631 | 8.66 |
| EN | integer std | 0.0918 | 0.0586 | 8.66 |
| EN | integer aHash | 0.0611 | 0.0490 | 8.55 |
| ZH | context hash, borrowed | 0.0407 | 0.0273 | 7.88 |
| ZH | context, borrowed | 0.0532 | 0.0341 | 8.11 |
| ZH | context, local | 0.0416 | 0.0339 | 7.73 |
| ZH | integer std | 0.0450 | 0.0323 | 7.59 |
| ZH | integer aHash | 0.0291 | 0.0199 | 8.06 |

Local scratch does not show a consistent net gain: EN slows in both worker configurations, while ZH W1 improves and ZH W4 is nearly unchanged. This experiment does not prove or disprove hardware false sharing. aHash improves all four quick cells, so a separately authorized two-repeat 4 MiB screen tested only that choice and a same-window direct scalar reference.

| Input | Mode | Workers | Median call s [min–max] | Median CPU s | Median training HWM MiB [min–max] |
|---|---|---:|---:|---:|---:|
| EN | integer std | 1 | 1.895 [1.872–1.917] | 1.885 | 81.42 [78.59–84.25] |
| EN | integer std | 4 | 0.769 [0.731–0.806] | 2.331 | 95.70 [94.60–96.81] |
| EN | integer aHash | 1 | 1.310 [1.254–1.366] | 1.307 | 84.38 [84.36–84.40] |
| EN | integer aHash | 4 | 0.533 [0.517–0.548] | 1.580 | 96.51 [94.54–98.49] |
| EN | direct scalar CF16 | 1 | 1.632 [1.609–1.655] | 1.629 | 86.41 [86.41–86.41] |
| ZH | integer std | 1 | 0.838 [0.783–0.892] | 0.836 | 81.92 [81.57–82.27] |
| ZH | integer std | 4 | 0.442 [0.403–0.481] | 1.236 | 100.37 [98.46–102.29] |
| ZH | integer aHash | 1 | 0.585 [0.561–0.609] | 0.584 | 82.14 [82.09–82.19] |
| ZH | integer aHash | 4 | 0.303 [0.285–0.321] | 0.835 | 94.09 [93.39–94.79] |
| ZH | direct scalar CF | 1 | 0.984 [0.970–0.997] | 0.981 | 100.42 [100.09–100.75] |

Using the two-run medians, aHash is 1.45× and 1.44× faster than std for EN W1/W4, and 1.43× and 1.46× for ZH W1/W4. Its W4 call is 3.06× faster than the EN direct scalar reference and 3.25× faster than the ZH reference **in this window**. Those direct references use different trainer paths; they do not isolate a hasher change. The aHash trainer's own W1→W4 speedups are 2.46× EN and 1.93× ZH, approximately the std mode's 2.47× and 1.90×. The result is a substantial hasher constant-factor improvement here, not a new parallel-scaling gain. Two repetitions and the printed ranges are not confidence intervals or a stable general ranking.

For CPU accounting, calculate median wall `T` and median process CPU `C` separately for each cell. `U=C/T` is average occupied cores, `I=C4/C1`, and `S=T1/T4=U4/(I×U1)`. EN aHash has `U4≈2.97`, `I≈1.21`; ZH aHash has `U4≈2.75`, `I≈1.43`. CPU time includes spin and stalls, so neither `U` nor `I` identifies useful compute or a particular bottleneck. Both screens pin W1 to CPU 5 and W4 to CPUs `[0,1,2,5]`. `train_vm_hwm_mib` is the process high-water mark sampled immediately after training; it includes parsing and transient allocation.

This build of aHash **selected its software fallback**, not its x86 AES implementation. `ahash-build-context.json` records aHash 0.8.12's Cargo fingerprint (default/getrandom/runtime-rng/std features, empty rustflags), unset `RUSTFLAGS` and `CARGO_ENCODED_RUSTFLAGS`, and the target cfg with no `aes` feature. The archived crate source selects `aes_hash` on x86 only when `cfg(target_feature="aes")` is true; otherwise it selects `fallback_hash`. `ahash-cargo-fingerprint.json` and `rustc-target-cfg.txt` preserve the build evidence.

Reproduction provenance is in `new-source-hashes.json`, `new-source-snapshot.tar.gz`, and `shared-source-provenance.json`. Check out the recorded base commit, then overlay the new-source snapshot. The integer-hash crate's lockfile was generated offline from locally cached dependencies and is part of that snapshot. Frozen release binaries are under ignored `rust/target/reruns/radical-local-hash-v1/`; `checks.json` records their hashes and validation status. `quick.jsonl` and `integer-long.jsonl` contain every raw observation, with adjacent `.environment.json` files documenting fixture hashes, affinity, binary hashes, and scheduling. `summary.json` and `integer-long-summary.json` provide the derived comparisons. Run `finalize.py` to verify completeness and provenance without repeating measurements.

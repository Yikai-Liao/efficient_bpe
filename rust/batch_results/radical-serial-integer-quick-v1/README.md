# Serial integer-hash light screen

This same-window, fixed-CPU-budget screen compares the frozen `serial_integer_hash` crate's CF32/CF16 × std/aHash × checked/unchecked modes. It also includes the frozen owner aHash trainer at W1/W4 and the native direct serial reference. The earlier serial crate had already passed six library tests, strict Clippy, and 156 full-trace oracle matches plus four expected Halfword alphabet rejections; this run does not repeat them.

All **22 training calls** on EN/ZH continuous 256 KiB fixtures matched the native reference's complete merge-rule trace, final tokens, and SHA-256 fingerprint. There is **one run per cell**, so the figures are a light screen rather than a stable ranking.

| Input | Mode | W | Call s | CPU s | Training HWM MiB |
|---|---|---:|---:|---:|---:|
| EN | CF32 std checked | 1 | 0.0580 | 0.0580 | 7.92 |
| EN | CF32 aHash checked | 1 | 0.0386 | 0.0362 | 7.85 |
| EN | CF32 std unchecked | 1 | 0.0594 | 0.0585 | 7.83 |
| EN | CF32 aHash unchecked | 1 | 0.0329 | 0.0327 | 8.02 |
| EN | CF16 std checked | 1 | 0.0595 | 0.0594 | 7.45 |
| EN | CF16 aHash checked | 1 | 0.0474 | 0.0464 | 7.45 |
| EN | CF16 std unchecked | 1 | 0.0642 | 0.0635 | 7.51 |
| EN | CF16 aHash unchecked | 1 | 0.0381 | 0.0381 | 7.50 |
| EN | owner aHash | 1 | 0.0555 | 0.0551 | 8.27 |
| EN | owner aHash | 4 | 0.0314 | 0.1045 | 8.58 |
| EN | native direct CF16 checked | 1 | 0.0783 | 0.0761 | 7.80 |
| ZH | CF32 std checked | 1 | 0.0267 | 0.0253 | 5.67 |
| ZH | CF32 aHash checked | 1 | 0.0198 | 0.0197 | 5.76 |
| ZH | CF32 std unchecked | 1 | 0.0261 | 0.0260 | 5.76 |
| ZH | CF32 aHash unchecked | 1 | 0.0207 | 0.0207 | 5.80 |
| ZH | CF16 std checked | 1 | 0.0276 | 0.0275 | 5.44 |
| ZH | CF16 aHash checked | 1 | 0.0217 | 0.0217 | 5.61 |
| ZH | CF16 std unchecked | 1 | 0.0265 | 0.0265 | 5.43 |
| ZH | CF16 aHash unchecked | 1 | 0.0279 | 0.0279 | 5.50 |
| ZH | owner aHash | 1 | 0.0430 | 0.0429 | 7.13 |
| ZH | owner aHash | 4 | 0.0180 | 0.0612 | 8.00 |
| ZH | native direct CF32 checked | 1 | 0.0302 | 0.0302 | 6.13 |

Within the **same serial backend and bounds mode**, aHash beats std in seven of eight n=1 cells; ZH CF16 unchecked is the exception (0.0279 versus 0.0265 s). EN CF32 checked changes 0.0580→0.0386 s, EN CF16 checked 0.0595→0.0474 s, ZH CF32 checked 0.0267→0.0198 s, and ZH CF16 checked 0.0276→0.0217 s. This is a within-kernel hasher comparison. The owner and native entries use different trainer paths, so their times do not isolate a hash algorithm change. The larger 4 MiB two-repeat owner result remains in the earlier [integer-hash archive](../radical-local-hash-v1/README.md); this screen does not extrapolate serial performance to 4 MiB.

W1 was pinned to CPU 5 and W4 to CPUs `[0,1,2,5]`. `call_seconds` times the whole training call; `call_cpu_seconds` is process CPU time, so CPU/wall is average occupied cores rather than useful-compute utilization. `train_vm_hwm_mib` is the process high-water mark sampled immediately after training, before fingerprint and trace construction. It includes parsing and transient allocations. The aHash 0.8.12 build uses its **software fallback**, documented with the compiler cfg and Cargo fingerprint in the [prior build context](../radical-local-hash-v1/ahash-build-context.json), not an x86 AES target feature.

`source-provenance.json` verifies 39 relevant source files byte for byte against Git commit `7d6eb8105683e5817cf981cf6b19822777aa4df0`, and records the earlier source archives, build flags, compiler version, and their hashes. `screen.jsonl` stores every raw output; its environment sidecar records fixture and binary hashes, CPU affinity, seed, and commands. `summary.json` gives per-cell and same-kernel comparisons. Frozen binaries remain under ignored `rust/target/reruns/`. Run `finalize.py` to recheck provenance and completeness without new timing.

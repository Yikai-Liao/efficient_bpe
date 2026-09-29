# Pair-owned pipelined batch, first screen

`parallel_pair_owned_pipeline` retains the pair-owned Plan16 layout and exact certified-prefix selection. It combines owner scalar reduction with disjoint endpoint writes in one worker phase, returns unused candidate leases at Prepare/GatherSelf, and finalizes staged new edges at the next candidate request or Finish. The old `parallel_pair_owned` variant remains the same-binary comparator.

Targeted Rust tests passed (4/4), as did `cargo clippy --all-targets -- -D warnings` and the release build. The Python full-recount oracle matched the complete rule trace and final tokens in 80/80 runs: 20 cases, including 8 seeded random cases, at workers 1 and 4 with checked and unchecked bounds. Both benchmark files have stable fingerprints within each case.

| Input | Workers | Old call (s) | Pipeline call (s) | Old / pipeline | Old VmHWM (MiB) | Pipeline VmHWM (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| EN 256 KiB | 1 | 0.1837 | 0.1557 | 1.180× | 9.47 | 9.79 |
| EN 256 KiB | 4 | 0.0677 | 0.0780 | 0.868× | 11.40 | 11.75 |
| ZH 256 KiB | 1 | 0.0936 | 0.0751 | 1.247× | 7.38 | 7.79 |
| ZH 256 KiB | 4 | 0.0396 | 0.0353 | 1.124× | 10.00 | 9.86 |
| EN 4 MiB | 1 | 3.1753 | 3.0750 | 1.033× | 111.38 | 110.86 |
| EN 4 MiB | 4 | 1.1621 | 1.1303 | 1.028× | 136.18 | 136.73 |
| ZH 4 MiB | 1 | 1.8916 | 1.8381 | 1.029× | 122.12 | 122.40 |
| ZH 4 MiB | 4 | 0.7469 | 0.6135 | 1.217× | 119.73 | 120.88 |

These are interleaved, one-shot screens (`n=1`), not stable speed estimates. The English 256 KiB four-worker result regresses despite the larger English four-worker result improving slightly. The pipeline reports fewer aggregate round messages: for EN 4 MiB, 10,160 → 6,144; for ZH 4 MiB, 11,240 → 6,840. This count includes protocol replies and candidate refills, so it is not a direct count of operating-system wakeups. Compare `call_seconds`; the pipeline's `merge_seconds` includes pending Finish cleanup while the old variant's `merge_seconds` stops before Finish. `VmHWM` is the native process high-water value, not `peak_rss_mib`.

The frozen binary is retained at `rust/target/reruns/pair-owned-pipeline-v1/ablation` (ignored by Git), with SHA-256 `25ad976ed8ede66c63b0ffc3e3956ac6999fdb2b11bf96b9459c68109e712147`. `native-source-snapshot.tar.gz` and `native-source-hashes.json` capture the Rust source at binary archival. Both benchmark sidecars match this snapshot for every compiled source; the sole hash difference is `parallel_spatial.rs`, an unregistered, uncompiled prototype edited concurrently after the build. The sidecars also record fixture and binary hashes, CPU affinity, tool versions, and commands.

Raw artifacts: `differential.json`, `quick.jsonl`, `smoke.jsonl`, their environment sidecars, and `checks.json`. The corrected sparse-dispatch RSS table is in the separate `sparse-dispatch-v1` archive; it was updated from raw `vm_hwm_mib` without rerunning its benchmark.

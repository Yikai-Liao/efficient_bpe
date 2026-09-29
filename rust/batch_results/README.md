# Batch training evidence

See [the report](../BATCH_PARALLEL_REPORT.md) for implementation, scope and limitations. Native implementation: `c2e8c69`. All completed timing archives here use binary SHA-256 `f2af9196829d5493aab21816c5556ad9c994fe2f3df4849e18325e028ada0df2`.

| Artifact | Status and use |
|---|---|
| [checks.json](checks.json) | Build, test, independent oracle and source hashes |
| [differential.json](differential.json) | 1,792 complete-trace oracle comparisons; all matched |
| [pilot-v1b/commands.json](pilot-v1b/commands.json) | 32 complete runs, one sample per configuration; preliminary diagnostics |
| [pilot-v1b/summary.json](pilot-v1b/summary.json) | Derived ratios and output checks; no pooling or final ranking |
| [quick-v1/completion.json](quick-v1/completion.json) | 10 lightweight runs, 1.472 seconds for benchmark stages; family-specific fingerprint checks passed |
| [quick-fixtures-262144-512.json](quick-fixtures-262144-512.json) | Frozen source slices and prepared-input hashes; 256 KiB, 512 rounds, weight 1 |
| [formal-v1/INTERRUPTED.json](formal-v1/INTERRUPTED.json) | Incomplete five-repeat matrix, stopped to adopt lightweight iteration; excluded from conclusions |
| [pilot-v1/commands.json](pilot-v1/commands.json) | Failed preflight because rustc was absent from PATH; no timings |

Each timing JSONL has a `.environment.json` sidecar. Samples are isolated child processes, interleaved within a stage; the entire child, including the coordinator, receives the stated CPU budget. `call_seconds` times the native training call. Process VmHWM includes input parsing and fingerprint generation as well. Hashes pin the executable and input independently of later documentation or runner edits.

`run_batch_matrix.py` now defaults to the quick screen. Use `--variants` for a focused quick/smoke comparison, `--smoke` for a few 4 MiB checks, and `--full` only once the candidate implementations converge. Build changed Rust sources before benchmarking; the runner uses the existing release executable. Generated fixtures and logs are ignored; keep future reruns in `batch_results/reruns/` and archive selected evidence deliberately.

Do not infer server scaling from quick samples. Do not combine different binaries, piece weights, incomplete matrices or exact/approximate semantics in a timing median. Legacy 4 MiB continuous fixtures have weight 2; the new 16 MiB and quick fixtures have weight 1.

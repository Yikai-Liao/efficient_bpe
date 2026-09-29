# Batch training evidence

See [the original batch report](../BATCH_PARALLEL_REPORT.md) and [the owner-sharding follow-up](../OWNER_PARALLEL_REPORT.md) for implementation, scope and limitations. The original batch archives below use native implementation `c2e8c69` and binary SHA-256 `f2af9196829d5493aab21816c5556ad9c994fe2f3df4849e18325e028ada0df2`. Later experiment directories pin their own binaries; do not pool them.

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

Subsequent focused experiments:

| Artifact | Status and use |
|---|---|
| [pair-owned-quick-v1/summary.json](pair-owned-quick-v1/summary.json) | Initial frequency/heap sharding, quick checked/unchecked screen |
| [pair-owned-smoke-v1/summary.json](pair-owned-smoke-v1/summary.json) | Same-binary comparison against the earlier certified batch kernel, 4 MiB, n=1 |
| [pair-owned-serial-reference-v1/summary.json](pair-owned-serial-reference-v1/summary.json) | Direct serial CF/CF16 reference for the initial sharding binary |
| [pair-owned-edges-v1.jsonl](pair-owned-edges-v1.jsonl) | Chain, AA and repeated AB diagnostics; exposed candidate overfetch cost |
| [pair-owned-differential-v1.json](pair-owned-differential-v1.json) | Initial sharding: 120 complete-trace comparisons |
| [pair-owned-compact-v1/README.md](pair-owned-compact-v1/README.md) | Independent same-binary Plan16 versus grouped Plan4 memory ablation; n=1 |
| [sparse-owner-v1/README.md](sparse-owner-v1/README.md) | Cached frontier and sparse dispatch versus a fair all-worker single-rule baseline; n=1, 240 trace comparisons |
| [sparse-dispatch-v1/README.md](sparse-dispatch-v1/README.md) | Same-kernel dispatch-mask isolation, 80 trace comparisons; n=3 timings overlapped another build and remain provisional |
| [pair-owned-pipeline-v1/README.md](pair-owned-pipeline-v1/README.md) | Three-phase exact batch pipeline versus the original sharded kernel; 80 trace comparisons, n=1 quick/smoke |
| [quality-quick-v1/commands.json](quality-quick-v1/commands.json) | Exact versus relaxed heldout token counts; disjoint 256 KiB train/heldout slices and 512 rules |
| [radical-fixed-budget-v1/README.md](radical-fixed-budget-v1/README.md) | Strict CPU-budget calibration, global batch arena and boxed postings; n=1 |
| [radical-owner-boxed-v1/README.md](radical-owner-boxed-v1/README.md) | Unique posting and frequency owner, lazy/eager heaps, and batched boxed storage; 120 complete oracle checks, quick and limited 4 MiB screen |
| [radical-owned-extensions-v1/README.md](radical-owned-extensions-v1/README.md) | Budgeted spatial certificate and counted deltas; 160 complete oracle checks, no stable speed winner in the n=1 quick screen |
| [radical-owned-4m-n2-v1/README.md](radical-owned-4m-n2-v1/README.md) | Frozen owned/probe/counts discriminant, 40 fingerprint-verified calls, two repeats per configuration; no full matrix |
| [radical-owner-routing-v1/README.md](radical-owner-routing-v1/README.md) | Grouped birth chains and logical owner shards; 200 complete oracle checks, quick and 12 limited W4 4 MiB calls |
| [radical-owned-inline-v1/README.md](radical-owned-inline-v1/README.md) | A reviewed 16-byte posting container with two inline positions; 80 complete oracle checks and 20 quick traces |
| [radical-layout-combo-v1/README.md](radical-layout-combo-v1/README.md) | Grouped+inline integration; 80 complete oracle checks, 16 quick traces, and final limited 36-call W1/W4 4 MiB n=2 comparison |

Each timing JSONL has a `.environment.json` sidecar. Samples are isolated child processes, interleaved within a stage; the entire child, including the coordinator, receives the stated CPU budget. `call_seconds` times the native training call. Legacy `vm_hwm_mib` includes input parsing and fingerprint preparation; the newer `train_vm_hwm_mib` is sampled before fingerprint/trace formatting and still includes startup and input parsing. Do not mix these two memory fields. The pipeline and earliest radical/spatial screens accidentally allowed all six CPUs even for W1; their READMEs flag this limitation. New strict-budget screens pin the whole process, coordinator included, to W CPUs. Hashes pin the executable and input independently of later documentation or runner edits.

`run_batch_matrix.py` now defaults to the quick screen. Use `--variants` for a focused quick/smoke comparison, `--smoke` for a few 4 MiB checks, and `--full` only once the candidate implementations converge. Build changed Rust sources before benchmarking; the runner uses the existing release executable. Generated fixtures and logs are ignored; keep future reruns in `batch_results/reruns/` and archive selected evidence deliberately.

Do not infer server scaling from quick samples. Do not combine different binaries, piece weights, incomplete matrices or exact/approximate semantics in a timing median. Legacy 4 MiB continuous fixtures have weight 2; the new 16 MiB and quick fixtures have weight 1.

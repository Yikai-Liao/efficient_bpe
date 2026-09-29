# Exact whole-piece multiprocessing prototype

`parallel_driver.train(prepared, workers=1|2|4, backend_class=LeanEndpoints,
max_merges=..., min_frequency=2, serial=False)` trains **one** vocabulary. The
master owns the global weighted frequency map and lazy max heap. Its packed
pair key `(left_id << 32) | right_id` preserves lexicographic pair tie order.
It broadcasts each chosen rule and fresh ID to persistent `spawn` workers;
workers do not choose rules independently or merge their vocabularies later.

The master partitions the `prepare` corpus at complete zero-separated pieces.
Shards are contiguous and greedily balanced by stored character positions.
Weighted deduplication changes frequency, not the number of stored positions
that a worker traverses. A worker owns its local corpus, fused backend,
historical occurrence arrays, and local weight pivots. It appends every fresh
ID length even when its shard has no occurrence. For the chosen pair, it
processes historical positions left-to-right, rejects stale positions,
updates adjacent pair occurrences, and returns sparse weighted count deltas
plus keys involving the fresh ID. The master sums deltas, lazily reconciles
old heap entries, and enqueues the new keys.

Local pivots are stored only when the weight changes, matching the shared
trainer's grouped pivots. After the master reduces initial pair counts, both
its received count maps and each worker's initial count map are released;
workers retain their occurrence lists for later rules.

This is exact under the separator rule: every counted edge belongs to exactly
one piece and therefore one worker. Initial global counts are the sum of
local counts. Each rule's worker deltas cover every changed edge in its piece;
there are no cross-shard edges. Summing the deltas preserves the global count
invariant, so the next greedy choice and its deterministic tie order match
`common_fused` and `packed_driver`. Left-to-right replacement order is also
preserved within every piece. `serial=True` runs the same `_WorkerState`
algorithm sequentially in the parent to expose IPC/process overhead.

## Timing, messages, and memory

`init_seconds` includes parent sharding, spawning, worker construction,
initial local counting, and global heap construction. `merge_seconds` runs
from the first global rule through the last worker delta; final extraction and
hashing are outside it. Wall-clock time is the meaningful parallel elapsed
time. `master_cpu_seconds` is **only the parent process CPU** in spawn mode;
it cannot establish speedup. `worker_process_cpu_seconds` reports each
spawned worker's cumulative CPU through final extraction, including Python
startup. `worker_merge_cpu_seconds` sums that worker's timed rule handlers.
In serial mode all worker work also appears in parent CPU, so those CPU
figures must not be added together.

Each rule sends one command and receives one sparse delta from every worker:
`messages_per_round = 2 × actual_workers`. This fixed barrier, per-round
pickling, and sparse-delta aggregation can dominate small pieces or many
rules. `active_worker_rounds`, `worker_merge_counts`, and
`merge_imbalance_max_minus_min_sum` expose idle workers and uneven work.
At most one shard is assigned per process; if there are fewer nonempty pieces
than requested workers, `actual_workers` is reduced accordingly.

The original prepared u32 corpus stays in the parent. Building shards creates
temporary parent u32 copies, measured by
`parent_temporary_shard_buffer_bytes`; they are released after worker setup.
Each spawned worker retains its own local input u32 array and makes a backend
copy, in addition to occurrence arrays, local dictionaries, pivots, and a
token-length list. Buffer fields count elements only, not Python container
overhead or pickling buffers. `parent_peak_rss_mib` and each
`worker_peak_rss_mib` are process high-water marks; their sum is **not** a
simultaneous memory peak. `worker_ready_rss_mib` and `worker_end_rss_mib` are
current Linux RSS snapshots at two barriers. No per-round process sampling is
performed.

This Python prototype uses separate processes to avoid the GIL but pays for
copied corpora and per-rule IPC. A native implementation could keep the same
global-rule barrier and per-shard ownership while placing corpus, token
lengths, and count-delta buffers in shared memory, with sparse worker-local
occurrences and a master reduction. That is a separate design and performance
claim; these results measure the Python process implementation.

Small reproducible checks:

```sh
python3 benchmarks/bpe_core_comparison/evolution/test_parallel.py
```

# Scatter-fill large grouped postings

`owned_scatter` is an independent copy of frozen `owned_grouped_inline`. The
same binary accepts `--fill-mode owner|scatter` (default `owner`) and
`--scatter-threshold N` (default 4096, minimum 3). Both modes keep the exact
batch certificate, AA parity, dynamic planning tasks, keyed birth chains,
owner-local frequency reduction, fresh-list allocation, and early selected-list
release. The only changed operation is filling eligible fresh postings after
apply and frequency reduction. No apply/reduce overlap is attempted.

Frequency reduction knows each fresh key's total physical occurrence count and
allocates one `SmallPosting` with that reservation. Scatter mode leaves keys
below threshold on the original owner-local append path. For each heavy key,
the coordinator looks up that key once in each producer→owner route, using the
route's exact `occurrences` to form at most W nonempty segments. It checks that
their sum equals the globally combined count, then temporarily moves the empty
heap posting out of the owner map. The source entry holds an empty inline
placeholder until fill succeeds. Segment metadata is O(W × heavy keys in the
batch); there is no per-position task record. One unusually large producer
chain remains a serial task and therefore a lower bound.

`SmallPosting::spare_capacity_mut` only exposes a fresh heap posting with
`len=0` as `&mut [MaybeUninit<u32>]`. The coordinator takes the exact counted
prefix and divides it with `split_at_mut`, so Rayon tasks have disjoint mutable
slices and no shared raw write pointer. Each task traverses its producer chain,
writes one slice, and verifies that the chain is neither shorter nor longer
than its count. All segment tasks join, including on error. Only after every
segment succeeds does the unsafe `finish_initialize(count)` set the visible
length. An error before that leaves the posting at `len=0`; its Vec allocation
is still freed safely by Drop, while the training call returns an error and
does not reuse the owner map. Successful postings are moved back into their
unique owner before the next selection. Debug builds also verify each planned
birth key against the final corpus. AA sorts the complete posting at selection;
non-AA matches remain order-independent.

The CLI additionally reports process `call_cpu_seconds` over the same training
call interval as `call_seconds`, sampled with `CLOCK_PROCESS_CPUTIME_ID` before
and after train. `train_vm_hwm_mib` is still sampled immediately after training
and before fingerprint formatting. Scatter metrics count heavy keys, positions,
producer-chain tasks, all W route lookups per heavy key, and peak segment
header capacity. `scatter_header_bytes_lower_bound_peak` counts explicit Rust
container/key payload capacities, excluding hash-table control bytes, allocator
overhead, and Rayon scheduling buffers; it is not RSS. `scatter_setup_seconds`
and `scatter_fill_seconds` are inside `birth_group_fill_seconds`, not additive
independent phases. Increasing parallel tasks may still lose to the extra
route lookups, split setup, and scheduling: grouped fill was already a small
part of the whole call. The test threshold 3 exercises many heavy paths;
benchmark screening keeps both modes at the fixed default 4096.

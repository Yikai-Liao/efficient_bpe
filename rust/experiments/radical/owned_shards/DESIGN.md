# Logical owner shards above a fixed worker pool

This crate copies the frozen owned v5 trainer. `--workers W` still creates W
Rayon threads and exactly W dynamic producer jobs. `--owner-shards S` chooses
the number of logical key owners; zero, the default, resolves to S=W. The
owner hash, frequency map, candidate heap, and unique historical posting
list all use S. Each producer has S route buckets, and owner reduction and
birth decoding use `owners.par_iter_mut()` on the same W-thread pool. No
worker receives a full corpus or full key index.

Only candidate coordination changes further. At the start of each batch,
the coordinator calls `peek_current` once per owner and constructs a
`BinaryHeap` from the resulting vector of at most S heads. Building this
heap is O(S). After selecting a candidate, only that candidate's owner heap
changes; the coordinator refreshes that owner's head and pushes it into the
frontier. A type-conflicting or non-first AA candidate remains at its owner
heap head when the batch stops; discarding the temporary frontier does not
discard the candidate. Stable frequency during the batch and the same
`(frequency descending, key ascending)` comparison preserve the v5 rule
order and fresh-ID order. The expected selection work is O(S) head scans per
batch plus O(log S) frontier operations per selected rule, apart from stale
candidate cleanup within owner heaps.

Increasing S can balance owners with hot keys but raises fixed costs. Each
producer materializes S route headers: O(W·S) headers and capacity, even if
many routes are empty. Initial owner construction and each batch's owner
reduction/birth decode iterate every owner over every producer's bucket;
their fixed scan work is O(W·S) per batch, or O(B·W·S) over B batches.
Routing payloads and historical posting positions remain unique by key,
but temporary route payloads overlap persistent postings as in v5. The CLI
rejects S>4096 or W·S>65,536 to keep the dense prototype bounded; it does
not introduce a sparse routing rewrite.

The JSON reports `route_bucket_headers`, frontier head/scanned-owner counts,
and sum/max per-owner counts for initial positions, reduced delta keys, and
birth positions. Initial position max and sum describe one initialization
phase, so they can be used together to assess that phase's owner skew. Delta
and birth sums accumulate over *all* epochs, while their peaks are the
largest owner count in *one* epoch; dividing those numbers does not yield an
average per-epoch imbalance. These counts record work, not worker CPU time.
More shards can distribute several hot keys across owners, but a single
dominant key's birth append still belongs to one owner task. Removing that
lower bound would require splitting payload filling below the key-owner
level while keeping one authoritative frequency and posting list. `S=W` within
this binary controls for the new frontier, while S=2W and S=4W isolate
sharding. Compare those settings with identical input, rule cap, chunk size,
heap policy, W, and CPU affinity. The frozen owned v5 binary remains a
separate baseline for the frontier's own overhead. Complete rule/final-token
oracle and train-time VmHWM are required before interpreting speed or memory.

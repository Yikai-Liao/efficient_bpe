# Owner-local accumulator reuse

This independent experiment copies the frozen `owned_integer_hash` trainer and
keeps its selection, planning, corpus writes, producer routes, per-key posting
layout, heap policy, and `std|ahash` call-wide dispatch. The only training
choice is `--owner-commit staged|fused-fresh|fused-reuse` (default `staged`).
All three choices run in one binary with the same input and CLI accounting.

`staged` is the source trainer's two owner phases: first combine every
producer's delta map and update frequencies, then scan the original route
maps to fill newborn postings. The two fused choices move each producer's
route headers into owner-exclusive vectors and perform those two phases
consecutively per owner. `fused-fresh` starts with an empty combined map;
`fused-reuse` moves the producer map with the largest number of distinct keys
into the combined slot. Moving transfers the map allocation and leaves that
producer's `born` vector in its route. Each owner is still executed by the
Rayon pool, and the corpus and permanent posting index are never copied.

For each pair, the combined delta's weight and occurrence count cover **all**
producers. In reuse mode its `head` keeps a narrower meaning: it indexes only
the chosen producer's `born` vector. An entry inserted by another producer
gets `head = u32::MAX`; foreign route maps retain their own heads and local
counts. The chosen producer is excluded from the foreign reduction and fill.
The frequency pass allocates an exact-capacity `SmallPosting` only for eligible
fresh keys. The fill pass first walks reused heads, then foreign heads, and
checks each final posting length against the combined occurrence count.
Foreign chains are also checked against their local occurrence counts. The
reused chain's local count was overwritten by the global total, so its runtime
checks instead enforce in-bounds indices, strictly decreasing nonterminal
`next`, at most `born.len()` steps, and no writes beyond the global total.
On an error the private training call returns; no incomplete posting is used
in another epoch.

The transposition moves `W²` route headers and leaves each delta table and
birth vector allocated exactly once. The extra owner-vector headers use
`O(W²)` space. A fresh combined table inserts `sum(k_i)` producer entries;
reuse inserts only `sum(k_i) - max(k_i)` entries, then both modes still visit
each distinct combined key and fill every eligible birth. Hash lookups are
expected constant time under the chosen hasher, not a worst-case guarantee.
Reusing the largest key-count map can trigger rehash if its capacity is too
small for the union; it also changes table iteration order. The experiment
measures whether saved insertions outweigh those costs.

Original route metrics are captured **before** moving any route. The reported
`accumulator_capacity_sum_peak` and `accumulator_entries_sum_peak` are the
largest per-epoch sums of owner accumulators *after* reduction; they are
capacity and entry proxies, not byte-accurate allocation peaks. In reuse
mode the accumulator is one of the original route maps, so these proxies
must not be added to `peak_route_delta_capacity` as though disjoint. The
transposed-header byte proxy conservatively sums old and new route-vector
capacities; it is an upper bound on this header overlap, not measured RSS.
`fused_commit_seconds` includes transposition, reduction and fill in fused
modes. The staged-only `frequency_reduce_seconds` and
`birth_group_fill_seconds` stay zero in fused modes and must not be added to
`fused_commit_seconds`. The CLI reports training VmHWM immediately after the
call, separately from final process VmHWM after trace/fingerprint output.

Same-binary full-trace tests cover weighted AA, adjacent rules, and frequency
thresholds with both hashers and one or four workers. A direct constant-hash
test forces collisions while checking a reused local head, a foreign-only
fresh key, old-key frequency subtraction, and threshold removal. Malformed
foreign counts and a cyclic reused chain must return errors. Full corpus
oracle and timing validation are delegated to the shared executor; this
document states the intended protocol, not measured performance.

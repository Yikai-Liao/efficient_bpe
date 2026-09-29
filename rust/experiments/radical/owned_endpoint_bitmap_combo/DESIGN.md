# Tagged endpoint and dense AA bitmap combination

This independent crate starts from the frozen `owned_fused_endpoint` trainer.
It adds the validated `owned_aa_bitmap_cache` word-cache AA path while leaving
owner commit, pair selection, the posting container, and non-AA planning
unchanged. The same binary accepts `--endpoint-plan
two-pass|tagged-two-pass|tagged-fused` and `--aa-order
sort|bitmap-adaptive`. Defaults are `two-pass` and `sort`. There is no
atomic-per-edge bitmap mode or owner-accumulator reuse in this experiment.

The endpoint mode is dispatched once for the whole call through `const
TAGGED`/`FUSED`; the bitmap AA functions inherit `TAGGED` as a const generic.
The AA order choice happens once per AA epoch, not per posting or corpus read.
The 31-bit ID domain check remains at the entry. A requested tagged mode
falls back for the **whole call** to ordinary two-pass u32 endpoints when the
maximum possible new ID touches the head bit. `bitmap-adaptive` remains
available in that fallback because its position bits do not encode token IDs.

For a selected AA key, the dense guard first requires historical posting
length `H >= ceil(N/16)` and a checked logical bitmap byte count no greater
than the selected posting's allocated payload bytes. It then tries one
`AtomicU64` bitmap allocation and rechecks actual Vec capacity; failure uses
the original sort/parity route. This is an adoption guard, not a strict
instantaneous RSS limit: an allocator can overallocate before the actual
capacity check rejects the bitmap. No full corpus or index is copied per
worker. Each posting chunk keeps one pending bitmap word on its stack and
flushes it with relaxed atomic OR when the word changes or the chunk ends.
All scatter tasks join before popcount, summary, route, or apply reads.

Only starts validated by `inspect::<TAGGED>` against a stable corpus enter
the bitmap. The summary and incoming-parity pass therefore sees the exact
valid AA occurrence set, including long tokens whose physical tail still
contains an ID. Its popcount must equal the number of validated records;
duplicate live posting starts return an error rather than disappearing under
OR. Route jobs reread `inspect::<TAGGED>` before any write. Given a selected
start `p` and old token length `L`, a bit at `p-L` means the preceding AA
candidate was skipped and the prior selected match owns the shared left
boundary. A bit at `p+2L` means the next selected match produces the right
neighbor's new ID. Both checks use complete valid starts and global parity.
Piece sentinel, stale starts, checked offsets, and weighted deltas retain
the original path's semantics.

All AA route jobs join before any endpoint write. Bitmap apply invokes the
same `write_merge_at::<TAGGED>` as ordinary Plan apply: in tagged modes it
publishes head, right start, then tail with Release stores; the untagged
fallback uses relaxed stores. Selected AA spans do not overlap. All apply
jobs join before owner commit and before the next epoch, so next-epoch
non-AA fused reads observe a completed AA rewrite. The fused non-AA decoder
is never used to read an in-progress AA apply. Bitmap operations are relaxed
because the phase joins, not the bitmap bits, carry the corpus barrier.

Bitmap mode replaces dense AA Plan Vecs with one bitmap and small per-chunk
parity summaries. Its `O(N/64 + H)` work follows from the density guard and
the fact that this selected AA pair retires after the epoch. It can still be
slower when history is stale or atomic ORs contend. The JSON separates
`aa_bitmap_*` phase/work metrics, actual bitmap capacity bytes, and sort
Plan Vec capacity bytes from the process train-HWM. `plan_seconds` contains
bitmap init/scatter/summary/route and must not be summed with those nested
timers. `peak_plan_len` continues to describe materialized Plans; bitmap
mode records selected matches in `planned_positions` instead.

Tests exercise the full endpoint × AA order matrix, alternating AA/non-AA
epochs, weighted threshold cuts, AA length above 255, sparse sort fallback,
duplicate valid starts under tagged reads, and 31-bit domain fallback with
bitmap AA. The shared executor runs full-trace oracle and measured workloads
after source freeze; no speed or memory benefit is assumed from combining
the independently validated ideas.

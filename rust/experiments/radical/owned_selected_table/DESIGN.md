# Read-only selected-pair table during planning

`owned_selected_table` copies the frozen `owned_grouped_inline` algorithm and
changes only how a batch's selected old pair maps to its fresh ID during
non-AA planning. `--selected-lookup hash|flat` defaults to `hash`, which keeps
`HashMap<u64,u32>` as the same-binary control. Candidate selection, heads/tails
certificate, owner partition, grouped route HashMaps, posting storage, AA parity,
apply, and owner commit are unchanged. The selected-key HashSet used by owner
commit also remains separate and unchanged.

Each batch has at most 256 rules. Flat mode builds one immutable table shared
by all workers, with `2 * next_power_of_two(batch_width)` slots, at most 512.
Each 16-byte slot stores the full packed `u64` pair key and its `u32` fresh ID;
key zero marks empty because a selectable pair has two nonzero token IDs.
A fixed full-width mixing function chooses the initial power-of-two slot;
linear probing tests the full key and stops at the first empty slot. There are
no deletions and load is at most one half, so every lookup or insertion checks
at most the finite slot count and a missing lookup must encounter an empty
slot. Mixing quality affects speed only, never answer correctness. The table
is built before planning, then read immutably by all dynamic worker tasks.
A generic selected-read interface dispatches once per batch to monomorphized
hash or flat planning; it adds no per-query mode switch or atomic query counter.
Flat mode's 8 KiB maximum slots are per batch, not per worker, and there is no
copy of the full pair index. `selected_table_slots_peak` records this bounded
storage. The CLI also records process CPU time over the same interval as wall
`call_seconds`, to distinguish real parallel work from scheduling overhead.

This is a local work experiment: every valid merge currently checks the
selected table at one or both neighbors while also updating worker-private
owner routes. Replacing the small selected lookup may remove expensive general
hashing but does not remove route HashMap updates, historical stale visits, or
the serial greedy rule choice. Exactness follows from full-key equality and
from preserving the same returned fresh ID or absence for every query. The
same complete rule/final-token oracle and weighted, AA, adjacent-rule, tie,
collision, absent-key, and maximum-load tests are required before timing.

A wider event/dependency DAG across the first conflicting candidate is not
implemented here. The old candidate order alone is insufficient: in separate
pieces `ABABABAB` and `CDCD`, merging AB first creates fresh ZZ with frequency
3, ahead of old CD at 2; `ABCABC` makes old BC disappear after the tied AB is
chosen. `AAAAA` requires left-to-right AA parity rather than treating all four
overlapping starts as independent. A possible bounded one-step lookahead must
first compute every predecessor's old-key decreases and fresh-key births,
select the true next global maximum with the key tie-break, and plan through a
sparse endpoint overlay while keeping the shared corpus unchanged. Plans can
be published only for a validated consecutive prefix; an extra-work budget is
needed so repeatedly discarded plans cannot cause unbounded rescans. Generic
epoch-free commits would additionally need frequency upper/lower certificates
for all pending keys and version-safe reads of complete postings, so the
sequential greedy dependency remains a real lower bound.

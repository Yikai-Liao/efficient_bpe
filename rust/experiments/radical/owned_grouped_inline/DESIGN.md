# Grouped birth routing with inline posting storage

`owned_grouped_inline` combines the frozen `owned_grouped` update algorithm with
the reviewed 16-byte `SmallPosting` from `owned_inline`. No spatial probe or
logical-owner sharding is added. Initial physical indexing still appends in
the original route order. Existing pair lists retain their historical positions;
the grouped birth chain, counted frequency update, exact batch ordering, AA
parity, and selected-list release are unchanged.

For each eligible fresh key, the grouped delta's exact `occurrences` constructs
`SmallPosting::with_capacity(count)`. Counts 0–2 use the two inline slots without
allocation. Counts above 2 allocate at least `max(count, 4)` slots once and start
with `len=0`; owner fill appends exactly the counted positions and checks the
final length. The heap pointer remains valid even at length zero, and Drop
reconstructs its Vec using the actual checked capacity. If a future push exceeds
the reservation, the same checked growth path as `owned_inline` applies. On
64-bit targets the container and `Entry` occupy 16 and 24 bytes respectively;
allocator overhead and HashMap capacity still affect process RSS.

The final retained-index walk records inline key/position counts, heap key count,
and allocated heap slots in one pass under `final_owner_stats_seconds`, separate
from final token construction. `owned_posting_capacity` counts heap slots only;
inline slots live in each map entry. These are end-of-training metrics, not peak
memory estimates. The CLI keeps both `train_vm_hwm_mib`, sampled immediately
after training, and the later whole-process `vm_hwm_mib`.

The remainder describes the inherited grouped-routing proof and tradeoff.

`owned_grouped` changes only the merge-update route in `owned_counts`. It preserves the same exact batch certificate, owner state, initial physical index, dynamic position tasks, AA parity, frequency counts, heap policies, and early release of selected postings. The question is whether extra temporary route bytes can remove the owner fill's per-birth corpus decode and HashMap lookup.

A producer route already maps each affected pair key to `Delta { weight, occurrences }`. This version adds a `u32` chain head to that value. A new edge appends `BirthNode { pos, next: old_head }` to that producer→owner route's arena and updates the head. The head sentinel is `u32::MAX`; route indices are checked against that reserved value. Each `next` points to an earlier node, so a valid chain cannot cycle. On this compiler `Delta { weight: u64, occurrences: u32, head: u32 }` remains 16 bytes, its `(u64, Delta)` pair remains 24 bytes, and each temporary birth node is 8 bytes. Initial physical scanning still routes plain `u32` positions with the same algorithm as `owned_counts`. Its temporary container is now a standalone producer×owner `Vec<Vec<u32>>`, so the small Vec-header layout differs; no extra per-position bytes are introduced in initialization.

All planning reads one stable pre-apply corpus. It computes the final key of each birth, including fresh/fresh edges between adjacent selected matches, and chooses a destination owner. Old key deltas have no birth chain. A batch's old keys use only IDs below its first fresh ID, while every birth key contains a new ID; the two directions cannot share a key. All endpoints are written before owner fill. The owner first combines checked absolute weights and counts, applies old decreases, and creates each eligible new posting with exact counted capacity. It then iterates each producer's new-key delta entries. One owner-map lookup obtains the destination posting for that producer/key; following its chain appends all positions. A final check compares every eligible new posting's length with its global occurrence count. AA lists are sorted on selection, so the chain's reverse producer-local order does not change greedy overlap resolution. Non-AA planning does not require posting order.

Debug builds also decode every chained position from the post-apply corpus and assert that its final key matches the planned key. Release builds omit that per-position corpus read; complete rule/final-token oracle comparison remains the release correctness gate. Route traversal still performs an indexed node read and a checked count per position. The old `birth_decode_seconds` JSON field remains zero in this variant; `birth_group_fill_seconds` measures the owner fill phase.

This layout trades memory for fewer random reads and hash lookups. A temporary birth position costs 8 bytes instead of `owned_counts`' 4, while a persistent posting position remains 4 bytes. Selected postings are dropped after planning, before apply, as in the parent. The CLI reports node size, total birth nodes and producer/key groups, route node capacity, final retained posting length/capacity, and training VmHWM. It does not claim lower peak RSS or faster training without a same-budget measurement. Both heap policies and the shared oracle cases, including near-limit and nonuniform weights, remain in the crate tests.

The first version deliberately retains the apply→owner-reduce barrier to isolate the grouped route's cost. Because release owner fill no longer reads corpus, a later variant could run endpoint apply and owner reduction/fill concurrently with `rayon::join`; each owner could also reduce then fill without a global barrier. Both sides must finish before the next candidate selection or corpus read. That scheduling change requires its own measurement and is not part of this prototype.

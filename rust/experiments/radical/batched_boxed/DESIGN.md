# v4 exact batches with boxed postings

This crate isolates the combination of v4's certified batch trainer and the
boxed storage experiment. Candidate order, type-safe batch prefix, AA run
parity, dynamic `FlatTask` planning, central delta reduction, and sorted
`(key, position)` birth construction follow v4. There is no owner routing or
delayed AA sort in this variant.

Each eligible pair has one immutable, exact-length `Box<[u32]>` and one
frequency in the hash map. Initialization uses a serial weighted count pass,
filters keys below `min_frequency`, allocates each surviving posting once,
then fills it in corpus order on a second pass. Born records are still sorted
as in v4; after reduction and threshold filtering, each eligible born key
receives one exact-length Box in sorted position order. Thus AA postings stay
ordered for the existing cross-chunk parity algorithm. Retiring a below-limit
old key immediately drops its whole posting. This retains historical stale
positions in keys that remain eligible.

When a batch is selected, its Box values move out of the entry map into a
rank-aligned `selected_postings` vector without copying their payloads.
`BatchRule` contains only Copy metadata, and each `FlatTask` indexes an offset
range *relative to its rule's Box*. All task reads complete against the stable
corpus before any endpoint write. The joined plans own their positions and
neighbor IDs, so selected Boxes are dropped after planning and before apply.
For AA, sorting is unnecessary because initialization and births remain
ordered; the same valid-position filtering and parity prefix run before the
drop. Rules, frequencies, final tokens, and tie order should match v4 exactly.

Posting metrics update at allocation/removal sites rather than traversing
the full map each round. `allocated_posting_records` includes temporarily
selected Boxes; `retained_entry_posting_len` counts map-held payloads.
`posting_payload_bytes_*` measures requested `u32` payload, while process
`train_vm_hwm_mib` measures the allocator, hash maps, task records, and
temporary arrays too. Initial count/fill/final maps briefly overlap. During
planning, selected Boxes coexist with `TaskPrepared` valid starts and births;
during birth construction the sorted birth vector coexists with new Boxes.
The v4 arena fields remain at zero for JSON comparison. Hash-map and heap
capacity are reported; no periodic shrink was added.

The comparison is meaningful only with identical fixtures, merge limit,
minimum frequency, worker/chunk settings, and complete rule/final-token
oracle. v4 and this crate share algorithmic stages, so differences in time
and RSS primarily measure boxed allocation, deallocation, and capacity
behavior rather than a new certificate or parallel reduction protocol.

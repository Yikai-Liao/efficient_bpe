# Boxed posting experiment

This independent crate changes the posting-list storage of radical v2. Rule
selection, AA run parity, Rayon chunk planning and application, frequency
updates, and `(key, position)` birth sorting follow v2. The experiment asks
whether promptly freeing a retired key's historical positions reduces memory
enough to offset many small allocations.

Each eligible key owns one `Entry { frequency: u64, posting: Box<[u32]> }`.
`Entry` is 24 bytes on the target 64-bit platform, the same inline size as
v2's `(frequency, arena offset, length)` entry. The posting payload has exactly
the requested number of `u32` slots; allocator metadata, size-class rounding,
fragmentation, and the hash map's capacity are additional memory. Neither
dropping a Box nor removing a hash-map entry guarantees a lower process RSS.

Initialization counts all pair occurrences and weighted frequencies, then
removes keys below `min_frequency`. It allocates one exact-length uninitialized
Box per remaining key and fills each in corpus order during the second scan.
The cursor check proves every slot was written before converting it to a
`Box<[u32]>`. Temporary count, fill, and final hash maps overlap during the
handoff, so the initial RSS peak can exceed the steady state. Their entry
counts and capacities are reported separately.

During an epoch the selected entry is removed from the map but remains owned
through parallel planning. All plans own coordinates and neighbor IDs; after
the planning barrier the selected Box is dropped before application. An old
key whose frequency falls below the threshold is removed and dropped during
reduction. A newly born eligible key receives one exact-length Box after the
existing v2 sort, preserving posting order. Keys below the threshold receive
no posting. A pair with a fresh ID can only be born in that ID's epoch, so no
posting must later grow or be rewritten. These lifetimes do not change the
stable-corpus planning barrier or the left-to-right AA selection rule.

All retained-record and allocation metrics update at insert/remove/allocate/
free sites; the trainer does not scan all entries at every epoch. In the JSON,
`allocated_posting_records` includes a temporarily owned selected posting,
whereas `retained_entry_posting_len` counts entries still in the map.
`peak_allocated_posting_records` includes that temporary interval and
`peak_retained_posting_records` excludes it. `posting_payload_bytes_*` is
exactly four times the logical record count, not an estimate of physical heap
usage. `posting_allocations_*` counts nonempty posting payloads and
`posting_frees_total` counts their drops. `entry_count_*` includes brief empty
birth placeholders between frequency reduction and posting construction;
those placeholders do not allocate a payload. The v2 arena fields remain in
the JSON at zero for schema comparison. Map and heap final/peak capacities
make persistent capacity visible; no periodic map shrinking was added.

This layout releases whole dead-key lists, but stale positions within a
still-eligible key remain until that key is selected or retired. In the worst
case, retained historical records are still O(total positions ever stored),
not O(current live token count). The benchmark must compare process VmHWM and
wall time against v2 on the same fixtures; logical payload counts alone do not
establish an RSS improvement.

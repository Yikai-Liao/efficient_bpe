# Python rewrite queue comparison

`queues.py` provides two queues over the same mutable pair-frequency mapping.
Both choose the highest current frequency, then the lexicographically smallest
pair. The driver may decrease an existing pair's frequency without notifying
the queue. It calls `add(pair)` **exactly once** for each new pair containing a
fresh token ID, after that pair's initial frequency is complete. A selected
pair is never added again. Re-adding an old pair is outside this interface.

`HeapQueue` uses standard `(-frequency, pair)` heap entries. When a stale entry
reaches the top, it is reinserted with the lower current frequency; entries
below `min_frequency` are discarded. `stats()` reports initial heapify entries,
subsequent pushes and pops.

`HighLowQueue` follows the queue split in YouTokenToMe's
[`bpe.cpp`](https://github.com/VKCOM/YouTokenToMe/blob/f4162d846057a3118222ca04a01b84297eb8a8db/youtokentome/cpp/bpe.cpp#L149-L314):
`B = floor(sqrt(initial_mass))`; frequencies below `B` go into a fixed array
of `B` buckets, and frequencies at least `B` go into a scanned high array.
Each `pop()` refreshes every high candidate, moving any that fell below `B`
into a low bucket. The low side tracks its highest potentially nonempty bucket
with one integer. A bucket is sorted in reverse lexicographic order only when
an insertion marks it dirty; `pop()` then takes the last item. A stale low
candidate is moved to its lower bucket when encountered. `stats()` reports
high scan visits, low candidate pops, empty-level pointer steps, and sorts.

This is a Python queue comparison under one trainer's semantics, not a full
port of YTTM's run encoding, merge tie handling, or parallel worker pipeline.
The deterministic tie rule is intentional for comparison with the shared
trainer; YTTM's default high/low queue does not enforce it unless compiled
with `DETERMINISTIC_QUEUE`.

The high scan costs `O(H)` per `pop()`, where `H` is the current high-array
length. Under the BPE mass invariant, high candidates are roughly bounded by
`initial_mass/B`, but Python tuple and dictionary costs matter. The low side
allocates `B` list headers and a `B`-byte dirty array even if most buckets are
empty; sorting costs are paid when a dirty bucket reaches the top. Heap push
and pop each cost `O(log K)` for current heap size `K`, with extra operations
for stale corrections. Neither strategy has a universal performance win.

Run bounded oracle tests with:

```bash
cd /root/code/efficient_bpe/benchmarks/bpe_core_comparison/python_rewrite
python3 -m unittest -v test_queues.py
```

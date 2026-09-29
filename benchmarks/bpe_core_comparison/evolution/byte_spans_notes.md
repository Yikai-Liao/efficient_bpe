# ByteSpans topology component

`ByteSpans` stores span boundaries in one `bytearray` of length `N`, so its
reported backing-buffer use is exactly `N` bytes. It contains no token IDs,
frequency information, occurrence lists, Python per-position objects, or
auxiliary length table. It is an isolated topology experiment; it does not
establish that a complete BPE trainer uses one byte per corpus position.

At a live start, tag 1 means a singleton, tags 2..63 encode short lengths
directly, and tag 126 announces a long span. At a live end, tag 1 means a
singleton, tags 64..125 encode lengths 2..63 as `tag - 62`, and tag 127
announces a long span. Long lengths are stored as five fixed base-128 digits
inside the span at both ends. Each digit has its high bit set (128..255), so
it cannot be mistaken for any start marker. Lengths are bounded by u32, and
the position space requires `N < 2^32`.

`length(pos)` is O(1); `next(pos)` adds that length. `prev(pos)` decodes the
length at the preceding end and cross-checks the corresponding start. Merge
accepts already-known adjacent boundaries `(pos, right, after)` and does not
scan the span or loop in proportion to its length. It clears the consumed
right start first, then writes the new start/end markers and, for a long span,
exactly five payload bytes at each end. The old left end can remain as stale
interior data. A long span is at least 64 positions, so its two five-byte
payload regions do not overlap each other or its endpoint markers. Payload
may overwrite stale interior bytes or the just-cleared right start; because
payload tags are >=128, `length()` still returns zero there.

The test suite uses an independent naive oracle: it keeps the live spans as an
ordered Python list, recomputes every current start by cumulative sums, and
checks all `length`, `prev`, and `next` results after random adjacent merges.
It also checks exact encoding boundaries 63/64, 255/256, 65535/65536, long
left/right merge chains, consumed-start overwrite by payload, invalid merge
boundaries, and exact backing-buffer size. The tests are topology microchecks;
they do not cover token-ID lookup or full BPE training costs.

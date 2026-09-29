# Direction-tagged hybrid layouts

These are experimental sequence backends for `python_rewrite/common_fused.py`.
They keep every original position fixed. A `u16` text array holds initial IDs;
each merged ID occupies the first two `u16` cells of its token, high half then
low half. A separate directional tag identifies live starts and token ends.
The input includes live ID-0 separators and an ID-0 EOF position. The driver
never merges either separator and appends each fresh token's length to the
shared `token_len` list before `merge_known`.

| Layout | Tag encoding | Backend buffers for N positions |
| --- | --- | ---: |
| H3 / fused H3 | One byte per position; lengths 2–127 inline | `3N` bytes |
| H2.5 | Two four-bit tags per byte; lengths 2–6 inline | `2N + ceil(N/2)` bytes |

These figures include text and tags. They exclude the driver's prepared input,
shared `token_len`, occurrence arrays, heap, and Python object headers. H3's
long lengths begin at 128; H2.5's begin at 7. A long start reads its ID's
length from `token_len`; the long end stores a 32-bit length in the token's
last two `u16` cells. Thus `next`, `prev`, occurrence inspection, and merge
use a fixed number of reads and writes. They do not scan long tokens or
maintain a per-long-token overflow table. The start ID and end length cannot
overlap at either threshold.

## Why the direction matters

A merged token's old right start becomes an interior position or its new end.
`merge_known` clears that right-start tag, writes the new start and end tags,
and can leave older end tags in the interior. Only a singleton or a start tag
may validate a cached occurrence. For every live start, `next` reads the start
length; for every live start after the first, `prev` reads the preceding live
token's end length. This preserves O(1) adjacency even when old end tags
remain inside a newer token.

An undirected length tag is unsafe as a cached-occurrence validity test. For
example, begin with `[0, 1, 1, 1, 1, 0]` and merge positions 1 and 2 into ID
65536 (`0x00010000`). The packed ID leaves `text[2] == 0`; position 2 is now
the merged token's end, and `text[3] == 1`. If an undirected length-2 tag at
position 2 were treated as a start, decoding two `u16` cells there would yield
ID 1. Its apparent next position would be 4, also ID 1. The stale pair
occurrence `(2, 1, 1)` would falsely pass. H3 marks position 2 as an **end**
tag, so `inspect_pair` rejects it before decoding an ID. The low half being
zero makes this collision particularly easy; it is valid ID data, not a blank
marker.

## Scope and fallback

Initial IDs must be below 65536 because a singleton has only one `u16` cell.
Fresh merged IDs and lengths may use all 32 bits, and original positions must
fit a 32-bit offset. If the initial alphabet has fewer than 65536 distinct
symbols, dense remapping can keep the `u16` path, preserving ID 0 for
separators. Otherwise use a u32-text backend such as the fused endpoint or
bitmap implementation. The endpoint layout has a 4N-byte backend buffer; the
u32 bitmap layout adds approximately 0.1875N bytes of boundary metadata.

`FusedHybridByteTags` keeps the same H3 representation and write path. Its
`inspect_pair` computes both neighbors and token IDs in one function without
calling `prev`, `next`, or `token`; short lengths come from tags and long-start
lengths from the shared list. H2.5 remains an exploratory space variant: its
nibble extraction and read-modify-write updates add Python work, so the
smaller buffer alone does not establish a speed advantage.

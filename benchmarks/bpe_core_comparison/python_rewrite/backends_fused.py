"""Fused occurrence inspection and merge updates for the Python rewrite.

The baseline API asks for the same successor several times per replacement.
These subclasses leave that API intact for compatibility but add a second,
trusted interface: inspect_pair returns the matching occurrence and its
context once; merge_known reuses the returned boundaries.  The caller still
appends token_len[new_id] before merging.  No baseline source is modified.
"""

from array import array

from backends_compact import (
    FastEbpeEndpoints, FastPrezzaBitmap, FastPrezzaHalfword,
)


class FusedEbpeEndpoints(FastEbpeEndpoints):
    def inspect_pair(self, pos, a, b):
        corpus, token_len = self.corpus, self.token_len
        if a == 0 or b == 0 or pos < 0 or pos >= self.last or corpus[pos] != a:
            return None
        right = pos + token_len[a]
        if right >= self.last or corpus[right] != b:
            return None
        before = pos - token_len[corpus[pos - 1]] if pos else None
        after = right + token_len[b]
        left_id = corpus[before] if before is not None else 0
        right_id = corpus[after] if after is not None else 0
        return before, left_id, right, after, right_id

    def merge_known(self, pos, right, after, new_id, new_len):
        corpus = self.corpus
        corpus[right - 1] = 0
        corpus[right] = 0
        corpus[pos] = new_id
        corpus[after - 1] = new_id
        return right


class FusedPrezzaBitmap(FastPrezzaBitmap):
    def inspect_pair(self, pos, a, b):
        if a == 0 or b == 0 or not self.alive(pos) or self.corpus[pos] != a:
            return None
        right = self.next(pos)
        if right is None or right >= self.last or self.corpus[right] != b:
            return None
        before = self.prev(pos)
        after = self.next(right)
        left_id = self.corpus[before] if before is not None else 0
        right_id = self.corpus[after] if after is not None else 0
        return before, left_id, right, after, right_id

    def merge_known(self, pos, right, after, new_id, new_len):
        bits, skips = self.bits, self.skips
        right_block = right // 64
        bits[right_block] &= ~(1 << (right % 64))
        first_block, after_block = pos // 64, after // 64
        if after_block > first_block + 1:
            gap = after - pos - 1
            skips[first_block + 1] = gap
            skips[after_block - 1] = gap
        self.corpus[pos] = new_id
        return right


class FusedPrezzaHalfword(FastPrezzaHalfword):
    """Halfword variant constructed directly from input into u16 storage."""

    def __init__(self, initial_ids, token_len=None):
        n = len(initial_ids)
        if n < 2 or initial_ids[0] != 0 or initial_ids[-1] != 0:
            raise ValueError("initial IDs need a leading separator and trailing EOF 0")
        if n >= 1 << 32:
            raise ValueError("u32 offsets require fewer than 2^32 positions")
        if any(token_id >= 1 << 16 or token_id < 0 for token_id in initial_ids):
            raise ValueError("halfword corpus requires initial IDs in [0, 65535]")
        self.corpus = array("H", initial_ids)
        if self.corpus.itemsize != 2:
            raise RuntimeError("array('H') is not 16 bits on this platform")
        if token_len is None:
            token_len = [1] * (max(self.corpus) + 1)
        elif not isinstance(token_len, list):
            raise TypeError("token_len must be a shared list")
        if len(token_len) <= max(self.corpus) or token_len[0] != 1:
            raise ValueError("token_len must cover all initial IDs, with token_len[0]=1")
        if any(token_len[token_id] != 1 for token_id in self.corpus):
            raise ValueError("all initial positions must have length 1")
        self.token_len = token_len
        self.last = n - 1
        words = (n + 63) // 64
        self.bits = array("Q", [(1 << 64) - 1]) * words
        if self.bits.itemsize != 8:
            raise RuntimeError("array('Q') is not 64 bits on this platform")
        self.bits[-1] = (1 << (self.last % 64 + 1)) - 1
        self.skips = array("I", [0]) * words

    def inspect_pair(self, pos, a, b):
        if a == 0 or b == 0 or not self.alive(pos) or self.token(pos) != a:
            return None
        right = self.next(pos)
        if right is None or right >= self.last or self.token(right) != b:
            return None
        before = self.prev(pos)
        after = self.next(right)
        left_id = self.token(before) if before is not None else 0
        right_id = self.token(after) if after is not None else 0
        return before, left_id, right, after, right_id

    def merge_known(self, pos, right, after, new_id, new_len):
        bits, skips = self.bits, self.skips
        right_block = right // 64
        bits[right_block] &= ~(1 << (right % 64))
        first_block, after_block = pos // 64, after // 64
        if after_block > first_block + 1:
            gap = after - pos - 1
            skips[first_block + 1] = gap
            skips[after_block - 1] = gap
        self.corpus[pos] = new_id >> 16
        self.corpus[pos + 1] = new_id & 0xFFFF
        return right


__all__ = ["FusedEbpeEndpoints", "FusedPrezzaBitmap", "FusedPrezzaHalfword"]

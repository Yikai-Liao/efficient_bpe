"""Array-backed boundary layouts for an identical BPE merge trace.

Both layouts receive an already flattened sequence of one-position token IDs.
ID 0 is a permanent separator, including the first and last (EOF) positions.
The caller only asks token/next/prev about known live token starts.  Occurrence
validation must use pair_matches: the endpoint layout deliberately has no
general-purpose live-position test because zero also denotes a separator.

This is an independent Python rewrite of a blocked bitmap/skip idea.  The
PrezzaBitmap variant uses a separate u32 ID corpus to isolate boundary costs;
FastPrezzaHalfword also tests the original halfword text-storage idea.
"""

from array import array


def _new_length(lengths, token_id, length):
    if not 0 < token_id < 1 << 32:
        raise ValueError("merged IDs must be nonzero u32 values")
    if token_id == len(lengths):
        lengths.append(length)
    elif token_id < len(lengths):
        old = lengths[token_id]
        if old not in (0, length):
            raise ValueError("new ID already has a different token length")
        lengths[token_id] = length
    else:
        raise ValueError("new IDs must be dense or already allocated")


def _setup(initial_ids, token_len):
    corpus = array("I", initial_ids)
    if corpus.itemsize != 4:
        raise RuntimeError("array('I') is not 32 bits on this platform")
    if len(corpus) < 2 or corpus[0] != 0 or corpus[-1] != 0:
        raise ValueError("initial IDs need a leading separator and trailing EOF 0")
    if len(corpus) >= 1 << 32:
        raise ValueError("u32 offsets require fewer than 2^32 positions")
    if token_len is None:
        token_len = [1] * (max(corpus) + 1)
    elif not isinstance(token_len, list):
        raise TypeError("token_len must be a shared list")
    if len(token_len) <= max(corpus) or token_len[0] != 1:
        raise ValueError("token_len must cover all initial IDs, with token_len[0]=1")
    if any(token_len[token_id] != 1 for token_id in corpus):
        raise ValueError("all initial positions must have length 1")
    return corpus, token_len


class EbpeEndpoints:
    """v2-style u32 IDs at both endpoints; zero at absorbed endpoints.

    Each merge writes the old left end and right start to zero, then the new
    start and end to new_id.  Interior positions were cleared by earlier
    merges, so a full-span zeroing pass is unnecessary.  ID 0 separators stay
    live by the caller's merge rule, although they are indistinguishable from
    dead positions to a general alive(pos) query.
    """

    def __init__(self, initial_ids, token_len=None):
        self.corpus, self.token_len = _setup(initial_ids, token_len)
        self.last = len(self.corpus) - 1

    def token(self, pos):
        return self.corpus[pos]

    def next(self, pos):
        if pos == self.last:
            return None
        return pos + self.token_len[self.corpus[pos]]

    def prev(self, pos):
        if pos == 0:
            return None
        return pos - self.token_len[self.corpus[pos - 1]]

    def pair_matches(self, pos, a, b):
        if a == 0 or b == 0 or not 0 <= pos < self.last:
            return False
        if self.corpus[pos] != a:
            return False
        right = pos + self.token_len[a]
        return right < self.last and self.corpus[right] == b

    def merge(self, pos, new_id, new_len):
        right = self.next(pos)
        if right is None or right == self.last or self.corpus[pos] == 0 or self.corpus[right] == 0:
            raise ValueError("merge requires two nonseparator live tokens")
        after = self.next(right)
        if after is None or after - pos != new_len:
            raise ValueError("new_len does not match adjacent token spans")
        _new_length(self.token_len, new_id, new_len)
        self.corpus[right - 1] = 0
        self.corpus[right] = 0
        self.corpus[pos] = new_id
        self.corpus[after - 1] = new_id
        return right

    def memory_details(self):
        """Allocated array buffers only; shared token_len and objects excluded."""
        return {"corpus_u32": len(self.corpus) * self.corpus.itemsize}

    def memory_bytes(self):
        return sum(self.memory_details().values())


class PrezzaBitmap:
    """u64 live bitmap plus one u32 skip per 64 original positions.

    The ID corpus is deliberately separate and stays untouched at dead
    positions.  A complete empty block between two adjacent live boundaries
    holds that gap's distance at its first and last block positions.
    """

    def __init__(self, initial_ids, token_len=None):
        self.corpus, self.token_len = _setup(initial_ids, token_len)
        self.last = len(self.corpus) - 1
        words = (len(self.corpus) + 63) // 64
        self.bits = array("Q", [(1 << 64) - 1]) * words
        if self.bits.itemsize != 8:
            raise RuntimeError("array('Q') is not 64 bits on this platform")
        self.bits[-1] = (1 << (self.last % 64 + 1)) - 1
        self.skips = array("I", [0]) * words

    def alive(self, pos):
        return 0 <= pos <= self.last and bool(self.bits[pos // 64] & (1 << (pos % 64)))

    def token(self, pos):
        return self.corpus[pos]

    def next(self, pos):
        if not self.alive(pos):
            raise ValueError("next requires a live position")
        if pos == self.last:
            return None
        block, offset = divmod(pos, 64)
        word = self.bits[block] >> (offset + 1)
        if word:
            return pos + (word & -word).bit_length()
        word = self.bits[block + 1]
        if word:
            return (block + 1) * 64 + (word & -word).bit_length() - 1
        return pos + self.skips[block + 1] + 1

    def prev(self, pos):
        if not self.alive(pos):
            raise ValueError("prev requires a live position")
        if pos == 0:
            return None
        block, offset = divmod(pos, 64)
        word = self.bits[block] & ((1 << offset) - 1)
        if word:
            return block * 64 + word.bit_length() - 1
        word = self.bits[block - 1]
        if word:
            return (block - 1) * 64 + word.bit_length() - 1
        return pos - self.skips[block - 1] - 1

    def pair_matches(self, pos, a, b):
        if a == 0 or b == 0 or not self.alive(pos) or self.corpus[pos] != a:
            return False
        right = self.next(pos)
        return right is not None and right < self.last and self.corpus[right] == b

    def merge(self, pos, new_id, new_len):
        right = self.next(pos)
        if right is None or right == self.last or self.corpus[pos] == 0 or self.corpus[right] == 0:
            raise ValueError("merge requires two nonseparator live tokens")
        after = self.next(right)
        if after is None or after - pos != new_len:
            raise ValueError("new_len does not match adjacent token spans")
        _new_length(self.token_len, new_id, new_len)
        right_block = right // 64
        self.bits[right_block] &= ~(1 << (right % 64))
        first_block, after_block = pos // 64, after // 64
        if after_block > first_block + 1:
            gap = after - pos - 1
            self.skips[first_block + 1] = gap
            self.skips[after_block - 1] = gap
        self.corpus[pos] = new_id
        return right

    def memory_details(self):
        """Allocated array buffers only; shared token_len and objects excluded."""
        return {
            "corpus_u32": len(self.corpus) * self.corpus.itemsize,
            "live_u64": len(self.bits) * self.bits.itemsize,
            "skips_u32": len(self.skips) * self.skips.itemsize,
        }

    def memory_bytes(self):
        return sum(self.memory_details().values())


class FastEbpeEndpoints(EbpeEndpoints):
    """Endpoint hot path; caller prevalidates pair and appends new token_len."""

    def merge(self, pos, new_id, new_len):
        right = self.next(pos)
        after = self.next(right)
        self.corpus[right - 1] = 0
        self.corpus[right] = 0
        self.corpus[pos] = new_id
        self.corpus[after - 1] = new_id
        return right


class FastPrezzaBitmap(PrezzaBitmap):
    """Bitmap hot path; caller supplies known-live positions and fresh IDs."""

    def next(self, pos):
        if pos == self.last:
            return None
        block, offset = divmod(pos, 64)
        word = self.bits[block] >> (offset + 1)
        if word:
            return pos + (word & -word).bit_length()
        word = self.bits[block + 1]
        if word:
            return (block + 1) * 64 + (word & -word).bit_length() - 1
        return pos + self.skips[block + 1] + 1

    def prev(self, pos):
        if pos == 0:
            return None
        block, offset = divmod(pos, 64)
        word = self.bits[block] & ((1 << offset) - 1)
        if word:
            return block * 64 + word.bit_length() - 1
        word = self.bits[block - 1]
        if word:
            return (block - 1) * 64 + word.bit_length() - 1
        return pos - self.skips[block - 1] - 1

    def merge(self, pos, new_id, new_len):
        right = self.next(pos)
        after = self.next(right)
        right_block = right // 64
        self.bits[right_block] &= ~(1 << (right % 64))
        first_block, after_block = pos // 64, after // 64
        if after_block > first_block + 1:
            gap = after - pos - 1
            self.skips[first_block + 1] = gap
            self.skips[after_block - 1] = gap
        self.corpus[pos] = new_id
        return right


class FastPrezzaHalfword(FastPrezzaBitmap):
    """Bitmap/skips with two u16 cells for a merged 32-bit token ID.

    Initial IDs must fit one u16.  At a live start, a live following position
    means its ID is the single u16 at that start.  Otherwise the next position
    was absorbed by that token, so the start and following dead cell contain
    its high and low ID halves.  This is the text packing omitted by the u32
    bitmap variant.  Caller prevalidates pairs and preappends token lengths.
    """

    def __init__(self, initial_ids, token_len=None):
        super().__init__(initial_ids, token_len)
        if max(self.corpus) >= 1 << 16:
            raise ValueError("halfword corpus requires every initial ID < 65536")
        self.corpus = array("H", self.corpus)
        if self.corpus.itemsize != 2:
            raise RuntimeError("array('H') is not 16 bits on this platform")

    def token(self, pos):
        first = self.corpus[pos]
        if pos == self.last or self.bits[(pos + 1) // 64] & (1 << ((pos + 1) % 64)):
            return first
        return (first << 16) | self.corpus[pos + 1]

    def pair_matches(self, pos, a, b):
        if a == 0 or b == 0 or not self.alive(pos) or self.token(pos) != a:
            return False
        right = self.next(pos)
        return right is not None and right < self.last and self.token(right) == b

    def merge(self, pos, new_id, new_len):
        right = self.next(pos)
        after = self.next(right)
        right_block = right // 64
        self.bits[right_block] &= ~(1 << (right % 64))
        first_block, after_block = pos // 64, after // 64
        if after_block > first_block + 1:
            gap = after - pos - 1
            self.skips[first_block + 1] = gap
            self.skips[after_block - 1] = gap
        self.corpus[pos] = new_id >> 16
        self.corpus[pos + 1] = new_id & 0xFFFF
        return right

    def memory_details(self):
        return {
            "corpus_u16": len(self.corpus) * self.corpus.itemsize,
            "live_u64": len(self.bits) * self.bits.itemsize,
            "skips_u32": len(self.skips) * self.skips.itemsize,
        }


__all__ = [
    "EbpeEndpoints", "PrezzaBitmap", "FastEbpeEndpoints",
    "FastPrezzaBitmap", "FastPrezzaHalfword",
]

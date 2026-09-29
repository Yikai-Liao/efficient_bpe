"""Array-backed linked corpus for isolating YTTM-style adjacency costs.

This is deliberately *not* YTTM's run-length representation. Every input
position gets one node, including the live zero-valued separators/sentinels.
The optional run-length array remains 1 for live tokens and exists only to
measure the four-array (16 bytes/position) layout against a three-array
(12 bytes/position) layout under the same merge semantics.
"""

from array import array


_DEAD = -2
_END = -1
_MAX_INDEX = (1 << 31) - 1


class LinkedBackend:
    """Mutable token chain stored in 32-bit arrays at stable input positions.

    ``initial_ids`` must contain zero at both ends and between segments.
    ``token_len`` is the driver's shared list; the driver appends each fresh
    token's length before calling ``merge``. Positions never move, and a dead
    right node is marked by ``next[right] == -2``. A live separator has value
    zero but a nonnegative next index (or -1 for the final sentinel).
    """

    def __init__(self, initial_ids, token_len, *, store_runlength=True):
        self.val = array("I", initial_ids)
        n = len(self.val)
        if not n or self.val[0] != 0 or self.val[-1] != 0:
            raise ValueError("initial_ids must begin and end with a 0 sentinel")
        if n > _MAX_INDEX:
            raise ValueError("array('i') links require fewer than 2^31 positions")
        if self.val.itemsize != 4 or array("i").itemsize != 4:
            raise RuntimeError("this backend requires 32-bit I and i arrays")

        self._prev = array("i", range(_END, n - 1))
        self._next = array("i", range(1, n))
        self._next.append(_END)
        self.runlength = array("I", (1 if value else 0 for value in self.val)) if store_runlength else None
        self.token_len = token_len

    def alive(self, pos):
        return 0 <= pos < len(self.val) and self._next[pos] != _DEAD

    def token(self, pos):
        return self.val[pos]

    def next(self, pos):
        if not self.alive(pos):
            return None
        right = self._next[pos]
        return None if right == _END else right

    def prev(self, pos):
        if not self.alive(pos):
            return None
        left = self._prev[pos]
        return None if left == _END else left

    def pair_matches(self, pos, a, b):
        if not a or not b or not self.alive(pos) or self.val[pos] != a:
            return False
        right = self._next[pos]
        return right >= 0 and self.val[right] == b

    def merge(self, pos, new_id, new_len):
        """Replace a pair prechecked by ``pair_matches``; return old right pos.

        The driver owns ``token_len`` and must append ``new_len`` for the fresh
        ``new_id`` before this call. This backend only changes adjacency, so it
        does not need token lengths in the hot merge path.
        """
        right = self._next[pos]
        following = self._next[right]
        self.val[pos] = new_id
        self._next[pos] = following
        if following >= 0:
            self._prev[following] = pos
        self.val[right] = 0
        self._prev[right] = _DEAD
        self._next[right] = _DEAD
        if self.runlength is not None:
            self.runlength[right] = 0
        return right

    def memory_bytes(self):
        """Allocated array buffers in bytes, excluding Python array headers."""
        arrays = (self.val, self._prev, self._next)
        if self.runlength is not None:
            arrays += (self.runlength,)
        return sum(items.__sizeof__() - array(items.typecode).__sizeof__() for items in arrays)

    def logical_bytes(self):
        """Used array elements only: 16 or 12 bytes per input position."""
        arrays = (self.val, self._prev, self._next)
        if self.runlength is not None:
            arrays += (self.runlength,)
        return sum(len(items) * items.itemsize for items in arrays)

    def memory_headers_bytes(self):
        """Python array object headers, separate from ``memory_bytes``."""
        arrays = (self.val, self._prev, self._next)
        if self.runlength is not None:
            arrays += (self.runlength,)
        return sum(array(items.typecode).__sizeof__() for items in arrays)


class CompactLinkedBackend(LinkedBackend):
    """The same adjacency semantics with no unused run-length array."""

    def __init__(self, initial_ids, token_len):
        super().__init__(initial_ids, token_len, store_runlength=False)


class FastLinkedBackend(LinkedBackend):
    """Trusted-call variant: neighbors are queried only at live positions.

    As with the checked layout, ``merge`` requires a successful prior
    ``pair_matches`` call. Dead nodes are never queried for neighbors, so their
    previous links need not be cleared.
    """

    def next(self, pos):
        right = self._next[pos]
        return None if right < 0 else right

    def prev(self, pos):
        left = self._prev[pos]
        return None if left < 0 else left

    def merge(self, pos, new_id, new_len):
        right = self._next[pos]
        following = self._next[right]
        self.val[pos] = new_id
        self._next[pos] = following
        if following >= 0:
            self._prev[following] = pos
        self.val[right] = 0
        self._next[right] = _DEAD
        if self.runlength is not None:
            self.runlength[right] = 0
        return right


class FastCompactLinkedBackend(FastLinkedBackend):
    """Trusted-call variant of the 12-byte, no-runlength layout."""

    def __init__(self, initial_ids, token_len):
        super().__init__(initial_ids, token_len, store_runlength=False)


# The unified driver can use either alias without adapter code.
Backend = LinkedBackend
CompactBackend = CompactLinkedBackend

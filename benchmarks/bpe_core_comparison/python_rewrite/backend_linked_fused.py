"""Fused adjacency access for the 12-byte linked-array comparison.

This backend has the same no-RLE corpus representation as
``FastCompactLinkedBackend``. The driver may call ``inspect_pair`` on stale
occurrence positions, but calls ``merge_known`` only after a successful
inspection and before changing any of the returned neighbors.
"""

from backend_linked import FastCompactLinkedBackend


class FusedLinked12(FastCompactLinkedBackend):
    """Return pair validity and both neighbors with one direct array read path."""

    def inspect_pair(self, pos, a, b):
        links = self._next
        right = links[pos]
        if not a or not b or right < 0 or self.val[pos] != a or self.val[right] != b:
            return None
        before = self._prev[pos]
        after = links[right]
        return (
            None if before < 0 else before,
            0 if before < 0 else self.val[before],
            right,
            None if after < 0 else after,
            0 if after < 0 else self.val[after],
        )

    def merge_known(self, pos, right, after, new_id, new_len):
        """Merge an inspected pair; ``after`` is its inspected successor."""
        self.val[pos] = new_id
        self._next[pos] = -1 if after is None else after
        if after is not None:
            self._prev[after] = pos
        self.val[right] = 0
        self._next[right] = -2
        return right


__all__ = ["FusedLinked12"]

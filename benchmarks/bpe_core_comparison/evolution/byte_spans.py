"""One-byte-per-position span boundaries for adjacent token merges.

This is a topology component only. It stores no token IDs and does not define
how callers distinguish token identities or decide which pair to merge.
"""


class ByteSpans:
    """Track live token starts and ends in exactly ``n`` bytes.

    Tags are 0 for dead/interior, 1 for a one-byte span, 2..63 for short
    starts, 64..125 for short ends, 126 for a long start, and 127 for a long
    end. Five base-128 payload bytes at each end encode lengths >=64. Payload
    bytes always have their high bit set and are never interpreted as starts.
    """

    __slots__ = ("_data",)

    _U32_MAX = (1 << 32) - 1
    _PAYLOAD_BYTES = 5

    def __init__(self, n):
        if not isinstance(n, int) or n < 0 or n >= 1 << 32:
            raise ValueError("n must be an integer in [0, 2^32)")
        self._data = bytearray(b"\x01") * n

    def __len__(self):
        return len(self._data)

    def _start_length(self, pos):
        tag = self._data[pos]
        if tag == 1:
            return 1
        if 2 <= tag <= 63:
            return tag
        if tag == 126:
            return self._decode(pos + 1, 1)
        return 0

    def _end_length(self, end):
        tag = self._data[end]
        if tag == 1:
            return 1
        if 64 <= tag <= 125:
            return tag - 62
        if tag == 127:
            return self._decode(end - 1, -1)
        return 0

    def _decode(self, first, step):
        value = 0
        for index in range(self._PAYLOAD_BYTES):
            value |= (self._data[first + step * index] & 0x7F) << (7 * index)
        return value

    def length(self, pos):
        """Return the span length if ``pos`` is a live start, otherwise 0."""
        if not isinstance(pos, int) or pos < 0 or pos >= len(self._data):
            return 0
        return self._start_length(pos)

    def next(self, pos):
        """Return the next start, or ``n`` for the end boundary."""
        span = self.length(pos)
        if span == 0:
            raise ValueError("next requires a live span start")
        return pos + span

    def prev(self, pos):
        """Return the previous start; ``pos`` must be a live start after 0."""
        if not isinstance(pos, int) or pos <= 0 or pos >= len(self._data) or not self.length(pos):
            raise ValueError("prev requires a live span start after position 0")
        span = self._end_length(pos - 1)
        if span == 0 or span > pos:
            raise ValueError("no valid preceding span boundary")
        start = pos - span
        if self.length(start) != span:
            raise ValueError("preceding start/end boundaries disagree")
        return start

    def merge(self, pos, right, after):
        """Merge adjacent spans at ``pos`` and ``right``; return ``right``.

        ``after`` is the start of the following span, or ``n``. Work is O(1):
        the only loops encode/decode the fixed five-byte u32 payload.
        """
        left_len = self.length(pos)
        right_len = self.length(right)
        n = len(self._data)
        if not left_len or not right_len:
            raise ValueError("merge requires two live span starts")
        if right != pos + left_len or after != right + right_len or after > n:
            raise ValueError("merge boundaries are not adjacent")
        merged_len = left_len + right_len
        if merged_len > self._U32_MAX:
            raise ValueError("span length exceeds u32")
        end = after - 1

        # Invalidate the absorbed right start before writing tags/payload. The
        # old left end is allowed to remain as stale interior data.
        self._data[right] = 0
        if merged_len <= 63:
            self._data[pos] = merged_len
            self._data[end] = merged_len + 62
        else:
            self._data[pos] = 126
            self._data[end] = 127
            for index in range(self._PAYLOAD_BYTES):
                digit = 128 | ((merged_len >> (7 * index)) & 0x7F)
                self._data[pos + 1 + index] = digit
                self._data[end - 1 - index] = digit
        return right

    def memory_bytes(self):
        """Return allocated payload bytes; exactly one byte per position."""
        return len(self._data)

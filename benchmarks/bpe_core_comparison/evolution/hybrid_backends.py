"""Direction-tagged endpoint experiments for the fused Python BPE driver.

The input is a flattened sequence of one-position u16 token IDs with live
zero-valued separators at both ends and between pieces.  The driver provides
fresh u32 IDs, appends their lengths to the shared token_len list, and calls
merge_known only after a successful inspect_pair.  Dead interior positions
retain arbitrary text and old end tags; only start tags identify occurrences.
"""

from array import array


def _initial_text(initial_ids, token_len):
    n = len(initial_ids)
    if n < 2 or initial_ids[0] != 0 or initial_ids[-1] != 0:
        raise ValueError("initial IDs require leading and trailing zero separators")
    if n >= 1 << 32:
        raise ValueError("u32 offsets require fewer than 2^32 positions")
    if any(x < 0 or x >= 1 << 16 for x in initial_ids):
        raise ValueError("initial token IDs must fit in u16")
    text = array("H", initial_ids)
    if text.itemsize != 2:
        raise RuntimeError("array('H') must have two-byte elements")
    if token_len is None:
        token_len = [1] * (max(text) + 1)
    if not isinstance(token_len, list) or len(token_len) <= max(text):
        raise ValueError("token_len must be a shared list covering initial IDs")
    if token_len[0] != 1 or any(token_len[x] != 1 for x in text):
        raise ValueError("every initial ID must have length one")
    return text, token_len


class HybridByteTags:
    """H3: 2N-byte text plus one directed tag byte per input position.

    Tag 0 is dead; 1 is a singleton; 2..127 are short starts with their
    length; 128..253 are short ends with length ``tag-126``; 254 and 255 are
    long starts and ends.  A long start obtains length from token_len[ID]; a
    long end stores its u32 length in the last two u16 text cells.  All merged
    IDs use the first two u16 cells, high half first.  Long spans begin at
    length 128, so start ID and end length cannot overlap.
    """

    __slots__ = ("text", "tags", "token_len", "last")

    def __init__(self, initial_ids, token_len=None):
        self.text, self.token_len = _initial_text(initial_ids, token_len)
        self.tags = bytearray(b"\x01") * len(self.text)
        self.last = len(self.text) - 1

    def token(self, pos):
        text = self.text
        if self.tags[pos] == 1:
            return text[pos]
        return (text[pos] << 16) | text[pos + 1]

    def next(self, pos):
        if pos == self.last:
            return None
        tag = self.tags[pos]
        if tag == 1:
            return pos + 1
        if tag != 254:
            return pos + tag
        text = self.text
        return pos + self.token_len[(text[pos] << 16) | text[pos + 1]]

    def prev(self, pos):
        if pos == 0:
            return None
        end = pos - 1
        tag = self.tags[end]
        if tag == 1:
            return end
        if tag != 255:
            return pos - (tag - 126)
        text = self.text
        return pos - ((text[end - 1] << 16) | text[end])

    def inspect_pair(self, pos, a, b):
        if a == 0 or b == 0 or pos < 0 or pos >= self.last:
            return None
        tags, text = self.tags, self.text
        tag = tags[pos]
        if tag == 1:
            if text[pos] != a:
                return None
            right = pos + 1
        elif 2 <= tag <= 127:
            if ((text[pos] << 16) | text[pos + 1]) != a:
                return None
            right = pos + tag
        elif tag == 254:
            if ((text[pos] << 16) | text[pos + 1]) != a:
                return None
            right = pos + self.token_len[a]
        else:
            return None
        if right >= self.last:
            return None
        right_tag = tags[right]
        if right_tag == 1:
            if text[right] != b:
                return None
        elif 2 <= right_tag <= 127 or right_tag == 254:
            if ((text[right] << 16) | text[right + 1]) != b:
                return None
        else:
            return None
        before = self.prev(pos)
        after = self.next(right)
        left_id = self.token(before) if before is not None else 0
        right_id = self.token(after) if after is not None else 0
        return before, left_id, right, after, right_id

    def merge_known(self, pos, right, after, new_id, new_len):
        tags, text = self.tags, self.text
        tags[right] = 0
        if new_len <= 127:
            tags[pos] = new_len
            tags[after - 1] = new_len + 126
        else:
            tags[pos] = 254
            tags[after - 1] = 255
            text[after - 2] = new_len >> 16
            text[after - 1] = new_len & 0xFFFF
        text[pos] = new_id >> 16
        text[pos + 1] = new_id & 0xFFFF
        return right

    def memory_details(self):
        return {"text_u16": len(self.text) * self.text.itemsize,
                "direction_tags_u8": len(self.tags)}

    def memory_bytes(self):
        return 2 * len(self.text) + len(self.tags)


class FusedHybridByteTags(HybridByteTags):
    """H3 with pair context resolved in one inline inspection pass.

    Direction tags reject dead positions and give short lengths directly.
    For a long start, its validated pair ID indexes the shared token_len.
    The preceding end tag supplies the predecessor length.  This avoids four
    method calls made by the exploratory H3 inspect_pair implementation.
    """

    __slots__ = ()

    def inspect_pair(self, pos, a, b):
        if a == 0 or b == 0 or pos < 0 or pos >= self.last:
            return None
        tags, text, token_len = self.tags, self.text, self.token_len
        start_tag = tags[pos]
        if start_tag == 1:
            if text[pos] != a:
                return None
            right = pos + 1
        elif 2 <= start_tag <= 127 or start_tag == 254:
            if ((text[pos] << 16) | text[pos + 1]) != a:
                return None
            right = pos + (start_tag if start_tag != 254 else token_len[a])
        else:
            return None

        if right >= self.last:
            return None
        right_tag = tags[right]
        if right_tag == 1:
            if text[right] != b:
                return None
            after = right + 1
        elif 2 <= right_tag <= 127 or right_tag == 254:
            if ((text[right] << 16) | text[right + 1]) != b:
                return None
            after = right + (right_tag if right_tag != 254 else token_len[b])
        else:
            return None

        if pos == 0:
            before = None
            left_id = 0
        else:
            end = pos - 1
            end_tag = tags[end]
            if end_tag == 1:
                before = end
            elif end_tag == 255:
                before = pos - ((text[end - 1] << 16) | text[end])
            else:
                before = pos - (end_tag - 126)
            before_tag = tags[before]
            left_id = (text[before] if before_tag == 1
                       else (text[before] << 16) | text[before + 1])

        after_tag = tags[after]
        right_id = (text[after] if after_tag == 1
                    else (text[after] << 16) | text[after + 1])
        return before, left_id, right, after, right_id


class HybridNibbleTags(HybridByteTags):
    """H2.5: 2N-byte text plus one directional four-bit tag per position.

    Nibble 0 is dead; 1 is singleton; 2..6 are short starts; 7 is long start;
    8..12 are short ends of length ``tag-6``; 13 is long end.  A long length
    starts at seven, leaving its first two u16 cells for ID and last two for
    length.  Nibbles 14 and 15 are unused.  Packing adds read-modify-write
    costs to Python tag updates, which the benchmarks should measure.
    """

    __slots__ = ()

    def __init__(self, initial_ids, token_len=None):
        self.text, self.token_len = _initial_text(initial_ids, token_len)
        n = len(self.text)
        self.tags = bytearray(b"\x11") * (n // 2)
        if n & 1:
            self.tags.append(1)
        self.last = n - 1

    def token(self, pos):
        tag = self.tags[pos >> 1] >> ((pos & 1) << 2) & 15
        text = self.text
        if tag == 1:
            return text[pos]
        return (text[pos] << 16) | text[pos + 1]

    def next(self, pos):
        if pos == self.last:
            return None
        tag = self.tags[pos >> 1] >> ((pos & 1) << 2) & 15
        if tag == 1:
            return pos + 1
        if tag != 7:
            return pos + tag
        text = self.text
        return pos + self.token_len[(text[pos] << 16) | text[pos + 1]]

    def prev(self, pos):
        if pos == 0:
            return None
        end = pos - 1
        tag = self.tags[end >> 1] >> ((end & 1) << 2) & 15
        if tag == 1:
            return end
        if tag != 13:
            return pos - (tag - 6)
        text = self.text
        return pos - ((text[end - 1] << 16) | text[end])

    def inspect_pair(self, pos, a, b):
        if a == 0 or b == 0 or pos < 0 or pos >= self.last:
            return None
        tags, text = self.tags, self.text
        tag = tags[pos >> 1] >> ((pos & 1) << 2) & 15
        if tag == 1:
            if text[pos] != a:
                return None
            right = pos + 1
        elif 2 <= tag <= 6:
            if ((text[pos] << 16) | text[pos + 1]) != a:
                return None
            right = pos + tag
        elif tag == 7:
            if ((text[pos] << 16) | text[pos + 1]) != a:
                return None
            right = pos + self.token_len[a]
        else:
            return None
        if right >= self.last:
            return None
        right_tag = tags[right >> 1] >> ((right & 1) << 2) & 15
        if right_tag == 1:
            if text[right] != b:
                return None
        elif 2 <= right_tag <= 7:
            if ((text[right] << 16) | text[right + 1]) != b:
                return None
        else:
            return None
        before = self.prev(pos)
        after = self.next(right)
        left_id = self.token(before) if before is not None else 0
        right_id = self.token(after) if after is not None else 0
        return before, left_id, right, after, right_id

    def merge_known(self, pos, right, after, new_id, new_len):
        tags, text = self.tags, self.text
        index = right >> 1
        shift = (right & 1) << 2
        tags[index] &= ~(15 << shift) & 255
        if new_len <= 6:
            start_tag, end_tag = new_len, new_len + 6
        else:
            start_tag, end_tag = 7, 13
            text[after - 2] = new_len >> 16
            text[after - 1] = new_len & 0xFFFF
        index = pos >> 1
        shift = (pos & 1) << 2
        tags[index] = (tags[index] & ~(15 << shift) & 255) | (start_tag << shift)
        end = after - 1
        index = end >> 1
        shift = (end & 1) << 2
        tags[index] = (tags[index] & ~(15 << shift) & 255) | (end_tag << shift)
        text[pos] = new_id >> 16
        text[pos + 1] = new_id & 0xFFFF
        return right

    def memory_details(self):
        return {"text_u16": len(self.text) * self.text.itemsize,
                "direction_tags_u4": len(self.tags)}

    def memory_bytes(self):
        return 2 * len(self.text) + len(self.tags)


__all__ = ["HybridByteTags", "FusedHybridByteTags", "HybridNibbleTags"]

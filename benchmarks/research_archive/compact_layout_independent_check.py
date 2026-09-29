"""Independent oracle checks for the compact boundary layout."""
from pathlib import Path as _AuditPath
REPO = _AuditPath(__file__).resolve().parents[2]
ARCHIVE = _AuditPath(__file__).resolve().parent

from array import array
import random


class Boundaries:
    def __init__(self, n):
        self.n = n
        self.bits = array('Q', [(1 << 64) - 1]) * ((n + 64) // 64)
        self.bits[-1] = (1 << ((n % 64) + 1)) - 1
        self.skips = array('I', [0]) * len(self.bits)

    def alive(self, p):
        assert 0 <= p <= self.n
        return bool(self.bits[p // 64] & (1 << (p % 64)))

    def nxt(self, p):
        assert self.alive(p)
        if p == self.n:
            return None
        b, o = divmod(p, 64)
        w = self.bits[b] >> (o + 1)
        if w:
            return p + (w & -w).bit_length()
        w = self.bits[b + 1]
        if w:
            return (b + 1) * 64 + (w & -w).bit_length() - 1
        return p + self.skips[b + 1] + 1

    def prev(self, p):
        assert self.alive(p)
        if p == 0:
            return None
        b, o = divmod(p, 64)
        w = self.bits[b] & ((1 << o) - 1)
        if w:
            return b * 64 + w.bit_length() - 1
        w = self.bits[b - 1]
        if w:
            return (b - 1) * 64 + w.bit_length() - 1
        return p - self.skips[b - 1] - 1

    def merge(self, p):
        j = self.nxt(p)
        assert j < self.n
        k = self.nxt(j)
        self.bits[j // 64] &= ~(1 << (j % 64))
        b1, b3 = p // 64, k // 64
        if b3 > b1 + 1:
            assert self.bits[b1 + 1] == self.bits[b3 - 1] == 0
            self.skips[b1 + 1] = self.skips[b3 - 1] = k - p - 1


def check(b, live):
    s = set(live)
    for p in range(b.n + 1):
        assert b.alive(p) == (p in s), (b.n, p, live)
    for t, p in enumerate(live):
        assert b.prev(p) == (live[t - 1] if t else None), (b.n, p, 'prev', live)
        assert b.nxt(p) == (live[t + 1] if t + 1 < len(live) else None), (b.n, p, 'next', live)
    # A wholly empty block belongs to one and only one live-boundary gap.
    owner = {}
    for lo, hi in zip(live, live[1:]):
        if hi // 64 > lo // 64 + 1:
            for block in range(lo // 64 + 1, hi // 64):
                assert block not in owner, (block, owner[block], (lo, hi))
                owner[block] = (lo, hi)
            gap = hi - lo - 1
            assert b.skips[lo // 64 + 1] == gap
            assert b.skips[hi // 64 - 1] == gap


rng = random.Random(303)
merges = 0
for n in [2, 3, 63, 64, 65, 126, 127, 128, 129, 191, 192, 193, 255, 256, 257, 513]:
    for trial in range(5):
        b = Boundaries(n)
        live = list(range(n + 1))
        check(b, live)
        while len(live) > 2:
            idx = (rng.randrange(len(live) - 2) if trial % 4 == 0 else
                   0 if trial % 4 == 1 else
                   len(live) - 3 if trial % 4 == 2 else
                   (len(live) - 2) // 2)
            b.merge(live[idx])
            del live[idx + 1]
            merges += 1
            check(b, live)
print({'merges': merges, 'all_dead_positions_checked_each_merge': True})

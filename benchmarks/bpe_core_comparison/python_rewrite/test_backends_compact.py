"""Small independent state-machine check; not a performance run."""

import random

from backends_compact import (
    EbpeEndpoints, PrezzaBitmap, FastEbpeEndpoints, FastPrezzaBitmap,
    FastPrezzaHalfword,
)


def exercise(initial, seed, fast=False):
    rng = random.Random(seed)
    classes = ([FastEbpeEndpoints, FastPrezzaBitmap, FastPrezzaHalfword]
               if fast else [EbpeEndpoints, PrezzaBitmap])
    backends = [cls(initial) for cls in classes]
    live = list(range(len(initial)))
    ids = dict(enumerate(initial))
    next_id = max(initial) + 1
    merges = 0
    while True:
        live_set = set(live)
        for backend in backends:
            for i, p in enumerate(live):
                assert backend.token(p) == ids[p]
                assert backend.prev(p) == (live[i - 1] if i else None)
                assert backend.next(p) == (live[i + 1] if i + 1 < len(live) else None)
            for p in range(len(initial) - 1):
                # Probe initial occurrences after arbitrary subsequent merges.
                a, b = initial[p:p + 2]
                expected = (p in live_set and a != 0 and b != 0 and
                            ids[p] == a and (p + 1) in live_set and ids[p + 1] == b)
                assert backend.pair_matches(p, a, b) == expected, (type(backend).__name__, p)
            if isinstance(backend, PrezzaBitmap):
                for p in range(len(initial)):
                    assert backend.alive(p) == (p in live_set)
            assert backend.memory_bytes() == sum(backend.memory_details().values())
            assert backend.memory_bytes() >= len(initial) * backend.corpus.itemsize
        choices = [i for i in range(len(live) - 1)
                   if ids[live[i]] != 0 and ids[live[i + 1]] != 0]
        if not choices:
            break
        i = rng.choice(choices)
        left, right = live[i:i + 2]
        after = live[i + 2]
        length = after - left
        for backend in backends:
            assert backend.pair_matches(left, ids[left], ids[right])
            if fast:
                assert next_id == len(backend.token_len)
                backend.token_len.append(length)
            assert backend.merge(left, next_id, length) == right
        ids[left] = next_id
        del ids[right]
        del live[i + 1]
        next_id += 1
        merges += 1
    return merges


total = 0
for n in [1, 2, 8, 32, 64, 65, 127, 128, 129, 256, 513]:
    for seed in range(4):
        initial = [0] + [1 + (i * 7 + seed) % 5 for i in range(n)] + [0]
        if n > 20:
            initial[n // 2] = 0
        total += exercise(initial, seed)
        total += exercise(initial, seed, fast=True)
# Force packed IDs across the 16-bit boundary in the halfword variant.
total += exercise([0, 65535, 1, 2, 0], 7, fast=True)
try:
    FastPrezzaHalfword([0, 65536, 0])
except ValueError:
    pass
else:
    raise AssertionError("halfword backend accepted an initial ID >= 65536")
print({"small_oracle_merges": total, "checked_backends": 2, "fast_backends": 3})

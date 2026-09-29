"""Independent oracle for fused occurrence inspection and known merge writes."""

import random

from backends_fused import (
    FusedEbpeEndpoints, FusedPrezzaBitmap, FusedPrezzaHalfword,
)


def run(initial, seed):
    rng = random.Random(seed)
    backends = [cls(initial) for cls in
                (FusedEbpeEndpoints, FusedPrezzaBitmap, FusedPrezzaHalfword)]
    live = list(range(len(initial)))
    ids = dict(enumerate(initial))
    next_id = max(initial) + 1
    merges = 0
    while True:
        live_set = set(live)
        for backend in backends:
            for i, pos in enumerate(live):
                assert backend.token(pos) == ids[pos]
                assert backend.prev(pos) == (live[i - 1] if i else None)
                assert backend.next(pos) == (live[i + 1] if i + 1 < len(live) else None)
            for pos in range(len(initial) - 1):
                a, b = initial[pos:pos + 2]
                valid = (pos in live_set and a != 0 and b != 0 and
                         ids[pos] == a and pos + 1 in live_set and ids[pos + 1] == b)
                context = backend.inspect_pair(pos, a, b)
                if not valid:
                    assert context is None, (type(backend).__name__, pos, context)
                else:
                    i = live.index(pos)
                    before = live[i - 1] if i else None
                    right = live[i + 1]
                    after = live[i + 2]
                    expected = (before, ids[before] if before is not None else 0,
                                right, after, ids[after])
                    assert context == expected, (type(backend).__name__, pos, context, expected)
            if isinstance(backend, (FusedPrezzaBitmap, FusedPrezzaHalfword)):
                for pos in range(len(initial)):
                    assert backend.alive(pos) == (pos in live_set)
        options = [i for i in range(len(live) - 1)
                   if ids[live[i]] and ids[live[i + 1]]]
        if not options:
            break
        i = rng.choice(options)
        pos, right = live[i:i + 2]
        after = live[i + 2]
        new_len = after - pos
        for backend in backends:
            context = backend.inspect_pair(pos, ids[pos], ids[right])
            assert context is not None
            assert (context[2], context[3]) == (right, after)
            assert len(backend.token_len) == next_id
            backend.token_len.append(new_len)
            assert backend.merge_known(pos, right, after, next_id, new_len) == right
        ids[pos] = next_id
        del ids[right]
        del live[i + 1]
        next_id += 1
        merges += 1
    return merges


total = 0
for n in (2, 8, 63, 64, 65, 127, 128, 129, 257):
    for seed in range(3):
        initial = [0] + [1 + (i * 7 + seed) % 5 for i in range(n)] + [0]
        if n > 20:
            initial[n // 2] = 0
        total += run(initial, seed)
total += run([0, 65535, 1, 2, 0], 3)
for merged_id in (65535, 65536, 0x12345678, 0xFFFFFFFF):
    packed = FusedPrezzaHalfword([0, 1, 2, 3, 0])
    assert packed.inspect_pair(1, 1, 2) == (0, 0, 2, 3, 3)
    assert packed.merge_known(1, 2, 3, merged_id, 2) == 2
    assert packed.token(1) == merged_id
    assert packed.next(1) == 3
    assert packed.inspect_pair(1, merged_id, 3) == (0, 0, 3, 4, 0)
    assert not packed.alive(2)
print({"fused_oracle_merges": total, "backends": 3})

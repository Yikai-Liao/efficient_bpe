"""Small oracle and directed boundary checks for hybrid endpoint layouts."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_rewrite"))

from common_fused import naive, prepare, train  # noqa: E402
from hybrid_backends import (  # noqa: E402
    FusedHybridByteTags, HybridByteTags, HybridNibbleTags,
)


BACKENDS = [HybridByteTags, FusedHybridByteTags, HybridNibbleTags]


def exercise(backend_class, initial, seed, thorough=True):
    backend = backend_class(initial)
    rng = random.Random(seed)
    live = list(range(len(initial)))
    ids = dict(enumerate(initial))
    historical = {(pos, initial[pos], initial[pos + 1])
                  for pos in range(len(initial) - 1)
                  if initial[pos] and initial[pos + 1]}
    next_id = max(initial) + 1
    merges = 0
    while True:
        live_set = set(live)
        if thorough or len(live) % 17 == 0:
            for i, pos in enumerate(live):
                assert backend.token(pos) == ids[pos]
                assert backend.prev(pos) == (live[i - 1] if i else None)
                assert backend.next(pos) == (live[i + 1] if i + 1 < len(live) else None)
            for pos, a, b in historical:
                valid = (pos in live_set and ids[pos] == a and a != 0 and
                         (i := live.index(pos)) + 1 < len(live) and
                         ids[live[i + 1]] == b and b != 0)
                context = backend.inspect_pair(pos, a, b)
                assert (context is not None) == valid, (pos, a, b, context, valid)
        options = [i for i in range(len(live) - 1)
                   if ids[live[i]] and ids[live[i + 1]]]
        if not options:
            break
        i = rng.choice(options)
        pos, right, after = live[i:i + 3]
        a, b = ids[pos], ids[right]
        historical.add((pos, a, b))
        context = backend.inspect_pair(pos, a, b)
        before = live[i - 1] if i else None
        expected = (before, ids[before] if before is not None else 0,
                    right, after, ids[after])
        assert context == expected, (context, expected)
        length = after - pos
        assert len(backend.token_len) == next_id
        backend.token_len.append(length)
        assert backend.merge_known(pos, right, after, next_id, length) == right
        ids[pos] = next_id
        del ids[right]
        del live[i + 1]
        if before is not None and ids[before] and ids[pos]:
            historical.add((before, ids[before], ids[pos]))
        if ids[pos] and ids[after]:
            historical.add((pos, ids[pos], ids[after]))
        next_id += 1
        merges += 1
    return backend, merges


class HybridTests(unittest.TestCase):
    def test_random_history_and_crossings(self):
        for cls in BACKENDS:
            for n in (2, 3, 6, 7, 8, 63, 64, 65, 127, 128, 129, 257):
                for seed in range(3):
                    initial = [0] + [1 + (i + seed) % 4 for i in range(n)] + [0]
                    if n > 20:
                        initial[n // 2] = 0
                    backend, _ = exercise(cls, initial, seed, thorough=n <= 65)
                    npos = len(initial)
                    expected_bytes = (2 * npos + (npos + 1) // 2
                                      if cls is HybridNibbleTags else 3 * npos)
                    self.assertEqual(backend.memory_bytes(), expected_bytes)

    def test_self_overlap_with_full_recount(self):
        for cls in BACKENDS:
            for words in (["aaaaaa", "aaaa", "a" * 17],
                          ["abababa", "babab", "abcabc"],
                          ["你好你好你好", "aaaa"]):
                prepared = prepare(words)
                expected_rules, expected_final = naive(prepared, 30, 1)
                actual = train(prepared, cls, 30, 1, capture=True)
                self.assertEqual(actual["merges"], expected_rules)
                self.assertEqual(actual["final"], expected_final)

    def test_u32_ids_and_singleton_edges(self):
        for cls in BACKENDS:
            for merged_id in (65535, 65536, 0x12345678, 0xFFFFFFFF):
                backend = cls([0, 1, 2, 3, 0])
                self.assertEqual(backend.inspect_pair(1, 1, 2), (0, 0, 2, 3, 3))
                backend.merge_known(1, 2, 3, merged_id, 2)
                self.assertEqual(backend.token(1), merged_id)
                self.assertEqual(backend.next(1), 3)
                self.assertEqual(backend.prev(3), 1)
                self.assertEqual(backend.inspect_pair(1, merged_id, 3),
                                 (0, 0, 3, 4, 0))
                self.assertIsNone(backend.inspect_pair(2, 2, 3))

    def test_chain_length_boundaries(self):
        # One linear chain reaches every requested threshold without O(N^2)
        # whole-state checking. Inspect the exact transition points only.
        targets = (2, 6, 7, 127, 128, 255, 256, 65535, 65536)
        for cls in BACKENDS:
            n = targets[-1]
            backend = cls([0] + [1] * n + [0])
            for length in range(2, n + 1):
                right = length
                after = length + 1
                new_id = len(backend.token_len)
                backend.token_len.append(length)
                backend.merge_known(1, right, after, new_id, length)
                if length in targets:
                    self.assertEqual(backend.token(1), new_id)
                    self.assertEqual(backend.next(1), after)
                    self.assertEqual(backend.prev(after), 1)
                    self.assertEqual(backend.inspect_pair(1, new_id, 1),
                                     (0, 0, after, after + 1,
                                      0 if after + 1 == backend.last else 1)
                                     if after < backend.last else None)

    def test_bad_initial_alphabet(self):
        for cls in BACKENDS:
            with self.assertRaises(ValueError):
                cls([0, 65536, 0])

    def test_old_end_never_validates_as_start(self):
        # An undirected tag at position 2 would misread low-half zero plus
        # text[3] as ID 1, yielding a false (1, 1) occurrence.
        for cls in BACKENDS:
            backend = cls([0, 1, 1, 1, 1, 0])
            backend.merge_known(1, 2, 3, 65536, 2)
            self.assertEqual(backend.token(1), 65536)
            self.assertIsNone(backend.inspect_pair(2, 1, 1))

    def test_right_growing_chain(self):
        for cls in BACKENDS:
            n = 256
            backend = cls([0] + [1] * n + [0])
            for pos in range(n - 1, 0, -1):
                right = pos + 1
                after = n + 1
                length = after - pos
                new_id = len(backend.token_len)
                backend.token_len.append(length)
                backend.merge_known(pos, right, after, new_id, length)
                self.assertEqual(backend.token(pos), new_id)
                self.assertEqual(backend.next(pos), after)
                self.assertEqual(backend.prev(after), pos)
                if pos > 1:
                    self.assertEqual(backend.inspect_pair(pos - 1, 1, new_id),
                                     (pos - 2, 0 if pos == 2 else 1,
                                      pos, after, 0))


if __name__ == "__main__":
    unittest.main()

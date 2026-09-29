"""Oracle checks for packed keys, weights and replacement ordering."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python_rewrite'))
from common_fused import prepare, naive, train as baseline
from backends_fused import FusedEbpeEndpoints
from lean_backend import LeanEndpoints
from packed_driver import train
from filtered_driver import train as filtered_train


class PackedTests(unittest.TestCase):
    def test_weighted_oracle(self):
        rng = random.Random(263009)
        cases = [[], [''], ['a'], ['aaa', 'aaaa', 'aaaaa'] * 3,
                 ['abababa', 'aba', 'bababab'] * 4]
        cases += [[''.join(rng.choices('abcde界', k=rng.randrange(0, 50)))
                   for _ in range(rng.randrange(1, 14))] * rng.randrange(1, 5)
                  for _ in range(300)]
        for i, pieces in enumerate(cases):
            for dedup in (True, False):
                prepared = prepare(pieces, dedup)
                minimum = 1 + i % 5
                expected = naive(prepared, 80, minimum)
                for backend in (FusedEbpeEndpoints, LeanEndpoints):
                    for driver in (train, filtered_train):
                        got = driver(prepared, backend, 80, minimum, capture=True)
                        self.assertEqual((got['merges'], got['final']), expected,
                                         (i, dedup, backend, driver))

    def test_integer_weights_have_no_u64_frequency_limit(self):
        prepared = prepare(['aaaaab', 'baaaaa'])
        huge = 1 << 70
        prepared = prepared[:3] + ([w * huge for w in prepared[3]],)
        expected = naive(prepared, 50, huge)
        for driver in (train, filtered_train):
            got = driver(prepared, LeanEndpoints, 50, huge, capture=True)
            self.assertEqual((got['merges'], got['final']), expected)

    def test_lexicographic_key_order_across_u16_boundary(self):
        pairs = [(1, 0xFFFFFFFF), (2, 1), (65535, 65536),
                 (65536, 65535), (0xFFFFFFFF, 0xFFFFFFFF)]
        rng = random.Random(42)
        rng.shuffle(pairs)
        packed = sorted((a << 32) | b for a, b in pairs)
        self.assertEqual([(p >> 32, p & 0xFFFFFFFF) for p in packed],
                         sorted(pairs))


if __name__ == '__main__':
    unittest.main()

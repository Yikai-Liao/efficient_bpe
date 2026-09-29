"""Small randomized oracle checks for the benchmark queue strategies."""

from __future__ import annotations

import random
import unittest

from queues import HighLowQueue, HeapQueue


class QueueOracleTests(unittest.TestCase):
    def test_equal_frequency_uses_lexicographically_smallest_pair(self) -> None:
        for queue_type in (HeapQueue, HighLowQueue):
            frequencies = {(8, 1): 5, (2, 9): 5, (2, 3): 5}
            queue = queue_type(frequencies, min_frequency=1, initial_mass=sum(frequencies.values()))
            self.assertEqual(queue.pop(), ((2, 3), 5))
            self.assertEqual(queue.pop(), ((2, 9), 5))
            self.assertEqual(queue.pop(), ((8, 1), 5))
            self.assertEqual(queue.pop(), (None, 0))

    def test_frequency_decreases_and_once_added_pairs_match_oracle(self) -> None:
        for queue_type in (HeapQueue, HighLowQueue):
            for seed in range(40):
                rng = random.Random(seed)
                frequencies = {
                    (i, i + 1): rng.randint(1, 90)
                    for i in range(1, rng.randint(8, 30))
                }
                min_frequency = rng.randint(1, 5)
                queue = queue_type(frequencies, min_frequency, sum(frequencies.values()))
                max_initial_frequency = max(frequencies.values())
                remaining = set(frequencies)
                next_id = 10_000

                for _ in range(180):
                    # Existing frequencies only decrease, without add().
                    for _ in range(rng.randint(0, 3)):
                        if not remaining:
                            break
                        pair = rng.choice(tuple(remaining))
                        old = frequencies[pair]
                        frequencies[pair] = max(0, old - rng.randint(1, max(1, old)))

                    # Every generated pair has a fresh token ID and is added
                    # exactly once, after its final initial frequency exists.
                    if rng.random() < 0.55:
                        pair = (next_id, next_id + 1)
                        next_id += 2
                        frequencies[pair] = rng.randint(1, max_initial_frequency)
                        queue.add(pair)
                        remaining.add(pair)

                    expected = min(
                        ((-frequencies[pair], pair) for pair in remaining
                         if frequencies[pair] >= min_frequency),
                        default=None,
                    )
                    actual = queue.pop()
                    if expected is None:
                        self.assertEqual(actual, (None, 0), (queue_type.__name__, seed))
                    else:
                        self.assertEqual(actual, (expected[1], -expected[0]), (queue_type.__name__, seed))
                        remaining.remove(actual[0])
                        frequencies[actual[0]] = 0

                # Drain once more to ensure stale events do not resurrect a
                # selected, deleted-below-minimum, or fully drained pair.
                while True:
                    expected = min(
                        ((-frequencies[pair], pair) for pair in remaining
                         if frequencies[pair] >= min_frequency),
                        default=None,
                    )
                    actual = queue.pop()
                    if expected is None:
                        self.assertEqual(actual, (None, 0), (queue_type.__name__, seed))
                        break
                    self.assertEqual(actual, (expected[1], -expected[0]), (queue_type.__name__, seed))
                    remaining.remove(actual[0])
                    frequencies[actual[0]] = 0

    def test_high_demotion_low_bucket_and_counters(self) -> None:
        frequencies = {(8, 1): 7, (2, 9): 6, (3, 1): 4}
        queue = HighLowQueue(frequencies, min_frequency=2, initial_mass=25)
        self.assertEqual(queue.threshold, 5)
        frequencies[(8, 1)] = 4  # High candidate moves to low on the scan.
        self.assertEqual(queue.pop(), ((2, 9), 6))
        self.assertEqual(queue.pop(), ((3, 1), 4))
        self.assertEqual(queue.pop(), ((8, 1), 4))
        self.assertEqual(queue.pop(), (None, 0))
        stats = queue.stats()
        self.assertGreaterEqual(stats["high_scan_visits"], 2)
        self.assertGreaterEqual(stats["low_candidate_pops"], 2)
        self.assertGreaterEqual(stats["low_sort_calls"], 1)

    def test_heap_counters_and_lazy_correction(self) -> None:
        frequencies = {(1, 2): 20, (2, 3): 12}
        queue = HeapQueue(frequencies, min_frequency=2, initial_mass=32)
        frequencies[(1, 2)] = 7
        self.assertEqual(queue.pop(), ((2, 3), 12))
        self.assertEqual(queue.pop(), ((1, 2), 7))
        self.assertEqual(queue.pop(), (None, 0))
        self.assertEqual(queue.stats(), {
            "heapify_entries": 2,
            "heap_pushes": 1,
            "heap_pops": 3,
        })


if __name__ == "__main__":
    unittest.main()

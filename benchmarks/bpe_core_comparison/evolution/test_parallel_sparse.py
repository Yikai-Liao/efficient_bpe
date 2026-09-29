"""Small differential checks for exact owner-mask worker dispatch."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_rewrite"))

from common_fused import naive, prepare  # noqa: E402
from lean_backend import LeanEndpoints  # noqa: E402
from parallel_driver import train as broadcast_train  # noqa: E402
from parallel_sparse_driver import train as sparse_train  # noqa: E402


class SparseParallelTests(unittest.TestCase):
    def check_case(self, pieces, workers=3, *, serial=False,
                   deduplicate=True, max_merges=24, min_frequency=2):
        prepared = prepare(pieces, deduplicate=deduplicate)
        expected_merges, expected_final = naive(prepared, max_merges,
                                                min_frequency)
        reference = broadcast_train(prepared, workers=workers,
                                    backend_class=LeanEndpoints,
                                    max_merges=max_merges,
                                    min_frequency=min_frequency,
                                    capture=True, serial=True)
        result = sparse_train(prepared, workers=workers,
                              backend_class=LeanEndpoints,
                              max_merges=max_merges,
                              min_frequency=min_frequency,
                              capture=True, serial=serial)
        self.assertEqual(result["merges"], expected_merges)
        self.assertEqual(result["final"], expected_final)
        self.assertEqual(result["fingerprint"], reference["fingerprint"])
        for field in ("rules", "actual_merges", "position_visits",
                      "stale_visits", "heap_pops", "max_token_length"):
            self.assertEqual(result[field], reference[field], field)
        self.assertEqual(sum(result["worker_dispatch_counts"])
                         + sum(result["worker_skip_counts"]),
                         result["rules"] * result["actual_workers"])
        self.assertEqual(result["round_messages"],
                         0 if serial else 2 * sum(result["worker_dispatch_counts"]))
        self.assertLessEqual(result["round_messages"],
                             2 * result["actual_workers"] * result["rules"])
        return result

    def test_small_random_shards(self):
        rng = random.Random(7204)
        for workers in (1, 2, 4):
            pieces = ["".join(rng.choices("abcde", k=rng.randrange(2, 13)))
                      for _ in range(9)]
            self.check_case(pieces, workers, max_merges=15)

    def test_weighted_overlap_and_stale_owners(self):
        pieces = (["aaaaaaa"] * 5 + ["abababa"] * 3 +
                  ["babab"] * 2 + ["abcabc", "ba", "cccccccc"])
        serial = self.check_case(pieces, 3, serial=True)
        spawned = self.check_case(pieces, 3)
        self.assertEqual(serial["fingerprint"], spawned["fingerprint"])
        self.assertGreater(spawned["stale_visits"], 0)

    def test_late_token_length_fill(self):
        # Heavy a-rules occupy one piece; the b-owner sits out several rules
        # and must catch up before its first late merge_round call.
        pieces = ["a" * 64] * 10 + ["b" * 7]
        result = self.check_case(pieces, 2, max_merges=18)
        self.assertGreater(result["max_late_token_fill"], 1)
        self.assertGreater(result["late_token_lengths_sent"], 0)
        self.assertLess(result["round_messages"],
                        2 * result["actual_workers"] * result["rules"])

    def test_many_pieces_with_idle_workers_and_empty(self):
        pieces = ["aaaaaa", "bc", "de", "fg", "hi", "jk", "lm", "no"]
        result = self.check_case(pieces, 4, deduplicate=False)
        self.assertTrue(any(n == 0 for n in result["worker_dispatch_counts"]))
        self.assertEqual(sum(result["shard_pieces"]), len(pieces))
        for pieces in ([], [""], ["", ""]):
            result = self.check_case(pieces, 4)
            self.assertEqual(result["actual_workers"], 0)
            self.assertEqual(result["round_messages"], 0)


if __name__ == "__main__":
    unittest.main()

"""Bounded correctness checks for serial and spawned whole-piece workers."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_rewrite"))

from common_fused import naive, prepare  # noqa: E402
from lean_backend import LeanEndpoints  # noqa: E402
from packed_driver import train as packed_train  # noqa: E402
from parallel_driver import train as parallel_train  # noqa: E402


class ParallelTests(unittest.TestCase):
    def assert_matches_oracles(self, pieces, worker_count, *, serial=False,
                               max_merges=20, deduplicate=True):
        prepared = prepare(pieces, deduplicate=deduplicate)
        expected_rules, expected_final = naive(prepared, max_merges, 2)
        packed = packed_train(prepared, LeanEndpoints, max_merges, 2, capture=True)
        result = parallel_train(prepared, workers=worker_count,
                                max_merges=max_merges, min_frequency=2,
                                capture=True, serial=serial)
        self.assertEqual(result["merges"], expected_rules)
        self.assertEqual(result["final"], expected_final)
        self.assertEqual(result["fingerprint"], packed["fingerprint"])
        self.assertEqual(result["actual_merges"], packed["actual_merges"])
        self.assertEqual(sum(result["worker_merge_counts"]),
                         result["actual_merges"])
        self.assertEqual(sum(result["shard_pieces"]),
                         sum(value == 0 for value in prepared[0]) - 1)
        self.assertTrue(all(0 <= rounds <= result["rules"]
                            for rounds in result["active_worker_rounds"]))
        self.assertEqual(result["messages_per_round"],
                         0 if serial else 2 * result["actual_workers"])
        return result

    def test_random_one_two_four_workers(self):
        rng = random.Random(9301)
        pieces = ["".join(rng.choices("abcde", k=7 + i % 9))
                  for i in range(12)]
        for count in (1, 2, 4):
            result = self.assert_matches_oracles(pieces, count)
            self.assertEqual(result["actual_workers"], count)
            self.assertEqual(len(result["worker_merge_cpu_seconds"]), count)
            self.assertEqual(len(result["worker_peak_rss_mib"]), count)

    def test_weighted_overlap_and_three_workers(self):
        pieces = (["aaaaaa"] * 4 + ["abababa"] * 3 +
                  ["babab"] * 2 + ["abcabc", "cabcab", "ba"])
        serial = self.assert_matches_oracles(pieces, 3, serial=True)
        spawned = self.assert_matches_oracles(pieces, 3)
        self.assertEqual(serial["fingerprint"], spawned["fingerprint"])
        self.assertEqual(spawned["round_messages"],
                         2 * spawned["actual_workers"] * spawned["rules"])

    def test_whole_piece_boundaries_and_no_dedup(self):
        pieces = ["abcabc", "a", "ba", "c" * 8, "ab" * 4,
                  "xyzxyz", "baba", "aca", "dddd", "abc"]
        result = self.assert_matches_oracles(pieces, 4, deduplicate=False)
        self.assertEqual(sum(result["shard_pieces"]), len(pieces))
        self.assertTrue(all(count > 0 for count in result["shard_pieces"]))
        self.assertEqual(sum(result["shard_chars"]), sum(map(len, pieces)))

    def test_empty(self):
        for pieces in ([], [""]):
            for serial in (False, True):
                result = self.assert_matches_oracles(pieces, 4, serial=serial)
                self.assertEqual(result["actual_workers"], 0)
                self.assertEqual(result["rules"], 0)


if __name__ == "__main__":
    unittest.main()

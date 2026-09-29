"""Bounded differential checks for the global occurrence arena."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_rewrite"))

from backends_fused import FusedEbpeEndpoints, FusedPrezzaHalfword  # noqa: E402
from backend_linked_fused import FusedLinked12  # noqa: E402
from common_fused import naive, prepare as baseline_prepare  # noqa: E402
from common_fused import train as baseline_train  # noqa: E402
from arena_driver import prepare, train  # noqa: E402
from arena_counted_driver import train as counted_train  # noqa: E402


class ArenaDriverTests(unittest.TestCase):
    def check_case(self, words, max_merges=40, min_frequency=1,
                   deduplicate=True, weight_scale=1):
        prepared = prepare(words, deduplicate=deduplicate)
        self.assertEqual(prepared, baseline_prepare(words, deduplicate=deduplicate))
        if weight_scale != 1:
            prepared[3][:] = [weight * weight_scale for weight in prepared[3]]
            min_frequency *= weight_scale
        expected_rules, expected_final = naive(prepared, max_merges, min_frequency)
        for backend in (FusedEbpeEndpoints, FusedPrezzaHalfword, FusedLinked12):
            baseline = baseline_train(prepared, backend, max_merges,
                                      min_frequency, capture=True)
            for trainer in (train, counted_train):
                with self.subTest(backend=backend.__name__,
                                  trainer=trainer.__module__, words=words[:2]):
                    result = trainer(prepared, backend, max_merges,
                                     min_frequency, capture=True)
                    self.assertEqual(result["merges"], expected_rules)
                    self.assertEqual(result["final"], expected_final)
                    self.assertEqual(result["fingerprint"], baseline["fingerprint"])
                    for field in ("rules", "actual_merges", "position_visits",
                                  "stale_visits", "heap_pops", "max_token_length"):
                        self.assertEqual(result[field], baseline[field], field)
                    self.assertGreaterEqual(result["arena_occurrence_capacity_bytes"],
                                            result["arena_occurrence_logical_bytes"])
                    self.assertGreaterEqual(result["arena_state_capacity_bytes"],
                                            result["arena_state_logical_bytes"])
        return result

    def test_random_oracle_and_baseline(self):
        rng = random.Random(2917)
        alphabet = "abcde"
        for _ in range(60):
            words = ["".join(rng.choices(alphabet, k=rng.randrange(1, 18)))
                     for _ in range(rng.randrange(1, 7))]
            words += rng.choices(words, k=2)
            self.check_case(words, max_merges=30,
                            min_frequency=rng.choice((1, 2, 3)))

    def test_self_overlap_and_high_weight(self):
        for words in (["aaaaaaa", "aaaa", "aa"],
                      ["ababababa", "babababa", "aaaabaaa"],
                      ["你好你好你好", "你好", "好好好好"]):
            self.check_case(words, 30, 1)
        # Frequencies exceed uint64; the state list intentionally uses Python int.
        self.check_case(["aaaabaaa", "aaaabaaa", "aaaacaaa"], 30, 2,
                        weight_scale=10**20)

    def test_stale_positions_and_pool_reuse(self):
        words = ["ab" * 24, "baba" * 13, "a" * 45 + "b" + "a" * 31]
        result = self.check_case(words, 40, 1)
        self.assertGreater(result["stale_visits"], 0)
        self.assertGreater(result["arena_occurrence_reuses"], 0)
        self.assertGreater(result["arena_state_reuses"], 0)
        self.assertLessEqual(result["arena_occurrence_active"],
                             result["arena_occurrence_high_water"])

    def test_empty_and_no_eligible_pair(self):
        for words in ([], [""], ["", ""]):
            prepared = prepare(words)
            for trainer in (train, counted_train):
                result = trainer(prepared, FusedEbpeEndpoints, capture=True)
                self.assertEqual((result["merges"], result["final"]), ([], [0]))
                self.assertEqual(result["rules"], 0)
        self.check_case(["a"], 10, 2)
        self.check_case(["abcd"], 10, 100)

    def test_rare_initial_pairs_do_not_allocate_counted_states(self):
        rare = "".join(chr(0x1000 + i) for i in range(100))
        prepared = prepare(["aa", "aa", "aa", rare])
        original = train(prepared, FusedLinked12, max_merges=0,
                         min_frequency=2, capture=True)
        counted = counted_train(prepared, FusedLinked12, max_merges=0,
                                min_frequency=2, capture=True)
        self.assertEqual(counted["fingerprint"], original["fingerprint"])
        self.assertEqual(counted["arena_state_high_water"], 1)
        self.assertEqual(original["arena_state_high_water"], 100)
        self.assertEqual(counted["initial_occurrence_bytes"], 8)
        self.assertEqual(counted["initial_unfiltered_offset_bytes"], 400)
        self.assertEqual(original["initial_occurrence_bytes"], 8)


if __name__ == "__main__":
    unittest.main()

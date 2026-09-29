"""Small correctness checks for LeanEndpoints' historical-start contract."""

from pathlib import Path
import random
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "python_rewrite"))
sys.path.insert(0, str(BASE / "evolution"))

from common_fused import naive, prepare, train  # noqa: E402
from lean_backend import LeanEndpoints  # noqa: E402


def check_history(test, initial_ids, seed, max_steps=80):
    """Exercise arbitrary valid merges while querying only recorded pair starts."""
    backend = LeanEndpoints(initial_ids)
    last = len(initial_ids) - 1
    live = list(range(1, last))
    ids = {pos: initial_ids[pos] for pos in live}
    history = set()
    rng = random.Random(seed)

    for _ in range(max_steps):
        # These are exactly the positions that a pair-position index may have
        # learned: true starts of adjacent, nonseparator tokens at some time.
        for i in range(len(live) - 1):
            left, right = live[i], live[i + 1]
            a, b = ids[left], ids[right]
            if a and b:
                history.add((left, a, b))

        live_index = {pos: i for i, pos in enumerate(live)}
        for pos, a, b in history:
            i = live_index.get(pos)
            valid = i is not None and ids[pos] == a and i + 1 < len(live) and ids[live[i + 1]] == b
            context = backend.inspect_pair(pos, a, b)
            test.assertEqual(context is not None, valid, (seed, pos, a, b, context, valid))

        for i, pos in enumerate(live):
            test.assertEqual(backend.token(pos), ids[pos])
            test.assertEqual(backend.prev(pos), live[i - 1] if i else 0)
            test.assertEqual(backend.next(pos), live[i + 1] if i + 1 < len(live) else last)

        mergeable = [i for i in range(len(live) - 1)
                     if ids[live[i]] != 0 and ids[live[i + 1]] != 0]
        if not mergeable:
            break

        i = rng.choice(mergeable)
        pos, right = live[i], live[i + 1]
        a, b = ids[pos], ids[right]
        before = live[i - 1] if i else 0
        after = live[i + 2] if i + 2 < len(live) else last
        expected = (before, ids.get(before, 0), right, after, ids.get(after, 0))
        context = backend.inspect_pair(pos, a, b)
        test.assertEqual(context, expected, (seed, pos, a, b, context, expected))

        new_id = len(backend.token_len)
        new_len = backend.token_len[a] + backend.token_len[b]
        backend.token_len.append(new_len)
        test.assertEqual(backend.merge_known(pos, right, after, new_id, new_len), right)
        ids[pos] = new_id
        del ids[right]
        del live[i + 1]

    # Include the pairs formed by the final merge before one last stale check.
    for i in range(len(live) - 1):
        a, b = ids[live[i]], ids[live[i + 1]]
        if a and b:
            history.add((live[i], a, b))
    live_index = {pos: i for i, pos in enumerate(live)}
    for pos, a, b in history:
        i = live_index.get(pos)
        valid = i is not None and ids[pos] == a and i + 1 < len(live) and ids[live[i + 1]] == b
        test.assertEqual(backend.inspect_pair(pos, a, b) is not None, valid,
                         (seed, pos, a, b, valid))
    return backend, live, ids, history


class LeanEndpointTests(unittest.TestCase):
    def test_random_training_matches_full_recount(self):
        # Includes weighted duplicate pieces, varied overlaps, and fresh IDs.
        for seed in range(30):
            rng = random.Random(seed)
            alphabet = "abcd"
            pieces = ["".join(rng.choice(alphabet) for _ in range(rng.randint(2, 10)))
                      for _ in range(rng.randint(4, 18))]
            if seed % 3 == 0:
                pieces.extend(pieces[:3])
            prepared = prepare(pieces, deduplicate=bool(seed % 2))
            max_merges = 45
            min_frequency = 1 + seed % 3
            expected_merges, expected_final = naive(prepared, max_merges, min_frequency)
            actual = train(prepared, LeanEndpoints, max_merges, min_frequency, capture=True)
            self.assertEqual(actual["merges"], expected_merges, seed)
            self.assertEqual(actual["final"], expected_final, seed)

    def test_self_overlap_aaa_and_repeated_runs(self):
        for pieces in (["aaa"], ["aaaa", "aaa", "aaaaa"], ["aaaaaa"] * 4):
            prepared = prepare(pieces)
            expected_merges, expected_final = naive(prepared, 32, 1)
            actual = train(prepared, LeanEndpoints, 32, 1, capture=True)
            self.assertEqual(actual["merges"], expected_merges, pieces)
            self.assertEqual(actual["final"], expected_final, pieces)

    def test_history_only_starts_survive_random_merges(self):
        # Cover many repeated symbols, overlapping aaa occurrences, and a
        # range of endpoint distances while retaining only historical starts.
        for n in (2, 3, 4, 7, 15, 31, 64, 65, 129):
            for seed in range(5):
                rng = random.Random(1000 * n + seed)
                alphabet_size = rng.randint(2, 6)
                body = [rng.randint(1, alphabet_size) for _ in range(n)]
                check_history(self, [0, *body, 0], seed)

        check_history(self, [0, 1, 1, 1, 0], 73)

    def test_two_long_tokens_merge_and_keep_only_indexed_occurrences(self):
        backend = LeanEndpoints([0, 1, 2, 3, 4, 5, 0])
        history = {(1, 1, 2), (3, 3, 4), (1, 5, 3)}

        # First make a length-2 token on each side of the later merge.
        for pos, right, after, new_id, new_len in (
            (1, 2, 3, 6, 2),
            (3, 4, 5, 7, 2),
        ):
            backend.token_len.append(new_len)
            backend.merge_known(pos, right, after, new_id, new_len)

        context = backend.inspect_pair(1, 6, 7)
        self.assertEqual(context, (0, 0, 3, 5, 5))
        new_id, new_len = len(backend.token_len), 4
        backend.token_len.append(new_len)
        backend.merge_known(1, 3, 5, new_id, new_len)
        self.assertEqual(backend.token(1), 8)
        self.assertEqual(backend.next(1), 5)
        self.assertEqual(backend.prev(5), 1)
        # The old left endpoint was intentionally left behind as stale data.
        self.assertEqual(backend.corpus[2], 6)
        self.assertEqual(backend.inspect_pair(3, 7, 5), None)

    def test_endpoint_inspection_is_not_a_general_alive_check(self):
        backend = LeanEndpoints([0, 1, 2, 3, 4, 0])
        # Merge (1, 2) to token 5, then merge token 5 with 3 to token 6.
        # Position 2 retains token 5 as the old left endpoint of the latter
        # merge. It was never a pair start for (5, 4), although an unrestricted
        # endpoint inspection can see those IDs at its computed offsets.
        backend.token_len.extend((2, 3))
        backend.merge_known(1, 2, 3, 5, 2)
        backend.merge_known(1, 3, 4, 6, 3)
        historical = {(1, 1, 2), (1, 5, 3), (2, 2, 3), (3, 3, 4)}
        self.assertNotIn((2, 5, 4), historical)
        self.assertEqual(backend.corpus[2], 5)
        # This is an intentional false positive outside the occurrence-index
        # contract. Correct training never asks inspect_pair for this position.
        self.assertIsNotNone(backend.inspect_pair(2, 5, 4))

    def test_fresh_id_crosses_65535(self):
        lengths = [1] * 65536
        backend = LeanEndpoints([0, 65535, 17, 0], lengths)
        context = backend.inspect_pair(1, 65535, 17)
        self.assertEqual(context, (0, 0, 2, 3, 0))
        new_id = len(lengths)
        self.assertEqual(new_id, 65536)
        lengths.append(2)
        backend.merge_known(1, 2, 3, new_id, 2)
        self.assertEqual(backend.token(1), 65536)
        self.assertEqual(backend.corpus[2], 65536)
        self.assertEqual(backend.next(1), 3)
        self.assertEqual(backend.prev(3), 1)


if __name__ == "__main__":
    unittest.main()

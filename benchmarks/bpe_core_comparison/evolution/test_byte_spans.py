"""Correctness tests for ByteSpans against an independent ordered-span model."""

import random
import unittest

from byte_spans import ByteSpans


def starts_from_lengths(lengths):
    result = []
    pos = 0
    for span in lengths:
        result.append(pos)
        pos += span
    return result


def assert_matches_model(test, spans, lengths, historical_starts=()):
    starts = starts_from_lengths(lengths)
    by_start = dict(zip(starts, lengths))
    test.assertEqual(sum(lengths), len(spans))
    for pos in range(len(spans)):
        test.assertEqual(spans.length(pos), by_start.get(pos, 0), (pos, lengths))
    for index, pos in enumerate(starts):
        test.assertEqual(spans.next(pos), starts[index + 1] if index + 1 < len(starts) else len(spans))
        if index:
            test.assertEqual(spans.prev(pos), starts[index - 1])
    for pos in historical_starts:
        if pos not in by_start:
            test.assertEqual(spans.length(pos), 0, ("stale start", pos, lengths))


def merge_model(spans, lengths, index):
    starts = starts_from_lengths(lengths)
    pos, right = starts[index], starts[index + 1]
    after = starts[index + 2] if index + 2 < len(starts) else len(spans)
    spans.merge(pos, right, after)
    lengths[index:index + 2] = [lengths[index] + lengths[index + 1]]


class ByteSpanTests(unittest.TestCase):
    def test_single_and_short_long_encoding_boundaries(self):
        for target in (1, 2, 62, 63, 64, 255, 256, 65535, 65536):
            with self.subTest(length=target):
                spans = ByteSpans(target + 1)
                # Fold a prefix into one token of exactly target bytes.
                for right in range(1, target):
                    spans.merge(0, right, right + 1)
                lengths = [target, 1]
                self.assertEqual(spans.length(0), target)
                self.assertEqual(spans.next(0), target)
                self.assertEqual(spans.prev(target), 0)
                self.assertEqual(spans.memory_bytes(), target + 1)
                assert_matches_model(self, spans, lengths)

    def test_left_and_right_long_chain_merges(self):
        for direction in ("left", "right"):
            spans = ByteSpans(257)
            lengths = [1] * 257
            while len(lengths) > 1:
                index = 0 if direction == "left" else len(lengths) - 2
                merge_model(spans, lengths, index)
            self.assertEqual(lengths, [257])
            self.assertEqual(spans.length(0), 257)
            self.assertEqual(spans.next(0), 257)
            self.assertEqual(spans.memory_bytes(), 257)
            assert_matches_model(self, spans, lengths)

    def test_random_merges_against_naive_live_boundary_oracle(self):
        for seed in range(40):
            rng = random.Random(seed)
            n = rng.randint(2, 180)
            spans = ByteSpans(n)
            lengths = [1] * n
            historical_starts = set(range(n))
            while len(lengths) > 1:
                merge_model(spans, lengths, rng.randrange(len(lengths) - 1))
                historical_starts.update(starts_from_lengths(lengths))
                assert_matches_model(self, spans, lengths, historical_starts)

    def test_payload_may_overwrite_a_consumed_start_but_is_not_a_start(self):
        # Build a 63-byte right span, then merge a singleton on its left.
        # The first long-start payload byte lands at the consumed right start.
        spans = ByteSpans(64)
        lengths = [1] * 64
        for right in range(2, 64):
            spans.merge(1, right, right + 1)
            lengths[1:right + 1] = [right]
        self.assertEqual(spans.length(1), 63)
        self.assertEqual(spans.length(0), 1)
        spans.merge(0, 1, 64)
        lengths[:2] = [64]
        self.assertGreaterEqual(spans._data[1], 128)
        self.assertEqual(spans.length(1), 0)
        assert_matches_model(self, spans, lengths, range(64))

    def test_merge_rejects_nonadjacent_and_nonlive_boundaries(self):
        spans = ByteSpans(8)
        with self.assertRaises(ValueError):
            spans.merge(0, 2, 3)
        spans.merge(0, 1, 2)
        with self.assertRaises(ValueError):
            spans.merge(1, 2, 3)
        with self.assertRaises(ValueError):
            spans.merge(0, 2, 4)

    def test_exactly_one_byte_per_position(self):
        for n in (0, 1, 2, 63, 64, 4097):
            spans = ByteSpans(n)
            self.assertEqual(spans.memory_bytes(), n)
            self.assertEqual(len(spans._data), n)


if __name__ == "__main__":
    unittest.main()

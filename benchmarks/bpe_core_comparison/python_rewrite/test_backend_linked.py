import unittest

from backend_linked import (
    CompactLinkedBackend,
    FastCompactLinkedBackend,
    FastLinkedBackend,
    LinkedBackend,
)


class LinkedBackendTest(unittest.TestCase):
    def test_both_layouts_preserve_neighbors_and_stale_occurrences(self):
        # Two segments with live zero sentinels, then an EOF sentinel.
        initial = [0, 1, 1, 1, 0, 2, 2, 0]
        lengths = [0, 1, 1, 2, 3]
        for cls in (
            LinkedBackend,
            CompactLinkedBackend,
            FastLinkedBackend,
            FastCompactLinkedBackend,
        ):
            with self.subTest(cls=cls.__name__):
                backend = cls(initial, lengths)
                self.assertIsNone(backend.prev(0))
                self.assertIsNone(backend.next(len(initial) - 1))
                self.assertTrue(backend.alive(4))  # Live separator has val 0.
                self.assertFalse(backend.pair_matches(3, 1, 2))
                self.assertFalse(backend.pair_matches(4, 0, 2))

                self.assertTrue(backend.pair_matches(1, 1, 1))
                self.assertEqual(backend.merge(1, 3, 2), 2)
                self.assertEqual(backend.token(1), 3)
                self.assertEqual(backend.next(1), 3)
                self.assertEqual(backend.prev(3), 1)
                self.assertFalse(backend.alive(2))
                self.assertFalse(backend.pair_matches(2, 1, 1))
                self.assertFalse(backend.pair_matches(1, 1, 1))

                self.assertTrue(backend.pair_matches(1, 3, 1))
                self.assertEqual(backend.merge(1, 4, 3), 3)
                self.assertEqual(backend.next(1), 4)
                self.assertEqual(backend.prev(4), 1)
                self.assertEqual(backend.token(4), 0)

                bytes_per_pos = 16 if backend.runlength is not None else 12
                self.assertEqual(backend.logical_bytes(), len(initial) * bytes_per_pos)
                self.assertGreaterEqual(backend.memory_bytes(), backend.logical_bytes())
                self.assertGreater(backend.memory_headers_bytes(), 0)

    def test_pair_validation_rejects_separator(self):
        backend = CompactLinkedBackend([0, 1, 0], [0, 1, 2])
        self.assertFalse(backend.pair_matches(1, 1, 0))
        self.assertTrue(backend.alive(2))

    def test_self_pair_positions_merge_without_overlap(self):
        for cls in (CompactLinkedBackend, FastCompactLinkedBackend):
            with self.subTest(cls=cls.__name__):
                backend = cls([0, 1, 1, 1, 1, 0], [0, 1, 2])
                self.assertTrue(backend.pair_matches(1, 1, 1))
                backend.merge(1, 2, 2)
                self.assertFalse(backend.pair_matches(2, 1, 1))
                self.assertTrue(backend.pair_matches(3, 1, 1))
                backend.merge(3, 2, 2)
                self.assertTrue(backend.pair_matches(1, 2, 2))


if __name__ == "__main__":
    unittest.main()

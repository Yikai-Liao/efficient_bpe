import random
import unittest

from common import prepare, train, naive
from backends_compact import EbpeEndpoints, PrezzaBitmap
from backend_linked import LinkedBackend, CompactLinkedBackend, FastLinkedBackend, FastCompactLinkedBackend


class TrainingTest(unittest.TestCase):
    def test_random_and_pathological(self):
        rng = random.Random(19990503)
        cases = [['aaa'], ['abababab'], ['abcabc', 'abcabc', 'abd'],
                 ['a'*1024], ['a'*64+'b'+'a'*130], ['x'], []]
        for _ in range(600):
            words = [''.join(rng.choices('abcde', k=rng.randrange(1, 40)))
                     for _ in range(rng.randrange(1, 14))]
            cases.append(words + rng.choices(words, k=rng.randrange(8)))
        classes = [EbpeEndpoints, PrezzaBitmap, LinkedBackend, CompactLinkedBackend,
                   FastLinkedBackend, FastCompactLinkedBackend]
        try:
            from backends_compact import FastEbpeEndpoints, FastPrezzaBitmap, FastPrezzaHalfword
            classes += [FastEbpeEndpoints, FastPrezzaBitmap, FastPrezzaHalfword]
        except ImportError:
            pass
        for words in cases:
            prepared = prepare(words)
            if len(prepared[0]) < 2:
                continue
            expected_merges, expected_final = naive(prepared, 100)
            for cls in classes:
                with self.subTest(cls=cls.__name__, words=words[:2]):
                    got = train(prepared, cls, 100, capture=True)
                    self.assertEqual(got['merges'], expected_merges)
                    self.assertEqual(got['final'], expected_final)


if __name__ == '__main__':
    unittest.main()

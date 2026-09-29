"""Compare fast backends with common.py's full-recount oracle."""

from backends_compact import FastEbpeEndpoints, FastPrezzaBitmap, FastPrezzaHalfword
from common import naive, prepare, train


cases = [
    ["abab", "aaaa", "abcabc", "aaaa"],
    ["a" * 72, "babababababa", "abac"],
    ["你好你好你好", "ab" * 42, "b" * 130],
]
checks = 0
for pieces in cases:
    prepared = prepare(pieces)
    expected_merges, expected_final = naive(prepared, max_merges=40, min_frequency=1)
    for backend in (FastEbpeEndpoints, FastPrezzaBitmap, FastPrezzaHalfword):
        actual = train(prepared, backend, max_merges=40, min_frequency=1, capture=True)
        assert actual["merges"] == expected_merges, (backend.__name__, pieces, "merges")
        assert actual["final"] == expected_final, (backend.__name__, pieces, "final")
        checks += 1
print({"naive_backend_case_checks": checks})

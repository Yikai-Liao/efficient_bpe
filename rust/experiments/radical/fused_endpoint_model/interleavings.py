"""Exhaustive SC schedules for the local fused-endpoint neighbor readers.

This is deliberately not a Rust memory-model test. The accompanying review
states the separate release/acquire obligation for a clear followed by reread.
"""

from __future__ import annotations

HEAD = 1 << 31
MASK = HEAD - 1
FRESH = 1000
K, L, A, B, C, D, E = 10, 11, 12, 13, 14, 15, 16


def old(raw: int, rules: dict[int, tuple[int, int]]) -> int:
    value = raw & MASK
    if value < FRESH:
        return value
    left, right = rules[value]
    return left if raw & HEAD else right


def schedules(initial, writes, read, expected):
    """Enumerate writer-prefix/read interleavings, preserving writer order."""
    count = 0

    def visit(cells, wi, stage, observations):
        nonlocal count
        if stage == "done":
            assert observations == expected, (initial, writes, observations, expected)
            count += 1
            return
        if wi < len(writes):
            cell, value = writes[wi]
            changed = dict(cells)
            changed[cell] = value
            visit(changed, wi + 1, stage, observations)
        next_stage, result = read(stage, cells, observations)
        visit(cells, wi, next_stage, result)

    visit(dict(initial), 0, "first", None)
    return count


def right_reader(rules, selected):
    def read(stage, cells, remembered):
        if stage == "first":
            raw = cells["t"]
            c = old(raw, rules)
            if (raw & MASK) >= FRESH and raw & HEAD:
                return "done", (c, selected[(C, D)])
            return "second", c
        if stage == "second":
            raw = cells["u"]
            if raw == 0:
                return "reread", remembered
            d = old(raw, rules)
            return "done", (remembered, selected.get((remembered, d), remembered))
        raw = cells["t"]
        if (raw & MASK) >= FRESH and raw & HEAD:
            return "done", (remembered, selected[(C, D)])
        # u is the piece boundary after C. C remains our right neighbor.
        return "done", (remembered, remembered)

    return read


def left_reader(rules, selected, k_end, l_end):
    def read(stage, cells, remembered):
        if stage == "first":
            return "second", old(cells[l_end], rules)
        k = old(cells[k_end], rules)
        return "done", (remembered, (k, remembered) in selected)

    return read


def main():
    cases = 0
    schedules_checked = 0
    for right_length in (1, 2, 257):
        right_writes = [("t", FRESH | HEAD)]
        if right_length == 1:
            right_writes.append(("u", FRESH))
        else:
            right_writes.extend([("u", 0), ("d_end", FRESH)])
        rules = {FRESH: (C, D)}
        schedules_checked += schedules(
            {"t": C | HEAD, "u": D | HEAD, "d_end": D},
            right_writes,
            right_reader(rules, {(C, D): FRESH}),
            (C, FRESH),
        )
        cases += 1

    # D starts another selected pair (D,E): u may be a fresh HEAD, but C
    # remains our old right neighbor because (C,D) is not selected.
    schedules_checked += schedules(
        {"t": C | HEAD, "u": D | HEAD, "e_start": E | HEAD},
        [("u", FRESH | HEAD), ("e_start", FRESH)],
        right_reader({FRESH: (D, E)}, {(D, E): FRESH}),
        (C, C),
    )
    cases += 1
    # End-of-piece sentinel remains zero; reread must retain old C.
    schedules_checked += schedules(
        {"t": C | HEAD, "u": 0},
        [],
        right_reader({}, {}),
        (C, C),
    )
    cases += 1

    for left_length in (1, 2, 257):
        for k_length in (1, 2, 257):
            cells = {"k_start": K | HEAD, "k_end": K, "l_start": L | HEAD, "l_end": L}
            if k_length == 1:
                cells["k_end"] = cells["k_start"]
                k_end = "k_start"
            else:
                k_end = "k_end"
            if left_length == 1:
                cells["l_end"] = cells["l_start"]
                l_end = "l_start"
                writes = [("k_start", FRESH | HEAD), ("l_start", FRESH)]
            else:
                l_end = "l_end"
                writes = [
                    ("k_start", FRESH | HEAD),
                    ("l_start", 0),
                    ("l_end", FRESH),
                ]
            schedules_checked += schedules(
                cells,
                writes,
                left_reader({FRESH: (K, L)}, {(K, L)}, k_end, l_end),
                (L, True),
            )
            cases += 1

    # K, the token just before L, may instead be the right constituent of
    # an outward selected (X,K). Its end becomes a bare fresh tail, but
    # selected[(K,L)] is false and the left birth must remain unsuppressed.
    for k_length in (1, 2, 257):
        cells = {"x_start": 20 | HEAD, "k_start": K | HEAD, "k_end": K, "l_end": L}
        if k_length == 1:
            k_end = "k_start"
            writes = [("x_start", FRESH | HEAD), ("k_start", FRESH)]
        else:
            k_end = "k_end"
            writes = [("x_start", FRESH | HEAD), ("k_start", 0), ("k_end", FRESH)]
        schedules_checked += schedules(
            cells,
            writes,
            left_reader({FRESH: (20, K)}, {(20, K)}, k_end, "l_end"),
            (L, False),
        )
        cases += 1

    # Writes never restore a selected old ID at a stale posting's p or q.
    for value in (0, FRESH, FRESH | HEAD):
        assert (value & MASK) not in (A, B)
    print(f"{cases} local cases, {schedules_checked} SC read/write prefixes checked")


if __name__ == "__main__":
    main()

"""Arena trainer that allocates initial pair states after frequency filtering.

This is an ablation of ``arena_driver``: only initial counting changes. The
merge loop and result fields retain its semantics and instrumentation.
"""

from array import array
from bisect import bisect_right
import hashlib
import heapq
import json
import time

try:
    from .arena_driver import _PairArena, _array_bytes, _NIL, _LOW_MASK, prepare
    from .support import empty_result
except ImportError:  # Direct execution from the evolution directory.
    from arena_driver import _PairArena, _array_bytes, _NIL, _LOW_MASK, prepare
    from support import empty_result


def train(prepared, backend_class, max_merges=1000, min_frequency=2,
          capture=False):
    """Train with the common_fused rule trace and a global occurrence pool."""
    corpus, initial_lengths, pivots, weights = prepared
    if len(corpus) == 1:
        return empty_result(capture)
    if len(corpus) - 1 >= _NIL:
        raise OverflowError("corpus positions exceed u32 occurrence offsets")
    start = time.perf_counter()
    cpu_start = time.process_time()
    token_len = initial_lengths.copy()
    backend = backend_class(corpus, token_len)
    arena = _PairArena()

    # Count into a short-lived map, then allocate states only for viable pairs.
    # Old IDs cannot form new old-ID pairs later, so rare initial pairs cannot
    # become eligible after this point.
    initial_counts = {}
    weight_i = 0
    initial_occurrences = 0
    for pos in range(1, len(corpus) - 1):
        while weight_i + 1 < len(pivots) and pos >= pivots[weight_i + 1]:
            weight_i += 1
        a, b = corpus[pos], corpus[pos + 1]
        if a and b:
            key = (a << 32) | b
            initial_counts[key] = initial_counts.get(key, 0) + weights[weight_i]
            initial_occurrences += 1

    for key, frequency in initial_counts.items():
        if frequency >= min_frequency:
            slot, _ = arena.state(key)
            arena.freq[slot] = frequency
    del initial_counts

    for pos in range(1, len(corpus) - 1):
        a, b = corpus[pos], corpus[pos + 1]
        if a and b:
            slot = arena.state_of.get((a << 32) | b)
            if slot is not None:
                arena.append(slot, pos)

    heap = [(-arena.freq[slot], key) for key, slot in arena.state_of.items()]
    heapq.heapify(heap)
    initialized = time.perf_counter()
    cpu_initialized = time.process_time()
    initial_position_bytes = len(arena.pos) * (arena.pos.itemsize + arena.next.itemsize)
    initial_unfiltered_offset_bytes = initial_occurrences * array("I").itemsize

    merges = []
    actual_merges = position_visits = stale_visits = heap_pops = 0
    for _ in range(max_merges):
        while heap:
            cached, key = heapq.heappop(heap)
            heap_pops += 1
            slot = arena.state_of.get(key)
            if slot is None:
                continue
            current = arena.freq[slot]
            if current < min_frequency:
                arena.discard(key, slot)
                continue
            if -cached != current:
                heapq.heappush(heap, (-current, key))
                continue
            break
        else:
            break

        a, b = key >> 32, key & _LOW_MASK
        new_id = len(token_len)
        if new_id > _LOW_MASK:
            raise OverflowError("token ID exceeds u32 pair-key encoding")
        length = token_len[a] + token_len[b]
        token_len.append(length)
        merges.append((a, b, current))
        node = arena.head[slot]
        arena.head[slot] = arena.tail[slot] = _NIL
        fresh_states = []  # (packed pair, slot), first-seen order only

        while node != _NIL:
            following = arena.next[node]
            pos = arena.pos[node]
            arena.recycle_node(node)
            node = following
            position_visits += 1
            context = backend.inspect_pair(pos, a, b)
            if context is None:
                stale_visits += 1
                continue
            before, left_id, right, after, right_id = context
            weight = weights[bisect_right(pivots, pos) - 1]
            arena.freq[slot] -= weight
            if left_id:
                old_slot = arena.state_of.get((left_id << 32) | a)
                if old_slot is not None:
                    arena.freq[old_slot] -= weight
            if right_id:
                old_slot = arena.state_of.get((b << 32) | right_id)
                if old_slot is not None:
                    arena.freq[old_slot] -= weight
            backend.merge_known(pos, right, after, new_id, length)
            actual_merges += 1
            if left_id:
                new_key = (left_id << 32) | new_id
                new_slot, created = arena.state(new_key)
                if created:
                    fresh_states.append((new_key, new_slot))
                arena.freq[new_slot] += weight
                arena.append(new_slot, before)
            if right_id:
                new_key = (new_id << 32) | right_id
                new_slot, created = arena.state(new_key)
                if created:
                    fresh_states.append((new_key, new_slot))
                arena.freq[new_slot] += weight
                arena.append(new_slot, pos)

        for new_key, new_slot in fresh_states:
            if arena.freq[new_slot] >= min_frequency:
                heapq.heappush(heap, (-arena.freq[new_slot], new_key))
            else:
                arena.discard(new_key, new_slot)
        arena.discard(key, slot)

    finished = time.perf_counter()
    cpu_finished = time.process_time()
    # Match the baseline: extraction and hashing are outside training time.
    final = []
    pos = 0
    while pos is not None:
        final.append(backend.token(pos))
        pos = backend.next(pos)
    fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
    occurrence_logical, occurrence_capacity = _array_bytes((arena.pos, arena.next))
    state_logical, state_capacity = _array_bytes((arena.head, arena.tail))
    result = {
        "init_seconds": initialized - start,
        "merge_seconds": finished - initialized,
        "train_seconds": finished - start,
        "init_cpu_seconds": cpu_initialized - cpu_start,
        "merge_cpu_seconds": cpu_finished - cpu_initialized,
        "train_cpu_seconds": cpu_finished - cpu_start,
        "rules": len(merges),
        "actual_merges": actual_merges,
        "position_visits": position_visits,
        "stale_visits": stale_visits,
        "heap_pops": heap_pops,
        "fingerprint": fingerprint,
        "max_token_length": max(token_len),
        "corpus_positions": len(corpus),
        "backend_buffer_bytes": backend.memory_bytes(),
        "initial_occurrence_bytes": initial_position_bytes,
        "initial_unfiltered_offset_bytes": initial_unfiltered_offset_bytes,
        "arena_occurrence_logical_bytes": occurrence_logical,
        "arena_occurrence_capacity_bytes": occurrence_capacity,
        "arena_state_logical_bytes": state_logical,
        "arena_state_capacity_bytes": state_capacity,
        "arena_frequency_list_capacity_bytes": arena.freq.__sizeof__() - [].__sizeof__(),
        "arena_occurrence_high_water": len(arena.pos),
        "arena_occurrence_active": arena.active_occ,
        "arena_occurrence_reuses": arena.occ_reuses,
        "arena_state_high_water": len(arena.head),
        "arena_state_active": len(arena.state_of),
        "arena_state_reuses": arena.state_reuses,
    }
    if capture:
        result["merges"], result["final"] = merges, final
    return result


__all__ = ["prepare", "train"]

"""Packed keys plus two-pass indexing of initially eligible pairs only.

Fresh-ID pairs accumulate for a complete rule before testing min_frequency.
Previously excluded pairs cannot increase: every new edge involves the newly
minted ID. This is separate from the arena so filtering and pooling can be
measured independently. No global rescan occurs inside the merge loop.
"""
from array import array
from bisect import bisect_right
from collections import defaultdict
import hashlib
import heapq
import json
import time
from support import empty_result


def train(prepared, backend_class, max_merges=1000, min_frequency=2,
          capture=False):
    corpus, initial_lengths, pivots, weights = prepared
    if len(corpus) == 1:
        return empty_result(capture)
    if len(initial_lengths) + max_merges > 1 << 32:
        raise ValueError("packed pair keys require u32 token IDs")
    start = time.perf_counter()
    cpu_start = time.process_time()
    token_len = initial_lengths.copy()
    backend = backend_class(corpus, token_len)
    pair_pos = defaultdict(lambda: array('I'))
    frequencies = defaultdict(int)
    weight_i = 0
    # Count before allocating occurrence arrays: old pairs can only decrease.
    for pos in range(1, len(corpus) - 1):
        while weight_i + 1 < len(pivots) and pos >= pivots[weight_i + 1]:
            weight_i += 1
        a, b = corpus[pos], corpus[pos + 1]
        if a and b:
            frequencies[(a << 32) | b] += weights[weight_i]
    for pair in list(frequencies):
        if frequencies[pair] < min_frequency:
            del frequencies[pair]
    for pos in range(1, len(corpus) - 1):
        a, b = corpus[pos], corpus[pos + 1]
        if a and b:
            pair = (a << 32) | b
            if pair in frequencies:
                pair_pos[pair].append(pos)
    heap = [(-f, p) for p, f in frequencies.items() if f >= min_frequency]
    heapq.heapify(heap)
    initialized = time.perf_counter()
    cpu_initialized = time.process_time()
    initial_position_bytes = sum(v.itemsize * len(v) for v in pair_pos.values())
    merges = []
    actual_merges = position_visits = stale_visits = heap_pops = 0
    for _ in range(max_merges):
        while heap:
            cached, pair = heapq.heappop(heap)
            heap_pops += 1
            current = frequencies.get(pair, 0)
            if current < min_frequency:
                pair_pos.pop(pair, None)
                frequencies.pop(pair, None)
                continue
            if -cached != current:
                heapq.heappush(heap, (-current, pair))
                continue
            break
        else:
            break
        a, b = pair >> 32, pair & 0xFFFFFFFF
        new_id = len(token_len)
        length = token_len[a] + token_len[b]
        token_len.append(length)
        merges.append((a, b, current))
        positions = pair_pos.pop(pair)
        new_pairs = set()
        for pos in positions:
            position_visits += 1
            context = backend.inspect_pair(pos, a, b)
            if context is None:
                stale_visits += 1
                continue
            before, left_id, right, after, right_id = context
            weight = weights[bisect_right(pivots, pos) - 1]
            frequencies[pair] -= weight
            if left_id:
                old = (left_id << 32) | a
                if old in frequencies:
                    frequencies[old] -= weight
            if right_id:
                old = (b << 32) | right_id
                if old in frequencies:
                    frequencies[old] -= weight
            backend.merge_known(pos, right, after, new_id, length)
            actual_merges += 1
            if left_id:
                p = (left_id << 32) | new_id
                frequencies[p] += weight
                pair_pos[p].append(before)
                new_pairs.add(p)
            if right_id:
                p = (new_id << 32) | right_id
                frequencies[p] += weight
                pair_pos[p].append(pos)
                new_pairs.add(p)
        for p in new_pairs:
            if frequencies[p] >= min_frequency:
                heapq.heappush(heap, (-frequencies[p], p))
            else:
                pair_pos.pop(p, None)
                frequencies.pop(p, None)
        frequencies.pop(pair, None)
    finished = time.perf_counter()
    cpu_finished = time.process_time()
    # Extraction and hashing are deliberately outside the training interval.
    final = []
    pos = 0
    while pos is not None:
        final.append(backend.token(pos))
        pos = backend.next(pos)
    fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
    result = {
        'init_seconds': initialized - start,
        'merge_seconds': finished - initialized,
        'train_seconds': finished - start,
        'init_cpu_seconds': cpu_initialized - cpu_start,
        'merge_cpu_seconds': cpu_finished - cpu_initialized,
        'train_cpu_seconds': cpu_finished - cpu_start,
        'rules': len(merges), 'actual_merges': actual_merges,
        'position_visits': position_visits, 'stale_visits': stale_visits,
        'heap_pops': heap_pops, 'fingerprint': fingerprint,
        'max_token_length': max(token_len),
        'corpus_positions': len(corpus),
        'backend_buffer_bytes': backend.memory_bytes(),
        'initial_occurrence_bytes': initial_position_bytes,
    }
    if capture:
        result['merges'], result['final'] = merges, final
    return result

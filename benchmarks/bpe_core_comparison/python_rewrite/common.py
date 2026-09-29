"""Shared deterministic trainer for isolating sequence representations.

This is an experimental Python rewrite, not a full port of any library.
Every backend receives exactly the same weighted pieces, IDs, merge ordering,
occurrence index and lazy heap. Pair frequencies count overlapping adjacencies;
actual replacements proceed left-to-right. IDs are fresh per merge rule.
"""
from array import array
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import time


def prepare(pieces, deduplicate=True):
    counts = Counter(pieces) if deduplicate else [(p, 1) for p in pieces]
    items = list(counts.items()) if deduplicate else counts
    items.sort(key=lambda x: (-x[1], x[0]))
    alphabet = sorted({c for w, _ in items for c in w})
    ids = {c: i + 1 for i, c in enumerate(alphabet)}
    corpus = array('I', [0])
    pivots, weights = [], []
    last_weight = None
    for word, weight in items:
        if not word:
            continue
        if weight != last_weight:
            pivots.append(len(corpus))
            weights.append(weight)
            last_weight = weight
        corpus.extend(ids[c] for c in word)
        corpus.append(0)
    return corpus, [1] * (len(alphabet) + 1), pivots, weights


def train(prepared, backend_class, max_merges=1000, min_frequency=2,
          capture=False):
    corpus, initial_lengths, pivots, weights = prepared
    start = time.perf_counter()
    cpu_start = time.process_time()
    token_len = initial_lengths.copy()
    backend = backend_class(corpus, token_len)
    pair_pos = defaultdict(lambda: array('I'))
    frequencies = defaultdict(int)
    weight_i = 0
    for pos in range(1, len(corpus) - 1):
        while weight_i + 1 < len(pivots) and pos >= pivots[weight_i + 1]:
            weight_i += 1
        a, b = corpus[pos], corpus[pos + 1]
        if a and b:
            pair_pos[a, b].append(pos)
            frequencies[a, b] += weights[weight_i]
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
                continue
            if -cached != current:
                heapq.heappush(heap, (-current, pair))
                continue
            break
        else:
            break
        a, b = pair
        new_id = len(token_len)
        length = token_len[a] + token_len[b]
        token_len.append(length)
        merges.append((a, b, current))
        positions = pair_pos.pop(pair)
        new_pairs = set()
        for pos in positions:
            position_visits += 1
            if not backend.pair_matches(pos, a, b):
                stale_visits += 1
                continue
            right = backend.next(pos)
            before = backend.prev(pos)
            after = backend.next(right)
            left_id = backend.token(before) if before is not None else 0
            right_id = backend.token(after) if after is not None else 0
            weight = weights[bisect_right(pivots, pos) - 1]
            frequencies[a, b] -= weight
            if left_id:
                frequencies[left_id, a] -= weight
            if right_id:
                frequencies[b, right_id] -= weight
            backend.merge(pos, new_id, length)
            actual_merges += 1
            if left_id:
                p = left_id, new_id
                frequencies[p] += weight
                pair_pos[p].append(before)
                new_pairs.add(p)
            if right_id:
                p = new_id, right_id
                frequencies[p] += weight
                pair_pos[p].append(pos)
                new_pairs.add(p)
        for p in new_pairs:
            if frequencies[p] >= min_frequency:
                heapq.heappush(heap, (-frequencies[p], p))
            else:
                pair_pos.pop(p, None)
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


def naive(prepared, max_merges=1000, min_frequency=2):
    """Independent full-recount oracle, including weighted self overlaps."""
    corpus, initial_lengths, pivots, weights = prepared
    sequences = []
    start = 1
    for pos in range(1, len(corpus)):
        if corpus[pos] == 0:
            if pos > start:
                sequences.append((list(corpus[start:pos]),
                                  weights[bisect_right(pivots, start) - 1]))
            start = pos + 1
    merges = []
    for new_id in range(len(initial_lengths), len(initial_lengths) + max_merges):
        counts = Counter()
        for seq, weight in sequences:
            for p in zip(seq, seq[1:]):
                counts[p] += weight
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        frequency = counts[pair]
        if frequency < min_frequency:
            break
        merges.append((*pair, frequency))
        updated = []
        for seq, weight in sequences:
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i+1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            updated.append((out, weight))
        sequences = updated
    final = [0]
    for seq, _ in sequences:
        final.extend(seq)
        final.append(0)
    return merges, final

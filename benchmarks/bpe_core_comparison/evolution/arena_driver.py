"""Weighted BPE trainer with one pair-state table and a reusable occurrence arena.

The training semantics and result fields follow ``python_rewrite.common_fused``.
Pair keys are packed u32 IDs, so integer order matches lexicographic pair order.
The global occurrence pool uses two u32 arrays (position, next); a pair state
owns only head/tail indices, not a separate Python array object. Frequencies
remain Python integers to support weights above uint64 without wraparound.

Old pair frequencies can only decrease. A pair containing the current fresh
token ID is accumulated during that rule, queued once at the end, and never
re-added after selection. This is the reason low-frequency states may be freed.
"""

from array import array
from bisect import bisect_right
from collections import Counter
import hashlib
import heapq
import json
import time

try:
    from .support import empty_result
except ImportError:  # Direct execution from the evolution directory.
    from support import empty_result


_NIL = 0xFFFFFFFF
_LOW_MASK = 0xFFFFFFFF


def prepare(pieces, deduplicate=True):
    """Use the same flattened weighted pieces as common_fused.prepare."""
    counts = Counter(pieces) if deduplicate else [(p, 1) for p in pieces]
    items = list(counts.items()) if deduplicate else counts
    items.sort(key=lambda x: (-x[1], x[0]))
    alphabet = sorted({c for word, _ in items for c in word})
    ids = {char: i + 1 for i, char in enumerate(alphabet)}
    corpus = array("I", [0])
    pivots, weights = [], []
    last_weight = None
    for word, weight in items:
        if not word:
            continue
        if weight != last_weight:
            pivots.append(len(corpus))
            weights.append(weight)
            last_weight = weight
        corpus.extend(ids[char] for char in word)
        corpus.append(0)
    return corpus, [1] * (len(alphabet) + 1), pivots, weights


class _PairArena:
    """State slots and position chains; live chains preserve append order."""

    def __init__(self):
        if array("I").itemsize != 4:
            raise RuntimeError("occurrence arena requires 32-bit array('I')")
        self.state_of = {}       # packed pair -> state slot
        self.freq = []           # Python int: no silent uint64 overflow
        self.head = array("I")
        self.tail = array("I")
        self.free_states = []
        self.pos = array("I")
        self.next = array("I")
        self.free_occ = _NIL
        self.active_occ = 0
        self.occ_reuses = 0
        self.state_reuses = 0

    def state(self, key):
        slot = self.state_of.get(key)
        if slot is not None:
            return slot, False
        if self.free_states:
            slot = self.free_states.pop()
            self.freq[slot] = 0
            self.head[slot] = _NIL
            self.tail[slot] = _NIL
            self.state_reuses += 1
        else:
            slot = len(self.freq)
            if slot == _NIL:
                raise OverflowError("pair-state arena exceeds u32 indices")
            self.freq.append(0)
            self.head.append(_NIL)
            self.tail.append(_NIL)
        self.state_of[key] = slot
        return slot, True

    def append(self, slot, position):
        if self.free_occ != _NIL:
            node = self.free_occ
            self.free_occ = self.next[node]
            self.pos[node] = position
            self.next[node] = _NIL
            self.occ_reuses += 1
        else:
            node = len(self.pos)
            if node == _NIL:
                raise OverflowError("occurrence arena exceeds u32 indices")
            self.pos.append(position)
            self.next.append(_NIL)
        tail = self.tail[slot]
        if tail == _NIL:
            self.head[slot] = node
        else:
            self.next[tail] = node
        self.tail[slot] = node
        self.active_occ += 1

    def recycle_node(self, node):
        self.next[node] = self.free_occ
        self.free_occ = node
        self.active_occ -= 1

    def discard(self, key, slot):
        """Release a state after its selected chain is consumed, or if too rare."""
        node = self.head[slot]
        while node != _NIL:
            following = self.next[node]
            self.recycle_node(node)
            node = following
        del self.state_of[key]
        self.head[slot] = _NIL
        self.tail[slot] = _NIL
        self.freq[slot] = 0
        self.free_states.append(slot)


def _array_bytes(arrays):
    logical = sum(len(values) * values.itemsize for values in arrays)
    capacity = sum(values.__sizeof__() - array(values.typecode).__sizeof__()
                   for values in arrays)
    return logical, capacity


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

    # First count weighted pairs, then store positions only for candidates that
    # can ever reach the threshold. Old IDs cannot form new old-ID pairs later.
    weight_i = 0
    initial_occurrences = 0
    for pos in range(1, len(corpus) - 1):
        while weight_i + 1 < len(pivots) and pos >= pivots[weight_i + 1]:
            weight_i += 1
        a, b = corpus[pos], corpus[pos + 1]
        if a and b:
            key = (a << 32) | b
            slot, _ = arena.state(key)
            arena.freq[slot] += weights[weight_i]
            initial_occurrences += 1

    for key, slot in list(arena.state_of.items()):
        if arena.freq[slot] < min_frequency:
            arena.discard(key, slot)

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

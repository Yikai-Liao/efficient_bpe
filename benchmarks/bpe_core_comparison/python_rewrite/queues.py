"""Queues for the shared Python BPE trainer.

The driver owns ``frequencies``. Each old pair's frequency only decreases;
``add(pair)`` is called once, after a fresh-token pair has its initial final
frequency. Selected pairs are never added again. These constraints allow lazy
refresh without per-pair generations or queue membership maps.

``HighLowQueue`` borrows YTTM's scanned high array and integer-indexed low
buckets. It does not implement YTTM's worker pipeline or its default tie order;
both queues here choose the lexicographically smallest pair on frequency ties.
"""

from __future__ import annotations

import heapq
from math import isqrt


class HeapQueue:
    """Ordinary lazy max heap, ordered by ``(-frequency, pair)``."""

    def __init__(self, frequencies, min_frequency, initial_mass):
        if min_frequency < 1 or initial_mass < 1:
            raise ValueError("min_frequency and initial_mass must be positive")
        self.frequencies = frequencies
        self.min_frequency = min_frequency
        self._heap = [(-freq, pair) for pair, freq in frequencies.items()
                      if freq >= min_frequency]
        self.heapify_entries = len(self._heap)
        heapq.heapify(self._heap)
        self.heap_pushes = 0
        self.heap_pops = 0

    def add(self, pair):
        """Add one never-before-queued fresh-token pair."""
        freq = self.frequencies.get(pair, 0)
        if freq >= self.min_frequency:
            heapq.heappush(self._heap, (-freq, pair))
            self.heap_pushes += 1

    def pop(self):
        while self._heap:
            cached_neg, pair = heapq.heappop(self._heap)
            self.heap_pops += 1
            current = self.frequencies.get(pair, 0)
            if current < self.min_frequency:
                continue
            if current != -cached_neg:
                heapq.heappush(self._heap, (-current, pair))
                self.heap_pushes += 1
                continue
            return pair, current
        return None, 0

    def stats(self):
        return {
            "heapify_entries": self.heapify_entries,
            "heap_pushes": self.heap_pushes,
            "heap_pops": self.heap_pops,
        }


class HighLowQueue:
    """High scan array plus low integer buckets, with lazy decreases.

    The fixed low array has ``floor(sqrt(initial_mass))`` buckets, indexed
    ``0..threshold-1``. Buckets are sorted only after insertion; reverse sort
    lets ``pop()`` choose the lexicographically smallest pair on ties.
    """

    def __init__(self, frequencies, min_frequency, initial_mass):
        if min_frequency < 1 or initial_mass < 1:
            raise ValueError("min_frequency and initial_mass must be positive")
        self.frequencies = frequencies
        self.min_frequency = min_frequency
        self.threshold = max(1, isqrt(initial_mass))
        self._low = [[] for _ in range(self.threshold)]
        self._dirty = bytearray(self.threshold)
        self._max_low = -1
        self._high = []  # (cached frequency, pair), with swap-pop removals
        self.high_scan_visits = 0
        self.low_candidate_pops = 0
        self.low_bucket_steps = 0
        self.low_sort_calls = 0
        for pair, freq in frequencies.items():
            self._place(pair, freq)

    def _place(self, pair, freq):
        if freq < self.min_frequency:
            return
        if freq >= self.threshold:
            self._high.append((freq, pair))
        else:
            self._low[freq].append(pair)
            self._dirty[freq] = 1
            if freq > self._max_low:
                self._max_low = freq

    def add(self, pair):
        """Add one never-before-queued fresh-token pair."""
        self._place(pair, self.frequencies.get(pair, 0))

    def _pop_low(self):
        while self._max_low >= self.min_frequency:
            level = self._max_low
            bucket = self._low[level]
            if not bucket:
                self._max_low -= 1
                self.low_bucket_steps += 1
                continue
            if self._dirty[level]:
                bucket.sort(reverse=True)
                self._dirty[level] = 0
                self.low_sort_calls += 1
            pair = bucket.pop()
            self.low_candidate_pops += 1
            current = self.frequencies.get(pair, 0)
            if current == level:
                return pair, current
            if current >= self.min_frequency:
                # Under the driver contract current < level; _place also keeps
                # a future higher insertion safe if the contract is relaxed.
                self._place(pair, current)
        return None, 0

    def pop(self):
        high = self._high
        best_index = -1
        best_key = None
        index = 0
        while index < len(high):
            _, pair = high[index]
            self.high_scan_visits += 1
            current = self.frequencies.get(pair, 0)
            if current < self.threshold or current < self.min_frequency:
                self._place(pair, current)
                high[index] = high[-1]
                high.pop()
                continue
            high[index] = (current, pair)
            key = (-current, pair)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
            index += 1
        if best_index >= 0:
            freq, pair = high[best_index]
            high[best_index] = high[-1]
            high.pop()
            return pair, freq
        return self._pop_low()

    def stats(self):
        return {
            "high_scan_visits": self.high_scan_visits,
            "low_candidate_pops": self.low_candidate_pops,
            "low_bucket_steps": self.low_bucket_steps,
            "low_sort_calls": self.low_sort_calls,
            "scan_count": self.high_scan_visits + self.low_candidate_pops,
        }


__all__ = ["HighLowQueue", "HeapQueue"]

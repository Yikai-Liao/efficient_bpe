"""Exact greedy BPE with a global heap and persistent whole-piece workers.

This is a runnable multiprocessing prototype, not a claim of speedup.  One
master selects each rule from global weighted counts.  Workers own disjoint
contiguous groups of complete pieces and apply the same rule to their local
historical occurrences.  A zero separator at every piece boundary prevents
cross-shard pairs.  Packed integer pair keys preserve lexicographic tie order.
"""

from array import array
from bisect import bisect_right
from collections import defaultdict
import hashlib
import heapq
import json
import multiprocessing as mp
import os
import resource
import time
import traceback

from lean_backend import LeanEndpoints
from support import empty_result


def _current_rss_mib():
    """Current Linux RSS at worker ready/end; no per-round sampling."""
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None


def _peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _piece_ranges(prepared):
    corpus, _, pivots, weights = prepared
    if len(corpus) == 1:
        return []
    if corpus[0] != 0 or corpus[-1] != 0:
        raise ValueError("prepared corpus must have zero separators at both ends")
    pieces = []
    start = 1
    for pos in range(1, len(corpus)):
        if corpus[pos] == 0:
            if pos > start:
                weight = weights[bisect_right(pivots, start) - 1]
                pieces.append((start, pos, weight))
            start = pos + 1
    return pieces


def _shard_ranges(pieces, requested_workers):
    """Contiguous whole-piece shards, greedily balanced by stored chars."""
    count = min(requested_workers, len(pieces))
    if not count:
        return []
    result = []
    i = 0
    remaining_chars = sum(end - start for start, end, _ in pieces)
    for shard_index in range(count):
        if shard_index == count - 1:
            stop = len(pieces)
        else:
            future = count - shard_index - 1
            target = remaining_chars / (future + 1)
            stop = i
            mass = 0
            while stop < len(pieces) - future:
                size = pieces[stop][1] - pieces[stop][0]
                if stop > i and abs(mass - target) <= abs(mass + size - target):
                    break
                mass += size
                stop += 1
        result.append((i, stop))
        remaining_chars -= sum(end - start for start, end, _ in pieces[i:stop])
        i = stop
    return result


def _local_shards(prepared, requested_workers):
    corpus, initial_lengths, _, _ = prepared
    pieces = _piece_ranges(prepared)
    ranges = _shard_ranges(pieces, requested_workers)
    shards = []
    for first, stop in ranges:
        local = array("I", [0])
        pivots, weights = [], []
        chars = 0
        for start, end, weight in pieces[first:stop]:
            if not weights or weights[-1] != weight:
                pivots.append(len(local))
                weights.append(weight)
            local.extend(corpus[start:end + 1])
            chars += end - start
        shards.append((local, initial_lengths.copy(), pivots, weights, chars,
                       stop - first))
    return shards


class _WorkerState:
    def __init__(self, shard, backend_class):
        corpus, token_len, pivots, weights, chars, pieces = shard
        self.backend = backend_class(corpus, token_len)
        self.token_len = token_len
        self.pivots = pivots
        self.weights = weights
        self.chars = chars
        self.pieces = pieces
        self.pair_pos = defaultdict(lambda: array("I"))
        counts = defaultdict(int)
        weight_i = 0
        for pos in range(1, len(corpus) - 1):
            while weight_i + 1 < len(pivots) and pos >= pivots[weight_i + 1]:
                weight_i += 1
            a, b = corpus[pos], corpus[pos + 1]
            if a and b:
                key = (a << 32) | b
                self.pair_pos[key].append(pos)
                counts[key] += weights[weight_i]
        self.initial_counts = dict(counts)
        self.initial_occurrence_bytes = sum(len(a) * a.itemsize
                                            for a in self.pair_pos.values())

    def merge_round(self, pair, new_id, new_len):
        started = time.process_time()
        a, b = pair >> 32, pair & 0xFFFFFFFF
        if new_id != len(self.token_len):
            raise ValueError("worker token IDs diverged from master")
        self.token_len.append(new_len)
        positions = self.pair_pos.pop(pair, ())
        delta = defaultdict(int)
        new_keys = set()
        visits = stale = merged = 0
        backend = self.backend
        pivots, weights = self.pivots, self.weights
        for pos in positions:
            visits += 1
            context = backend.inspect_pair(pos, a, b)
            if context is None:
                stale += 1
                continue
            before, left_id, right, after, right_id = context
            weight = weights[bisect_right(pivots, pos) - 1]
            delta[pair] -= weight
            if left_id:
                delta[(left_id << 32) | a] -= weight
            if right_id:
                delta[(b << 32) | right_id] -= weight
            backend.merge_known(pos, right, after, new_id, new_len)
            merged += 1
            if left_id:
                key = (left_id << 32) | new_id
                delta[key] += weight
                self.pair_pos[key].append(before)
                new_keys.add(key)
            if right_id:
                key = (new_id << 32) | right_id
                delta[key] += weight
                self.pair_pos[key].append(pos)
                new_keys.add(key)
        return (dict(delta), tuple(new_keys), visits, stale, merged,
                time.process_time() - started)

    def finish(self):
        cpu_start = time.process_time()
        final = []
        pos = 0
        while pos is not None:
            final.append(self.backend.token(pos))
            pos = self.backend.next(pos)
        return (final, self.backend.memory_bytes(), _peak_rss_mib(),
                _current_rss_mib(), time.process_time() - cpu_start,
                time.process_time())


def _worker_main(connection, shard, backend_class):
    try:
        cpu_start = time.process_time()
        state = _WorkerState(shard, backend_class)
        connection.send(("ready", state.initial_counts,
                         state.initial_occurrence_bytes,
                         state.backend.memory_bytes(),
                         time.process_time() - cpu_start,
                         _peak_rss_mib(), _current_rss_mib(),
                         state.chars, state.pieces))
        state.initial_counts = None
        while True:
            command = connection.recv()
            if command[0] == "merge":
                _, pair, new_id, new_len = command
                connection.send(("round", state.merge_round(pair, new_id, new_len)))
            elif command[0] == "finish":
                connection.send(("finish", state.finish()))
                break
            else:
                raise ValueError(f"unknown worker command: {command[0]}")
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def _receive(connection, kind):
    message = connection.recv()
    if message[0] == "error":
        raise RuntimeError(message[1])
    if message[0] != kind:
        raise RuntimeError(f"expected {kind}, received {message[0]}")
    return message[1:]


def train(prepared, workers=2, *, backend_class=LeanEndpoints,
          max_merges=1000, min_frequency=2, capture=False, serial=False):
    """Train one exact global vocabulary using serial or spawned shard workers."""
    if workers < 1 or min_frequency < 1 or max_merges < 0:
        raise ValueError("workers/min_frequency must be positive; max_merges nonnegative")
    corpus, initial_lengths, _, _ = prepared
    if len(corpus) == 1:
        result = empty_result(capture)
        result.update(requested_workers=workers, actual_workers=0,
                      execution="serial" if serial else "spawn",
                      master_cpu_seconds=0.0, worker_init_cpu_seconds=[],
                      worker_merge_cpu_seconds=[], worker_finish_cpu_seconds=[],
                      worker_process_cpu_seconds=[], worker_peak_rss_mib=[],
                      worker_ready_rss_mib=[], worker_end_rss_mib=[],
                      parent_peak_rss_mib=_peak_rss_mib(),
                      active_worker_rounds=[], worker_merge_counts=[],
                      round_messages=0, messages_per_round=0,
                      total_ipc_messages=0,
                      shard_chars=[], shard_pieces=[],
                      prepared_input_buffer_bytes=corpus.itemsize * len(corpus),
                      parent_temporary_shard_buffer_bytes=0,
                      worker_local_input_buffer_bytes=[],
                      worker_initial_occurrence_bytes=[])
        return result
    if len(initial_lengths) + max_merges > 1 << 32:
        raise ValueError("packed pair keys require u32 token IDs")

    start = time.perf_counter()
    cpu_start = time.process_time()
    shards = _local_shards(prepared, workers)
    count = len(shards)
    shard_input_bytes = [len(shard[0]) * shard[0].itemsize for shard in shards]
    processes = []
    connections = []
    states = []
    try:
        if serial:
            ready = []
            for shard in shards:
                worker_cpu_start = time.process_time()
                state = _WorkerState(shard, backend_class)
                states.append(state)
                ready.append((state.initial_counts, state.initial_occurrence_bytes,
                              state.backend.memory_bytes(),
                              time.process_time() - worker_cpu_start,
                              _peak_rss_mib(), _current_rss_mib(),
                              state.chars, state.pieces))
        else:
            context = mp.get_context("spawn")
            for shard in shards:
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=_worker_main,
                                          args=(child, shard, backend_class))
                process.start()
                child.close()
                processes.append(process)
                connections.append(parent)
            ready = [_receive(conn, "ready") for conn in connections]
        # Spawned processes now own their deserialized local shards. The
        # parent's temporary shard copies are unnecessary after initialization.
        del shards
        del shard

        frequencies = defaultdict(int)
        for local_counts, *_ in ready:
            for key, value in local_counts.items():
                frequencies[key] += value
        del local_counts
        if serial:
            for state in states:
                state.initial_counts = None
        ready = [(None, *row[1:]) for row in ready]
        heap = [(-freq, key) for key, freq in frequencies.items()
                if freq >= min_frequency]
        heapq.heapify(heap)
        initialized = time.perf_counter()
        cpu_initialized = time.process_time()

        token_len = initial_lengths.copy()
        merges = []
        actual_merges = position_visits = stale_visits = heap_pops = 0
        active_rounds = [0] * count
        worker_merges = [0] * count
        worker_merge_cpu = [0.0] * count
        round_spread = 0
        for _ in range(max_merges):
            while heap:
                cached, pair = heapq.heappop(heap)
                heap_pops += 1
                current = frequencies.get(pair, 0)
                if current < min_frequency:
                    continue
                if -cached != current:
                    heapq.heappush(heap, (-current, pair))
                    continue
                break
            else:
                break
            a, b = pair >> 32, pair & 0xFFFFFFFF
            new_id = len(token_len)
            new_len = token_len[a] + token_len[b]
            token_len.append(new_len)
            merges.append((a, b, current))
            if serial:
                round_results = [state.merge_round(pair, new_id, new_len)
                                 for state in states]
            else:
                for conn in connections:
                    conn.send(("merge", pair, new_id, new_len))
                round_results = [_receive(conn, "round")[0]
                                 for conn in connections]

            fresh_keys = set()
            per_worker = []
            for worker_i, result in enumerate(round_results):
                delta, new_keys, visits, stale, merged, cpu = result
                for key, amount in delta.items():
                    frequencies[key] += amount
                fresh_keys.update(new_keys)
                position_visits += visits
                stale_visits += stale
                actual_merges += merged
                worker_merges[worker_i] += merged
                worker_merge_cpu[worker_i] += cpu
                if merged:
                    active_rounds[worker_i] += 1
                per_worker.append(merged)
            round_spread += max(per_worker) - min(per_worker)
            for key in fresh_keys:
                if frequencies[key] >= min_frequency:
                    heapq.heappush(heap, (-frequencies[key], key))
            frequencies.pop(pair, None)

        finished = time.perf_counter()
        cpu_finished = time.process_time()
        if serial:
            endings = [state.finish() for state in states]
        else:
            for conn in connections:
                conn.send(("finish",))
            endings = [_receive(conn, "finish")[0] for conn in connections]
        final = [0]
        for local_final, *_ in endings:
            final.extend(local_final[1:])
        fingerprint = hashlib.sha256(json.dumps([merges, final]).encode()).hexdigest()
        result = {
            "init_seconds": initialized - start,
            "merge_seconds": finished - initialized,
            "train_seconds": finished - start,
            "init_cpu_seconds": cpu_initialized - cpu_start,
            "merge_cpu_seconds": cpu_finished - cpu_initialized,
            "train_cpu_seconds": cpu_finished - cpu_start,
            "master_cpu_seconds": cpu_finished - cpu_start,
            "worker_init_cpu_seconds": [row[3] for row in ready],
            "worker_merge_cpu_seconds": worker_merge_cpu,
            "worker_finish_cpu_seconds": [row[4] for row in endings],
            "worker_process_cpu_seconds": ([ready[i][3] + worker_merge_cpu[i] +
                                            endings[i][4] for i in range(count)]
                                           if serial else [row[5] for row in endings]),
            "parent_peak_rss_mib": _peak_rss_mib(),
            "worker_peak_rss_mib": ([] if serial else
                                    [row[2] for row in endings]),
            "worker_ready_rss_mib": ([] if serial else
                                     [row[5] for row in ready]),
            "worker_end_rss_mib": ([] if serial else
                                   [row[3] for row in endings]),
            "requested_workers": workers,
            "actual_workers": count,
            "execution": "serial" if serial else "spawn",
            "rules": len(merges),
            "actual_merges": actual_merges,
            "position_visits": position_visits,
            "stale_visits": stale_visits,
            "heap_pops": heap_pops,
            "fingerprint": fingerprint,
            "max_token_length": max(token_len),
            "corpus_positions": len(corpus),
            "backend_buffer_bytes": sum(row[1] for row in endings),
            "initial_occurrence_bytes": sum(row[1] for row in ready),
            "prepared_input_buffer_bytes": len(corpus) * corpus.itemsize,
            "parent_temporary_shard_buffer_bytes": sum(shard_input_bytes),
            "worker_local_input_buffer_bytes": (shard_input_bytes if not serial
                                                 else []),
            "worker_initial_occurrence_bytes": [row[1] for row in ready],
            "shard_chars": [row[6] for row in ready],
            "shard_pieces": [row[7] for row in ready],
            "active_worker_rounds": active_rounds,
            "worker_merge_counts": worker_merges,
            "merge_imbalance_max_minus_min_sum": round_spread,
            "messages_per_round": 0 if serial else 2 * count,
            "round_messages": 0 if serial else 2 * count * len(merges),
            "total_ipc_messages": 0 if serial else count * (2 * len(merges) + 3),
        }
        if capture:
            result.update(merges=merges, final=final)
        return result
    finally:
        for conn in connections:
            conn.close()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()


__all__ = ["train"]

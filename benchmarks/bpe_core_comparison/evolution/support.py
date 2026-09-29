"""Small edge-case helpers shared by experimental drivers."""
import hashlib
import json


def empty_result(capture=False):
    """prepare([]) has a single separator and needs no mutable backend."""
    result = {name: 0.0 for name in ('init_seconds', 'merge_seconds',
              'train_seconds', 'init_cpu_seconds', 'merge_cpu_seconds',
              'train_cpu_seconds')}
    result.update(rules=0, actual_merges=0, position_visits=0, stale_visits=0,
                  heap_pops=0, max_token_length=1, corpus_positions=1,
                  backend_buffer_bytes=0, initial_occurrence_bytes=0,
                  fingerprint=hashlib.sha256(json.dumps([[], [0]]).encode()).hexdigest())
    if capture:
        result.update(merges=[], final=[0])
    return result

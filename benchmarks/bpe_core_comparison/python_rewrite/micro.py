"""Replay identical adjacency/merge traces, excluding occurrence/heap costs.

v1_u8 measures boundary operations only: reconstructing token identities by
string slicing is NOT included. Other layouts keep their ID write. This test
therefore isolates topology rather than claiming a complete v1 trainer win.
"""
from array import array
import argparse
import json
import random
import resource
import time
from bench import backends


class ByteLengths:
    def __init__(self, initial, lengths):
        self.seg = array('B', [1]) * len(initial)
        self.last = len(initial)-1

    def next(self, pos):
        return None if pos == self.last else pos + self.seg[pos]

    def prev(self, pos):
        return None if pos == 0 else pos - self.seg[pos-1]

    def merge(self, pos, new_id, new_len):
        right = pos + self.seg[pos]
        right_len = self.seg[right]
        self.seg[pos] = new_len
        if right_len == 1:
            self.seg[right] = new_len
        else:
            self.seg[right] = 0
            self.seg[right+right_len-1] = new_len
        return right

    def memory_bytes(self):
        return len(self.seg)


def make_trace(length, positions, pattern):
    rng = random.Random(202309)
    nblocks = max(1, positions // length)
    initial = array('I', [0])
    traces = []
    lengths = [1, 1]
    for block in range(nblocks):
        start = len(initial)
        initial.extend(array('I', [1]) * length)
        initial.append(0)
        live = list(range(start, start+length))
        if pattern == 'balanced':
            while len(live) > 1:
                remaining = []
                for k in range(0, len(live), 2):
                    pos = live[k]
                    remaining.append(pos)
                    if k+1 < len(live):
                        end = live[k+2] if k+2 < len(live) else start+length
                        new_len = end-pos
                        traces.append((pos, len(lengths), new_len))
                        lengths.append(new_len)
                live = remaining
        else:
            while len(live) > 1:
                k = 0 if pattern == 'chain' else rng.randrange(len(live)-1)
                pos = live[k]
                end = live[k+2] if k+2 < len(live) else start+length
                new_len = end-pos
                traces.append((pos, len(lengths), new_len))
                lengths.append(new_len)
                del live[k+1]
    return initial, traces, lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--positions', type=int, default=131072)
    p.add_argument('--pattern', choices=['balanced','random','chain'], default='random')
    a = p.parse_args()
    initial, trace, lengths = make_trace(a.length, a.positions, a.pattern)
    cls = ByteLengths if a.backend == 'v1_u8' else backends()[a.backend]
    backend = cls(initial, lengths)
    wall, cpu = time.perf_counter(), time.process_time()
    checksum = 0
    for pos, new_id, new_len in trace:
        before = backend.prev(pos)
        right = backend.next(pos)
        after = backend.next(right)
        checksum += before + right + after
        backend.merge(pos, new_id, new_len)
    cpu_elapsed = time.process_time()-cpu
    wall_elapsed = time.perf_counter()-wall
    starts=[]
    pos=0
    while pos is not None:
        starts.append(pos)
        pos=backend.next(pos)
    expected=[0]
    for start in range(1,len(initial)-1,a.length+1):
        expected.extend([start,start+a.length])
    assert starts == expected
    print(json.dumps({'backend':a.backend,'length':a.length,
                      'pattern':a.pattern,'positions':len(initial),
                      'operations':len(trace),'checksum':checksum,
                      'seconds':wall_elapsed,'cpu_seconds':cpu_elapsed,
                      'cpu_ns_per_merge':cpu_elapsed/len(trace)*1e9,
                      'buffer_bytes':backend.memory_bytes(),
                      'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}))


if __name__=='__main__':
    main()

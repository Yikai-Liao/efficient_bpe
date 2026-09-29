"""Checked boundary-only long-chain timing; excludes token IDs and BPE training."""
import argparse
import json
import time
from byte_spans import ByteSpans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--length', type=int, required=True)
    a = p.parse_args()
    span = ByteSpans(a.length + 2)
    wall, cpu = time.perf_counter(), time.process_time()
    for right in range(2, a.length + 1):
        before = span.prev(1)
        following = span.next(right)
        assert before == 0 and span.next(1) == right
        span.merge(1, right, following)
        assert span.prev(following) == 1
    elapsed_cpu, elapsed_wall = time.process_time()-cpu, time.perf_counter()-wall
    assert span.length(1) == a.length and span.next(1) == a.length+1
    assert span.memory_bytes() == a.length+2
    print(json.dumps(dict(length=a.length, merges=a.length-1,
                          buffer_bytes=span.memory_bytes(),
                          cpu_seconds=elapsed_cpu, seconds=elapsed_wall,
                          cpu_ns_per_merge=elapsed_cpu/(a.length-1)*1e9)))


if __name__ == '__main__':
    main()

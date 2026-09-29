"""Wall-clock and whole-process CPU accounting for exact parallel training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'python_rewrite'))
from common_fused import prepare
from bench_fused import load_pieces
from lean_backend import LeanEndpoints
from packed_driver import train as packed_train
from parallel_driver import train as parallel_train
from parallel_sparse_driver import train as sparse_train


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['packed','serial1','serial4','spawn1','spawn2','spawn4',
                                        'sparse1','sparse2','sparse4'],
                   required=True)
    p.add_argument('--dataset', default='en-4m')
    p.add_argument('--split', default='regex')
    p.add_argument('--rules', type=int, default=3000)
    a = p.parse_args()
    if a.dataset.startswith('chain-'):
        n = int(a.dataset.split('-')[1])
        word = ''.join(chr(0x1000+i) for i in range(n, 0, -1))
        pieces = [word, word]
        input_bytes = len(word.encode()) * 2
        digest = hashlib.sha256(word.encode()).hexdigest()
        a.rules = n
    else:
        pieces, input_bytes, digest = load_pieces(a.dataset, a.split)
    prepared = prepare(pieces)
    del pieces
    gc.collect()
    start, cpu_start, children_start = time.perf_counter(), time.process_time(), child_cpu()
    if a.variant == 'packed':
        result = packed_train(prepared, LeanEndpoints, a.rules)
    else:
        driver = sparse_train if a.variant.startswith('sparse') else parallel_train
        result = driver(prepared, workers=int(a.variant[-1]),
                        max_merges=a.rules, serial=a.variant.startswith('serial'))
    result.update(call_wall_seconds=time.perf_counter()-start,
                  call_master_cpu_seconds=time.process_time()-cpu_start,
                  call_children_cpu_seconds=child_cpu()-children_start)
    result['call_total_cpu_seconds'] = (result['call_master_cpu_seconds'] +
                                       result['call_children_cpu_seconds'])
    result.update(variant=a.variant, dataset=a.dataset, split=a.split,
                  input_bytes=input_bytes, input_sha256=digest,
                  requested_rules=a.rules, python=sys.version.split()[0],
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    if a.variant.startswith(('spawn','sparse')):
        result['sum_process_peak_rss_mib'] = (result['parent_peak_rss_mib'] +
                                            sum(result['worker_peak_rss_mib']))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

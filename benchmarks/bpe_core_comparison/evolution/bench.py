"""One evolution experiment per process; import the frozen baseline directly."""
import argparse
from array import array
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'python_rewrite'))
from common_fused import prepare, train as original_train
from backends_fused import FusedEbpeEndpoints, FusedPrezzaHalfword
from bench_fused import load_pieces
from lean_backend import LeanEndpoints
from packed_driver import train as packed_train
from filtered_driver import train as filtered_train
from hybrid_backends import HybridByteTags, HybridNibbleTags


def variants():
    result = {
        'baseline': (original_train, FusedEbpeEndpoints),
        'lean': (original_train, LeanEndpoints),
        'packed': (packed_train, LeanEndpoints),
        'filtered': (filtered_train, LeanEndpoints),
        'h3': (original_train, HybridByteTags),
        'h25': (original_train, HybridNibbleTags),
        'halfword': (original_train, FusedPrezzaHalfword),
    }
    try:
        from arena_driver import train as arena_train
        from arena_counted_driver import train as counted_train
        result['arena'] = arena_train, LeanEndpoints
        result['arena_counted'] = counted_train, LeanEndpoints
        result['arena_h3'] = arena_train, HybridByteTags
    except ImportError:
        pass
    try:
        from hybrid_backends import FusedHybridByteTags
        result['h3_fused'] = original_train, FusedHybridByteTags
        result['packed_h3'] = packed_train, FusedHybridByteTags
        result['filtered_h3'] = filtered_train, FusedHybridByteTags
        if 'arena' in result:
            result['arena_h3'] = arena_train, FusedHybridByteTags
    except ImportError:
        pass
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=sorted(variants()))
    p.add_argument('--dataset', default='en-1m')
    p.add_argument('--split', default='regex', choices=['regex', 'paragraph'])
    p.add_argument('--rules', type=int, default=3000)
    p.add_argument('--weight-scale', type=int, default=1)
    p.add_argument('--no-dedup', action='store_true')
    a = p.parse_args()
    train, backend = variants()[a.variant]
    started = time.perf_counter()
    if a.dataset.startswith('chain-'):
        length = int(a.dataset.split('-')[1])
        word = ''.join(chr(0x1000+i) for i in range(length, 0, -1))
        pieces = [word, word]
        input_bytes = len(word.encode()) * 2
        digest = hashlib.sha256(word.encode()).hexdigest()
        a.rules = length
    else:
        pieces, input_bytes, digest = load_pieces(a.dataset, a.split)
    num_pieces, chars = len(pieces), sum(map(len, pieces))
    prepared = prepare(pieces, not a.no_dedup)
    prepared = prepared[:3] + ([w * a.weight_scale for w in prepared[3]],)
    del pieces
    gc.collect()
    preprocess = time.perf_counter() - started
    result = train(prepared, backend, a.rules, 2 * a.weight_scale)
    if a.dataset.startswith('chain-'):
        assert result['actual_merges'] == result['rules'] == length - 1
        assert result['max_token_length'] == length
    result.update(variant=a.variant, dataset=a.dataset, split=a.split,
                  requested_rules=a.rules, weight_scale=a.weight_scale,
                  deduplicate=not a.no_dedup, input_bytes=input_bytes,
                  input_sha256=digest, original_pieces=num_pieces,
                  original_chars=chars, initial_alphabet=len(prepared[1])-1,
                  preprocessing_seconds=preprocess, python=sys.version.split()[0],
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

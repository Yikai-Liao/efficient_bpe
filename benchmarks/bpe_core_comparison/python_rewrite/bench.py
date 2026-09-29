"""Isolated-process benchmark; input snapshots live outside shared benchmark."""
import argparse
from array import array
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import resource
import sys
import time

from common import prepare, train

HERE = Path(__file__).resolve().parent


def backends():
    from backends_compact import FastEbpeEndpoints, FastPrezzaBitmap, FastPrezzaHalfword
    from backend_linked import FastLinkedBackend, FastCompactLinkedBackend
    return {'endpoints': FastEbpeEndpoints, 'bitmap_u32': FastPrezzaBitmap,
            'bitmap_halfword': FastPrezzaHalfword, 'linked16': FastLinkedBackend,
            'linked12': FastCompactLinkedBackend}


def load_pieces(name, split):
    if name.startswith('random-'):
        n = int(name.split('-')[1])
        rng = random.Random(261029)
        text = ''.join(rng.choices('abcdefghijkl', k=n))
        pieces = [text[i:i+128] for i in range(0, n, 128)]
        digest = hashlib.sha256(text.encode()).hexdigest()
        return pieces, n, digest
    if name.startswith('runs-'):
        n = int(name.split('-')[1])
        # Keep distinct surroundings so whole-piece dedup does not erase runs.
        pieces = [chr(0x4e00+i) + 'a' * 1024 + chr(0x7000+i)
                  for i in range(n // 1026)]
        text = '\n'.join(pieces)
        return pieces, len(text.encode()), hashlib.sha256(text.encode()).hexdigest()
    path = HERE.parent / 'data' / (name + '.txt')
    data = path.read_bytes()
    text = data.decode('utf-8')
    if split == 'regex':
        pieces = re.findall(r'\w+|[^\w\s]+', text)
    elif split == 'paragraph':
        pieces = [s for s in text.splitlines() if s]
    else:
        raise ValueError(split)
    return pieces, len(data), hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', default='regex')
    parser.add_argument('--rules', type=int, default=3000)
    parser.add_argument('--no-dedup', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    pieces, input_bytes, input_hash = load_pieces(args.dataset, args.split)
    num_pieces = len(pieces)
    original_chars = sum(map(len, pieces))
    prepared = prepare(pieces, not args.no_dedup)
    unique_pieces = sum(t == 0 for t in prepared[0]) - 1
    del pieces
    preprocessing = time.perf_counter() - start
    gc.collect()
    result = train(prepared, backends()[args.backend], args.rules)
    result.update({
        'backend': args.backend, 'dataset': args.dataset, 'split': args.split,
        'deduplicate': not args.no_dedup, 'queue': 'heap',
        'requested_rules': args.rules, 'python': sys.version.split()[0],
        'input_bytes': input_bytes, 'input_sha256': input_hash,
        'original_chars': original_chars, 'original_pieces': num_pieces,
        'stored_pieces': unique_pieces,
        'initial_alphabet': len(prepared[1])-1,
        'preprocessing_seconds': preprocessing,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

"""Measure the whole-process RSS floor before any mutable trainer is built."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'python_rewrite'))
from common_fused import prepare
from bench_fused import load_pieces


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    a = p.parse_args()
    pieces, size, digest = load_pieces(a.dataset, 'regex')
    prepared = prepare(pieces)
    del pieces
    gc.collect()
    resident = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    print(json.dumps(dict(dataset=a.dataset,input_sha256=digest,
                          input_bytes=size,corpus_positions=len(prepared[0]),
                          prepared_buffer_bytes=4*len(prepared[0]),
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                          current_rss_mib=resident/1024**2)))


if __name__ == '__main__':
    main()

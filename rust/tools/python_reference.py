"""Read the same numeric input as Rust and time the existing packed Python core."""
from array import array
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/bpe_core_comparison/python_rewrite'))
sys.path.insert(0,str(ROOT/'benchmarks/bpe_core_comparison/evolution'))
from lean_backend import LeanEndpoints
from packed_driver import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--rules',type=int,default=3000)
    a = p.parse_args()
    raw = a.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    del raw
    corpus = array('I',data.pop('corpus'))
    prepared = (corpus,data['initial_lengths'],data['pivots'],data['weights'])
    del data
    wall,cpu = time.perf_counter(),time.process_time()
    result = train(prepared,LeanEndpoints,a.rules)
    result.update(call_seconds=time.perf_counter()-wall,
                  call_cpu_seconds=time.process_time()-cpu,
                  fixture_sha256=digest,python=sys.version.split()[0],
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    result['vm_hwm_mib'] = float(next(line.split()[1]
        for line in Path('/proc/self/status').read_text().splitlines()
        if line.startswith('VmHWM:')))/1024
    print(json.dumps(result))


if __name__ == '__main__':
    main()

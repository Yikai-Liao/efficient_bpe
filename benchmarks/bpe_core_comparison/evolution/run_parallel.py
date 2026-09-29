"""Run parallel jobs sequentially, allowing all available CPUs in each job."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', choices=['main','edge'], default='main')
    p.add_argument('--variants', default='packed,serial1,spawn1,spawn2,spawn4,sparse2,sparse4')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4:
        raise RuntimeError('Parallel comparison requires at least four available CPUs')
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    cases = [('en-4m','regex'), ('zh-4m','regex'), ('en-1m','paragraph')]
    if a.profile == 'edge':
        cases = [('chain-2000','regex'), ('runs-131072','regex')]
    jobs = [(rep, dataset, split, variant) for rep in range(a.repeats)
            for dataset,split in cases for variant in a.variants.split(',')]
    random.Random(20261001).shuffle(jobs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sources = list(HERE.glob('*.py')) + list((HERE.parent/'python_rewrite').glob('*.py'))
    metadata = dict(python=sys.version, executable=sys.executable,
                    platform=platform.platform(), cpu_affinity=allowed,
                    command=sys.argv, seed=20261001,
                    files_sha256={str(f.relative_to(HERE.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in sources})
    a.output.with_suffix('.environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    fingerprints = defaultdict(set)
    with a.output.open('x', buffering=1) as f:
        for i,(rep,dataset,split,variant) in enumerate(jobs):
            cmd = [sys.executable,'parallel_bench.py','--variant',variant,
                   '--dataset',dataset,'--split',split]
            proc = subprocess.run(cmd,cwd=HERE,env=env,text=True,
                                  capture_output=True,timeout=240)
            if proc.returncode:
                raise RuntimeError((cmd,proc.stdout,proc.stderr))
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            fingerprints[dataset,split].add(row['fingerprint'])
            if len(fingerprints[dataset,split]) != 1:
                raise AssertionError(('semantic divergence',dataset,variant))
            row.update(repetition=rep,command=cmd,profile=a.profile)
            f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(completed=i+1,total=len(jobs),variant=variant,
                                  dataset=dataset,wall_s=row['train_seconds'],
                                  cpu_s=row['call_total_cpu_seconds'])),flush=True)


if __name__ == '__main__':
    main()

"""Sequential fresh processes, one CPU, shuffled variants, validated fingerprints."""
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
    p.add_argument('--profile', choices=['pilot', 'real', 'edge', 'chain', 'scale'],
                   default='pilot')
    p.add_argument('--variants', default='baseline,lean,packed,h3,h25,halfword')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    cases = {
        'pilot': [('en-1m','regex',1), ('zh-1m','regex',1)],
        'real': [('en-1m','regex',1), ('zh-1m','regex',1),
                 ('de-1m','regex',1), ('ja-1m','regex',1),
                 ('en-1m','paragraph',1)],
        'edge': [('random-131072','regex',1), ('runs-131072','regex',1),
                 ('en-1m','regex',64)],
        'chain': [(f'chain-{n}','regex',1) for n in (2000,4000,8000,16000)],
        'scale': [('en-4m','regex',1), ('zh-4m','regex',1)],
    }[a.profile]
    jobs = [(rep, case, v) for rep in range(a.repeats) for case in cases
            for v in a.variants.split(',')]
    random.Random(20260930).shuffle(jobs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sources = list(HERE.glob('*.py')) + list((HERE.parent/'python_rewrite').glob('*.py'))
    metadata = dict(python=sys.version, executable=sys.executable,
                    platform=platform.platform(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                    command=sys.argv, seed=20260930,
                    files_sha256={str(f.relative_to(HERE.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in sources})
    a.output.with_suffix('.environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    fingerprints = defaultdict(set)
    with a.output.open('x', buffering=1) as f:
        for i, (rep, (dataset, split, scale), variant) in enumerate(jobs):
            cmd = [sys.executable, 'bench.py', '--variant', variant,
                   '--dataset', dataset, '--split', split,
                   '--weight-scale', str(scale)]
            proc = subprocess.run(cmd, cwd=HERE, env=env, text=True,
                                  capture_output=True, timeout=240)
            if proc.returncode:
                raise RuntimeError((cmd, proc.stdout, proc.stderr))
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            fingerprints[dataset,split,scale].add(row['fingerprint'])
            if len(fingerprints[dataset,split,scale]) != 1:
                raise AssertionError(('semantic divergence', dataset, variant))
            row.update(repetition=rep, command=cmd, profile=a.profile,
                       cpu_affinity=sorted(os.sched_getaffinity(0)))
            f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(completed=i+1,total=len(jobs),dataset=dataset,
                                  variant=variant,cpu_s=row['train_cpu_seconds'],
                                  rss_mib=row['peak_rss_mib'])), flush=True)


if __name__ == '__main__':
    main()

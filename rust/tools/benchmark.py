"""Sequential numeric-input comparison: Python, checked Rust, unchecked Rust."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'rust'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,default=HERE/'target/release/efficient-bpe-rust')
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',type=Path,default=HERE/'results/baseline.jsonl')
    a = p.parse_args()
    fixtures = json.loads((HERE/'results/fixtures.json').read_text())
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    env = dict(os.environ,PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')
    jobs = [(rep,case,variant) for rep in range(a.repeats) for case in fixtures
            for variant in ('python','checked','unchecked')]
    random.Random(20261003).shuffle(jobs)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    sources = [HERE/'Cargo.toml',HERE/'Cargo.lock',*list((HERE/'src').glob('*.rs'))]
    sources += list((HERE/'tools').glob('*.py'))
    sources += [ROOT/'benchmarks/bpe_core_comparison'/name for name in (
        'evolution/packed_driver.py', 'evolution/lean_backend.py',
        'python_rewrite/common_fused.py')]
    environment = dict(python=sys.version,
                       rustc=subprocess.check_output(['/root/.cargo/bin/rustc','-Vv'],text=True),
                       command=sys.argv,cpu_affinity=sorted(os.sched_getaffinity(0)),
                       seed=20261003,profiling_enabled=False,
                       binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                       sources_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                       for f in sources})
    fingerprints = defaultdict(set)
    with a.output.open('x',buffering=1) as output:
        with a.output.with_suffix('.environment.json').open('x') as metadata:
            metadata.write(json.dumps(environment,indent=2)+'\n')
        for i,(rep,case,variant) in enumerate(jobs):
            input_path = HERE/case['file']
            if variant=='python':
                cmd=[sys.executable,str(HERE/'tools/python_reference.py')]
            else:
                cmd=[str(a.binary),'--bounds',variant]
            cmd.extend(['--input',str(input_path),'--rules',str(case['rules'])])
            run=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=240)
            if run.returncode:
                raise RuntimeError((cmd,run.stdout,run.stderr))
            result=json.loads(run.stdout.strip().splitlines()[-1])
            if variant != 'python':
                assert result['profiling_enabled'] is False, 'profiling binary cannot be benchmarked'
            assert result['fixture_sha256']==case['fixture_sha256']
            key=case['dataset'],case['split']
            fingerprints[key].add(result['fingerprint'])
            assert len(fingerprints[key])==1,('semantic difference',key,variant)
            result.update(variant=variant,dataset=case['dataset'],split=case['split'],
                          repetition=rep,command=cmd,requested_rules=case['rules'],
                          input_sha256=case['input_sha256'],
                          cpu_affinity=sorted(os.sched_getaffinity(0)))
            output.write(json.dumps(result)+'\n')
            print(json.dumps(dict(completed=i+1,total=len(jobs),dataset=case['dataset'],
                                  split=case['split'],variant=variant,
                                  core_wall_s=result['train_seconds'])),flush=True)


if __name__ == '__main__':
    main()

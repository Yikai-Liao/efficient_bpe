"""Cross-language full-rule oracle checks, including both bounds implementations."""
from array import array
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/bpe_core_comparison/python_rewrite'))
from common_fused import prepare, naive


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,default=ROOT/'rust/target/release/efficient-bpe-rust')
    p.add_argument('--cases',type=int,default=200)
    a = p.parse_args()
    rng = random.Random(20260930)
    cases = [([],True),([''],True),(['a'],True),(['aaa','aaaa','aaaaa']*3,True),
             (['abababab','babababa','aaaaab']*4,True)]
    for _ in range(a.cases):
        pieces = [''.join(rng.choices('abcd界文🙂',k=rng.randrange(1,60)))
                  for _ in range(rng.randrange(1,12))]*rng.randrange(1,5)
        cases.append((pieces,rng.choice((True,False))))
    checked = 0
    with tempfile.TemporaryDirectory(prefix='bpe-rust-differential-') as temp:
        directory = Path(temp)
        for i,(pieces,dedup) in enumerate(cases):
            prepared = prepare(pieces,dedup)
            minimum = 1+i%5
            if i%13==0:
                scale = 1 << 40
                prepared = prepared[:3]+([w*scale for w in prepared[3]],)
                minimum *= scale
            wire = dict(zip(('corpus','initial_lengths','pivots','weights'),
                            (list(value) for value in prepared)))
            fixture = directory/'input.json'
            fixture.write_text(json.dumps(wire))
            merges,final = naive(prepared,80,minimum)
            fingerprint = hashlib.sha256(json.dumps([merges,final]).encode()).hexdigest()
            for bounds in ('checked','unchecked'):
                trace = directory/'trace.json'
                cmd = [str(a.binary),'--input',str(fixture),'--bounds',bounds,
                       '--rules','80','--min-frequency',str(minimum),'--trace',str(trace)]
                run = subprocess.run(cmd,text=True,capture_output=True,check=True)
                result = json.loads(run.stdout.strip().splitlines()[-1])
                observed = json.loads(trace.read_text())
                assert observed=={'merges':[list(rule) for rule in merges],'final':final},(i,bounds)
                assert result['fingerprint']==fingerprint,(i,bounds,'fingerprint')
                checked += 1
    report = dict(cases=len(cases),rust_runs=checked,oracle='full-recount Python naive',
                  full_rule_traces_match=True,final_tokens_match=True,
                  fingerprints_match=True,random_seed=20260930,
                  binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest())
    (ROOT/'rust/results/differential.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

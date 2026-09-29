"""Export the exact Python prepared input, with no Rust regex/tokenization drift."""
from array import array
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks/bpe_core_comparison/python_rewrite'))
from common_fused import prepare
from bench_fused import load_pieces


def export_case(dataset, split, output):
    if dataset.startswith('chain-'):
        n = int(dataset.split('-')[1])
        word = ''.join(chr(0x1000+i) for i in range(n,0,-1))
        pieces = [word, word]
        size = len(word.encode())*2
        digest = hashlib.sha256(word.encode()).hexdigest()
        rules = n
    else:
        pieces, size, digest = load_pieces(dataset, split)
        rules = 3000
    prepared = prepare(pieces)
    wire = dict(zip(('corpus','initial_lengths','pivots','weights'),
                    (list(value) for value in prepared)))
    body = json.dumps(wire,separators=(',',':')).encode()+b'\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(body)
    return dict(dataset=dataset,split=split,rules=rules,
                file=str(output.relative_to(ROOT/'rust')),
                fixture_sha256=hashlib.sha256(body).hexdigest(),
                input_sha256=digest,input_bytes=size,
                corpus_positions=len(prepared[0]),
                initial_alphabet=len(prepared[1])-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=ROOT/'rust/fixtures')
    a = p.parse_args()
    cases = [('en-1m','regex'),('zh-1m','regex'),('de-1m','regex'),
             ('ja-1m','regex'),('en-1m','paragraph'),('en-4m','regex'),
             ('zh-4m','regex'),('random-131072','regex'),('runs-131072','regex'),
             ('chain-8000','regex'),('chain-16000','regex')]
    rows = [export_case(ds,split,a.output_dir/f'{ds}-{split}.json')
            for ds,split in cases]
    manifest = ROOT/'rust/results/fixtures.json'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'exported':len(rows),'manifest':str(manifest)}))


if __name__ == '__main__':
    main()

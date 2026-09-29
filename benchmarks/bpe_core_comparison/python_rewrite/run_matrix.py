"""Sequential isolated jobs, fixed shuffled order, append-only local results."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=['real','scale','micro','queue','fused','chain'],default='real')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output-dir',type=Path,default=None)
    a=p.parse_args()
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    env=dict(os.environ,PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')
    jobs=[]
    names=['endpoints','linked12','linked16','bitmap_u32','bitmap_halfword']
    if a.profile=='real':
        for dataset,split in [('en-1m','regex'),('zh-1m','regex'),('de-1m','regex'),('ja-1m','regex'),('en-1m','paragraph')]:
            for backend in names:
                jobs.append(['bench.py','--dataset',dataset,'--split',split,'--backend',backend,'--rules','3000'])
    elif a.profile=='scale':
        for n in [32768,131072,524288]:
            for backend in ['endpoints','linked12','bitmap_halfword']:
                jobs.append(['bench.py','--dataset',f'random-{n}','--backend',backend,'--rules','256','--no-dedup'])
        for dataset in ['en-4m','zh-4m']:
            for backend in ['endpoints','linked12','bitmap_halfword']:
                jobs.append(['bench.py','--dataset',dataset,'--backend',backend,'--rules','3000'])
    elif a.profile=='micro':
        for length,pattern in [(32,'random'),(128,'balanced'),(128,'chain'),(1024,'balanced'),(8192,'chain')]:
            for backend in names+(['v1_u8'] if length<=255 else []):
                positions=16384 if length==8192 else 131072
                jobs.append(['micro.py','--backend',backend,'--length',str(length),'--pattern',pattern,'--positions',str(positions)])
    elif a.profile=='chain':
        for length in [1000,2000,4000,8000]:
            for backend in ['full_clear','endpoints','linked12','bitmap_halfword']:
                jobs.append(['chain_bench.py','--length',str(length),'--backend',backend])
    elif a.profile=='fused':
        for dataset,split in [('en-1m','regex'),('zh-1m','regex'),('de-1m','regex'),('ja-1m','regex'),('en-1m','paragraph')]:
            for backend in ['endpoints','linked12','bitmap_halfword']:
                jobs.append(['bench_fused.py','--dataset',dataset,'--split',split,'--backend',backend,'--rules','3000'])
    elif a.profile=='queue':
        for dataset,weight_scale in [('en-1m',1),('zh-1m',1),('random-131072',1),('en-1m',64)]:
            for queue in ['heap','bucket']:
                jobs.append(['queue_bench.py','--dataset',dataset,'--rules','3000','--queue',queue,'--weight-scale',str(weight_scale)])
    all_jobs=[(rep,job) for rep in range(a.repeats) for job in jobs]
    random.Random(20260929).shuffle(all_jobs)
    output_dir=a.output_dir.resolve() if a.output_dir else HERE.parent
    output_dir.mkdir(parents=True,exist_ok=True)
    target=output_dir/(a.profile+'-results.jsonl')
    with target.open('a',buffering=1) as f:
        for i,(rep,job) in enumerate(all_jobs):
            cmd=[sys.executable,*job]
            result=subprocess.run(cmd,cwd=HERE,env=env,text=True,capture_output=True,timeout=180)
            if result.returncode:
                raise RuntimeError((cmd,result.stdout,result.stderr))
            row=json.loads(result.stdout.strip().splitlines()[-1])
            row.update(repetition=rep,command=cmd,cpu_affinity=sorted(os.sched_getaffinity(0)))
            f.write(json.dumps(row)+'\n')
            print(json.dumps({'completed':i+1,'total':len(all_jobs),'case':job,
                              'cpu_s':row.get('train_cpu_seconds',row.get('cpu_seconds'))}),flush=True)


if __name__=='__main__':
    main()

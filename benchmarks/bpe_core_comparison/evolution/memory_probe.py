"""Separate untimed process-tree memory sampling, including the spawn tracker.

The sampled sum of PSS apportions shared mappings, unlike a sum of individual
ru_maxrss high-water marks. Sampling is not an exact peak measurement.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def family(root_pid):
    found, pending = [], [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.append(pid)
        try:
            child_file = Path(f'/proc/{pid}/task/{pid}/children')
            pending.extend(map(int, child_file.read_text().split()))
        except OSError:
            pass
    return found


def sample(root_pid):
    rows = []
    for pid in family(root_pid):
        try:
            fields = {}
            for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in ('Rss:','Pss:','Private_Clean:','Private_Dirty:'):
                    fields[parts[0][:-1]] = int(parts[1])
            rows.append(dict(pid=pid, **fields))
        except OSError:
            pass
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cases = [('en-4m','regex'),('zh-4m','regex'),('en-1m','paragraph')]
    with a.output.open('x',buffering=1) as out:
        for dataset,split in cases:
            for variant in ('packed','spawn4','sparse4'):
                cmd = [sys.executable,'parallel_bench.py','--dataset',dataset,
                       '--split',split,'--variant',variant]
                proc = subprocess.Popen(cmd,cwd=HERE,stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,text=True,
                                        env=dict(os.environ,PYTHONHASHSEED='0',
                                                 PYTHONDONTWRITEBYTECODE='1'))
                peak_pss = peak_rss = samples = max_family = 0
                largest_gap = 0.0
                last_sample = time.perf_counter()
                peak_rows = []
                while proc.poll() is None:
                    now = time.perf_counter()
                    largest_gap = max(largest_gap, now-last_sample)
                    last_sample = now
                    rows = sample(proc.pid)
                    total_pss = sum(r.get('Pss',0) for r in rows)
                    total_rss = sum(r.get('Rss',0) for r in rows)
                    if total_pss > peak_pss:
                        peak_pss, peak_rows = total_pss, rows
                    peak_rss = max(peak_rss,total_rss)
                    max_family = max(max_family,len(rows))
                    samples += 1
                    time.sleep(0.02)
                stdout,stderr = proc.communicate()
                if proc.returncode:
                    raise RuntimeError((cmd,stdout,stderr))
                result = json.loads(stdout.strip().splitlines()[-1])
                row = dict(dataset=dataset,split=split,variant=variant,
                           sampled_peak_pss_mib=peak_pss/1024,
                           sampled_peak_rss_sum_mib=peak_rss/1024,
                           sample_count=samples,max_processes=max_family,
                           requested_interval_seconds=0.02,
                           largest_sample_gap_seconds=largest_gap,
                           peak_pss_processes=peak_rows,
                           fingerprint=result['fingerprint'],
                           input_sha256=result['input_sha256'], command=cmd)
                out.write(json.dumps(row)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in
                                  ('peak_pss_processes','command')}),flush=True)


if __name__ == '__main__':
    main()

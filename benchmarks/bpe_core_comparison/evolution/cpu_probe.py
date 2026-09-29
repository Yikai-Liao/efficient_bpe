"""Check that spawned workers can actually consume more than one CPU."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def work(iterations):
    start = time.process_time()
    value = 17
    for _ in range(iterations):
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return time.process_time()-start, value


def main():
    rows = []
    for workers in (1,4):
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            outputs = list(pool.map(work, [10_000_000]*workers))
        wall = time.perf_counter()-start
        assert len({value for _,value in outputs}) == 1
        cpu = sum(duration for duration,_ in outputs)
        rows.append(dict(workers=workers,wall_seconds=wall,worker_cpu_seconds=cpu,
                         observed_parallelism=cpu/wall,
                         affinity=sorted(os.sched_getaffinity(0))))
    path = Path(__file__).resolve().parent/'cpu-probe.json'
    path.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()

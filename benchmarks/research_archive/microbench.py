from pathlib import Path as _AuditPath
REPO = _AuditPath(__file__).resolve().parents[2]
ARCHIVE = _AuditPath(__file__).resolve().parent
import sys,inspect,time,statistics
sys.path.insert(0,str(REPO))
import ebpe_v2 as v2
from array import array
old=v2.merge_token_pair
source=inspect.getsource(old).replace('        corpus[pos_x] = corpus[pos_end - 1] = new_token\n        for i in range(pos_x + 1, pos_end - 1):\n            corpus[i] = 0', '        corpus[pos_y - 1] = corpus[pos_y] = 0\n        corpus[pos_x] = corpus[pos_end - 1] = new_token')
ns=dict(vars(v2));exec(source,ns);new=ns['merge_token_pair']

def bench(fn,n):
    corpus=array('I',[0])+array('I',[2])*n+array('I',[0])
    lengths=[1,1,1]
    current=2
    start=time.perf_counter()
    for length in range(2,n+1):
        next_id=len(lengths);lengths.append(length)
        fn(corpus,(current,2),next_id,[1],[0,n+2],[1,0],lengths,n+1)
        current=next_id
    elapsed=time.perf_counter()-start
    assert corpus[1]==corpus[n]==current
    assert all(x==0 for x in corpus[2:n])
    return elapsed
for n in (1000,2000,4000,8000):
    a=statistics.median(bench(old,n) for _ in range(3));b=statistics.median(bench(new,n) for _ in range(3))
    print({'n':n,'old_seconds':round(a,6),'boundary_seconds':round(b,6),'ratio':round(a/b,2)})

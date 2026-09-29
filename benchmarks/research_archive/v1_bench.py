from pathlib import Path as _AuditPath
REPO = _AuditPath(__file__).resolve().parents[2]
ARCHIVE = _AuditPath(__file__).resolve().parent
import sys,time,resource,hashlib,json,pathlib
sys.path.insert(0,str(REPO))
source=pathlib.Path(str(REPO / 'ebpe.py')).read_text()
variant=sys.argv[1]
if variant=='array_init':
 source=source.replace("array('B', [1] * (len(self.corpus) + 2))", "array('B', [1]) * (len(self.corpus) + 2)")
ns={'__name__':'audit'};exec(compile(source,'ebpe.py','exec'),ns)
t=ns['BPETrainer'](10000,min_freq=15,compress_threshold=.3,single_char=False)
start=time.perf_counter();b=t.train_from_file(str(REPO / 'data/train_BPE.txt'));elapsed=time.perf_counter()-start
print(json.dumps({'variant':variant,'seconds':round(elapsed,4),'maxrss_MiB':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),'vocab_hash':hashlib.sha256(json.dumps(b.vocab,sort_keys=True).encode()).hexdigest()}))

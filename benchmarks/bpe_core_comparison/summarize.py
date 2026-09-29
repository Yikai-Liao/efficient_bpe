"""Validate recorded comparisons and render the report from raw JSONL."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parent
EXPECTED={'real':75,'fused':45,'scale':45,'micro':84,'queue':24,'chain':48}
DATA={name:[json.loads(s) for s in (ROOT/(name+'-results.jsonl')).read_text().splitlines()]
      for name in EXPECTED}
for name,count in EXPECTED.items():
    assert len(DATA[name])==count,(name,len(DATA[name]),count)

checks=defaultdict(set)
for name,rows in DATA.items():
    for r in rows:
        if name=='micro':
            key=('micro',r['length'],r['pattern'],r['positions'])
            value=(r['operations'],r['checksum'])
        elif name=='chain':
            key=('chain',r['length'])
            value=r['fingerprint']
        elif name=='queue':
            key=('queue',r['dataset'],r['weight_scale'])
            value=r['fingerprint']
        else:
            key=('training',r['dataset'],r['split'],r['requested_rules'],r['deduplicate'])
            value=r['fingerprint']
        checks[key].add(value)
for key,values in checks.items():
    assert len(values)==1,(key,values)
for row in json.loads((ROOT/'data/snapshot-manifest.json').read_text()):
    assert hashlib.sha256((ROOT/'data'/row['snapshot']).read_bytes()).hexdigest()==row['sha256']

def select(name,**criteria):
    return [r for r in DATA[name] if all(r.get(k)==v for k,v in criteria.items())]

def med(name,field,**criteria):
    rows=select(name,**criteria)
    assert len(rows)==3,(name,criteria,len(rows))
    return median(r[field] for r in rows)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

labels=[('en-1m','regex','英文分片'),('zh-1m','regex','中文分片'),
        ('de-1m','regex','德文分片'),('ja-1m','regex','日文分片'),
        ('en-1m','paragraph','英文整段')]
backends=['endpoints','linked12','bitmap_halfword']
fused=[];rss=[]
for ds,sp,label in labels:
    fused.append([label]+[f'{med("fused","merge_cpu_seconds",dataset=ds,split=sp,backend=b):.3f}' for b in backends])
    rss.append([label]+[f'{med("fused","peak_rss_mib",dataset=ds,split=sp,backend=b):.1f}' for b in backends])
scale=[]
for ds in ['random-32768','random-131072','random-524288','en-4m','zh-4m']:
    row=select('scale',dataset=ds)[0]
    scale.append([ds,row['requested_rules'],row['corpus_positions']]+[
        f'{med("scale","train_cpu_seconds",dataset=ds,backend=b):.3f}' for b in backends])
micro=[]
for le,pa in [(32,'random'),(128,'balanced'),(128,'chain'),(1024,'balanced'),(8192,'chain')]:
    columns=[]
    for b in ['v1_u8','endpoints','linked12','bitmap_halfword']:
        xs=select('micro',length=le,pattern=pa,backend=b)
        columns.append(f'{median(r["cpu_ns_per_merge"] for r in xs):.0f}' if xs else '—')
    micro.append([le,pa]+columns)
queues=[]
for ds,sc in [('en-1m',1),('zh-1m',1),('random-131072',1),('en-1m',64)]:
    qs=select('queue',dataset=ds,weight_scale=sc,queue='bucket')[0]['queue_stats']
    queues.append([ds,sc]+[f'{med("queue","merge_cpu_seconds",dataset=ds,weight_scale=sc,queue=q):.3f}' for q in ['heap','bucket']]+[qs['high_scan_visits']])
chain=[]
for le in [1000,2000,4000,8000]:
    chain.append([le]+[f'{med("chain","merge_cpu_seconds",length=le,backend=b):.4f}' for b in ['full_clear','endpoints','linked12','bitmap_halfword']])
rle=[]
for r in json.loads((ROOT/'input-layout-counts.json').read_text()):
    rle.append([r['dataset'],r['physical_characters'],r['runs'],f'{r["positions_per_run"]:.3f}×',r['flat_initial_occurrences'],r['yttm_initial_occurrences']])
tables={
    'FUSED_TABLE':table(['约 1 MiB 输入','双端 ID','数组链 12B','半字＋位图'],fused),
    'RSS_TABLE':table(['输入','双端 ID / MiB','数组链 / MiB','半字位图 / MiB'],rss),
    'SCALE_TABLE':table(['输入','规则数','物理位置 U','双端 / 秒','数组链 / 秒','半字 / 秒'],scale),
    'MICRO_TABLE':table(['片段长度','轨迹','v1_u8','双端 ID','数组链 12B','半字位图'],micro),
    'QUEUE_TABLE':table(['输入','权重倍数','堆 / 秒','高低频队列 / 秒','高频扫描记录数'],queues),
    'CHAIN_TABLE':table(['链长','整段清零','四端点写','数组链','半字位图'],chain),
    'RLE_TABLE':table(['输入','唯一片段字符数','run 数','字符数 / run 数','普通初始 occurrence','YTTM 初始 occurrence'],rle),
}
report=(ROOT/'REPORT.template.md').read_text()
for key,value in tables.items():report=report.replace('{{'+key+'}}',value)
assert '{{' not in report
(ROOT/'REPORT.md').write_text(report)
(ROOT/'tables.md').write_text('\n\n'.join(tables.values()))
summary={'timed_processes':sum(EXPECTED.values()),'rows':EXPECTED,
         'matching_semantic_groups':len(checks),'all_fingerprints_match':True,
         'all_snapshots_match':True}
(ROOT/'verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
print(tables['CHAIN_TABLE'])

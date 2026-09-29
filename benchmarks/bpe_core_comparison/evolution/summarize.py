"""Validate final rows and render the experiment report from measured medians."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
FINAL_FILES = ['real-results.jsonl', 'edge-results.jsonl', 'chain-results.jsonl',
               'scale-results.jsonl', 'parallel-results.jsonl',
               'parallel-edge-results.jsonl']
LABELS = {'baseline':'基线：4 次端点写入', 'lean':'减少端点写入',
          'packed':'整数 pair key', 'filtered':'整数 key＋预筛索引',
          'arena':'位置池 v1', 'arena_counted':'位置池 v2',
          'h3_fused':'H3：3 字节布局', 'h25':'H2.5：2.5 字节布局',
          'halfword':'论文 halfword＋bitmap',
          'filtered_h3':'预筛索引＋H3'}


def main():
    all_rows, matrices = [], {}
    for name in FINAL_FILES:
        rows = [json.loads(line) for line in (HERE/name).read_text().splitlines()]
        all_rows.extend(rows)
        groups = defaultdict(list)
        for row in rows:
            key = (row['dataset'], row['split'], row.get('weight_scale',1), row['variant'])
            groups[key].append(row)
        for key, group in groups.items():
            assert len(group) == 3 and {r['repetition'] for r in group} == {0,1,2}, (name,key)
        matrices[name] = groups
    semantic = defaultdict(set)
    for row in all_rows:
        key = (row['dataset'],row['split'],row.get('weight_scale',1),row['requested_rules'])
        semantic[key].add(row['fingerprint'])
        data = HERE.parent/'data'/(row['dataset']+'.txt')
        if data.exists():
            assert hashlib.sha256(data.read_bytes()).hexdigest() == row['input_sha256']
    assert all(len(fingerprints) == 1 for fingerprints in semantic.values())

    def values(matrix, dataset, variant, split='regex', weight=1):
        return matrices[matrix][dataset,split,weight,variant]

    def med(rows, key):
        return median(row[key] for row in rows)

    real = matrices['real-results.jsonl']
    variants = list(LABELS)
    table = ['| 版本 | 英文 CPU 秒 | 英文 RSS MiB | 中文 CPU 秒 | 中文 RSS MiB |',
             '|---|---:|---:|---:|---:|']
    for variant in variants:
        en, zh = real['en-1m','regex',1,variant], real['zh-1m','regex',1,variant]
        table.append(f'| {LABELS[variant]} | {med(en,"train_cpu_seconds"):.3f} | '
                     f'{med(en,"peak_rss_mib"):.1f} | {med(zh,"train_cpu_seconds"):.3f} | '
                     f'{med(zh,"peak_rss_mib"):.1f} |')
    expanded = ['| 数据 | 版本 | 初始化 CPU 秒 | 合并 CPU 秒 | 总 CPU 秒 | RSS MiB |',
                '|---|---|---:|---:|---:|---:|']
    summary = []
    for name, groups in matrices.items():
        for key, rows in sorted(groups.items()):
            metrics = {metric:median(r[metric] for r in rows)
                       for metric in ('init_cpu_seconds','merge_cpu_seconds',
                                      'train_cpu_seconds','train_seconds','peak_rss_mib')}
            record = dict(matrix=name, dataset=key[0], split=key[1],weight_scale=key[2],
                          variant=key[3],repetitions=len(rows),**metrics)
            record['train_cpu_min_max'] = [min(r['train_cpu_seconds'] for r in rows),
                                           max(r['train_cpu_seconds'] for r in rows)]
            summary.append(record)
            if name == 'real-results.jsonl':
                expanded.append(f'| {key[0]} / {key[1]} | {key[3]} | '
                                f'{metrics["init_cpu_seconds"]:.3f} | '
                                f'{metrics["merge_cpu_seconds"]:.3f} | '
                                f'{metrics["train_cpu_seconds"]:.3f} | '
                                f'{metrics["peak_rss_mib"]:.1f} |')

    parallel = ['| 数据 | 执行方式 | 训练墙钟秒 | 合并墙钟秒 | 整次调用总 CPU 秒 |',
                '|---|---|---:|---:|---:|']
    for name in ('parallel-results.jsonl','parallel-edge-results.jsonl'):
        for (dataset,split,weight,variant),rows in sorted(matrices[name].items()):
            parallel.append(f'| {dataset}/{split} | {variant} | {med(rows,"train_seconds"):.3f} | '
                            f'{med(rows,"merge_seconds"):.3f} | {med(rows,"call_total_cpu_seconds"):.3f} |')

    memory_rows = [json.loads(line) for line in (HERE/'parallel-memory.jsonl').read_text().splitlines()]
    memory_groups = defaultdict(dict)
    for row in memory_rows:
        assert row['fingerprint'] in semantic[row['dataset'],row['split'],1,3000]
        memory_groups[row['dataset'],row['split']][row['variant']] = row['sampled_peak_pss_mib']
    assert len(memory_rows)==9 and all(len(group)==3 for group in memory_groups.values())
    memory_table = ['| 数据 | 直接单进程 PSS MiB | 广播 4 worker PSS MiB | owner 4 worker PSS MiB |',
                    '|---|---:|---:|---:|']
    for (dataset,split),group in sorted(memory_groups.items()):
        memory_table.append(f'| {dataset}/{split} | '+
                            ' | '.join(f'{group[v]:.1f}' for v in ('packed','spawn4','sparse4'))+' |')

    chain = ['| 长度 | 基线合并 CPU 秒 | 整数 key | 预筛索引 | 位置池 v2 | H3 |',
             '|---:|---:|---:|---:|---:|---:|']
    for n in (2000,4000,8000,16000):
        nums = [med(values('chain-results.jsonl',f'chain-{n}',v),'merge_cpu_seconds')
                for v in ('baseline','packed','filtered','arena_counted','h3_fused')]
        chain.append(f'| {n} | '+' | '.join(f'{number:.4f}' for number in nums)+' |')
    span_rows = [json.loads(line) for line in (HERE/'span-results.jsonl').read_text().splitlines()]
    spans = defaultdict(list)
    for row in span_rows:
        spans[row['length']].append(row)
    span_table = ['| 长度 | 边界数组字节（含两端哨兵） | 合并 CPU 秒 | 每合并微秒 |',
                  '|---:|---:|---:|---:|']
    for n,rows in sorted(spans.items()):
        assert len(rows)==3 and {r['repetition'] for r in rows}=={0,1,2}
        assert all(r['buffer_bytes']==n+2 for r in rows)
        span_table.append(f'| {n} | {n+2} | {med(rows,"cpu_seconds"):.4f} | '
                          f'{med(rows,"cpu_ns_per_merge")/1000:.2f} |')
    placeholders = {'REAL_TABLE':'\n'.join(table),'EXPANDED_TABLE':'\n'.join(expanded),
                    'PARALLEL_TABLE':'\n'.join(parallel),'CHAIN_TABLE':'\n'.join(chain),
                    'SPAN_TABLE':'\n'.join(span_table),'MEMORY_TABLE':'\n'.join(memory_table)}
    report = (HERE/'REPORT.template.md').read_text()
    for name,value in placeholders.items():
        assert '{{'+name+'}}' in report
        report = report.replace('{{'+name+'}}',value)
    (HERE/'REPORT.md').write_text(report)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    verification = dict(timed_training_processes=len(all_rows),
                        boundary_micro_processes=len(span_rows),
                        separate_memory_profiles=len(memory_rows),
                        semantic_groups=len(semantic),all_fingerprints_match=True,
                        all_snapshots_match=True,
                        result_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                       for name in FINAL_FILES+['span-results.jsonl','parallel-memory.jsonl']})
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()

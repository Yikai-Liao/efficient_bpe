"""Validate the frozen matrix and derive tables without hand-entered timings."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parents[1]


def main():
    results = HERE / 'results'
    raw = results / 'baseline.jsonl'
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    fixtures = json.loads((results / 'fixtures.json').read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['split'], row['variant']].append(row)
    assert len(rows) == len(fixtures) * 3 * 5
    summary, timing, phases, memory = [], [], [], []
    for fixture in fixtures:
        ds, split = fixture['dataset'], fixture['split']
        case = dict(dataset=ds, split=split, variants={})
        fingerprints = set()
        counters = set()
        for variant in ('python', 'checked', 'unchecked'):
            samples = groups[ds, split, variant]
            assert len(samples) == 5
            assert {r['repetition'] for r in samples} == set(range(5))
            for row in samples:
                assert row['fixture_sha256'] == fixture['fixture_sha256']
                assert row['input_sha256'] == fixture['input_sha256']
                fingerprints.add(row['fingerprint'])
                counters.add(tuple(row[k] for k in (
                    'rules', 'actual_merges', 'position_visits', 'stale_visits',
                    'heap_pops', 'max_token_length', 'backend_buffer_bytes',
                    'initial_occurrence_bytes')))
            case['variants'][variant] = {
                metric: dict(median=statistics.median(r[metric] for r in samples),
                             minimum=min(r[metric] for r in samples),
                             maximum=max(r[metric] for r in samples))
                for metric in ('train_seconds', 'init_seconds', 'merge_seconds',
                               'call_seconds', 'call_cpu_seconds', 'vm_hwm_mib')
            }
        assert len(fingerprints) == 1, (ds, split, 'fingerprints')
        assert len(counters) == 1, (ds, split, 'operation counts')
        case['fingerprint'] = fingerprints.pop()
        py, checked, unchecked = (case['variants'][v] for v in ('python', 'checked', 'unchecked'))
        median = lambda result, key='train_seconds': result[key]['median']
        case['python_over_checked'] = median(py) / median(checked)
        case['checked_over_unchecked'] = median(checked) / median(unchecked)
        label = f'{ds} / {split}'
        timing.append(f'| {label} | {median(py):.4f} | {median(checked):.4f} | '
                      f'{median(unchecked):.4f} | {case["python_over_checked"]:.2f}× | '
                      f'{case["checked_over_unchecked"]:.3f}× |')
        phases.append(f'| {label} | {median(checked, "init_seconds"):.4f} | '
                      f'{median(checked, "merge_seconds"):.4f} | '
                      f'{checked["train_seconds"]["minimum"]:.4f}–{checked["train_seconds"]["maximum"]:.4f} | '
                      f'{unchecked["train_seconds"]["minimum"]:.4f}–{unchecked["train_seconds"]["maximum"]:.4f} |')
        memory.append(f'| {label} | {median(py, "vm_hwm_mib"):.1f} | '
                      f'{median(checked, "vm_hwm_mib"):.1f} | {median(unchecked, "vm_hwm_mib"):.1f} |')
        summary.append(case)
    verification = dict(training_processes=len(rows), semantic_groups=len(fixtures),
                        repeats=5, all_fingerprints_match=True, all_operation_counts_match=True,
                        baseline_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    (results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (results / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    template = (HERE / 'REPORT.template.md').read_text()
    for key, value in (('TIMING', timing), ('PHASES', phases), ('MEMORY', memory)):
        template = template.replace('{{' + key + '}}', '\n'.join(value))
    (HERE / 'REPORT.md').write_text(template)
    print(json.dumps(verification))


if __name__ == '__main__':
    main()

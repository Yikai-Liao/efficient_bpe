# Compact pair-owned variant: light screening

The compact variant passed the focused Rust checks and all 120 independent full-trace oracle comparisons: 20 cases (including 8 random cases), workers 1/2/4, and checked/unchecked access. There were no skipped cases.

The one-shot screen produced 16 quick runs (`n=1`, both bounds), 8 smoke runs (`n=1`, checked), and 12 edge-case runs (`n=1`, checked). Each benchmark sidecar records the binary, fixture manifest, and complete source hashes; all runs used one binary and matching source hashes. Fingerprints were stable across configurations for every benchmark case.

## Smoke measurements

Each cell shows `call_seconds / VmHWM MiB / plan+group capacity MiB` for baseline then compact; each is a single run (`n=1`). Compact group capacity was 1 KiB at one worker and 4 KiB at four workers; baseline group capacity was 0.

| Input | Workers | 16-byte baseline | 4-byte compact |
|---|---:|---:|---:|
| English 4 MiB | 1 | 2.872 / 111.98 / 8.000 | 2.782 / 107.09 / 2.001 |
| English 4 MiB | 4 | 1.221 / 137.59 / 8.000 | 1.119 / 135.87 / 2.004 |
| Chinese 4 MiB | 1 | 1.525 / 123.15 / 0.500 | 1.573 / 123.38 / 0.126 |
| Chinese 4 MiB | 4 | 0.712 / 120.99 / 0.500 | 0.692 / 120.13 / 0.129 |

Compact reduced reported plan capacity by 75% on English and Chinese; the group buffer added only 1–4 KiB. English was faster and used less RSS at both worker counts. Chinese compact was 3.2% slower and 0.22 MiB higher RSS at one worker, while at four workers it was 2.8% faster and 0.86 MiB lower RSS.

The three edge cases show one apparent regression: `chain-2000` at one worker rose from 0.1168 to 0.1362 s (+16.6%), with RSS falling from 3.90 to 3.63 MiB. At four workers it was effectively flat (0.34653 vs 0.34666 s). `single-run-a-65536` and `single-piece-ab-65536` were faster with compact at both worker counts. All edge figures are `n=1`, so treat the small changes as screening signals.

## Phase timing within the compact-v1 binary

This comparison uses both variants from the same compact-v1 smoke binary and source-hash set. It reuses one-shot rows (`n=1`), checked bounds; each pair is baseline 16B → compact 4B. Fingerprints match for all four fixture/worker pairs.

| Input | W | Call s / RSS MiB | Init s | Merge s | Train s | Call − train s |
|---|---:|---:|---:|---:|---:|---:|
| English 4 MiB | 1 | 2.872→2.782 / 111.980→107.086 | 0.484→0.449 | 2.275→2.222 | 2.759→2.671 | 0.113→0.112 |
| English 4 MiB | 4 | 1.221→1.119 / 137.594→135.871 | 0.139→0.125 | 0.995→0.916 | 1.135→1.041 | 0.087→0.078 |
| Chinese 4 MiB | 1 | 1.525→1.573 / 123.152→123.375 | 0.547→0.569 | 0.825→0.860 | 1.372→1.429 | 0.152→0.144 |
| Chinese 4 MiB | 4 | 0.712→0.692 / 120.988→120.129 | 0.219→0.208 | 0.410→0.429 | 0.628→0.637 | 0.084→0.055 |

The call residual is not a separately measured final phase: the binary exposes no `validation_seconds` or `final_seconds`. `train_seconds` is exactly `init_seconds + merge_seconds`; `call_seconds - train_seconds` includes uninstrumented validation/setup, worker shutdown, final token traversal, and result assembly.

| Input | W | Select / plan / owner-reduce / apply wall seconds (16B → 4B) | Plan / reduce / apply worker-work seconds sum (16B → 4B) | Batch epochs / max width |
|---|---:|---|---|---:|
| English | 1 | 0.037→0.036 / 1.199→1.178 / 0.312→0.309 / 0.727→0.698 | 1.171→1.149 / 0.305→0.302 / 0.720→0.692 | 251 / 47 |
| English | 4 | 0.041→0.046 / 0.461→0.448 / 0.158→0.127 / 0.333→0.294 | 1.262→1.285 / 0.378→0.359 / 0.944→0.925 | 251 / 47 |
| Chinese | 1 | 0.027→0.030 / 0.281→0.298 / 0.242→0.261 / 0.274→0.270 | 0.262→0.277 / 0.237→0.255 / 0.268→0.264 | 275 / 50 |
| Chinese | 4 | 0.031→0.033 / 0.140→0.151 / 0.110→0.114 / 0.127→0.130 | 0.312→0.298 / 0.296→0.291 / 0.366→0.355 | 275 / 50 |

Batch epochs and maximum width are identical between layouts for each fixture: English 251 / 47; Chinese 275 / 50. English compact is faster at both worker counts; Chinese is slower at one worker and faster at four. The phase differences are small one-shot observations and are not a formal ranking.

See [phase-comparison.json](phase-comparison.json) for full-precision fields.

These are screening observations, not a formal performance ranking. See [checks.json](checks.json), [differential.json](differential.json), [quick results](quick/summary.json), [smoke results](smoke/summary.json), and [edge results](edges.jsonl).

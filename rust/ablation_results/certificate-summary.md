# Certified-prefix diagnostic

Native source: `1c2fffc`. The probe preselects a certified prefix, then still applies each rule serially. These counters measure potential epoch width, not achieved parallel speedup. Each configuration was run once; no timing comparison is drawn from this diagnostic.

Validated 80 runs across 20 fixtures: probe and `combined_filtered`, each with checked and unchecked access. Complete-output fingerprints agree, and batch counters are identical between bounds modes.

| Input | Rules | Certified epochs | Rules / epoch | Maximum width | Singleton epochs |
|---|---:|---:|---:|---:|---:|
| chain-16000--regex | 15,999 | 15,999 | 1.00 | 1 | 15999 |
| chain-2000 | 1,999 | 1,999 | 1.00 | 1 | 1999 |
| chain-4000 | 3,999 | 3,999 | 1.00 | 1 | 3999 |
| chain-8000--regex | 7,999 | 7,999 | 1.00 | 1 | 7999 |
| de-1m--regex | 3,000 | 237 | 12.66 | 55 | 26 |
| en-1m--paragraph | 3,000 | 253 | 11.86 | 59 | 17 |
| en-1m--regex | 3,000 | 243 | 12.35 | 62 | 20 |
| en-4m--regex | 3,000 | 269 | 11.15 | 44 | 23 |
| en-4m-continuous | 3,000 | 251 | 11.95 | 47 | 16 |
| ja-1m--regex | 3,000 | 219 | 13.70 | 63 | 25 |
| mixed-long-short-rare | 24 | 23 | 1.04 | 2 | 22 |
| random-131072--regex | 3,000 | 217 | 13.82 | 87 | 37 |
| rare-pair-pressure-512 | 3 | 3 | 1.00 | 1 | 3 |
| runs-131072--regex | 10 | 10 | 1.00 | 1 | 10 |
| single-piece-ab-65536 | 16 | 16 | 1.00 | 1 | 16 |
| single-run-a-65536 | 16 | 16 | 1.00 | 1 | 16 |
| weighted64-en-1m-regex | 3,000 | 243 | 12.35 | 62 | 20 |
| zh-1m--regex | 3,000 | 214 | 14.02 | 66 | 27 |
| zh-4m--regex | 3,000 | 237 | 12.66 | 62 | 41 |
| zh-4m-continuous | 3,000 | 275 | 10.91 | 50 | 53 |

The cap is min(256, remaining rules); the hit-cap counter includes the final truncated batch. No real-data maximum above reaches 256. Long chains and self-pair runs mostly remain singleton epochs. Weighted64 preserves the same grouping as its unscaled counterpart.

Semantic checks are separate: 448 full-trace Python-oracle comparisons, 24 library tests and six independent recount reference tests, including 29,523 exhaustive ternary strings and 2,000 weighted random cases. The reference tests also compare direct batch-end count deltas with a complete recount after every batch. See `certificate-differential.json` and `certificate-checks.json`.

Reproduce with `python rust/tools/certificate_summary.py --output-prefix rust/ablation_results/reruns/certificate-summary`. Raw archives are never overwritten.

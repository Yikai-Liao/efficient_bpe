# Lazy exact neighbor certificate gate and quick screen

`owned_lazy_neighbor` keeps the same exact owner trainer and compares three certificates in one binary: the original type rule, a birth-neighbor64 summary built for every retained pair, and a lazy-neighbor64 summary built only when a type conflict needs it. The lazy mode can build a cold mask in parallel above a configurable threshold. This archive validates both the serial and forced-parallel construction paths; performance is limited to one-repeat 256 KiB cells.

The final source passed **17/17** debug library tests, strict all-target Clippy, and release build. An independent Python recount matched **200/200 complete rule traces and final token sequences**: 20 cases × W1/W4 × type, eager birth, lazy default, lazy forced parallel, and lazy forced serial. The 14 authorized quick runs also matched a same-binary type/W1 full-trace reference. Exact batch widths change some intermediate birth counts, so the complete output trace is the semantic criterion.

| Input | W | Mode | Call s | CPU s | HWM MiB | Select / plan ms | Rounds |
|---|---:|---|---:|---:|---:|---:|---:|
| EN | 1 | type | 0.1005 | 0.0992 | 7.78 | 1.69 / 48.47 | 69 |
| EN | 1 | eager birth | 0.1212 | 0.1160 | 8.71 | 1.67 / 55.81 | 60 |
| EN | 1 | lazy default | 0.0894 | 0.0880 | 8.22 | 2.26 / 42.62 | 61 |
| EN | 4 | type | 0.0611 | 0.1644 | 8.89 | 1.44 / 24.04 | 69 |
| EN | 4 | eager birth | 0.0516 | 0.1625 | 8.94 | 1.73 / 18.59 | 60 |
| EN | 4 | lazy default | 0.0584 | 0.1726 | 8.51 | 3.02 / 26.09 | 61 |
| EN | 4 | lazy forced serial | 0.0574 | 0.1714 | 8.73 | 2.98 / 25.54 | 61 |
| ZH | 1 | type | 0.0433 | 0.0433 | 7.16 | 0.90 / 11.09 | 87 |
| ZH | 1 | eager birth | 0.0500 | 0.0499 | 7.75 | 1.05 / 11.37 | 82 |
| ZH | 1 | lazy default | 0.0427 | 0.0427 | 7.05 | 1.01 / 10.82 | 82 |
| ZH | 4 | type | 0.0223 | 0.0774 | 7.90 | 1.01 / 5.82 | 87 |
| ZH | 4 | eager birth | 0.0266 | 0.0748 | 8.99 | 0.97 / 6.12 | 82 |
| ZH | 4 | lazy default | 0.0295 | 0.0780 | 8.02 | 1.54 / 8.18 | 82 |
| ZH | 4 | lazy forced serial | 0.0284 | 0.0695 | 7.98 | 1.34 / 7.07 | 82 |

The lazy certificate records 16 negative admissions on EN and 9 on ZH, versus 15 and 9 for the eager birth summary. EN's 68 mask builds and 10 cache hits visit 32,801 historical positions (8,244 stale); ZH's 66 builds and 2 hits visit 4,751 (160 stale). The eager summary instead reads EN 260,980 initial plus 269,745 birth positions, and ZH 108,314 plus 40,874. Lazy maximum single-mask scans are 4,091 EN and 550 ZH. Its peak cache capacity-scaled payload proxy is 224 bytes EN and 896 ZH, versus the eager retained-summary proxy of about 224 KiB EN and 412 KiB ZH at W4. These proxies are **not** measured HashMap allocation or simultaneous process memory; process HWM is reported separately above.

The default parallel threshold is 8,192. **Neither quick input crosses it:** `lazy_parallel_builds=0` in every default cell. The forced-serial W4 controls therefore exercise the same mask-build path as default and cannot isolate a parallel-build performance effect. The forced-parallel path is covered by the 200-case correctness oracle and directed unit tests. In this n=1 screen, lazy lowers EN W4 HWM versus eager birth (8.94→8.51 MiB) but its call is slower (0.0516→0.0584 s); ZH shows 8.99→8.02 MiB and 0.0266→0.0295 s. These observations support a space-saving mechanism without establishing a throughput improvement or a stable ranking.

W1 was pinned to CPU 5 and W4 to CPUs `[0,1,2,5]`. `call_seconds` covers the full training call, `call_cpu_seconds` is process CPU time, and `train_vm_hwm_mib` is the process high-water mark sampled before fingerprint/trace construction. `select_seconds` and `plan_seconds` are individual reported phases; their values must not be added to other possibly nested phases without checking implementation boundaries. Raw metrics, including mask seconds, hits, capacity proxies, and visits, are in `quick.jsonl`; `summary.json` extracts the comparisons.

The independent source is preserved in `new-source-snapshot.tar.gz` with per-file hashes in `new-source-hashes.json`; `shared-source-provenance.json` records the byte-identical shared Rust base. The frozen release binary is under ignored `rust/target/reruns/radical-lazy-neighbor-gate-v1/`. `checks.json` records hashes and validation; `finalize.py` rechecks provenance and result completeness without another training run.

# First spatial-probe screen

The frozen `owned_probe` binary passed 8/8 local tests, strict Clippy, a
release build, and 80/80 complete rule/final-token oracle cases (the shared
archive has 160/160 including the separate counted-delta variant). The
fixed-budget 256 KiB quick runs used the same binary for `off` and
`budgeted`, with 512 rules, `min_frequency=2`, chunk size 4096, lazy heap,
and W1/W4 CPU pinning. Full commands, hashes, sidecars, and raw JSON are in
[`radical-owned-extensions-v1`](../../../batch_results/radical-owned-extensions-v1/README.md).

| Fixture | Workers | Off call / train HWM | Budgeted call / train HWM | Epochs off → budgeted |
|---|---:|---:|---:|---:|
| EN | 1 | 0.09847 s / 8.92 MiB | 0.11449 s / 9.02 MiB | 69 → 53 |
| EN | 4 | 0.04828 s / 9.15 MiB | 0.05504 s / 8.93 MiB | 69 → 53 |
| ZH | 1 | 0.05416 s / 8.45 MiB | 0.05287 s / 8.25 MiB | 87 → 80 |
| ZH | 4 | 0.03485 s / 9.22 MiB | 0.02698 s / 9.19 MiB | 87 → 80 |

EN's maximum width increased 22 → 32 and generated births fell 284,407 →
284,098. ZH's maximum width stayed 26 and births fell 51,001 → 50,899.
Historical posting visits were unchanged at 202,877 EN and 29,670 ZH;
probe visits are separate. At W4, budgeted mode probed 8,998 EN positions
(5,275 stale) and 873 ZH positions (185 stale), with 49/56 conflict stops,
32/12 disjoint proofs, and no budget stops. Probe wall time was 0.00060 s
EN and 0.00009 s ZH within selection. The extra visits were well below the
configured twice-selected-history limit.

The certificate genuinely widens some exact batches and reduces intermediate
births with little direct probe time. Total call time moved in opposite
directions on EN and ZH; each cell is one short run, so this screen does not
establish a net speedup or RSS benefit. The coordinator still performs every
witness scan serially. A repeated or larger-input comparison would be needed
before making this the default certificate.

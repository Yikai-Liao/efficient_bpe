# Sparse pair-owner: light screen

The fair single-rule control is `parallel_pair_owned_single`: pair frequencies and candidate heaps remain sharded, but each merge wakes all workers; its prefetch page is capped at one entry. The candidate `parallel_sparse_owner` wakes only the workers that hold the selected pair. All timings below are one-shot (`n=1`) checked-bound results.

## Smoke: 4 MiB continuous fixtures

| Input | Workers | Single control: s / MiB / round messages | Sparse owner: s / MiB / round messages | Time change | Sparse active plan / owner / apply totals (rules; applications) | Prune mailbox locks |
|---|---:|---:|---:|---:|---:|---:|
| English | 1 | 3.042 / 93.62 / 30,030 | 2.862 / 109.91 / 18,030 | -5.9% | 3,000 / 3,000 / 3,000 (3,000; 2,893,635) | 80,692 |
| English | 4 | 2.052 / 121.96 / 120,120 | 1.320 / 136.48 / 71,968 | -35.7% | 11,962 / 11,991 / 11,971 (3,000; 2,893,635) | 138,971 |
| Chinese | 1 | 2.010 / 106.57 / 30,060 | 1.637 / 123.33 / 18,060 | -18.6% | 3,000 / 3,000 / 3,000 (3,000; 646,325) | 122,217 |
| Chinese | 4 | 1.318 / 118.38 / 120,240 | 0.894 / 143.01 / 71,856 | -32.2% | 11,921 / 11,958 / 11,929 (3,000; 646,325) | 178,612 |

Sparse owner reduced call time in all four smoke configurations: about 5.9%/35.7% on English and 18.6%/32.2% on Chinese at one/four workers. RSS increased by about 16.3/14.5 MiB on English and 16.8/24.6 MiB on Chinese. Round messages fell by roughly 40% in every case. For the four-worker English run, sparse owner activated about 12k worker-participations in each phase across 3,000 merge rounds, versus 12,000 worker-rounds for an all-worker dispatch; the Chinese totals are similar.

`prune_mailbox_locks` counts data mailbox locks used to publish pruned holder entries; it is not a control wakeup message count. `round_messages` is the control-message counter used for direct comparison. `CoreStats.heap_pops` only counts pops from owner-local pair heaps; it excludes coordinator `Frontier` stale-entry pops and rebuild work.


## Chain edge case and activation interpretation

The `chain-2000` rows compare `parallel_pair_owned_single` (dense single-rule dispatch, prefetch cap 1) with `parallel_sparse_owner`; both are `n=1`. Dense plan/reduce/apply dispatch slots are `workers × 1,999 rule rounds`. Sparse owner totals are measured active worker participations.

| Workers | Dense single: seconds / RSS MiB / round messages / heap pops | Sparse: seconds / RSS MiB / round messages / heap pops | Sparse active plan / owner-reduce / apply totals | Rules / actual applications / prune locks |
|---:|---:|---:|---:|---:|
| 1 | 0.108081 / 3.711 / 20,002 / 3,997 | 0.071595 / 3.652 / 12,002 / 3,997 | 1,999 / 1,999 / 1,999 | 1,999 / 1,999 / 3,997 |
| 4 | 0.335355 / 4.125 / 80,008 / 9,977 | 0.117647 / 4.031 / 20,212 / 3,997 | 1,999 / 4,592 / 3,499 | 1,999 / 1,999 / 3,997 |

Sparse owner was 33.8% faster at one worker and 64.9% faster at four; RSS was 0.059 and 0.094 MiB lower. At four workers, planning touched one worker per rule round, versus four dense dispatch slots; reduction and apply touched 4,592 and 3,499 workers across the 1,999 rounds. The one-worker message reduction (8,000 messages) exactly matches the dense single variant's 8,000 candidate-refill messages. The four-worker chain also benefits from dispatching fewer location workers.

The natural smoke cases show a different mechanism: plan/owner/apply participation is near all-worker dispatch. Round messages fall by 12,000 at one worker, exactly the baseline's candidate-refill messages; at four workers they fall by 48,152 (English) and 48,384 (Chinese), compared with 48,000 candidate-refill messages. This points to removed Prefetch/Return phases as the main natural-corpus message savings, rather than substantially fewer location-worker activations.

Capacity counters only partly explain higher smoke RSS. Index-position peaks, final position capacity, plan, edge, and owner-heap capacities match between variants; scalar capacities differ by at most a small amount. Sparse Router buckets are 32-byte `Delta` records: 0.5/1 MiB on English and 16 MiB on Chinese at one/four workers. That is close to the Chinese RSS increases (+16.75/+24.64 MiB), but far below English's +16.29/+14.52 MiB. Sparse adds 16-64 KiB for shared lengths and under 1 KiB for the frontier. The measured capacities therefore leave English's RSS increase unexplained and do not form a complete peak-allocation ledger.

`heap_pops` is the `CoreStats` counter for owner-local pair-heap pops. It does not include coordinator `Frontier` stale-entry pops or rebuild work, so it is not a count of all heap operations. `prune_mailbox_locks` counts data-mailbox locks while publishing pruned holder entries; it is separate from control wakeup messages.

## Quick and edge results

The 8-run quick screen (English/Chinese 256 KiB, one and four workers) passed with stable fingerprints. Sparse owner was faster on English at both worker counts and on Chinese at four workers; Chinese one-worker time was nearly even (+1.5%, within one-shot noise). RSS was higher in all four cases. The 12 edge runs also passed:

| Quick input | Workers | Single control s / MiB | Sparse owner s / MiB | Time change |
|---|---:|---:|---:|---:|
| English 256 KiB | 1 | 0.2086 / 9.69 | 0.1943 / 9.84 | -6.9% |
| English 256 KiB | 4 | 0.1671 / 11.20 | 0.1158 / 12.24 | -30.7% |
| Chinese 256 KiB | 1 | 0.1005 / 6.76 | 0.1020 / 8.94 | +1.4% |
| Chinese 256 KiB | 4 | 0.1426 / 9.41 | 0.0815 / 11.89 | -42.8% |


| Case | Workers | Single control s / MiB | Sparse owner s / MiB | Time change |
|---|---:|---:|---:|---:|
| chain-2000 | 1 | 0.108 / 3.71 | 0.072 / 3.65 | -33.8% |
| chain-2000 | 4 | 0.335 / 4.12 | 0.118 / 4.03 | -64.9% |
| single-run-a-65536 | 1 | 0.028 / 4.79 | 0.026 / 4.91 | -6.5% |
| single-run-a-65536 | 4 | 0.017 / 5.14 | 0.017 / 4.78 | -0.1% |
| single-piece-ab-65536 | 1 | 0.035 / 4.84 | 0.026 / 4.70 | -26.5% |
| single-piece-ab-65536 | 4 | 0.019 / 5.02 | 0.016 / 4.73 | -15.7% |

No edge configuration regressed in this one-shot screen. These data are a lightweight algorithm-selection signal, not a formal ranking. See [checks.json](checks.json), [differential.json](differential.json), [quick results](quick/summary.json), [smoke results](smoke/summary.json), and [edge results](edges.jsonl).

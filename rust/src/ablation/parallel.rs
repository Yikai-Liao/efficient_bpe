//! Persistent-thread BPE ablations with a global exact-greedy coordinator.
//!
//! `broadcast` and `owner` partition only at permanent zero separators.
//! `occurrence` keeps one continuous corpus and partitions each selected
//! rule's independent, preplanned replacements instead. The latter pays a
//! serial planning cost, reported separately; it is not a claim of speedup.

#[path = "parallel_certified.rs"]
mod certified;
#[path = "parallel_occurrence.rs"]
mod occurrence;
#[path = "parallel_occurrence_snapshot.rs"]
mod occurrence_snapshot;
#[path = "parallel_piece.rs"]
mod piece;
#[path = "parallel_sharded.rs"]
mod sharded;
#[path = "parallel_sparse_owner.rs"]
mod sparse_owner;
#[path = "parallel_spatial.rs"]
mod spatial;

use super::{Options, Result, validate};
use crate::{Prepared, Rule, TrainError, TrainOptions, TrainResult};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

#[derive(Clone, Copy, Eq, PartialEq)]
struct HeapEntry {
    frequency: u64,
    key: u64,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.key.cmp(&self.key))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

fn initial_heap(frequencies: &HashMap<u64, u64>, min_frequency: u64) -> BinaryHeap<HeapEntry> {
    frequencies
        .iter()
        .filter_map(|(&key, &frequency)| {
            (frequency >= min_frequency).then_some(HeapEntry { frequency, key })
        })
        .collect()
}

fn pop_best(
    heap: &mut BinaryHeap<HeapEntry>,
    frequencies: &HashMap<u64, u64>,
    min_frequency: u64,
    heap_pops: &mut usize,
) -> Option<(u64, u64)> {
    loop {
        let entry = heap.pop()?;
        *heap_pops += 1;
        let current = frequencies.get(&entry.key).copied().unwrap_or(0);
        if current < min_frequency {
            continue;
        }
        if current != entry.frequency {
            heap.push(HeapEntry {
                frequency: current,
                key: entry.key,
            });
            continue;
        }
        return Some((entry.key, current));
    }
}

fn apply_delta(
    frequencies: &mut HashMap<u64, u64>,
    key: u64,
    amount: i128,
) -> std::result::Result<(), TrainError> {
    let current = frequencies.get(&key).copied().unwrap_or(0);
    let updated = if amount >= 0 {
        current.checked_add(
            u64::try_from(amount).map_err(|_| TrainError::Overflow("pair delta exceeds u64"))?,
        )
    } else {
        current.checked_sub(
            u64::try_from(-amount)
                .map_err(|_| TrainError::Overflow("pair delta magnitude exceeds u64"))?,
        )
    }
    .ok_or(TrainError::InternalInvariant(
        "global pair frequency underflow/overflow",
    ))?;
    frequencies.insert(key, updated);
    Ok(())
}

fn add_delta(deltas: &mut HashMap<u64, i128>, key: u64, amount: i128) {
    *deltas.entry(key).or_default() += amount;
}

struct CoreStats {
    init_seconds: f64,
    merge_seconds: f64,
    actual_merges: usize,
    position_visits: usize,
    stale_visits: usize,
    heap_pops: usize,
    backend_buffer_bytes: usize,
    initial_occurrence_bytes: usize,
    max_token_length: u32,
    corpus_positions: usize,
}

fn core_result(merges: Vec<Rule>, final_tokens: Vec<u32>, stats: CoreStats) -> TrainResult {
    TrainResult {
        rules: merges.len(),
        merges,
        final_tokens,
        init_seconds: stats.init_seconds,
        merge_seconds: stats.merge_seconds,
        train_seconds: stats.init_seconds + stats.merge_seconds,
        actual_merges: stats.actual_merges,
        position_visits: stats.position_visits,
        stale_visits: stats.stale_visits,
        heap_pops: stats.heap_pops,
        backend_buffer_bytes: stats.backend_buffer_bytes,
        initial_occurrence_bytes: stats.initial_occurrence_bytes,
        max_token_length: stats.max_token_length,
        corpus_positions: stats.corpus_positions,
    }
}

fn metric(metrics: &mut BTreeMap<String, f64>, name: &str, value: impl Into<f64>) {
    metrics.insert(name.to_owned(), value.into());
}

pub fn train(
    input: Prepared,
    options: Options,
    mode: &str,
) -> std::result::Result<Result, TrainError> {
    if options.workers == 0 {
        return Err(TrainError::InvalidInput("workers must be positive"));
    }
    if !matches!(
        mode,
        "broadcast"
            | "owner"
            | "serial"
            | "occurrence"
            | "occurrence_snapshot"
            | "occurrence_adaptive"
            | "occurrence_adaptive_256"
            | "occurrence_adaptive_4096"
            | "certified"
            | "certified_single"
            | "batch_relaxed"
            | "pair_owned"
            | "pair_owned_pipeline"
            | "pair_owned_spatial"
            | "pair_owned_single"
            | "pair_owned_compact"
            | "sparse_owner"
            | "sparse_owner_all"
    ) {
        return Err(TrainError::InvalidInput("unknown parallel mode"));
    }
    validate(&input, options)?;
    if input.corpus.len() == 1 {
        let core = crate::train(
            input,
            TrainOptions {
                max_merges: options.max_merges,
                min_frequency: options.min_frequency,
                bounds: options.bounds,
            },
        )?;
        let mut metrics = BTreeMap::new();
        metric(&mut metrics, "workers_requested", options.workers as f64);
        metric(&mut metrics, "workers_actual", 0.0);
        return Ok(Result { core, metrics });
    }
    match mode {
        "pair_owned_spatial" => {
            if options.bounds == crate::Bounds::Unchecked {
                spatial::train::<true>(input, options, 256)
            } else {
                spatial::train::<false>(input, options, 256)
            }
        }
        "pair_owned_pipeline" => {
            if options.bounds == crate::Bounds::Unchecked {
                sharded::train_pipeline::<true>(input, options, 256)
            } else {
                sharded::train_pipeline::<false>(input, options, 256)
            }
        }
        "sparse_owner_all" => {
            if options.bounds == crate::Bounds::Unchecked {
                sparse_owner::train_all::<true>(input, options)
            } else {
                sparse_owner::train_all::<false>(input, options)
            }
        }
        "pair_owned_single" => {
            if options.bounds == crate::Bounds::Unchecked {
                sharded::train::<true>(input, options, 1)
            } else {
                sharded::train::<false>(input, options, 1)
            }
        }
        "sparse_owner" => {
            if options.bounds == crate::Bounds::Unchecked {
                sparse_owner::train::<true>(input, options)
            } else {
                sparse_owner::train::<false>(input, options)
            }
        }
        "pair_owned_compact" => {
            if options.bounds == crate::Bounds::Unchecked {
                sharded::train_compact::<true>(input, options, 256)
            } else {
                sharded::train_compact::<false>(input, options, 256)
            }
        }
        "pair_owned" => {
            if options.bounds == crate::Bounds::Unchecked {
                sharded::train::<true>(input, options, 256)
            } else {
                sharded::train::<false>(input, options, 256)
            }
        }
        "batch_relaxed" => {
            if options.bounds == crate::Bounds::Unchecked {
                certified::train_relaxed::<true>(input, options, 256)
            } else {
                certified::train_relaxed::<false>(input, options, 256)
            }
        }
        "certified" | "certified_single" => {
            let cap = if mode == "certified_single" { 1 } else { 256 };
            if options.bounds == crate::Bounds::Unchecked {
                certified::train::<true>(input, options, cap)
            } else {
                certified::train::<false>(input, options, cap)
            }
        }
        "broadcast" => piece::train(input, options, false, false, false),
        "owner" if options.workers > 128 => piece::train(input, options, false, false, true),
        "owner" => piece::train(input, options, true, false, false),
        "serial" => piece::train(input, options, false, true, false),
        "occurrence" => occurrence::train(input, options),
        "occurrence_snapshot" => occurrence_snapshot::train(input, options),
        "occurrence_adaptive" => occurrence_snapshot::train_adaptive(input, options, 1024),
        "occurrence_adaptive_256" => occurrence_snapshot::train_adaptive(input, options, 256),
        "occurrence_adaptive_4096" => occurrence_snapshot::train_adaptive(input, options, 4096),
        _ => Err(TrainError::InvalidInput("unknown parallel mode")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, TrainOptions};

    fn prepared(words: &[(&[u32], u64)], alphabet: usize) -> Prepared {
        let mut corpus = vec![0];
        let mut pivots = Vec::new();
        let mut weights = Vec::new();
        for &(word, weight) in words {
            pivots.push(corpus.len() as u32);
            weights.push(weight);
            corpus.extend_from_slice(word);
            corpus.push(0);
        }
        Prepared {
            corpus,
            initial_lengths: vec![1; alphabet + 1],
            pivots,
            weights,
        }
    }

    fn check(input: Prepared, workers: usize, max_merges: usize, min_frequency: u64) {
        let baseline = crate::train(
            input.clone(),
            TrainOptions {
                max_merges,
                min_frequency,
                bounds: Bounds::Checked,
            },
        )
        .unwrap();
        for mode in [
            "serial",
            "broadcast",
            "owner",
            "occurrence",
            "occurrence_snapshot",
            "occurrence_adaptive",
            "occurrence_adaptive_256",
            "occurrence_adaptive_4096",
        ] {
            let options = Options {
                max_merges,
                min_frequency,
                bounds: Bounds::Checked,
                workers,
            };
            let result = train(input.clone(), options, mode).unwrap();
            assert_eq!(result.core.merges, baseline.merges, "{mode}");
            assert_eq!(result.core.final_tokens, baseline.final_tokens, "{mode}");
            assert_eq!(result.core.actual_merges, baseline.actual_merges, "{mode}");
            assert_eq!(
                result.core.position_visits, baseline.position_visits,
                "{mode}"
            );
            assert_eq!(result.core.stale_visits, baseline.stale_visits, "{mode}");
        }
        for mode in ["occurrence_snapshot", "occurrence_adaptive"] {
            let unchecked = train(
                input.clone(),
                Options {
                    max_merges,
                    min_frequency,
                    bounds: Bounds::Unchecked,
                    workers,
                },
                mode,
            )
            .unwrap();
            assert_eq!(unchecked.core.merges, baseline.merges, "{mode}");
            assert_eq!(unchecked.core.final_tokens, baseline.final_tokens, "{mode}");
            assert_eq!(
                unchecked.core.actual_merges, baseline.actual_merges,
                "{mode}"
            );
            assert_eq!(
                unchecked.core.position_visits, baseline.position_visits,
                "{mode}"
            );
            assert_eq!(unchecked.core.stale_visits, baseline.stale_visits, "{mode}");
            assert_eq!(unchecked.metrics["unchecked_corpus_access"], 1.0);
        }
    }

    #[test]
    fn weighted_overlap_and_adjacent_abab() {
        let input = prepared(
            &[
                (&[1, 1, 1, 1, 1, 1, 1], 5),
                (&[1, 2, 1, 2, 1, 2, 1, 2], 3),
                (&[2, 1, 2, 1, 2, 1, 2, 1], 2),
            ],
            2,
        );
        for workers in [1, 2, 4, 6] {
            check(input.clone(), workers, 20, 1);
        }
    }

    #[test]
    fn one_long_piece_and_many_pieces() {
        let mut long = vec![1; 64];
        long.extend((0..32).flat_map(|_| [1, 2]));
        check(prepared(&[(&long, 1)], 2), 6, 25, 1);
        let words: Vec<(Vec<u32>, u64)> = (0..12)
            .map(|i| {
                (
                    (0..(6 + i % 5)).map(|j| (1 + (i + j) % 3) as u32).collect(),
                    (1 + i % 4) as u64,
                )
            })
            .collect();
        let references: Vec<(&[u32], u64)> =
            words.iter().map(|(w, f)| (w.as_slice(), *f)).collect();
        check(prepared(&references, 3), 6, 20, 2);
    }

    #[test]
    fn random_continuous_and_uneven_pieces() {
        let mut seed = 0x60ba_1963_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize
        };
        for case in 0..120 {
            let count = if case % 2 == 0 { 1 } else { 1 + next() % 8 };
            let mut words = Vec::new();
            let mut seen = [false; 4];
            for index in 0..count {
                let length = if index == 0 {
                    30 + next() % 80
                } else {
                    1 + next() % 12
                };
                let word: Vec<u32> = (0..length)
                    .map(|_| {
                        let id = 1 + next() % 3;
                        seen[id] = true;
                        id as u32
                    })
                    .collect();
                words.push((word, (1 + next() % 4) as u64));
            }
            for (id, present) in seen.iter().enumerate().skip(1) {
                if !present {
                    words.push((vec![id as u32], 1));
                }
            }
            let refs: Vec<(&[u32], u64)> = words.iter().map(|(w, f)| (w.as_slice(), *f)).collect();
            check(
                prepared(&refs, 3),
                [1, 2, 4, 6][case % 4],
                15,
                (1 + next() % 3) as u64,
            );
        }
    }

    #[test]
    fn maximum_valid_weight_and_unchecked_bounds() {
        let input = prepared(&[(&[1, 1], u64::MAX)], 1);
        let baseline = crate::train(
            input.clone(),
            TrainOptions {
                max_merges: 1,
                min_frequency: u64::MAX,
                bounds: Bounds::Unchecked,
            },
        )
        .unwrap();
        for mode in [
            "serial",
            "broadcast",
            "owner",
            "occurrence",
            "occurrence_snapshot",
            "occurrence_adaptive",
            "occurrence_adaptive_256",
            "occurrence_adaptive_4096",
        ] {
            let result = train(
                input.clone(),
                Options {
                    max_merges: 1,
                    min_frequency: u64::MAX,
                    bounds: Bounds::Unchecked,
                    workers: 6,
                },
                mode,
            )
            .unwrap();
            assert_eq!(result.core.merges, baseline.merges, "{mode}");
            assert_eq!(result.core.final_tokens, baseline.final_tokens, "{mode}");
        }
    }

    #[test]
    fn owner_really_skips_idle_workers_and_continuous_mode_uses_them() {
        let words: Vec<(Vec<u32>, u64)> = std::iter::once((vec![1; 48], 5))
            .chain((2..=8).map(|id| (vec![id], 1)))
            .collect();
        let refs: Vec<(&[u32], u64)> = words.iter().map(|(w, f)| (w.as_slice(), *f)).collect();
        let input = prepared(&refs, 8);
        let options = Options {
            max_merges: 5,
            min_frequency: 2,
            bounds: Bounds::Checked,
            workers: 6,
        };
        let broadcast = train(input.clone(), options, "broadcast").unwrap();
        let owner = train(input, options, "owner").unwrap();
        assert_eq!(owner.core.merges, broadcast.core.merges);
        assert_eq!(owner.core.final_tokens, broadcast.core.final_tokens);
        assert!(owner.metrics["round_messages"] < broadcast.metrics["round_messages"]);
        assert_eq!(broadcast.metrics["owner_directory_peak"], 0.0);

        let fallback = train(
            prepared(&refs, 8),
            Options {
                workers: 129,
                ..options
            },
            "owner",
        )
        .unwrap();
        assert_eq!(fallback.core.merges, broadcast.core.merges);
        assert_eq!(fallback.core.final_tokens, broadcast.core.final_tokens);
        assert_eq!(fallback.metrics["owner_fallback_broadcast"], 1.0);
        assert_eq!(fallback.metrics["owner_directory_peak"], 0.0);

        let one_piece = prepared(&[(&[1; 48], 1)], 1);
        let owner = train(one_piece.clone(), options, "owner").unwrap();
        let occurrence = train(one_piece, options, "occurrence").unwrap();
        let snapshot = train(
            prepared(&[(&[1; 48], 1)], 1),
            options,
            "occurrence_snapshot",
        )
        .unwrap();
        assert_eq!(owner.metrics["workers_actual"], 1.0);
        assert_eq!(occurrence.metrics["workers_actual"], 6.0);
        assert_eq!(snapshot.metrics["workers_actual"], 6.0);
        assert_eq!(owner.core.final_tokens, occurrence.core.final_tokens);
        assert_eq!(owner.core.final_tokens, snapshot.core.final_tokens);
    }

    #[test]
    fn snapshot_aa_and_abab_cross_worker_chunk_boundaries() {
        let cases = [
            prepared(&[(&vec![1; 61], 1)], 1),
            prepared(&[(&(0..41).flat_map(|_| [1, 2]).collect::<Vec<_>>(), 1)], 2),
        ];
        for input in cases {
            let baseline = crate::train(
                input.clone(),
                TrainOptions {
                    max_merges: 1,
                    min_frequency: 1,
                    bounds: Bounds::Checked,
                },
            )
            .unwrap();
            for workers in [1, 2, 4, 6] {
                let options = Options {
                    max_merges: 1,
                    min_frequency: 1,
                    bounds: Bounds::Checked,
                    workers,
                };
                let old = train(input.clone(), options, "occurrence").unwrap();
                let snapshot = train(input.clone(), options, "occurrence_snapshot").unwrap();
                assert_eq!(snapshot.core.merges, baseline.merges);
                assert_eq!(snapshot.core.final_tokens, baseline.final_tokens);
                assert_eq!(snapshot.core.actual_merges, baseline.actual_merges);
                assert_eq!(snapshot.core.position_visits, baseline.position_visits);
                assert_eq!(snapshot.core.stale_visits, baseline.stale_visits);
                assert_eq!(snapshot.core.final_tokens, old.core.final_tokens);
                assert_eq!(snapshot.metrics["barriers"], 2.0);
            }
        }
    }

    #[test]
    fn adaptive_dispatches_on_fixed_grain_boundary() {
        let options = Options {
            max_merges: 1,
            min_frequency: 1,
            bounds: Bounds::Checked,
            workers: 2,
        };
        for (characters, expected_parallel) in [(512, 0.0), (513, 1.0)] {
            let input = prepared(&[(&vec![1; characters], 1)], 1);
            let baseline = crate::train(
                input.clone(),
                TrainOptions {
                    max_merges: 1,
                    min_frequency: 1,
                    bounds: Bounds::Checked,
                },
            )
            .unwrap();
            for bounds in [Bounds::Checked, Bounds::Unchecked] {
                let result = train(
                    input.clone(),
                    Options { bounds, ..options },
                    "occurrence_adaptive_256",
                )
                .unwrap();
                assert_eq!(result.core.merges, baseline.merges);
                assert_eq!(result.core.final_tokens, baseline.final_tokens);
                assert_eq!(result.core.actual_merges, baseline.actual_merges);
                assert_eq!(result.core.position_visits, baseline.position_visits);
                assert_eq!(result.core.stale_visits, baseline.stale_visits);
                assert_eq!(result.metrics["parallel_rounds"], expected_parallel);
                assert_eq!(result.metrics["serial_rounds"], 1.0 - expected_parallel);
                assert_eq!(result.metrics["actualbarriers"], 2.0 * expected_parallel);
                assert_eq!(result.metrics["adaptive_grain"], 256.0);
            }
        }
    }
}

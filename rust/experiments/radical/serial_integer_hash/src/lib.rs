//! Original counted Combined scalar BPE with one selectable integer-map builder.

pub use efficient_bpe_rust::TrainError;
use efficient_bpe_rust::{Bounds, Prepared, Rule, TrainOptions, TrainResult, validate_prepared};
use std::collections::hash_map::RandomState as StdRandomState;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::BuildHasher;
use std::time::Instant;

#[allow(dead_code)]
#[path = "../../../../src/ablation/backends.rs"]
mod backends;
mod index;
#[allow(dead_code)]
#[path = "../../../../src/ablation/queue.rs"]
mod queue;

use backends::{Corpus, Endpoint, Halfword};
use index::{Combined, Key};
use queue::Queue;

type Result<T> = std::result::Result<T, TrainError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    CombinedFiltered,
    CombinedFilteredHalfword,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerHash {
    Std,
    AHash,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub backend: Backend,
    pub integer_hash: IntegerHash,
    pub workers: usize,
}

#[derive(Debug)]
pub struct Output {
    pub core: TrainResult,
    pub metrics: BTreeMap<String, f64>,
}

/// Validate in the timed call, then select the complete monomorphized scalar kernel.
pub fn train(input: Prepared, options: TrainOptions, config: Config) -> Result<Output> {
    if config.workers != 1 {
        return Err(TrainError::InvalidInput("serial kernel requires workers=1"));
    }
    validate_prepared(&input, options)?;
    match (config.backend, options.bounds, config.integer_hash) {
        (Backend::CombinedFiltered, Bounds::Checked, IntegerHash::Std) => {
            run::<Endpoint<2, false>, StdRandomState>(input, options)
        }
        (Backend::CombinedFiltered, Bounds::Unchecked, IntegerHash::Std) => {
            run::<Endpoint<2, true>, StdRandomState>(input, options)
        }
        (Backend::CombinedFiltered, Bounds::Checked, IntegerHash::AHash) => {
            run::<Endpoint<2, false>, ahash::RandomState>(input, options)
        }
        (Backend::CombinedFiltered, Bounds::Unchecked, IntegerHash::AHash) => {
            run::<Endpoint<2, true>, ahash::RandomState>(input, options)
        }
        (Backend::CombinedFilteredHalfword, Bounds::Checked, IntegerHash::Std) => {
            run::<Halfword<false>, StdRandomState>(input, options)
        }
        (Backend::CombinedFilteredHalfword, Bounds::Unchecked, IntegerHash::Std) => {
            run::<Halfword<true>, StdRandomState>(input, options)
        }
        (Backend::CombinedFilteredHalfword, Bounds::Checked, IntegerHash::AHash) => {
            run::<Halfword<false>, ahash::RandomState>(input, options)
        }
        (Backend::CombinedFilteredHalfword, Bounds::Unchecked, IntegerHash::AHash) => {
            run::<Halfword<true>, ahash::RandomState>(input, options)
        }
    }
}

fn run<B: Corpus, H: BuildHasher + Default>(
    input: Prepared,
    options: TrainOptions,
) -> Result<Output> {
    let started = Instant::now();
    let Prepared {
        corpus,
        initial_lengths: mut lengths,
        pivots,
        weights,
    } = input;
    let n = corpus.len();
    let mut backend = B::new(corpus, lengths.len() - 1)?;
    debug_assert!(!backend.is_empty() && backend.len() == n);
    let mut index = Combined::<H>::default();
    let mut initial_edges = 0;
    let mut initial_stored = 0;
    let mut weight_i = 0;
    let mut counts = HashMap::<u64, u64, H>::with_hasher(H::default());
    for pos in 1..n.saturating_sub(1) {
        while weight_i + 1 < pivots.len() && pos >= pivots[weight_i + 1] as usize {
            weight_i += 1;
        }
        let (a, b) = (backend.initial_token(pos), backend.initial_token(pos + 1));
        if a != 0 && b != 0 {
            let frequency = counts.entry(u64::pair(a, b)).or_default();
            *frequency = frequency
                .checked_add(weights[weight_i])
                .ok_or(TrainError::Overflow("initial count exceeds u64"))?;
            initial_edges += 1;
        }
    }
    for (key, frequency) in counts {
        if frequency >= options.min_frequency {
            index.add(key, frequency)?;
        }
    }
    for pos in 1..n.saturating_sub(1) {
        let (a, b) = (backend.initial_token(pos), backend.initial_token(pos + 1));
        if a != 0 && b != 0 && index.append_if_tracked(u64::pair(a, b), pos as u32) {
            initial_stored += 1;
        }
    }
    let initial_occurrence_bytes = initial_stored * Combined::<H>::RECORD_BYTES;
    let entries = index.entries();
    let mut queue = Queue::new(entries, options.min_frequency, 0, 0, 1, n);
    let init_seconds = started.elapsed().as_secs_f64();
    let merge_started = Instant::now();
    let mut merges = Vec::new();
    let mut actual_merges = 0;
    let mut position_visits = 0;
    let mut stale_visits = 0;
    for _ in 0..options.max_merges {
        let choice = queue.pop(|key| {
            let frequency = index.frequency(key);
            if frequency < options.min_frequency {
                index.discard(key);
            }
            frequency
        });
        let Some((key, frequency)) = choice else {
            break;
        };
        let (a, b) = key.tokens();
        let new_id = u32::try_from(lengths.len())
            .map_err(|_| TrainError::Overflow("new token ID exceeds u32"))?;
        let length = lengths[a as usize]
            .checked_add(lengths[b as usize])
            .ok_or(TrainError::Overflow("token length exceeds u32"))?;
        lengths.push(length);
        merges.push(Rule {
            left: a,
            right: b,
            frequency,
        });
        let mut batch = index.detach(key);
        let mut fresh = HashSet::<u64, H>::with_hasher(H::default());
        for position in &mut batch {
            position_visits += 1;
            let pos = position as usize;
            let Some(ctx) = backend.inspect_pair(pos, a, b, &lengths) else {
                stale_visits += 1;
                continue;
            };
            let wi = pivots.partition_point(|&pivot| pivot as usize <= pos) - 1;
            let weight = weights[wi];
            index.subtract(key, weight)?;
            if ctx.left_id != 0 {
                index.subtract(u64::pair(ctx.left_id, a), weight)?;
            }
            if ctx.right_id != 0 {
                index.subtract(u64::pair(b, ctx.right_id), weight)?;
            }
            backend.merge_with_lengths(pos, ctx, new_id, length, &lengths);
            actual_merges += 1;
            if ctx.left_id != 0 {
                let left_key = u64::pair(ctx.left_id, new_id);
                index.record(
                    left_key,
                    weight,
                    ctx.before
                        .ok_or(TrainError::InternalInvariant("left boundary absent"))?
                        as u32,
                )?;
                fresh.insert(left_key);
            }
            if ctx.right_id != 0 {
                let right_key = u64::pair(new_id, ctx.right_id);
                index.record(right_key, weight, pos as u32)?;
                fresh.insert(right_key);
            }
        }
        for key in fresh {
            let frequency = index.frequency(key);
            if frequency >= options.min_frequency {
                queue.add(key, frequency);
            } else {
                index.discard(key);
            }
        }
        index.discard(key);
    }
    let merge_seconds = merge_started.elapsed().as_secs_f64();
    let train_seconds = started.elapsed().as_secs_f64();
    let final_tokens = backend.final_tokens(&lengths);
    let mut metrics = index.metrics();
    metrics.extend(queue.metrics());
    metrics.insert(
        "backend_capacity_bytes".into(),
        backend.capacity_bytes() as f64,
    );
    metrics.insert(
        "initial_unfiltered_offset_bytes".into(),
        (initial_edges * 4) as f64,
    );
    metrics.insert(
        "final_occurrence_logical_bytes".into(),
        index.occurrence_bytes() as f64,
    );
    metrics.insert("key_size_bytes".into(), std::mem::size_of::<u64>() as f64);
    Ok(Output {
        core: TrainResult {
            rules: merges.len(),
            merges,
            final_tokens,
            init_seconds,
            merge_seconds,
            train_seconds,
            actual_merges,
            position_visits,
            stale_visits,
            heap_pops: queue.pops,
            backend_buffer_bytes: backend.logical_bytes(),
            initial_occurrence_bytes,
            corpus_positions: n,
            max_token_length: lengths.iter().copied().max().unwrap_or(1),
        },
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use efficient_bpe_rust::ablation::{Options, train_variant};

    fn prepared(words: &[(Vec<u32>, u64)], alphabet: usize) -> Prepared {
        let mut corpus = vec![0];
        let mut pivots = Vec::new();
        let mut weights = Vec::new();
        for (word, weight) in words {
            pivots.push(corpus.len() as u32);
            weights.push(*weight);
            corpus.extend(word);
            corpus.push(0);
        }
        Prepared {
            corpus,
            initial_lengths: vec![1; alphabet + 1],
            pivots,
            weights,
        }
    }

    fn compare(input: Prepared, max_merges: usize, min_frequency: u64) {
        for backend in [Backend::CombinedFiltered, Backend::CombinedFilteredHalfword] {
            let variant = match backend {
                Backend::CombinedFiltered => "combined_filtered",
                Backend::CombinedFilteredHalfword => "combined_filtered_halfword",
            };
            for bounds in [Bounds::Checked, Bounds::Unchecked] {
                let options = TrainOptions {
                    max_merges,
                    min_frequency,
                    bounds,
                };
                let expected = train_variant(
                    input.clone(),
                    Options {
                        max_merges,
                        min_frequency,
                        bounds,
                        workers: 1,
                    },
                    variant,
                )
                .unwrap();
                for integer_hash in [IntegerHash::Std, IntegerHash::AHash] {
                    let actual = train(
                        input.clone(),
                        options,
                        Config {
                            backend,
                            integer_hash,
                            workers: 1,
                        },
                    )
                    .unwrap();
                    assert_eq!(actual.core.merges, expected.core.merges);
                    assert_eq!(actual.core.final_tokens, expected.core.final_tokens);
                    assert_eq!(actual.core.actual_merges, expected.core.actual_merges);
                    assert_eq!(actual.core.position_visits, expected.core.position_visits);
                    assert_eq!(actual.core.stale_visits, expected.core.stale_visits);
                    assert_eq!(actual.core.heap_pops, expected.core.heap_pops);
                    assert_eq!(
                        actual.core.backend_buffer_bytes,
                        expected.core.backend_buffer_bytes
                    );
                    assert_eq!(
                        actual.core.initial_occurrence_bytes,
                        expected.core.initial_occurrence_bytes
                    );
                    assert_eq!(actual.core.max_token_length, expected.core.max_token_length);
                    assert_eq!(actual.core.corpus_positions, expected.core.corpus_positions);
                }
            }
        }
    }

    #[test]
    fn weighted_overlap_adjacent_and_long_token() {
        compare(prepared(&[(vec![1; 12], 3), (vec![1; 7], 2)], 1), 10, 1);
        compare(
            prepared(&[(vec![1, 2, 3, 4], 5), (vec![1, 2], 1)], 4),
            12,
            1,
        );
        compare(
            prepared(&[(vec![1, 2, 1, 2, 1], 9), (vec![2, 1, 2], 2)], 2),
            12,
            1,
        );
        compare(prepared(&[(vec![1; 513], 1)], 1), 10, 1);
        compare(
            Prepared {
                corpus: vec![0],
                initial_lengths: vec![1],
                pivots: vec![],
                weights: vec![],
            },
            10,
            1,
        );
    }

    #[test]
    fn random_weighted_traces() {
        let mut state = 0x3719_4561_ac31_890e_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as usize
        };
        for _ in 0..40 {
            let mut present = [false; 5];
            let mut words = Vec::new();
            for _ in 0..1 + next() % 4 {
                let word = (0..1 + next() % 18)
                    .map(|_| {
                        let id = 1 + next() % 4;
                        present[id] = true;
                        id as u32
                    })
                    .collect();
                words.push((word, (1 + next() % 7) as u64));
            }
            for (id, &seen) in present.iter().enumerate().skip(1) {
                if !seen {
                    words.push((vec![id as u32], 1));
                }
            }
            compare(prepared(&words, 4), 20, (1 + next() % 4) as u64);
        }
    }

    #[test]
    fn rejects_parallel_label() {
        let error = train(
            prepared(&[(vec![1, 1], 1)], 1),
            TrainOptions {
                max_merges: 1,
                min_frequency: 1,
                bounds: Bounds::Checked,
            },
            Config {
                backend: Backend::CombinedFiltered,
                integer_hash: IntegerHash::Std,
                workers: 4,
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            TrainError::InvalidInput("serial kernel requires workers=1")
        );
    }
}

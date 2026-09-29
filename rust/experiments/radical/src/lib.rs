//! Exact BPE on immutable, globally stored posting spans.
//!
//! A pair is born only once: initially or in the epoch in which a fresh token
//! ID is introduced. Its posting is therefore written once and never extended.
//! All historical positions live in one arena, independent of worker count.

use efficient_bpe_rust::{Prepared, Rule, TrainError, TrainOptions, validate_prepared};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

#[path = "../../aa_parity.rs"]
mod aa_parity;

type Result<T> = std::result::Result<T, TrainError>;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub workers: usize,
    pub chunk_size: usize,
}

#[derive(Default, Clone, Debug)]
pub struct Metrics {
    pub validation_seconds: f64,
    pub pool_seconds: f64,
    pub init_seconds: f64,
    pub initial_count_seconds: f64,
    pub initial_fill_seconds: f64,
    pub plan_seconds: f64,
    pub apply_seconds: f64,
    pub frequency_reduce_seconds: f64,
    pub birth_sort_seconds: f64,
    pub birth_append_seconds: f64,
    pub final_seconds: f64,
    pub posting_visits: usize,
    pub stale_visits: usize,
    pub actual_merges: usize,
    pub posting_arena_len: usize,
    pub posting_arena_capacity: usize,
    pub retained_entry_posting_len: usize,
    pub eligible_posting_len: usize,
    pub final_live_edges: usize,
    pub stored_born_postings: usize,
    pub generated_birth_records: usize,
    pub peak_plan_len: usize,
    pub peak_birth_records: usize,
    pub peak_delta_keys: usize,
}

#[derive(Debug)]
pub struct Output {
    pub rules: Vec<Rule>,
    pub final_tokens: Vec<u32>,
    pub metrics: Metrics,
}

#[derive(Clone, Copy)]
struct Entry {
    start: usize,
    len: usize,
    frequency: u64,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Candidate {
    frequency: u64,
    key: u64,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.key.cmp(&self.key))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct Plan {
    pos: u32,
    right: u32,
    after: u32,
    before: u32,
    left_id: u32,
    right_id: u32,
    weight: u64,
}

#[derive(Default)]
struct Changes {
    delta: HashMap<u64, i128>,
    born: Vec<(u64, u32)>,
    merges: usize,
}

#[inline]
fn key(a: u32, b: u32) -> u64 {
    (u64::from(a) << 32) | u64::from(b)
}

#[inline]
fn read(corpus: &[AtomicU32], pos: usize) -> u32 {
    corpus[pos].load(Ordering::Relaxed)
}

fn inspect(corpus: &[AtomicU32], lengths: &[u32], pos: usize, a: u32, b: u32) -> Option<Plan> {
    let last = corpus.len() - 1;
    if pos == 0 || pos >= last || read(corpus, pos) != a {
        return None;
    }
    let right = pos.checked_add(lengths[a as usize] as usize)?;
    if right >= last || read(corpus, right) != b {
        return None;
    }
    let after = right.checked_add(lengths[b as usize] as usize)?;
    if after > last {
        return None;
    }
    let prior_length = lengths[read(corpus, pos - 1) as usize] as usize;
    let before = pos.checked_sub(prior_length)?;
    Some(Plan {
        pos: pos as u32,
        right: right as u32,
        after: after as u32,
        before: before as u32,
        left_id: read(corpus, before),
        right_id: read(corpus, after),
        weight: 0,
    })
}

#[inline]
fn add_delta(delta: &mut HashMap<u64, i128>, pair: u64, value: i128) {
    *delta.entry(pair).or_default() += value;
}

fn write_merge(corpus: &[AtomicU32], plan: Plan, b_length: usize, new_id: u32) {
    let pos = plan.pos as usize;
    let right = plan.right as usize;
    let after = plan.after as usize;
    corpus[pos].store(new_id, Ordering::Relaxed);
    if b_length == 1 {
        corpus[right].store(new_id, Ordering::Relaxed);
    } else {
        corpus[right].store(0, Ordering::Relaxed);
        corpus[after - 1].store(new_id, Ordering::Relaxed);
    }
}

fn weight_at(pivots: &[u32], weights: &[u64], pos: u32) -> u64 {
    let i = pivots.partition_point(|&pivot| pivot <= pos) - 1;
    weights[i]
}

fn initial_index(
    corpus: &[AtomicU32], pivots: &[u32], weights: &[u64], metrics: &mut Metrics,
) -> Result<(HashMap<u64, Entry>, Vec<u32>)> {
    let start = Instant::now();
    let mut entries: HashMap<u64, Entry> = HashMap::new();
    let mut weight_i = 0;
    let mut occurrences = 0;
    for pos in 1..corpus.len() - 1 {
        while weight_i + 1 < pivots.len() && pos >= pivots[weight_i + 1] as usize {
            weight_i += 1;
        }
        let a = read(corpus, pos);
        let b = read(corpus, pos + 1);
        if a == 0 || b == 0 {
            continue;
        }
        let entry = entries.entry(key(a, b)).or_insert(Entry {
            start: 0,
            len: 0,
            frequency: 0,
        });
        entry.len += 1;
        entry.frequency = entry
            .frequency
            .checked_add(weights[weight_i])
            .ok_or(TrainError::Overflow("pair frequency exceeds u64"))?;
        occurrences += 1;
    }
    let mut next = 0;
    for entry in entries.values_mut() {
        entry.start = next;
        next += entry.len;
        entry.len = 0;
    }
    metrics.initial_count_seconds = start.elapsed().as_secs_f64();
    let fill_started = Instant::now();
    let mut arena = vec![0_u32; occurrences];
    for pos in 1..corpus.len() - 1 {
        let a = read(corpus, pos);
        let b = read(corpus, pos + 1);
        if a == 0 || b == 0 {
            continue;
        }
        let entry = entries.get_mut(&key(a, b)).unwrap();
        arena[entry.start + entry.len] = pos as u32;
        entry.len += 1;
    }
    metrics.initial_fill_seconds = fill_started.elapsed().as_secs_f64();
    Ok((entries, arena))
}

fn combine_changes(parts: Vec<Changes>, metrics: &mut Metrics) -> Changes {
    let mut result = Changes::default();
    for part in parts {
        result.merges += part.merges;
        result.born.extend(part.born);
        for (pair, value) in part.delta {
            add_delta(&mut result.delta, pair, value);
        }
    }
    metrics.peak_birth_records = metrics.peak_birth_records.max(result.born.len());
    metrics.peak_delta_keys = metrics.peak_delta_keys.max(result.delta.len());
    result
}

fn apply_plans(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    plans: &[Plan],
    a: u32,
    b: u32,
    new_id: u32,
    b_length: usize,
    chunk_size: usize,
    metrics: &mut Metrics,
) -> Changes {
    let parts = pool.install(|| {
        plans
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_i, chunk)| {
                let mut changes = Changes::default();
                let offset = chunk_i * chunk_size;
                for (local_i, &plan) in chunk.iter().enumerate() {
                    let i = offset + local_i;
                    let previous_selected = i > 0 && plans[i - 1].right == plan.before;
                    let next_selected = i + 1 < plans.len() && plans[i + 1].pos == plan.after;
                    let w = i128::from(plan.weight);
                    if plan.left_id != 0 && !previous_selected {
                        let old_key = key(plan.left_id, a);
                        if old_key != key(a, b) {
                            add_delta(&mut changes.delta, old_key, -w);
                        }
                        let new_key = key(plan.left_id, new_id);
                        add_delta(&mut changes.delta, new_key, w);
                        changes.born.push((new_key, plan.before));
                    }
                    if plan.right_id != 0 {
                        let old_key = key(b, plan.right_id);
                        if old_key != key(a, b) {
                            add_delta(&mut changes.delta, old_key, -w);
                        }
                        let right_id = if next_selected { new_id } else { plan.right_id };
                        let new_key = key(new_id, right_id);
                        add_delta(&mut changes.delta, new_key, w);
                        changes.born.push((new_key, plan.pos));
                    }
                    write_merge(corpus, plan, b_length, new_id);
                    changes.merges += 1;
                }
                changes
            })
            .collect::<Vec<_>>()
    });
    combine_changes(parts, metrics)
}

/// Use the parent crate's full input validation, including frequency overflow.
pub fn train(input: Prepared, options: TrainOptions, config: Config) -> Result<Output> {
    if config.workers == 0 || config.chunk_size == 0 || options.min_frequency == 0 {
        return Err(TrainError::InvalidInput("workers, chunk_size and min_frequency must be positive"));
    }
    let started = Instant::now();
    validate_prepared(&input, options)?;
    let mut metrics = Metrics::default();
    metrics.validation_seconds = started.elapsed().as_secs_f64();
    let pool_started = Instant::now();
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()
        .map_err(|_| TrainError::InvalidInput("cannot create worker pool"))?;
    metrics.pool_seconds = pool_started.elapsed().as_secs_f64();
    let Prepared { corpus, mut initial_lengths, pivots, weights } = input;
    let corpus: Vec<AtomicU32> = corpus.into_iter().map(AtomicU32::new).collect();
    let (mut entries, mut arena) = initial_index(&corpus, &pivots, &weights, &mut metrics)?;
    let initial_postings = arena.len();
    let mut heap = BinaryHeap::from(
        entries.iter().filter_map(|(&key, entry)| {
            (entry.frequency >= options.min_frequency).then_some(Candidate { frequency: entry.frequency, key })
        }).collect::<Vec<_>>()
    );
    metrics.init_seconds = started.elapsed().as_secs_f64();
    let mut rules = Vec::new();
    while rules.len() < options.max_merges {
        let (pair, frequency) = loop {
            let Some(candidate) = heap.pop() else { break (0, 0); };
            let current = entries.get(&candidate.key).map_or(0, |entry| entry.frequency);
            if current < options.min_frequency { continue; }
            if current != candidate.frequency {
                heap.push(Candidate { frequency: current, key: candidate.key });
                continue;
            }
            break (candidate.key, current);
        };
        if frequency == 0 { break; }
        let a = (pair >> 32) as u32;
        let b = pair as u32;
        let new_id = u32::try_from(initial_lengths.len())
            .map_err(|_| TrainError::Overflow("fresh token ID exceeds u32"))?;
        let new_length = initial_lengths[a as usize].checked_add(initial_lengths[b as usize])
            .ok_or(TrainError::Overflow("merged token length exceeds u32"))?;
        let b_length = initial_lengths[b as usize] as usize;
        initial_lengths.push(new_length);
        let selected = entries.remove(&pair).ok_or(TrainError::InternalInvariant("selected posting absent"))?;
        let postings = &arena[selected.start..selected.start + selected.len];
        let t = Instant::now();
        let plans = if a == b {
            // The posting is globally ordered. Empty chunks retain the previous
            // run parity; only O(number of chunks) summaries are serialized.
            let valid_chunks = pool.install(|| postings.par_chunks(config.chunk_size).map(|chunk| {
                chunk.iter().copied().filter(|&pos| {
                    inspect(&corpus, &initial_lengths, pos as usize, a, b).is_some()
                }).collect::<Vec<_>>()
            }).collect::<Vec<_>>());
            let valid: usize = valid_chunks.iter().map(Vec::len).sum();
            let summaries = valid_chunks.iter().map(|chunk| aa_parity::summarize(chunk, b_length as u32)).collect::<Vec<_>>();
            let incoming = aa_parity::incoming_parities(&summaries, b_length as u32);
            let chunks = pool.install(|| valid_chunks.par_iter().zip(incoming.par_iter()).map(|(chunk, &odd)| {
                let mut local = Vec::new();
                aa_parity::for_each_selected(chunk, b_length as u32, odd, |pos| {
                    let mut plan = inspect(&corpus, &initial_lengths, pos as usize, a, b).unwrap();
                    plan.weight = weight_at(&pivots, &weights, pos);
                    local.push(plan);
                });
                local
            }).collect::<Vec<_>>());
            metrics.posting_visits += postings.len();
            metrics.stale_visits += postings.len() - valid;
            chunks.into_iter().flatten().collect::<Vec<_>>()
        } else {
            let chunks = pool.install(|| postings.par_chunks(config.chunk_size).map(|chunk| {
                let mut local = Vec::new();
                for &pos in chunk {
                    if let Some(mut plan) = inspect(&corpus, &initial_lengths, pos as usize, a, b) {
                        plan.weight = weight_at(&pivots, &weights, pos);
                        local.push(plan);
                    }
                }
                local
            }).collect::<Vec<_>>());
            let valid: usize = chunks.iter().map(Vec::len).sum();
            metrics.posting_visits += postings.len();
            metrics.stale_visits += postings.len() - valid;
            chunks.into_iter().flatten().collect::<Vec<_>>()
        };
        metrics.peak_plan_len = metrics.peak_plan_len.max(plans.len());
        metrics.plan_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut changes = apply_plans(&pool, &corpus, &plans, a, b, new_id, b_length, config.chunk_size, &mut metrics);
        metrics.apply_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        metrics.actual_merges += changes.merges;
        metrics.generated_birth_records += changes.born.len();
        for (changed_key, delta) in changes.delta.drain() {
            if changed_key == pair { continue; }
            if let Some(entry) = entries.get_mut(&changed_key) {
                if delta > 0 { return Err(TrainError::InternalInvariant("old pair frequency increased")); }
                let decrease = u64::try_from(-delta)
                    .map_err(|_| TrainError::Overflow("frequency decrease exceeds u64"))?;
                entry.frequency = entry.frequency.checked_sub(decrease)
                    .ok_or(TrainError::InternalInvariant("negative old pair frequency"))?;
            } else if delta < 0 {
                return Err(TrainError::InternalInvariant("new pair has negative frequency"));
            } else {
                // This key is born below. Store its exact net count in a new entry.
                let freq = u64::try_from(delta).map_err(|_| TrainError::Overflow("new frequency exceeds u64"))?;
                entries.insert(changed_key, Entry { start: 0, len: 0, frequency: freq });
            }
        }
        metrics.frequency_reduce_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        pool.install(|| changes.born.par_sort_unstable_by_key(|&(born_key, pos)| (born_key, pos)));
        metrics.birth_sort_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        for group in changes.born.chunk_by(|a, b| a.0 == b.0) {
            let born_key = group[0].0;
            let freq = entries.get(&born_key).map_or(0, |entry| entry.frequency);
            // Keep a scalar for sub-threshold keys in this first version so
            // later old-edge deletions use the same update path.
            if freq < options.min_frequency { continue; }
            let start = arena.len();
            arena.extend(group.iter().map(|&(_, pos)| pos));
            let entry = entries.get_mut(&born_key).unwrap();
            entry.start = start;
            entry.len = group.len();
            heap.push(Candidate { key: born_key, frequency: freq });
        }
        metrics.birth_append_seconds += t.elapsed().as_secs_f64();
        rules.push(Rule { left: a, right: b, frequency });
    }
    let t = Instant::now();
    let mut final_tokens = Vec::new();
    let mut pos = 0;
    let mut live_edges = 0;
    loop {
        let id = read(&corpus, pos);
        final_tokens.push(id);
        if pos == corpus.len() - 1 { break; }
        let next = pos + initial_lengths[id as usize] as usize;
        if next >= corpus.len() { return Err(TrainError::InternalInvariant("invalid final boundary")); }
        if id != 0 && read(&corpus, next) != 0 { live_edges += 1; }
        pos = next;
    }
    metrics.final_seconds = t.elapsed().as_secs_f64();
    metrics.posting_arena_len = arena.len();
    metrics.posting_arena_capacity = arena.capacity();
    metrics.retained_entry_posting_len = entries.values().map(|entry| entry.len).sum();
    metrics.eligible_posting_len = entries.values()
        .filter(|entry| entry.frequency >= options.min_frequency).map(|entry| entry.len).sum();
    metrics.final_live_edges = live_edges;
    metrics.stored_born_postings = arena.len() - initial_postings;
    Ok(Output { rules, final_tokens, metrics })
}

#[cfg(test)]
mod tests {
    use super::*;
    use efficient_bpe_rust::{Bounds, train as reference};

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
        Prepared { corpus, initial_lengths: vec![1; alphabet + 1], pivots, weights }
    }

    fn compare(input: Prepared, merges: usize, min_frequency: u64) {
        let options = TrainOptions { max_merges: merges, min_frequency, bounds: Bounds::Checked };
        let expected = reference(input.clone(), options).unwrap();
        for workers in [1, 2, 4] {
            for chunk_size in [1, 5, 32] {
                let actual = train(input.clone(), options, Config { workers, chunk_size }).unwrap();
                assert_eq!(actual.rules, expected.merges, "rules: workers={workers}, chunk={chunk_size}");
                assert_eq!(actual.final_tokens, expected.final_tokens, "tokens: workers={workers}, chunk={chunk_size}");
            }
        }
    }

    #[test]
    fn overlap_adjacent_ties_and_weights() {
        compare(prepared(&[(vec![1; 11], 3), (vec![1; 7], 2)], 1), 12, 1);
        compare(prepared(&[(vec![1, 2, 1, 2, 1, 2], 3), (vec![2, 1, 2, 1], 4)], 2), 12, 1);
        compare(prepared(&[(vec![1, 2, 3], 3), (vec![4, 5], 2)], 5), 10, 1);
        compare(prepared(&[(vec![1, 2, 3, 4, 1, 2, 3, 4], 1)], 4), 12, 1);
    }

    #[test]
    fn random_weighted_traces() {
        let mut state = 0x55aa_12ff_7819_036d_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as usize
        };
        for _ in 0..100 {
            let mut words = Vec::new();
            let mut present = [false; 5];
            for _ in 0..1 + next() % 5 {
                let word = (0..1 + next() % 14)
                    .map(|_| { let id = 1 + next() % 4; present[id] = true; id as u32 })
                    .collect();
                words.push((word, (1 + next() % 5) as u64));
            }
            for id in 1..=4 { if !present[id] { words.push((vec![id as u32], 1)); } }
            compare(prepared(&words, 4), 16, (1 + next() % 4) as u64);
        }
    }
}

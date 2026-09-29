//! Exact BPE on immutable, globally stored posting spans.
//!
//! A pair is born only once: initially or in the epoch in which a fresh token
//! ID is introduced. Its posting is therefore written once and never extended.
//! All historical positions live in one arena, independent of worker count.

use efficient_bpe_rust::{Prepared, Rule, TrainError, TrainOptions, validate_prepared};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

#[path = "../../../aa_parity.rs"]
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
    pub select_seconds: f64,
    pub plan_seconds: f64,
    pub chunk_summary_seconds: f64,
    pub apply_seconds: f64,
    pub combine_seconds: f64,
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
    pub initial_all_postings: usize,
    pub initial_eligible_postings: usize,
    pub peak_plan_len: usize,
    pub peak_birth_records: usize,
    pub peak_delta_keys: usize,
    pub batch_rounds: usize,
    pub batch_rules: usize,
    pub max_batch_width: usize,
    pub singleton_rounds: usize,
    pub flat_tasks: usize,
    pub planned_positions: usize,
    pub peak_flat_tasks: usize,
    pub peak_task_starts: usize,
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

#[derive(Clone, Copy)]
struct BatchRule {
    a: u32,
    b: u32,
    new_id: u32,
    frequency: u64,
    a_length: usize,
    b_length: usize,
    span: Entry,
}

#[derive(Clone, Copy)]
struct FlatTask {
    rank: usize,
    start: usize,
    end: usize,
}

#[derive(Default)]
struct TaskPrepared {
    valid: Vec<u32>,
    changes: Changes,
    visits: usize,
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
    minimum: u64,
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
    metrics.initial_all_postings = occurrences;
    // An old pair can only lose frequency. Below-minimum pairs therefore
    // need neither a posting nor a scalar: they can never enter the heap.
    entries.retain(|_, entry| entry.frequency >= minimum);
    occurrences = entries.values().map(|entry| entry.len).sum();
    metrics.initial_eligible_postings = occurrences;
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
        if let Some(entry) = entries.get_mut(&key(a, b)) {
            arena[entry.start + entry.len] = pos as u32;
            entry.len += 1;
        }
    }
    metrics.initial_fill_seconds = fill_started.elapsed().as_secs_f64();
    Ok((entries, arena))
}

fn combine_changes(parts: Vec<Changes>, metrics: &mut Metrics) -> Changes {
    let started = Instant::now();
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
    metrics.combine_seconds += started.elapsed().as_secs_f64();
    result
}

#[allow(clippy::too_many_arguments)]
fn apply_plan_chunks(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    chunks: &[Vec<Plan>],
    a: u32,
    b: u32,
    new_id: u32,
    b_length: usize,
    metrics: &mut Metrics,
) -> Changes {
    let summary_started = Instant::now();
    let mut previous_right = vec![None; chunks.len()];
    let mut last_right = None;
    for (i, chunk) in chunks.iter().enumerate() {
        previous_right[i] = last_right;
        if let Some(plan) = chunk.last() { last_right = Some(plan.right); }
    }
    let mut next_pos = vec![None; chunks.len()];
    let mut first_pos = None;
    for (i, chunk) in chunks.iter().enumerate().rev() {
        next_pos[i] = first_pos;
        if let Some(plan) = chunk.first() { first_pos = Some(plan.pos); }
    }
    metrics.chunk_summary_seconds += summary_started.elapsed().as_secs_f64();
    let parts = pool.install(|| {
        chunks
            .par_iter()
            .enumerate()
            .map(|(chunk_i, chunk)| {
                let mut changes = Changes::default();
                for (local_i, &plan) in chunk.iter().enumerate() {
                    let previous_selected = if local_i > 0 {
                        chunk[local_i - 1].right == plan.before
                    } else {
                        previous_right[chunk_i] == Some(plan.before)
                    };
                    let next_selected = if local_i + 1 < chunk.len() {
                        chunk[local_i + 1].pos == plan.after
                    } else {
                        next_pos[chunk_i] == Some(plan.after)
                    };
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

fn pop_current_candidate(
    heap: &mut BinaryHeap<Candidate>,
    entries: &HashMap<u64, Entry>,
    minimum: u64,
) -> Option<Candidate> {
    loop {
        let candidate = heap.pop()?;
        let current = entries.get(&candidate.key).map_or(0, |entry| entry.frequency);
        if current < minimum { continue; }
        if current != candidate.frequency {
            heap.push(Candidate { frequency: current, key: candidate.key });
            continue;
        }
        return Some(candidate);
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_batch(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    arena: &[u32],
    batch: &[BatchRule],
    tasks: &[FlatTask],
    selected: &HashMap<u64, u32>,
) -> Vec<TaskPrepared> {
    pool.install(|| tasks.par_iter().map(|task| {
        let rule = batch[task.rank];
        let mut output = TaskPrepared {
            visits: task.end - task.start,
            ..TaskPrepared::default()
        };
        for &pos in &arena[task.start..task.end] {
            let Some(plan) = inspect(corpus, lengths, pos as usize, rule.a, rule.b) else {
                continue;
            };
            let w = i128::from(weight_at(pivots, weights, pos));
            output.valid.push(pos);
            output.changes.merges += 1;
            if plan.left_id != 0 {
                let before = plan.before as usize;
                let prior_id = read(corpus, before - 1);
                let left_selected = prior_id != 0 && selected.contains_key(&key(prior_id, plan.left_id));
                if !left_selected {
                    add_delta(&mut output.changes.delta, key(plan.left_id, rule.a), -w);
                    let born_key = key(plan.left_id, rule.new_id);
                    add_delta(&mut output.changes.delta, born_key, w);
                    output.changes.born.push((born_key, plan.before));
                }
            }
            if plan.right_id != 0 {
                add_delta(&mut output.changes.delta, key(rule.b, plan.right_id), -w);
                let after = plan.after as usize;
                let next = after + lengths[plan.right_id as usize] as usize;
                let next_id = read(corpus, next);
                let final_right = selected.get(&key(plan.right_id, next_id))
                    .copied().unwrap_or(plan.right_id);
                let born_key = key(rule.new_id, final_right);
                add_delta(&mut output.changes.delta, born_key, w);
                output.changes.born.push((born_key, plan.pos));
            }
        }
        output
    }).collect::<Vec<_>>())
}

fn apply_batch(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    batch: &[BatchRule],
    tasks: &[FlatTask],
    prepared: &[TaskPrepared],
) {
    pool.install(|| tasks.par_iter().zip(prepared.par_iter()).for_each(|(task, result)| {
        let rule = batch[task.rank];
        for &pos in &result.valid {
            let right = pos as usize + rule.a_length;
            let after = right + rule.b_length;
            write_merge(corpus, Plan {
                pos, right: right as u32, after: after as u32,
                before: 0, left_id: 0, right_id: 0, weight: 0,
            }, rule.b_length, rule.new_id);
        }
    }));
}

#[allow(clippy::too_many_arguments)]
fn commit_changes(
    pool: &ThreadPool,
    entries: &mut HashMap<u64, Entry>,
    heap: &mut BinaryHeap<Candidate>,
    arena: &mut Vec<u32>,
    selected: &HashSet<u64>,
    mut changes: Changes,
    minimum: u64,
    metrics: &mut Metrics,
) -> Result<()> {
    let t = Instant::now();
    metrics.actual_merges += changes.merges;
    metrics.generated_birth_records += changes.born.len();
    for (changed_key, delta) in changes.delta.drain() {
        if selected.contains(&changed_key) { continue; }
        if let Some(entry) = entries.get_mut(&changed_key) {
            if delta > 0 { return Err(TrainError::InternalInvariant("old pair frequency increased")); }
            let decrease = u64::try_from(-delta)
                .map_err(|_| TrainError::Overflow("frequency decrease exceeds u64"))?;
            entry.frequency = entry.frequency.checked_sub(decrease)
                .ok_or(TrainError::InternalInvariant("negative old pair frequency"))?;
            if entry.frequency < minimum { entries.remove(&changed_key); }
        } else if delta > 0 {
            let freq = u64::try_from(delta)
                .map_err(|_| TrainError::Overflow("new frequency exceeds u64"))?;
            if freq >= minimum {
                entries.insert(changed_key, Entry { start: 0, len: 0, frequency: freq });
            }
        }
    }
    metrics.frequency_reduce_seconds += t.elapsed().as_secs_f64();
    let t = Instant::now();
    pool.install(|| changes.born.par_sort_unstable_by_key(|&(born_key, pos)| (born_key, pos)));
    metrics.birth_sort_seconds += t.elapsed().as_secs_f64();
    let t = Instant::now();
    for group in changes.born.chunk_by(|a, b| a.0 == b.0) {
        let born_key = group[0].0;
        let Some(entry) = entries.get_mut(&born_key) else { continue; };
        let start = arena.len();
        arena.extend(group.iter().map(|&(_, pos)| pos));
        entry.start = start;
        entry.len = group.len();
        heap.push(Candidate { key: born_key, frequency: entry.frequency });
    }
    metrics.birth_append_seconds += t.elapsed().as_secs_f64();
    Ok(())
}

/// Use the parent crate's full input validation, including frequency overflow.
pub fn train(input: Prepared, options: TrainOptions, config: Config) -> Result<Output> {
    if config.workers == 0 || config.chunk_size == 0 || options.min_frequency == 0 {
        return Err(TrainError::InvalidInput("workers, chunk_size and min_frequency must be positive"));
    }
    let started = Instant::now();
    validate_prepared(&input, options)?;
    let mut metrics = Metrics {
        validation_seconds: started.elapsed().as_secs_f64(),
        ..Metrics::default()
    };
    let pool_started = Instant::now();
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()
        .map_err(|_| TrainError::InvalidInput("cannot create worker pool"))?;
    metrics.pool_seconds = pool_started.elapsed().as_secs_f64();
    let Prepared { corpus, mut initial_lengths, pivots, weights } = input;
    let corpus: Vec<AtomicU32> = corpus.into_iter().map(AtomicU32::new).collect();
    let (mut entries, mut arena) = initial_index(&corpus, &pivots, &weights, &mut metrics, options.min_frequency)?;
    let initial_postings = arena.len();
    let mut heap = BinaryHeap::from(
        entries.iter().filter_map(|(&key, entry)| {
            (entry.frequency >= options.min_frequency).then_some(Candidate { frequency: entry.frequency, key })
        }).collect::<Vec<_>>()
    );
    metrics.init_seconds = started.elapsed().as_secs_f64();
    let mut rules = Vec::new();
    while rules.len() < options.max_merges {
        let t = Instant::now();
        let mut chosen = Vec::<Candidate>::new();
        let mut heads = HashSet::<u32>::new();
        let mut tails = HashSet::<u32>::new();
        let cap = (options.max_merges - rules.len()).min(256);
        while chosen.len() < cap {
            let Some(candidate) = pop_current_candidate(&mut heap, &entries, options.min_frequency) else { break; };
            let a = (candidate.key >> 32) as u32;
            let b = candidate.key as u32;
            if a == b || tails.contains(&a) || heads.contains(&b) {
                if chosen.is_empty() { chosen.push(candidate); } else { heap.push(candidate); }
                break;
            }
            chosen.push(candidate);
            heads.insert(a);
            tails.insert(b);
        }
        metrics.select_seconds += t.elapsed().as_secs_f64();
        if chosen.is_empty() { break; }
        let mut batch = Vec::<BatchRule>::with_capacity(chosen.len());
        let mut selected = HashMap::<u64, u32>::with_capacity(chosen.len());
        let mut selected_keys = HashSet::<u64>::with_capacity(chosen.len());
        for candidate in chosen {
            let a = (candidate.key >> 32) as u32;
            let b = candidate.key as u32;
            let new_id = u32::try_from(initial_lengths.len())
                .map_err(|_| TrainError::Overflow("fresh token ID exceeds u32"))?;
            let a_length = initial_lengths[a as usize] as usize;
            let b_length = initial_lengths[b as usize] as usize;
            let new_length = initial_lengths[a as usize].checked_add(initial_lengths[b as usize])
                .ok_or(TrainError::Overflow("merged token length exceeds u32"))?;
            initial_lengths.push(new_length);
            let span = entries.remove(&candidate.key)
                .ok_or(TrainError::InternalInvariant("selected posting absent"))?;
            selected.insert(candidate.key, new_id);
            selected_keys.insert(candidate.key);
            batch.push(BatchRule { a, b, new_id,
                frequency: candidate.frequency, a_length, b_length, span });
        }
        metrics.batch_rounds += 1;
        metrics.batch_rules += batch.len();
        metrics.max_batch_width = metrics.max_batch_width.max(batch.len());
        if batch.len() == 1 { metrics.singleton_rounds += 1; }

        let changes = if batch[0].a == batch[0].b {
            assert_eq!(batch.len(), 1);
            let rule = batch[0];
            let postings = &arena[rule.span.start..rule.span.start + rule.span.len];
            let t = Instant::now();
            let valid_chunks = pool.install(|| postings.par_chunks(config.chunk_size).map(|chunk| {
                chunk.iter().copied().filter(|&pos| {
                    inspect(&corpus, &initial_lengths, pos as usize, rule.a, rule.b).is_some()
                }).collect::<Vec<_>>()
            }).collect::<Vec<_>>());
            let valid: usize = valid_chunks.iter().map(Vec::len).sum();
            let summaries = valid_chunks.iter().map(|chunk| {
                aa_parity::summarize(chunk, rule.b_length as u32)
            }).collect::<Vec<_>>();
            let incoming = aa_parity::incoming_parities(&summaries, rule.b_length as u32);
            let plan_chunks = pool.install(|| valid_chunks.par_iter().zip(incoming.par_iter()).map(|(chunk, &odd)| {
                let mut local = Vec::new();
                aa_parity::for_each_selected(chunk, rule.b_length as u32, odd, |pos| {
                    let mut plan = inspect(&corpus, &initial_lengths, pos as usize, rule.a, rule.b).unwrap();
                    plan.weight = weight_at(&pivots, &weights, pos);
                    local.push(plan);
                });
                local
            }).collect::<Vec<_>>());
            let planned: usize = plan_chunks.iter().map(Vec::len).sum();
            metrics.posting_visits += postings.len();
            metrics.stale_visits += postings.len() - valid;
            metrics.planned_positions += planned;
            metrics.flat_tasks += plan_chunks.len();
            metrics.peak_flat_tasks = metrics.peak_flat_tasks.max(plan_chunks.len());
            metrics.peak_plan_len = metrics.peak_plan_len.max(planned);
            metrics.peak_task_starts = metrics.peak_task_starts.max(planned);
            metrics.plan_seconds += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let changes = apply_plan_chunks(&pool, &corpus, &plan_chunks,
                rule.a, rule.b, rule.new_id, rule.b_length, &mut metrics);
            metrics.apply_seconds += t.elapsed().as_secs_f64();
            changes
        } else {
            let mut tasks = Vec::<FlatTask>::new();
            for (rank, rule) in batch.iter().enumerate() {
                let end = rule.span.start + rule.span.len;
                let mut start = rule.span.start;
                while start < end {
                    let next = start + config.chunk_size.min(end - start);
                    tasks.push(FlatTask { rank, start, end: next });
                    start = next;
                }
            }
            metrics.flat_tasks += tasks.len();
            metrics.peak_flat_tasks = metrics.peak_flat_tasks.max(tasks.len());
            let t = Instant::now();
            let prepared = prepare_batch(&pool, &corpus, &initial_lengths,
                &pivots, &weights, &arena, &batch, &tasks, &selected);
            let visited: usize = prepared.iter().map(|part| part.visits).sum();
            let planned: usize = prepared.iter().map(|part| part.valid.len()).sum();
            metrics.posting_visits += visited;
            metrics.stale_visits += visited - planned;
            metrics.planned_positions += planned;
            metrics.peak_plan_len = metrics.peak_plan_len.max(planned);
            metrics.peak_task_starts = metrics.peak_task_starts.max(planned);
            metrics.plan_seconds += t.elapsed().as_secs_f64();
            let t = Instant::now();
            apply_batch(&pool, &corpus, &batch, &tasks, &prepared);
            metrics.apply_seconds += t.elapsed().as_secs_f64();
            combine_changes(prepared.into_iter().map(|part| part.changes).collect(), &mut metrics)
        };
        commit_changes(&pool, &mut entries, &mut heap, &mut arena,
            &selected_keys, changes, options.min_frequency, &mut metrics)?;
        for rule in batch {
            rules.push(Rule { left: rule.a, right: rule.b, frequency: rule.frequency });
        }
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
        let adjacent = prepared(&[
            (vec![1, 2, 3, 4], 1), (vec![1, 2], 3), (vec![3, 4], 2),
        ], 4);
        compare(adjacent.clone(), 12, 1);
        let output = train(adjacent.clone(), TrainOptions {
            max_merges: 12, min_frequency: 1, bounds: Bounds::Checked,
        }, Config { workers: 4, chunk_size: 1 }).unwrap();
        assert!(output.metrics.max_batch_width >= 2);
        let huge_chunk = train(adjacent, TrainOptions {
            max_merges: 12, min_frequency: 1, bounds: Bounds::Checked,
        }, Config { workers: 2, chunk_size: usize::MAX }).unwrap();
        assert_eq!(huge_chunk.rules, output.rules);
        assert_eq!(huge_chunk.final_tokens, output.final_tokens);
        compare(prepared(&[(vec![1; 512], 1)], 1), 10, 1);
        compare(Prepared {
            corpus: vec![0], initial_lengths: vec![1], pivots: vec![], weights: vec![],
        }, 10, 1);
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
            for (id, &seen) in present.iter().enumerate().skip(1) {
                if !seen { words.push((vec![id as u32], 1)); }
            }
            compare(prepared(&words, 4), 16, (1 + next() % 4) as u64);
        }
    }
}

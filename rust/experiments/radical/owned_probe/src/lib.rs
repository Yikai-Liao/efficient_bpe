//! Exact owner-routed BPE with bounded, sequential spatial conflict probes.
//! Position chunks are dynamically executed by W jobs, then routed to owners.

use efficient_bpe_rust::{Prepared, Rule, TrainError, TrainOptions, validate_prepared};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../aa_parity.rs"]
mod aa_parity;

type Result<T> = std::result::Result<T, TrainError>;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub workers: usize,
    pub chunk_size: usize,
    pub heap_policy: HeapPolicy,
    pub spatial_probe: SpatialProbe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeapPolicy {
    Eager,
    Lazy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpatialProbe {
    Off,
    Budgeted,
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
    pub initial_route_seconds: f64,
    pub initial_owner_seconds: f64,
    pub aa_sort_seconds: f64,
    pub birth_decode_seconds: f64,
    pub heap_pops: usize,
    pub heap_refreshes: usize,
    pub heap_reinsertions: usize,
    pub peak_heap_len: usize,
    pub peak_heap_capacity: usize,
    pub owned_posting_len: usize,
    pub owned_posting_capacity: usize,
    pub owner_entry_count: usize,
    pub peak_route_born_len: usize,
    pub peak_route_born_capacity: usize,
    pub probe_calls: usize,
    pub probe_visited: usize,
    pub probe_stale_visits: usize,
    pub probe_conflict_stops: usize,
    pub probe_budget_stops: usize,
    pub probe_proved_disjoint: usize,
    pub probe_seconds: f64,
    pub probe_epoch_history_total: usize,
    pub probe_epoch_budget_total: usize,
}

#[derive(Debug)]
pub struct Output {
    pub rules: Vec<Rule>,
    pub final_tokens: Vec<u32>,
    pub metrics: Metrics,
}

struct Entry {
    frequency: u64,
    positions: Vec<u32>,
}

#[derive(Default)]
struct Owner {
    entries: HashMap<u64, Entry>,
    heap: BinaryHeap<Candidate>,
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
struct Route {
    delta: HashMap<u64, i128>,
    born: Vec<u32>,
}

struct BatchRule {
    a: u32,
    b: u32,
    new_id: u32,
    frequency: u64,
    a_length: usize,
    b_length: usize,
    posting: Vec<u32>,
}

#[derive(Clone, Copy)]
struct FlatTask {
    rank: usize,
    start: usize,
    end: usize,
}

struct WorkerOutput {
    routes: Vec<Route>,
    results: Vec<(usize, Vec<u32>)>,
    visits: usize,
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

#[inline]
fn owner_for(pair: u64, workers: usize) -> usize {
    let mixed = (pair ^ (pair >> 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    ((mixed >> 32) as usize) % workers
}

fn empty_worker(workers: usize) -> WorkerOutput {
    WorkerOutput {
        routes: (0..workers).map(|_| Route::default()).collect(),
        results: Vec::new(),
        visits: 0,
        merges: 0,
    }
}

fn route_delta(output: &mut WorkerOutput, workers: usize, pair: u64, delta: i128) {
    add_delta(
        &mut output.routes[owner_for(pair, workers)].delta,
        pair,
        delta,
    );
}

fn route_birth(output: &mut WorkerOutput, workers: usize, pair: u64, pos: u32, weight: i128) {
    let route = &mut output.routes[owner_for(pair, workers)];
    add_delta(&mut route.delta, pair, weight);
    route.born.push(pos);
}

#[allow(clippy::too_many_arguments)]
fn initial_index(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    pivots: &[u32],
    weights: &[u64],
    workers: usize,
    chunk_size: usize,
    minimum: u64,
    metrics: &mut Metrics,
) -> Result<Vec<Owner>> {
    let started = Instant::now();
    let last = corpus.len() - 1;
    let positions = last.saturating_sub(1);
    let task_count = positions.div_ceil(chunk_size);
    let cursor = AtomicUsize::new(0);
    let outputs = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| {
                let mut output = empty_worker(workers);
                loop {
                    let task = cursor.fetch_add(1, Ordering::Relaxed);
                    if task >= task_count {
                        break;
                    }
                    let start = 1 + task * chunk_size;
                    let end = start + chunk_size.min(last - start);
                    for pos in start..end {
                        let a = read(corpus, pos);
                        let b = read(corpus, pos + 1);
                        if a != 0 && b != 0 {
                            output.routes[owner_for(key(a, b), workers)]
                                .born
                                .push(pos as u32);
                        }
                    }
                }
                output
            })
            .collect::<Vec<_>>()
    });
    metrics.initial_route_seconds = started.elapsed().as_secs_f64();
    metrics.initial_all_postings = outputs
        .iter()
        .flat_map(|output| &output.routes)
        .map(|route| route.born.len())
        .sum();
    let started = Instant::now();
    let mut owners: Vec<Owner> = (0..workers).map(|_| Owner::default()).collect();
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .map(|(owner_i, owner)| -> Result<()> {
                for output in &outputs {
                    for &pos in &output.routes[owner_i].born {
                        let a = read(corpus, pos as usize);
                        let b = read(corpus, pos as usize + 1);
                        let pair = key(a, b);
                        if owner_for(pair, workers) != owner_i {
                            return Err(TrainError::InternalInvariant(
                                "initial owner route mismatch",
                            ));
                        }
                        let entry = owner.entries.entry(pair).or_insert_with(|| Entry {
                            frequency: 0,
                            positions: Vec::new(),
                        });
                        entry.frequency = entry
                            .frequency
                            .checked_add(weight_at(pivots, weights, pos))
                            .ok_or(TrainError::Overflow("pair frequency exceeds u64"))?;
                        entry.positions.push(pos);
                    }
                }
                owner.entries.retain(|_, entry| entry.frequency >= minimum);
                owner.heap = BinaryHeap::from(
                    owner
                        .entries
                        .iter()
                        .map(|(&pair, entry)| Candidate {
                            key: pair,
                            frequency: entry.frequency,
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(())
            })
            .collect::<Vec<_>>()
    });
    for check in checks {
        check?;
    }
    metrics.initial_owner_seconds = started.elapsed().as_secs_f64();
    metrics.initial_eligible_postings = owners
        .iter()
        .flat_map(|owner| owner.entries.values())
        .map(|entry| entry.positions.len())
        .sum();
    Ok(owners)
}

fn peek_current(
    owner: &mut Owner,
    minimum: u64,
    policy: HeapPolicy,
    metrics: &mut Metrics,
) -> Option<Candidate> {
    loop {
        let candidate = *owner.heap.peek()?;
        let current = owner
            .entries
            .get(&candidate.key)
            .map(|entry| entry.frequency);
        if current == Some(candidate.frequency) && candidate.frequency >= minimum {
            return Some(candidate);
        }
        owner.heap.pop();
        metrics.heap_pops += 1;
        metrics.heap_refreshes += 1;
        if let (HeapPolicy::Lazy, Some(frequency)) =
            (policy, current.filter(|&freq| freq >= minimum))
        {
            owner.heap.push(Candidate {
                key: candidate.key,
                frequency,
            });
            metrics.heap_reinsertions += 1;
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ProbeOutcome {
    Disjoint,
    Conflict,
    Budget,
}

fn probe_candidate(
    corpus: &[AtomicU32],
    lengths: &[u32],
    positions: &[u32],
    pair: u64,
    selected: &HashSet<u64>,
    remaining_budget: usize,
    metrics: &mut Metrics,
) -> (ProbeOutcome, usize) {
    let started = Instant::now();
    let a = (pair >> 32) as u32;
    let b = pair as u32;
    let mut visited = 0;
    let mut stale = 0;
    let mut outcome = ProbeOutcome::Disjoint;
    for &pos in positions {
        if visited == remaining_budget {
            outcome = ProbeOutcome::Budget;
            break;
        }
        visited += 1;
        let Some(plan) = inspect(corpus, lengths, pos as usize, a, b) else {
            stale += 1;
            continue;
        };
        // Two adjacent matches share a token iff the stable left or right
        // neighbor pair is one of the earlier selected keys. A zero sentinel
        // cuts the piece and cannot form such a pair.
        if (plan.left_id != 0 && selected.contains(&key(plan.left_id, a)))
            || (plan.right_id != 0 && selected.contains(&key(b, plan.right_id)))
        {
            outcome = ProbeOutcome::Conflict;
            break;
        }
    }
    metrics.probe_calls += 1;
    metrics.probe_visited = metrics.probe_visited.saturating_add(visited);
    metrics.probe_stale_visits = metrics.probe_stale_visits.saturating_add(stale);
    match outcome {
        ProbeOutcome::Disjoint => metrics.probe_proved_disjoint += 1,
        ProbeOutcome::Conflict => metrics.probe_conflict_stops += 1,
        ProbeOutcome::Budget => metrics.probe_budget_stops += 1,
    }
    metrics.probe_seconds += started.elapsed().as_secs_f64();
    (outcome, visited)
}

fn route_aa(
    pool: &ThreadPool,
    chunks: &[Vec<Plan>],
    a: u32,
    b: u32,
    new_id: u32,
    workers: usize,
    metrics: &mut Metrics,
) -> Vec<WorkerOutput> {
    let started = Instant::now();
    let mut previous_right = vec![None; chunks.len()];
    let mut last_right = None;
    for (i, chunk) in chunks.iter().enumerate() {
        previous_right[i] = last_right;
        if let Some(plan) = chunk.last() {
            last_right = Some(plan.right);
        }
    }
    let mut next_pos = vec![None; chunks.len()];
    let mut first_pos = None;
    for (i, chunk) in chunks.iter().enumerate().rev() {
        next_pos[i] = first_pos;
        if let Some(plan) = chunk.first() {
            first_pos = Some(plan.pos);
        }
    }
    metrics.chunk_summary_seconds += started.elapsed().as_secs_f64();
    let cursor = AtomicUsize::new(0);
    pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| {
                let mut output = empty_worker(workers);
                loop {
                    let chunk_i = cursor.fetch_add(1, Ordering::Relaxed);
                    if chunk_i >= chunks.len() {
                        break;
                    }
                    let chunk = &chunks[chunk_i];
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
                                route_delta(&mut output, workers, old_key, -w);
                            }
                            route_birth(
                                &mut output,
                                workers,
                                key(plan.left_id, new_id),
                                plan.before,
                                w,
                            );
                        }
                        if plan.right_id != 0 {
                            let old_key = key(b, plan.right_id);
                            if old_key != key(a, b) {
                                route_delta(&mut output, workers, old_key, -w);
                            }
                            let final_right = if next_selected { new_id } else { plan.right_id };
                            route_birth(
                                &mut output,
                                workers,
                                key(new_id, final_right),
                                plan.pos,
                                w,
                            );
                        }
                        output.merges += 1;
                    }
                }
                output
            })
            .collect::<Vec<_>>()
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_batch(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    batch: &[BatchRule],
    tasks: &[FlatTask],
    selected: &HashMap<u64, u32>,
    workers: usize,
) -> Vec<WorkerOutput> {
    let cursor = AtomicUsize::new(0);
    pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| {
                let mut output = empty_worker(workers);
                loop {
                    let task_i = cursor.fetch_add(1, Ordering::Relaxed);
                    if task_i >= tasks.len() {
                        break;
                    }
                    let task = tasks[task_i];
                    let rule = &batch[task.rank];
                    output.visits += task.end - task.start;
                    let mut valid = Vec::new();
                    for &pos in &rule.posting[task.start..task.end] {
                        let Some(plan) = inspect(corpus, lengths, pos as usize, rule.a, rule.b)
                        else {
                            continue;
                        };
                        valid.push(pos);
                        output.merges += 1;
                        let w = i128::from(weight_at(pivots, weights, pos));
                        if plan.left_id != 0 {
                            let before = plan.before as usize;
                            let prior_id = read(corpus, before - 1);
                            let left_selected = prior_id != 0
                                && selected.contains_key(&key(prior_id, plan.left_id));
                            if !left_selected {
                                route_delta(&mut output, workers, key(plan.left_id, rule.a), -w);
                                route_birth(
                                    &mut output,
                                    workers,
                                    key(plan.left_id, rule.new_id),
                                    plan.before,
                                    w,
                                );
                            }
                        }
                        if plan.right_id != 0 {
                            route_delta(&mut output, workers, key(rule.b, plan.right_id), -w);
                            let after = plan.after as usize;
                            let next = after + lengths[plan.right_id as usize] as usize;
                            let next_id = read(corpus, next);
                            let final_right = selected
                                .get(&key(plan.right_id, next_id))
                                .copied()
                                .unwrap_or(plan.right_id);
                            route_birth(
                                &mut output,
                                workers,
                                key(rule.new_id, final_right),
                                plan.pos,
                                w,
                            );
                        }
                    }
                    output.results.push((task_i, valid));
                }
                output
            })
            .collect::<Vec<_>>()
    })
}

fn apply_batch(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    batch: &[BatchRule],
    tasks: &[FlatTask],
    outputs: &mut [WorkerOutput],
) {
    let mut ordered = vec![Vec::<u32>::new(); tasks.len()];
    for output in outputs {
        for (task_i, valid) in output.results.drain(..) {
            ordered[task_i] = valid;
        }
    }
    pool.install(|| {
        tasks
            .par_iter()
            .zip(ordered.par_iter())
            .for_each(|(task, valid)| {
                let rule = &batch[task.rank];
                for &pos in valid {
                    let right = pos as usize + rule.a_length;
                    let after = right + rule.b_length;
                    write_merge(
                        corpus,
                        Plan {
                            pos,
                            right: right as u32,
                            after: after as u32,
                            before: 0,
                            left_id: 0,
                            right_id: 0,
                            weight: 0,
                        },
                        rule.b_length,
                        rule.new_id,
                    );
                }
            })
    });
}

#[allow(clippy::too_many_arguments)]
fn commit_routes(
    pool: &ThreadPool,
    owners: &mut [Owner],
    outputs: &[WorkerOutput],
    corpus: &[AtomicU32],
    lengths: &[u32],
    selected: &HashSet<u64>,
    minimum: u64,
    policy: HeapPolicy,
    metrics: &mut Metrics,
) -> Result<()> {
    metrics.actual_merges += outputs.iter().map(|output| output.merges).sum::<usize>();
    let born_len: usize = outputs
        .iter()
        .flat_map(|output| &output.routes)
        .map(|route| route.born.len())
        .sum();
    let born_capacity: usize = outputs
        .iter()
        .flat_map(|output| &output.routes)
        .map(|route| route.born.capacity())
        .sum();
    metrics.generated_birth_records += born_len;
    metrics.peak_birth_records = metrics.peak_birth_records.max(born_len);
    metrics.peak_route_born_len = metrics.peak_route_born_len.max(born_len);
    metrics.peak_route_born_capacity = metrics.peak_route_born_capacity.max(born_capacity);
    metrics.peak_delta_keys = metrics.peak_delta_keys.max(
        outputs
            .iter()
            .flat_map(|output| &output.routes)
            .map(|route| route.delta.len())
            .sum(),
    );
    let started = Instant::now();
    let workers = owners.len();
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .map(|(owner_i, owner)| -> Result<()> {
                let mut combined = HashMap::<u64, i128>::new();
                for output in outputs {
                    for (&pair, &delta) in &output.routes[owner_i].delta {
                        add_delta(&mut combined, pair, delta);
                    }
                }
                for (pair, delta) in combined {
                    if selected.contains(&pair) {
                        continue;
                    }
                    if let Some(entry) = owner.entries.get_mut(&pair) {
                        if delta > 0 {
                            return Err(TrainError::InternalInvariant(
                                "old pair frequency increased",
                            ));
                        }
                        let decrease = u64::try_from(
                            delta
                                .checked_neg()
                                .ok_or(TrainError::Overflow("frequency delta overflow"))?,
                        )
                        .map_err(|_| TrainError::Overflow("frequency decrease exceeds u64"))?;
                        entry.frequency = entry
                            .frequency
                            .checked_sub(decrease)
                            .ok_or(TrainError::InternalInvariant("negative old pair frequency"))?;
                        if entry.frequency < minimum {
                            owner.entries.remove(&pair);
                        } else if policy == HeapPolicy::Eager {
                            owner.heap.push(Candidate {
                                key: pair,
                                frequency: entry.frequency,
                            });
                        }
                    } else if delta > 0 {
                        let frequency = u64::try_from(delta)
                            .map_err(|_| TrainError::Overflow("new frequency exceeds u64"))?;
                        if frequency >= minimum {
                            owner.entries.insert(
                                pair,
                                Entry {
                                    frequency,
                                    positions: Vec::new(),
                                },
                            );
                            owner.heap.push(Candidate {
                                key: pair,
                                frequency,
                            });
                        }
                    }
                }
                Ok(())
            })
            .collect::<Vec<_>>()
    });
    for check in checks {
        check?;
    }
    metrics.frequency_reduce_seconds += started.elapsed().as_secs_f64();
    let started = Instant::now();
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .map(|(owner_i, owner)| -> Result<usize> {
                let mut stored = 0;
                for output in outputs {
                    for &pos in &output.routes[owner_i].born {
                        let p = pos as usize;
                        let a = read(corpus, p);
                        if a == 0 {
                            return Err(TrainError::InternalInvariant(
                                "birth anchor no longer live",
                            ));
                        }
                        let next = p + lengths[a as usize] as usize;
                        if next >= corpus.len() {
                            return Err(TrainError::InternalInvariant("birth next outside corpus"));
                        }
                        let b = read(corpus, next);
                        if b == 0 {
                            return Err(TrainError::InternalInvariant(
                                "birth crosses piece boundary",
                            ));
                        }
                        let pair = key(a, b);
                        if owner_for(pair, workers) != owner_i {
                            return Err(TrainError::InternalInvariant(
                                "birth owner route mismatch",
                            ));
                        }
                        if let Some(entry) = owner.entries.get_mut(&pair) {
                            entry.positions.push(pos);
                            stored += 1;
                        }
                    }
                }
                Ok(stored)
            })
            .collect::<Vec<_>>()
    });
    for check in checks {
        metrics.stored_born_postings += check?;
    }
    metrics.birth_decode_seconds += started.elapsed().as_secs_f64();
    Ok(())
}

/// Validate the shared input contract inside the timed call.
pub fn train(input: Prepared, options: TrainOptions, config: Config) -> Result<Output> {
    if config.workers == 0 || config.chunk_size == 0 || options.min_frequency == 0 {
        return Err(TrainError::InvalidInput(
            "workers, chunk_size and min_frequency must be positive",
        ));
    }
    let started = Instant::now();
    validate_prepared(&input, options)?;
    let mut metrics = Metrics {
        validation_seconds: started.elapsed().as_secs_f64(),
        ..Metrics::default()
    };
    let started = Instant::now();
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()
        .map_err(|_| TrainError::InvalidInput("cannot create worker pool"))?;
    metrics.pool_seconds = started.elapsed().as_secs_f64();
    let Prepared {
        corpus,
        mut initial_lengths,
        pivots,
        weights,
    } = input;
    let corpus: Vec<AtomicU32> = corpus.into_iter().map(AtomicU32::new).collect();
    let started = Instant::now();
    let mut owners = initial_index(
        &pool,
        &corpus,
        &pivots,
        &weights,
        config.workers,
        config.chunk_size,
        options.min_frequency,
        &mut metrics,
    )?;
    metrics.init_seconds = started.elapsed().as_secs_f64();
    metrics.peak_heap_len = owners.iter().map(|owner| owner.heap.len()).sum();
    metrics.peak_heap_capacity = owners.iter().map(|owner| owner.heap.capacity()).sum();
    let mut rules = Vec::new();
    while rules.len() < options.max_merges {
        let started = Instant::now();
        let mut chosen = Vec::<(Candidate, Entry)>::new();
        let mut heads = HashSet::<u32>::new();
        let mut tails = HashSet::<u32>::new();
        let mut witness_selected = HashSet::<u64>::new();
        let mut selected_history = 0_usize;
        let mut probe_spent = 0_usize;
        let cap = (options.max_merges - rules.len()).min(256);
        while chosen.len() < cap {
            let mut best: Option<(usize, Candidate)> = None;
            for (owner_i, owner) in owners.iter_mut().enumerate() {
                match peek_current(
                    owner,
                    options.min_frequency,
                    config.heap_policy,
                    &mut metrics,
                ) {
                    Some(candidate) if best.is_none_or(|(_, prior)| candidate > prior) => {
                        best = Some((owner_i, candidate));
                    }
                    _ => {}
                }
            }
            let Some((owner_i, candidate)) = best else {
                break;
            };
            let a = (candidate.key >> 32) as u32;
            let b = candidate.key as u32;
            if !chosen.is_empty() {
                if a == b {
                    break;
                }
                if tails.contains(&a) || heads.contains(&b) {
                    if config.spatial_probe == SpatialProbe::Off {
                        break;
                    }
                    let budget = selected_history.saturating_mul(2);
                    let remaining = budget.saturating_sub(probe_spent);
                    let positions = &owners[owner_i]
                        .entries
                        .get(&candidate.key)
                        .ok_or(TrainError::InternalInvariant("probe posting absent"))?
                        .positions;
                    let (outcome, visited) = probe_candidate(
                        &corpus,
                        &initial_lengths,
                        positions,
                        candidate.key,
                        &witness_selected,
                        remaining,
                        &mut metrics,
                    );
                    probe_spent = probe_spent.saturating_add(visited);
                    if outcome != ProbeOutcome::Disjoint {
                        break;
                    }
                }
            }
            owners[owner_i].heap.pop();
            metrics.heap_pops += 1;
            let entry = owners[owner_i]
                .entries
                .remove(&candidate.key)
                .ok_or(TrainError::InternalInvariant("selected posting absent"))?;
            if config.spatial_probe == SpatialProbe::Budgeted {
                selected_history = selected_history.saturating_add(entry.positions.len());
                witness_selected.insert(candidate.key);
            }
            chosen.push((candidate, entry));
            heads.insert(a);
            tails.insert(b);
            if a == b {
                break;
            }
        }
        if config.spatial_probe == SpatialProbe::Budgeted {
            debug_assert!(probe_spent <= selected_history.saturating_mul(2));
            metrics.probe_epoch_history_total = metrics
                .probe_epoch_history_total
                .saturating_add(selected_history);
            metrics.probe_epoch_budget_total = metrics
                .probe_epoch_budget_total
                .saturating_add(selected_history.saturating_mul(2));
        }
        metrics.select_seconds += started.elapsed().as_secs_f64();
        if chosen.is_empty() {
            break;
        }
        let mut batch = Vec::<BatchRule>::with_capacity(chosen.len());
        let mut selected = HashMap::<u64, u32>::with_capacity(chosen.len());
        let mut selected_keys = HashSet::<u64>::with_capacity(chosen.len());
        for (candidate, entry) in chosen {
            let a = (candidate.key >> 32) as u32;
            let b = candidate.key as u32;
            let new_id = u32::try_from(initial_lengths.len())
                .map_err(|_| TrainError::Overflow("fresh token ID exceeds u32"))?;
            let a_length = initial_lengths[a as usize] as usize;
            let b_length = initial_lengths[b as usize] as usize;
            let new_length = initial_lengths[a as usize]
                .checked_add(initial_lengths[b as usize])
                .ok_or(TrainError::Overflow("merged token length exceeds u32"))?;
            initial_lengths.push(new_length);
            selected.insert(candidate.key, new_id);
            selected_keys.insert(candidate.key);
            batch.push(BatchRule {
                a,
                b,
                new_id,
                frequency: candidate.frequency,
                a_length,
                b_length,
                posting: entry.positions,
            });
        }
        metrics.batch_rounds += 1;
        metrics.batch_rules += batch.len();
        metrics.max_batch_width = metrics.max_batch_width.max(batch.len());
        if batch.len() == 1 {
            metrics.singleton_rounds += 1;
        }
        let mut outputs = if batch[0].a == batch[0].b {
            assert_eq!(batch.len(), 1);
            let rule = &mut batch[0];
            let started = Instant::now();
            pool.install(|| rule.posting.par_sort_unstable());
            metrics.aa_sort_seconds += started.elapsed().as_secs_f64();
            let started = Instant::now();
            let valid_chunks = pool.install(|| {
                rule.posting
                    .par_chunks(config.chunk_size)
                    .map(|chunk| {
                        chunk
                            .iter()
                            .copied()
                            .filter(|&pos| {
                                inspect(&corpus, &initial_lengths, pos as usize, rule.a, rule.b)
                                    .is_some()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            let valid: usize = valid_chunks.iter().map(Vec::len).sum();
            let summaries = valid_chunks
                .iter()
                .map(|chunk| aa_parity::summarize(chunk, rule.b_length as u32))
                .collect::<Vec<_>>();
            let incoming = aa_parity::incoming_parities(&summaries, rule.b_length as u32);
            let plan_chunks = pool.install(|| {
                valid_chunks
                    .par_iter()
                    .zip(incoming.par_iter())
                    .map(|(chunk, &odd)| {
                        let mut local = Vec::new();
                        aa_parity::for_each_selected(chunk, rule.b_length as u32, odd, |pos| {
                            let mut plan =
                                inspect(&corpus, &initial_lengths, pos as usize, rule.a, rule.b)
                                    .unwrap();
                            plan.weight = weight_at(&pivots, &weights, pos);
                            local.push(plan);
                        });
                        local
                    })
                    .collect::<Vec<_>>()
            });
            let planned: usize = plan_chunks.iter().map(Vec::len).sum();
            metrics.posting_visits += rule.posting.len();
            metrics.stale_visits += rule.posting.len() - valid;
            metrics.planned_positions += planned;
            metrics.flat_tasks += plan_chunks.len();
            metrics.peak_flat_tasks = metrics.peak_flat_tasks.max(plan_chunks.len());
            metrics.peak_plan_len = metrics.peak_plan_len.max(planned);
            metrics.peak_task_starts = metrics.peak_task_starts.max(planned);
            let outputs = route_aa(
                &pool,
                &plan_chunks,
                rule.a,
                rule.b,
                rule.new_id,
                config.workers,
                &mut metrics,
            );
            metrics.plan_seconds += started.elapsed().as_secs_f64();
            // The ordered Plan chunks retain every selected AA start needed by apply.
            drop(std::mem::take(&mut rule.posting));
            let started = Instant::now();
            pool.install(|| {
                plan_chunks.par_iter().for_each(|chunk| {
                    for &plan in chunk {
                        write_merge(&corpus, plan, rule.b_length, rule.new_id);
                    }
                })
            });
            metrics.apply_seconds += started.elapsed().as_secs_f64();
            outputs
        } else {
            let mut tasks = Vec::<FlatTask>::new();
            for (rank, rule) in batch.iter().enumerate() {
                let end = rule.posting.len();
                let mut start = 0;
                while start < end {
                    let next = start + config.chunk_size.min(end - start);
                    tasks.push(FlatTask {
                        rank,
                        start,
                        end: next,
                    });
                    start = next;
                }
            }
            metrics.flat_tasks += tasks.len();
            metrics.peak_flat_tasks = metrics.peak_flat_tasks.max(tasks.len());
            let started = Instant::now();
            let mut outputs = prepare_batch(
                &pool,
                &corpus,
                &initial_lengths,
                &pivots,
                &weights,
                &batch,
                &tasks,
                &selected,
                config.workers,
            );
            let visited: usize = outputs.iter().map(|output| output.visits).sum();
            let planned: usize = outputs.iter().map(|output| output.merges).sum();
            metrics.posting_visits += visited;
            metrics.stale_visits += visited - planned;
            metrics.planned_positions += planned;
            metrics.peak_plan_len = metrics.peak_plan_len.max(planned);
            metrics.peak_task_starts = metrics.peak_task_starts.max(planned);
            metrics.plan_seconds += started.elapsed().as_secs_f64();
            // Apply reads only rule metadata and the valid starts in `outputs`.
            for rule in &mut batch {
                drop(std::mem::take(&mut rule.posting));
            }
            let started = Instant::now();
            apply_batch(&pool, &corpus, &batch, &tasks, &mut outputs);
            metrics.apply_seconds += started.elapsed().as_secs_f64();
            outputs
        };
        commit_routes(
            &pool,
            &mut owners,
            &outputs,
            &corpus,
            &initial_lengths,
            &selected_keys,
            options.min_frequency,
            config.heap_policy,
            &mut metrics,
        )?;
        metrics.peak_heap_len = metrics
            .peak_heap_len
            .max(owners.iter().map(|owner| owner.heap.len()).sum());
        metrics.peak_heap_capacity = metrics
            .peak_heap_capacity
            .max(owners.iter().map(|owner| owner.heap.capacity()).sum());
        outputs.clear();
        for rule in batch {
            rules.push(Rule {
                left: rule.a,
                right: rule.b,
                frequency: rule.frequency,
            });
        }
    }
    let started = Instant::now();
    let mut final_tokens = Vec::new();
    let mut pos = 0;
    let mut live_edges = 0;
    loop {
        let id = read(&corpus, pos);
        final_tokens.push(id);
        if pos == corpus.len() - 1 {
            break;
        }
        let next = pos + initial_lengths[id as usize] as usize;
        if next >= corpus.len() {
            return Err(TrainError::InternalInvariant("invalid final boundary"));
        }
        if id != 0 && read(&corpus, next) != 0 {
            live_edges += 1;
        }
        pos = next;
    }
    metrics.final_seconds = started.elapsed().as_secs_f64();
    metrics.owned_posting_len = owners
        .iter()
        .flat_map(|owner| owner.entries.values())
        .map(|entry| entry.positions.len())
        .sum();
    metrics.owned_posting_capacity = owners
        .iter()
        .flat_map(|owner| owner.entries.values())
        .map(|entry| entry.positions.capacity())
        .sum();
    metrics.owner_entry_count = owners.iter().map(|owner| owner.entries.len()).sum();
    metrics.retained_entry_posting_len = metrics.owned_posting_len;
    metrics.eligible_posting_len = metrics.owned_posting_len;
    metrics.final_live_edges = live_edges;
    Ok(Output {
        rules,
        final_tokens,
        metrics,
    })
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
        Prepared {
            corpus,
            initial_lengths: vec![1; alphabet + 1],
            pivots,
            weights,
        }
    }

    fn compare(input: Prepared, merges: usize, min_frequency: u64) {
        let options = TrainOptions {
            max_merges: merges,
            min_frequency,
            bounds: Bounds::Checked,
        };
        let expected = reference(input.clone(), options).unwrap();
        for heap_policy in [HeapPolicy::Lazy, HeapPolicy::Eager] {
            for workers in [1, 2, 4] {
                for chunk_size in [1, 5, 32] {
                    let actual = train(
                        input.clone(),
                        options,
                        Config {
                            workers,
                            chunk_size,
                            heap_policy,
                            spatial_probe: SpatialProbe::Budgeted,
                        },
                    )
                    .unwrap();
                    assert_eq!(
                        actual.rules, expected.merges,
                        "rules: policy={heap_policy:?}, workers={workers}, chunk={chunk_size}"
                    );
                    assert_eq!(
                        actual.final_tokens, expected.final_tokens,
                        "tokens: policy={heap_policy:?}, workers={workers}, chunk={chunk_size}"
                    );
                }
            }
        }
    }

    fn probe_trace(input: Prepared, merges: usize, spatial_probe: SpatialProbe) -> Output {
        train(
            input,
            TrainOptions {
                max_merges: merges,
                min_frequency: 1,
                bounds: Bounds::Checked,
            },
            Config {
                workers: 2,
                chunk_size: 1,
                heap_policy: HeapPolicy::Lazy,
                spatial_probe,
            },
        )
        .unwrap()
    }

    #[test]
    fn spatial_probe_widens_only_when_valid_occurrences_are_disjoint() {
        let input = prepared(&[(vec![1, 2, 4, 2, 3], 1)], 4);
        let off = probe_trace(input.clone(), 2, SpatialProbe::Off);
        let budgeted = probe_trace(input, 2, SpatialProbe::Budgeted);
        assert_eq!(budgeted.rules, off.rules);
        assert_eq!(budgeted.final_tokens, off.final_tokens);
        assert_eq!(off.metrics.max_batch_width, 1);
        assert_eq!(budgeted.metrics.max_batch_width, 2);
        assert_eq!(budgeted.metrics.probe_proved_disjoint, 1);
        assert_eq!(budgeted.metrics.probe_visited, 1);
        assert!(budgeted.metrics.probe_visited <= budgeted.metrics.probe_epoch_budget_total);

        let across_pieces = probe_trace(
            prepared(&[(vec![1, 2], 1), (vec![2, 3], 1)], 3),
            2,
            SpatialProbe::Budgeted,
        );
        assert_eq!(across_pieces.metrics.max_batch_width, 2);
        assert_eq!(across_pieces.metrics.probe_proved_disjoint, 1);
    }

    #[test]
    fn spatial_probe_detects_both_overlap_directions_and_long_tokens() {
        let left = probe_trace(
            prepared(&[(vec![1, 2, 3], 1)], 3),
            2,
            SpatialProbe::Budgeted,
        );
        let right = probe_trace(
            prepared(&[(vec![3, 1, 2], 1)], 3),
            2,
            SpatialProbe::Budgeted,
        );
        assert!(left.metrics.probe_conflict_stops >= 1);
        assert!(right.metrics.probe_conflict_stops >= 1);

        let long = probe_trace(
            prepared(&[(vec![1, 2, 3, 4], 1), (vec![1, 2], 1)], 4),
            3,
            SpatialProbe::Budgeted,
        );
        assert_eq!((long.rules[0].left, long.rules[0].right), (1, 2));
        assert_eq!((long.rules[1].left, long.rules[1].right), (3, 4));
        // The second probe sees the fresh length-two token immediately left
        // of 3, then finds the selected (3,4) on the candidate's right.
        assert!(long.metrics.probe_conflict_stops >= 2);
    }

    #[test]
    fn spatial_probe_preserves_budget_stop_and_aa_singleton() {
        let budget_input = prepared(
            &[
                (vec![1, 2], 10),
                (vec![2, 3], 1),
                (vec![2, 3], 1),
                (vec![2, 3], 1),
            ],
            3,
        );
        let off = probe_trace(budget_input.clone(), 2, SpatialProbe::Off);
        let budgeted = probe_trace(budget_input, 2, SpatialProbe::Budgeted);
        assert_eq!(budgeted.rules, off.rules);
        assert_eq!(budgeted.final_tokens, off.final_tokens);
        assert_eq!(budgeted.metrics.probe_budget_stops, 1);
        assert_eq!(budgeted.metrics.probe_visited, 2);

        let aa = probe_trace(
            prepared(&[(vec![1, 1, 1], 3), (vec![2, 3], 2)], 3),
            2,
            SpatialProbe::Budgeted,
        );
        assert_eq!((aa.rules[0].left, aa.rules[0].right), (1, 1));
        assert_eq!(aa.metrics.batch_rounds, 2);
        assert_eq!(aa.metrics.probe_calls, 0);
    }

    #[test]
    fn overlap_adjacent_ties_and_weights() {
        compare(prepared(&[(vec![1; 11], 3), (vec![1; 7], 2)], 1), 12, 1);
        // The first AA candidate must finish its epoch alone, even when an
        // unrelated second candidate would satisfy the token-disjoint rule.
        compare(prepared(&[(vec![1, 1, 1], 3), (vec![2, 3], 2)], 3), 8, 1);
        compare(
            prepared(&[(vec![1, 2, 1, 2, 1, 2], 3), (vec![2, 1, 2, 1], 4)], 2),
            12,
            1,
        );
        compare(prepared(&[(vec![1, 2, 3], 3), (vec![4, 5], 2)], 5), 10, 1);
        compare(prepared(&[(vec![1, 2, 3, 4, 1, 2, 3, 4], 1)], 4), 12, 1);
        let adjacent = prepared(
            &[(vec![1, 2, 3, 4], 1), (vec![1, 2], 3), (vec![3, 4], 2)],
            4,
        );
        compare(adjacent.clone(), 12, 1);
        let output = train(
            adjacent.clone(),
            TrainOptions {
                max_merges: 12,
                min_frequency: 1,
                bounds: Bounds::Checked,
            },
            Config {
                workers: 4,
                chunk_size: 1,
                heap_policy: HeapPolicy::Lazy,
                spatial_probe: SpatialProbe::Budgeted,
            },
        )
        .unwrap();
        assert!(output.metrics.max_batch_width >= 2);
        let huge_chunk = train(
            adjacent,
            TrainOptions {
                max_merges: 12,
                min_frequency: 1,
                bounds: Bounds::Checked,
            },
            Config {
                workers: 2,
                chunk_size: usize::MAX,
                heap_policy: HeapPolicy::Lazy,
                spatial_probe: SpatialProbe::Budgeted,
            },
        )
        .unwrap();
        assert_eq!(huge_chunk.rules, output.rules);
        assert_eq!(huge_chunk.final_tokens, output.final_tokens);
        compare(prepared(&[(vec![1; 512], 1)], 1), 10, 1);
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
                    .map(|_| {
                        let id = 1 + next() % 4;
                        present[id] = true;
                        id as u32
                    })
                    .collect();
                words.push((word, (1 + next() % 5) as u64));
            }
            for (id, &seen) in present.iter().enumerate().skip(1) {
                if !seen {
                    words.push((vec![id as u32], 1));
                }
            }
            compare(prepared(&words, 4), 16, (1 + next() % 4) as u64);
        }
    }
}

//! Exact batched BPE with owner-local counted deltas and keyed birth chains.

use efficient_bpe_rust::{Prepared, Rule, TrainError, TrainOptions, validate_prepared};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../aa_parity.rs"]
mod aa_parity;
mod small_posting;

use small_posting::SmallPosting;

type Result<T> = std::result::Result<T, TrainError>;

trait CorpusCell: Send + Sync {
    fn read(&self) -> u32;
    fn write(&self, value: u32);
}

impl CorpusCell for AtomicU32 {
    #[inline]
    fn read(&self) -> u32 {
        self.load(Ordering::Relaxed)
    }

    #[inline]
    fn write(&self, value: u32) {
        self.store(value, Ordering::Relaxed);
    }
}

impl CorpusCell for AtomicU16 {
    #[inline]
    fn read(&self) -> u32 {
        u32::from(self.load(Ordering::Relaxed))
    }

    #[inline]
    fn write(&self, value: u32) {
        // write_merge stores only zero or a fresh ID, both within the
        // checked full-call L + R <= 65536 domain.
        debug_assert!(value <= u32::from(u16::MAX));
        self.store(value as u16, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub workers: usize,
    pub chunk_size: usize,
    pub heap_policy: HeapPolicy,
    pub corpus_width: CorpusWidth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorpusWidth {
    U32,
    U32Exact,
    U16,
    Auto,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CorpusAllocation {
    #[default]
    ConsumingCollect,
    ExactNew,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeapPolicy {
    Eager,
    Lazy,
}

#[derive(Default, Clone, Debug)]
pub struct Metrics {
    pub validation_seconds: f64,
    pub conversion_seconds: f64,
    pub corpus_allocation: CorpusAllocation,
    pub corpus_width_bits: u8,
    pub corpus_positions: usize,
    pub input_corpus_capacity: usize,
    pub endpoint_corpus_capacity: usize,
    pub endpoint_corpus_bytes: usize,
    pub conversion_simultaneous_capacity_bytes: usize,
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
    pub final_owner_stats_seconds: f64,
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
    pub peak_route_delta_capacity: usize,
    pub delta_value_bytes: usize,
    pub delta_entry_bytes: usize,
    pub birth_node_bytes: usize,
    pub grouped_birth_keys: usize,
    pub grouped_birth_nodes: usize,
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
    pub birth_group_fill_seconds: f64,
    pub heap_pops: usize,
    pub heap_refreshes: usize,
    pub heap_reinsertions: usize,
    pub peak_heap_len: usize,
    pub peak_heap_capacity: usize,
    pub owned_posting_len: usize,
    pub owned_posting_capacity: usize,
    pub owner_entry_count: usize,
    pub inline_posting_keys: usize,
    pub inline_posting_positions: usize,
    pub heap_posting_keys: usize,
    pub peak_route_born_len: usize,
    pub peak_route_born_capacity: usize,
}

#[derive(Debug)]
pub struct Output {
    pub rules: Vec<Rule>,
    pub final_tokens: Vec<u32>,
    pub metrics: Metrics,
}

struct Entry {
    frequency: u64,
    positions: SmallPosting,
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
    delta: HashMap<u64, Delta>,
    born: Vec<BirthNode>,
}

#[derive(Clone, Copy)]
struct Delta {
    weight: u64,
    occurrences: u32,
    head: u32,
}

impl Default for Delta {
    fn default() -> Self {
        Self {
            weight: 0,
            occurrences: 0,
            head: u32::MAX,
        }
    }
}

#[derive(Clone, Copy)]
struct BirthNode {
    pos: u32,
    next: u32,
}

struct BatchRule {
    a: u32,
    b: u32,
    new_id: u32,
    frequency: u64,
    a_length: usize,
    b_length: usize,
    posting: SmallPosting,
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
fn read<C: CorpusCell>(corpus: &[C], pos: usize) -> u32 {
    corpus[pos].read()
}

fn inspect<C: CorpusCell>(
    corpus: &[C],
    lengths: &[u32],
    pos: usize,
    a: u32,
    b: u32,
) -> Option<Plan> {
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
fn add_delta(delta: &mut HashMap<u64, Delta>, pair: u64, weight: u64) -> Result<()> {
    accumulate_delta(
        delta,
        pair,
        Delta {
            weight,
            occurrences: 1,
            head: u32::MAX,
        },
    )
}

fn accumulate_delta(delta: &mut HashMap<u64, Delta>, pair: u64, incoming: Delta) -> Result<()> {
    let entry = delta.entry(pair).or_default();
    entry.weight = entry
        .weight
        .checked_add(incoming.weight)
        .ok_or(TrainError::Overflow("routed delta weight exceeds u64"))?;
    entry.occurrences = entry
        .occurrences
        .checked_add(incoming.occurrences)
        .ok_or(TrainError::Overflow("routed occurrence count exceeds u32"))?;
    Ok(())
}

fn write_merge<C: CorpusCell>(corpus: &[C], plan: Plan, b_length: usize, new_id: u32) {
    let pos = plan.pos as usize;
    let right = plan.right as usize;
    let after = plan.after as usize;
    corpus[pos].write(new_id);
    if b_length == 1 {
        corpus[right].write(new_id);
    } else {
        corpus[right].write(0);
        corpus[after - 1].write(new_id);
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

#[inline]
fn is_new_pair(pair: u64, fresh_start: u32) -> bool {
    (pair >> 32) as u32 >= fresh_start || (pair as u32) >= fresh_start
}

fn empty_worker(workers: usize) -> WorkerOutput {
    WorkerOutput {
        routes: (0..workers).map(|_| Route::default()).collect(),
        results: Vec::new(),
        visits: 0,
        merges: 0,
    }
}

fn route_delta(output: &mut WorkerOutput, workers: usize, pair: u64, weight: u64) -> Result<()> {
    add_delta(
        &mut output.routes[owner_for(pair, workers)].delta,
        pair,
        weight,
    )
}

fn route_birth(
    output: &mut WorkerOutput,
    workers: usize,
    pair: u64,
    pos: u32,
    weight: u64,
) -> Result<()> {
    let route = &mut output.routes[owner_for(pair, workers)];
    let index = u32::try_from(route.born.len())
        .map_err(|_| TrainError::Overflow("birth route index exceeds u32"))?;
    if index == u32::MAX {
        return Err(TrainError::Overflow(
            "birth route index collides with empty sentinel",
        ));
    }
    let entry = route.delta.entry(pair).or_default();
    let new_weight = entry
        .weight
        .checked_add(weight)
        .ok_or(TrainError::Overflow("routed delta weight exceeds u64"))?;
    let new_count = entry
        .occurrences
        .checked_add(1)
        .ok_or(TrainError::Overflow("routed occurrence count exceeds u32"))?;
    route.born.push(BirthNode {
        pos,
        next: entry.head,
    });
    entry.weight = new_weight;
    entry.occurrences = new_count;
    entry.head = index;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn initial_index<C: CorpusCell>(
    pool: &ThreadPool,
    corpus: &[C],
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
                let mut positions_by_owner = vec![Vec::<u32>::new(); workers];
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
                            positions_by_owner[owner_for(key(a, b), workers)].push(pos as u32);
                        }
                    }
                }
                positions_by_owner
            })
            .collect::<Vec<_>>()
    });
    metrics.initial_route_seconds = started.elapsed().as_secs_f64();
    metrics.initial_all_postings = outputs.iter().flatten().map(Vec::len).sum();
    let started = Instant::now();
    let mut owners: Vec<Owner> = (0..workers).map(|_| Owner::default()).collect();
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .map(|(owner_i, owner)| -> Result<()> {
                for output in &outputs {
                    for &pos in &output[owner_i] {
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
                            positions: SmallPosting::default(),
                        });
                        entry.frequency = entry
                            .frequency
                            .checked_add(weight_at(pivots, weights, pos))
                            .ok_or(TrainError::Overflow("pair frequency exceeds u64"))?;
                        entry.positions.push(pos)?;
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

fn route_aa(
    pool: &ThreadPool,
    chunks: &[Vec<Plan>],
    a: u32,
    b: u32,
    new_id: u32,
    workers: usize,
    metrics: &mut Metrics,
) -> Result<Vec<WorkerOutput>> {
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
    let results = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| -> Result<WorkerOutput> {
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
                        let w = plan.weight;
                        if plan.left_id != 0 && !previous_selected {
                            let old_key = key(plan.left_id, a);
                            if old_key != key(a, b) {
                                route_delta(&mut output, workers, old_key, w)?;
                            }
                            route_birth(
                                &mut output,
                                workers,
                                key(plan.left_id, new_id),
                                plan.before,
                                w,
                            )?;
                        }
                        if plan.right_id != 0 {
                            let old_key = key(b, plan.right_id);
                            if old_key != key(a, b) {
                                route_delta(&mut output, workers, old_key, w)?;
                            }
                            let final_right = if next_selected { new_id } else { plan.right_id };
                            route_birth(
                                &mut output,
                                workers,
                                key(new_id, final_right),
                                plan.pos,
                                w,
                            )?;
                        }
                        output.merges += 1;
                    }
                }
                Ok(output)
            })
            .collect::<Vec<_>>()
    });
    results.into_iter().collect()
}

#[allow(clippy::too_many_arguments)]
fn prepare_batch<C: CorpusCell>(
    pool: &ThreadPool,
    corpus: &[C],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    batch: &[BatchRule],
    tasks: &[FlatTask],
    selected: &HashMap<u64, u32>,
    workers: usize,
) -> Result<Vec<WorkerOutput>> {
    let cursor = AtomicUsize::new(0);
    let results = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| -> Result<WorkerOutput> {
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
                    for &pos in &rule.posting.as_slice()[task.start..task.end] {
                        let Some(plan) = inspect(corpus, lengths, pos as usize, rule.a, rule.b)
                        else {
                            continue;
                        };
                        valid.push(pos);
                        output.merges += 1;
                        let w = weight_at(pivots, weights, pos);
                        if plan.left_id != 0 {
                            let before = plan.before as usize;
                            let prior_id = read(corpus, before - 1);
                            let left_selected = prior_id != 0
                                && selected.contains_key(&key(prior_id, plan.left_id));
                            if !left_selected {
                                route_delta(&mut output, workers, key(plan.left_id, rule.a), w)?;
                                route_birth(
                                    &mut output,
                                    workers,
                                    key(plan.left_id, rule.new_id),
                                    plan.before,
                                    w,
                                )?;
                            }
                        }
                        if plan.right_id != 0 {
                            route_delta(&mut output, workers, key(rule.b, plan.right_id), w)?;
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
                            )?;
                        }
                    }
                    output.results.push((task_i, valid));
                }
                Ok(output)
            })
            .collect::<Vec<_>>()
    });
    results.into_iter().collect()
}

fn apply_batch<C: CorpusCell>(
    pool: &ThreadPool,
    corpus: &[C],
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
fn commit_routes<C: CorpusCell>(
    pool: &ThreadPool,
    owners: &mut [Owner],
    outputs: &[WorkerOutput],
    corpus: &[C],
    lengths: &[u32],
    selected: &HashSet<u64>,
    fresh_start: u32,
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
    metrics.grouped_birth_nodes += born_len;
    metrics.grouped_birth_keys += outputs
        .iter()
        .flat_map(|output| &output.routes)
        .flat_map(|route| route.delta.values())
        .filter(|delta| delta.head != u32::MAX)
        .count();
    let route_keys = outputs
        .iter()
        .flat_map(|output| &output.routes)
        .map(|route| route.delta.len())
        .sum();
    metrics.peak_delta_keys = metrics.peak_delta_keys.max(route_keys);
    metrics.peak_route_delta_capacity = metrics.peak_route_delta_capacity.max(
        outputs
            .iter()
            .flat_map(|output| &output.routes)
            .map(|route| route.delta.capacity())
            .sum(),
    );
    let started = Instant::now();
    let workers = owners.len();
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .map(|(owner_i, owner)| -> Result<Vec<(u64, u32)>> {
                let mut combined = HashMap::<u64, Delta>::new();
                for output in outputs {
                    for (&pair, &delta) in &output.routes[owner_i].delta {
                        accumulate_delta(&mut combined, pair, delta)?;
                    }
                }
                let mut expected = Vec::new();
                for (pair, delta) in combined {
                    if selected.contains(&pair) {
                        continue;
                    }
                    if is_new_pair(pair, fresh_start) {
                        if owner.entries.contains_key(&pair) {
                            return Err(TrainError::InternalInvariant(
                                "fresh pair already in owner",
                            ));
                        }
                        if delta.weight >= minimum {
                            owner.entries.insert(
                                pair,
                                Entry {
                                    frequency: delta.weight,
                                    positions: SmallPosting::with_capacity(delta.occurrences)?,
                                },
                            );
                            owner.heap.push(Candidate {
                                key: pair,
                                frequency: delta.weight,
                            });
                            expected.push((pair, delta.occurrences));
                        }
                    } else if let Some(entry) = owner.entries.get_mut(&pair) {
                        entry.frequency = entry
                            .frequency
                            .checked_sub(delta.weight)
                            .ok_or(TrainError::InternalInvariant("negative old pair frequency"))?;
                        if entry.frequency < minimum {
                            owner.entries.remove(&pair);
                        } else if policy == HeapPolicy::Eager {
                            owner.heap.push(Candidate {
                                key: pair,
                                frequency: entry.frequency,
                            });
                        }
                    }
                }
                Ok(expected)
            })
            .collect::<Vec<_>>()
    });
    let expected = checks.into_iter().collect::<Result<Vec<_>>>()?;
    metrics.frequency_reduce_seconds += started.elapsed().as_secs_f64();
    let started = Instant::now();
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .map(|(owner_i, owner)| -> Result<usize> {
                let mut stored = 0;
                for output in outputs {
                    let route = &output.routes[owner_i];
                    for (&pair, delta) in &route.delta {
                        if !is_new_pair(pair, fresh_start) {
                            if delta.head != u32::MAX {
                                return Err(TrainError::InternalInvariant(
                                    "old delta owns birth chain",
                                ));
                            }
                            continue;
                        }
                        if owner_for(pair, workers) != owner_i {
                            return Err(TrainError::InternalInvariant(
                                "birth owner route mismatch",
                            ));
                        }
                        let Some(entry) = owner.entries.get_mut(&pair) else {
                            continue;
                        };
                        let mut cursor = delta.head;
                        let mut traversed = 0_u32;
                        while cursor != u32::MAX {
                            let node = *route.born.get(cursor as usize).ok_or(
                                TrainError::InternalInvariant("birth chain index outside route"),
                            )?;
                            debug_assert!(node.next == u32::MAX || node.next < cursor);
                            debug_assert_eq!(
                                {
                                    let p = node.pos as usize;
                                    let a = read(corpus, p);
                                    let next = p + lengths[a as usize] as usize;
                                    key(a, read(corpus, next))
                                },
                                pair,
                                "planned birth key differs from final corpus"
                            );
                            entry.positions.push(node.pos)?;
                            stored += 1;
                            traversed = traversed
                                .checked_add(1)
                                .ok_or(TrainError::Overflow("birth chain length exceeds u32"))?;
                            if traversed > delta.occurrences {
                                return Err(TrainError::InternalInvariant(
                                    "birth chain exceeds counted occurrences",
                                ));
                            }
                            cursor = node.next;
                        }
                        if traversed != delta.occurrences {
                            return Err(TrainError::InternalInvariant(
                                "birth chain differs from counted occurrences",
                            ));
                        }
                    }
                }
                for &(pair, count) in &expected[owner_i] {
                    let entry = owner
                        .entries
                        .get(&pair)
                        .ok_or(TrainError::InternalInvariant(
                            "fresh eligible posting disappeared",
                        ))?;
                    if entry.positions.len() != count as usize {
                        return Err(TrainError::InternalInvariant(
                            "birth count differs from posting length",
                        ));
                    }
                }
                Ok(stored)
            })
            .collect::<Vec<_>>()
    });
    for check in checks {
        metrics.stored_born_postings += check?;
    }
    metrics.birth_group_fill_seconds += started.elapsed().as_secs_f64();
    Ok(())
}

fn fits_u16_full_call(initial_ids: usize, max_merges: usize) -> bool {
    initial_ids
        .checked_add(max_merges)
        .is_some_and(|limit| limit <= usize::from(u16::MAX) + 1)
}

fn record_corpus_capacity<C>(
    metrics: &mut Metrics,
    source_capacity: usize,
    corpus_positions: usize,
    endpoint_capacity: usize,
) {
    metrics.corpus_width_bits = (std::mem::size_of::<C>() * 8) as u8;
    metrics.corpus_positions = corpus_positions;
    metrics.input_corpus_capacity = source_capacity;
    metrics.endpoint_corpus_capacity = endpoint_capacity;
    metrics.endpoint_corpus_bytes = endpoint_capacity.saturating_mul(std::mem::size_of::<C>());
    metrics.conversion_simultaneous_capacity_bytes = source_capacity
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(endpoint_capacity.saturating_mul(std::mem::size_of::<C>()));
}

/// Validate the shared input contract and choose one endpoint type for the full call.
pub fn train(input: Prepared, options: TrainOptions, config: Config) -> Result<Output> {
    if config.workers == 0 || config.chunk_size == 0 || options.min_frequency == 0 {
        return Err(TrainError::InvalidInput(
            "workers, chunk_size and min_frequency must be positive",
        ));
    }
    let started = Instant::now();
    validate_prepared(&input, options)?;
    let u16_eligible = fits_u16_full_call(input.initial_lengths.len(), options.max_merges);
    let use_u16 = match config.corpus_width {
        CorpusWidth::U32 | CorpusWidth::U32Exact => false,
        CorpusWidth::U16 if u16_eligible => true,
        CorpusWidth::U16 => {
            return Err(TrainError::InvalidInput(
                "u16 corpus width requires initial IDs plus max_merges <= 65536",
            ));
        }
        CorpusWidth::Auto => u16_eligible,
    };
    let mut metrics = Metrics {
        validation_seconds: started.elapsed().as_secs_f64(),
        ..Metrics::default()
    };
    metrics.delta_value_bytes = std::mem::size_of::<Delta>();
    metrics.delta_entry_bytes = std::mem::size_of::<(u64, Delta)>();
    metrics.birth_node_bytes = std::mem::size_of::<BirthNode>();
    let started = Instant::now();
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .build()
        .map_err(|_| TrainError::InvalidInput("cannot create worker pool"))?;
    metrics.pool_seconds = started.elapsed().as_secs_f64();
    let Prepared {
        corpus: raw_corpus,
        initial_lengths,
        pivots,
        weights,
    } = input;
    let source_capacity = raw_corpus.capacity();
    let started = Instant::now();
    if use_u16 {
        // Keep the original u32 Vec alive until this new u16 allocation is filled.
        // This makes the conversion's simultaneous capacity explicit.
        let mut corpus = Vec::<AtomicU16>::with_capacity(raw_corpus.len());
        for &id in &raw_corpus {
            // Shared validation requires initial IDs < L, and L <= 65536.
            debug_assert!(id <= u32::from(u16::MAX));
            corpus.push(AtomicU16::new(id as u16));
        }
        record_corpus_capacity::<AtomicU16>(
            &mut metrics,
            source_capacity,
            corpus.len(),
            corpus.capacity(),
        );
        metrics.corpus_allocation = CorpusAllocation::ExactNew;
        drop(raw_corpus);
        metrics.conversion_seconds = started.elapsed().as_secs_f64();
        train_impl(
            corpus,
            initial_lengths,
            pivots,
            weights,
            options,
            config,
            metrics,
            pool,
        )
    } else if config.corpus_width == CorpusWidth::U32Exact {
        // Match the u16 conversion's allocation lifecycle while retaining
        // 32-bit atomic endpoints: both source and destination coexist.
        let mut corpus = Vec::<AtomicU32>::with_capacity(raw_corpus.len());
        for &id in &raw_corpus {
            corpus.push(AtomicU32::new(id));
        }
        record_corpus_capacity::<AtomicU32>(
            &mut metrics,
            source_capacity,
            corpus.len(),
            corpus.capacity(),
        );
        metrics.corpus_allocation = CorpusAllocation::ExactNew;
        drop(raw_corpus);
        metrics.conversion_seconds = started.elapsed().as_secs_f64();
        train_impl(
            corpus,
            initial_lengths,
            pivots,
            weights,
            options,
            config,
            metrics,
            pool,
        )
    } else {
        let corpus: Vec<AtomicU32> = raw_corpus.into_iter().map(AtomicU32::new).collect();
        record_corpus_capacity::<AtomicU32>(
            &mut metrics,
            source_capacity,
            corpus.len(),
            corpus.capacity(),
        );
        metrics.conversion_seconds = started.elapsed().as_secs_f64();
        train_impl(
            corpus,
            initial_lengths,
            pivots,
            weights,
            options,
            config,
            metrics,
            pool,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn train_impl<C: CorpusCell>(
    corpus: Vec<C>,
    mut initial_lengths: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
    options: TrainOptions,
    config: Config,
    mut metrics: Metrics,
    pool: ThreadPool,
) -> Result<Output> {
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
            if !chosen.is_empty() && (a == b || tails.contains(&a) || heads.contains(&b)) {
                break;
            }
            owners[owner_i].heap.pop();
            metrics.heap_pops += 1;
            let entry = owners[owner_i]
                .entries
                .remove(&candidate.key)
                .ok_or(TrainError::InternalInvariant("selected posting absent"))?;
            chosen.push((candidate, entry));
            heads.insert(a);
            tails.insert(b);
            if a == b {
                break;
            }
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
            pool.install(|| rule.posting.as_mut_slice().par_sort_unstable());
            metrics.aa_sort_seconds += started.elapsed().as_secs_f64();
            let started = Instant::now();
            let valid_chunks = pool.install(|| {
                rule.posting
                    .as_slice()
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
            )?;
            metrics.plan_seconds += started.elapsed().as_secs_f64();
            // The ordered Plan chunks retain every selected AA start needed by apply.
            drop(rule.posting.take());
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
            )?;
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
                drop(rule.posting.take());
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
            batch[0].new_id,
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
    let owner_stats_started = Instant::now();
    for owner in &owners {
        metrics.owner_entry_count += owner.entries.len();
        for entry in owner.entries.values() {
            debug_assert!(!entry.positions.is_empty());
            metrics.owned_posting_len += entry.positions.len();
            metrics.owned_posting_capacity += entry.positions.allocated_capacity();
            if entry.positions.is_inline() {
                metrics.inline_posting_keys += 1;
                metrics.inline_posting_positions += entry.positions.len();
            } else {
                metrics.heap_posting_keys += 1;
            }
        }
    }
    metrics.retained_entry_posting_len = metrics.owned_posting_len;
    metrics.eligible_posting_len = metrics.owned_posting_len;
    metrics.final_live_edges = live_edges;
    metrics.final_owner_stats_seconds = owner_stats_started.elapsed().as_secs_f64();
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
        for corpus_width in [
            CorpusWidth::U32,
            CorpusWidth::U32Exact,
            CorpusWidth::U16,
            CorpusWidth::Auto,
        ] {
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
                                corpus_width,
                            },
                        )
                        .unwrap();
                        assert_eq!(
                            actual.rules, expected.merges,
                            "rules: width={corpus_width:?}, policy={heap_policy:?}, workers={workers}, chunk={chunk_size}"
                        );
                        assert_eq!(
                            actual.final_tokens, expected.final_tokens,
                            "tokens: width={corpus_width:?}, policy={heap_policy:?}, workers={workers}, chunk={chunk_size}"
                        );
                        assert_eq!(
                            actual.metrics.corpus_width_bits,
                            if matches!(corpus_width, CorpusWidth::U32 | CorpusWidth::U32Exact) {
                                32
                            } else {
                                16
                            }
                        );
                        assert_eq!(
                            actual.metrics.corpus_allocation,
                            if corpus_width == CorpusWidth::U32 {
                                CorpusAllocation::ConsumingCollect
                            } else {
                                CorpusAllocation::ExactNew
                            }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn overlap_adjacent_ties_and_weights() {
        assert_eq!(std::mem::size_of::<Delta>(), 16);
        assert_eq!(std::mem::size_of::<(u64, Delta)>(), 24);
        assert_eq!(std::mem::size_of::<BirthNode>(), 8);
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
        compare(
            prepared(&[(vec![1, 2, 3], u64::MAX / 2), (vec![3], 1)], 3),
            5,
            1,
        );
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
                corpus_width: CorpusWidth::U16,
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
                corpus_width: CorpusWidth::U16,
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

    #[test]
    fn width_boundaries_and_fresh_id_range() {
        assert!(fits_u16_full_call(65_535, 0));
        assert!(fits_u16_full_call(65_535, 1));
        assert!(fits_u16_full_call(65_536, 0));
        assert!(!fits_u16_full_call(65_536, 1));
        assert!(!fits_u16_full_call(usize::MAX, 1));

        for (alphabet, expected_bits) in [(65_534, 16), (65_535, 32)] {
            let mut word = (1..=alphabet).collect::<Vec<u32>>();
            word.extend([1, 2]);
            let input = prepared(&[(word, 1)], alphabet as usize);
            let options = TrainOptions {
                max_merges: 1,
                min_frequency: 1,
                bounds: Bounds::Checked,
            };
            let output = train(
                input.clone(),
                options,
                Config {
                    workers: 2,
                    chunk_size: 512,
                    heap_policy: HeapPolicy::Lazy,
                    corpus_width: CorpusWidth::Auto,
                },
            )
            .unwrap();
            assert_eq!(output.metrics.corpus_width_bits, expected_bits);
            assert_eq!(output.rules.len(), 1);
            assert_eq!((output.rules[0].left, output.rules[0].right), (1, 2));
            assert_eq!(output.rules[0].frequency, 2);
            let fresh_id = alphabet + 1;
            assert_eq!(
                output
                    .final_tokens
                    .iter()
                    .filter(|&&id| id == fresh_id)
                    .count(),
                2
            );

            let strict = train(
                input,
                options,
                Config {
                    workers: 1,
                    chunk_size: 512,
                    heap_policy: HeapPolicy::Lazy,
                    corpus_width: CorpusWidth::U16,
                },
            );
            if expected_bits == 16 {
                assert_eq!(strict.unwrap().final_tokens, output.final_tokens);
            } else {
                assert!(matches!(strict, Err(TrainError::InvalidInput(_))));
            }
        }
    }

    #[test]
    fn long_token_lengths_do_not_limit_endpoint_width() {
        let input = prepared(&[(vec![1; 513], 1)], 1);
        let options = TrainOptions {
            max_merges: 10,
            min_frequency: 1,
            bounds: Bounds::Checked,
        };
        let narrow = train(
            input.clone(),
            options,
            Config {
                workers: 2,
                chunk_size: 31,
                heap_policy: HeapPolicy::Lazy,
                corpus_width: CorpusWidth::U16,
            },
        )
        .unwrap();
        let wide = train(
            input.clone(),
            options,
            Config {
                workers: 2,
                chunk_size: 31,
                heap_policy: HeapPolicy::Lazy,
                corpus_width: CorpusWidth::U32,
            },
        )
        .unwrap();
        let wide_exact = train(
            input.clone(),
            options,
            Config {
                workers: 2,
                chunk_size: 31,
                heap_policy: HeapPolicy::Lazy,
                corpus_width: CorpusWidth::U32Exact,
            },
        )
        .unwrap();
        let auto = train(
            input,
            options,
            Config {
                workers: 2,
                chunk_size: 31,
                heap_policy: HeapPolicy::Lazy,
                corpus_width: CorpusWidth::Auto,
            },
        )
        .unwrap();
        assert_eq!(narrow.rules, wide.rules);
        assert_eq!(narrow.final_tokens, wide.final_tokens);
        assert_eq!(auto.rules, narrow.rules);
        assert_eq!(auto.final_tokens, narrow.final_tokens);
        assert_eq!(auto.metrics.corpus_width_bits, 16);
        assert_eq!(auto.metrics.corpus_allocation, CorpusAllocation::ExactNew);
        assert_eq!(wide_exact.rules, wide.rules);
        assert_eq!(wide_exact.final_tokens, wide.final_tokens);
        assert_eq!(
            narrow.final_tokens,
            vec![0, (narrow.rules.len() + 1) as u32, 0]
        );
        assert_eq!(narrow.metrics.corpus_width_bits, 16);
        assert_eq!(wide_exact.metrics.corpus_width_bits, 32);
        assert_eq!(
            wide_exact.metrics.corpus_allocation,
            CorpusAllocation::ExactNew
        );
        assert_eq!(
            wide.metrics.corpus_allocation,
            CorpusAllocation::ConsumingCollect
        );
        assert_eq!(
            narrow.metrics.endpoint_corpus_bytes,
            narrow.metrics.endpoint_corpus_capacity * 2
        );
        assert_eq!(
            narrow.metrics.conversion_simultaneous_capacity_bytes,
            narrow.metrics.input_corpus_capacity * 4 + narrow.metrics.endpoint_corpus_capacity * 2
        );
        assert_eq!(
            wide_exact.metrics.conversion_simultaneous_capacity_bytes,
            wide_exact.metrics.input_corpus_capacity * 4
                + wide_exact.metrics.endpoint_corpus_capacity * 4
        );
    }
}

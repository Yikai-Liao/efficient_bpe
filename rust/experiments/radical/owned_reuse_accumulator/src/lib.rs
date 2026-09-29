//! Exact batched BPE with owner-local counted deltas and keyed birth chains.

use efficient_bpe_rust::{Prepared, Rule, TrainError, TrainOptions, validate_prepared};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cmp::Ordering as CmpOrdering;
use std::collections::hash_map::RandomState as StdRandomState;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../aa_parity.rs"]
mod aa_parity;
mod small_posting;

use small_posting::SmallPosting;

type Result<T> = std::result::Result<T, TrainError>;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub workers: usize,
    pub chunk_size: usize,
    pub heap_policy: HeapPolicy,
    pub integer_hash: IntegerHash,
    pub owner_commit: OwnerCommit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnerCommit {
    Staged,
    FusedFresh,
    FusedReuse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerHash {
    Std,
    AHash,
}

trait HashBuild: BuildHasher + Default + Clone + Send + Sync {}
impl<T: BuildHasher + Default + Clone + Send + Sync> HashBuild for T {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeapPolicy {
    Eager,
    Lazy,
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
    pub fused_commit_seconds: f64,
    pub accumulator_capacity_sum_peak: usize,
    pub accumulator_entries_sum_peak: usize,
    pub accumulator_reused_input_keys: usize,
    pub accumulator_foreign_entry_visits: usize,
    pub transposed_route_header_bytes_peak: usize,
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

struct Owner<H: HashBuild> {
    entries: HashMap<u64, Entry, H>,
    heap: BinaryHeap<Candidate>,
}

impl<H: HashBuild> Default for Owner<H> {
    fn default() -> Self {
        Self {
            entries: HashMap::with_hasher(H::default()),
            heap: BinaryHeap::new(),
        }
    }
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

struct Route<H: HashBuild> {
    delta: HashMap<u64, Delta, H>,
    born: Vec<BirthNode>,
}

impl<H: HashBuild> Default for Route<H> {
    fn default() -> Self {
        Self {
            delta: HashMap::with_hasher(H::default()),
            born: Vec::new(),
        }
    }
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

struct WorkerOutput<H: HashBuild> {
    routes: Vec<Route<H>>,
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
fn add_delta<H: HashBuild>(
    delta: &mut HashMap<u64, Delta, H>,
    pair: u64,
    weight: u64,
) -> Result<()> {
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

fn accumulate_delta<H: HashBuild>(
    delta: &mut HashMap<u64, Delta, H>,
    pair: u64,
    incoming: Delta,
) -> Result<()> {
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

#[inline]
fn is_new_pair(pair: u64, fresh_start: u32) -> bool {
    (pair >> 32) as u32 >= fresh_start || (pair as u32) >= fresh_start
}

fn empty_worker<H: HashBuild>(workers: usize) -> WorkerOutput<H> {
    WorkerOutput {
        routes: (0..workers).map(|_| Route::<H>::default()).collect(),
        results: Vec::new(),
        visits: 0,
        merges: 0,
    }
}

fn route_delta<H: HashBuild>(
    output: &mut WorkerOutput<H>,
    workers: usize,
    pair: u64,
    weight: u64,
) -> Result<()> {
    add_delta(
        &mut output.routes[owner_for(pair, workers)].delta,
        pair,
        weight,
    )
}

fn route_birth<H: HashBuild>(
    output: &mut WorkerOutput<H>,
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
fn initial_index<H: HashBuild>(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    pivots: &[u32],
    weights: &[u64],
    workers: usize,
    chunk_size: usize,
    minimum: u64,
    metrics: &mut Metrics,
) -> Result<Vec<Owner<H>>> {
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
    let mut owners: Vec<Owner<H>> = (0..workers).map(|_| Owner::<H>::default()).collect();
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

fn peek_current<H: HashBuild>(
    owner: &mut Owner<H>,
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

fn route_aa<H: HashBuild>(
    pool: &ThreadPool,
    chunks: &[Vec<Plan>],
    a: u32,
    b: u32,
    new_id: u32,
    workers: usize,
    metrics: &mut Metrics,
) -> Result<Vec<WorkerOutput<H>>> {
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
            .map(|_| -> Result<WorkerOutput<H>> {
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
fn prepare_batch<H: HashBuild>(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    batch: &[BatchRule],
    tasks: &[FlatTask],
    selected: &HashMap<u64, u32, H>,
    workers: usize,
) -> Result<Vec<WorkerOutput<H>>> {
    let cursor = AtomicUsize::new(0);
    let results = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| -> Result<WorkerOutput<H>> {
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

fn apply_batch<H: HashBuild>(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    batch: &[BatchRule],
    tasks: &[FlatTask],
    outputs: &mut [WorkerOutput<H>],
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

/// Must run before fused modes move the producer routes into owner-local
/// vectors. In particular, the original route-capacity proxy includes every
/// producer table rather than silently omitting the reused accumulator.
fn record_route_metrics<H: HashBuild>(outputs: &[WorkerOutput<H>], metrics: &mut Metrics) {
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
}

#[allow(clippy::too_many_arguments)]
fn commit_routes<H: HashBuild>(
    pool: &ThreadPool,
    owners: &mut [Owner<H>],
    outputs: &[WorkerOutput<H>],
    corpus: &[AtomicU32],
    lengths: &[u32],
    selected: &HashSet<u64, H>,
    fresh_start: u32,
    minimum: u64,
    policy: HeapPolicy,
    metrics: &mut Metrics,
) -> Result<()> {
    record_route_metrics(outputs, metrics);
    let started = Instant::now();
    let workers = owners.len();
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .map(|(owner_i, owner)| -> Result<Vec<(u64, u32)>> {
                let mut combined = HashMap::<u64, Delta, H>::with_hasher(H::default());
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

/// Walk one producer's private chain. The strict decrease of next indices,
/// bounds check and total-posting limit all remain active in release builds.
/// The reused producer's local occurrence count was overwritten by the
/// owner-wide total, so its chain is checked against that final total rather
/// than a false local count. Foreign producers retain their local counts.
#[allow(clippy::too_many_arguments)]
fn fill_chain<H: HashBuild>(
    owner: &mut Owner<H>,
    pair: u64,
    mut cursor: u32,
    born: &[BirthNode],
    local_count: Option<u32>,
    total_count: u32,
    corpus: &[AtomicU32],
    lengths: &[u32],
) -> Result<usize> {
    let entry = owner
        .entries
        .get_mut(&pair)
        .ok_or(TrainError::InternalInvariant(
            "eligible fresh posting disappeared during fill",
        ))?;
    let mut traversed = 0_u32;
    while cursor != u32::MAX {
        let node = *born
            .get(cursor as usize)
            .ok_or(TrainError::InternalInvariant(
                "birth chain index outside its producer route",
            ))?;
        if node.next != u32::MAX && node.next >= cursor {
            return Err(TrainError::InternalInvariant(
                "birth chain does not strictly decrease",
            ));
        }
        if entry.positions.len() >= total_count as usize {
            return Err(TrainError::InternalInvariant(
                "birth chain exceeds total occurrences",
            ));
        }
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
        traversed = traversed
            .checked_add(1)
            .ok_or(TrainError::Overflow("birth chain length exceeds u32"))?;
        if traversed as usize > born.len() {
            return Err(TrainError::InternalInvariant(
                "birth chain exceeds producer born Vec",
            ));
        }
        cursor = node.next;
    }
    if local_count.is_some_and(|count| count != traversed) {
        return Err(TrainError::InternalInvariant(
            "foreign birth chain differs from local occurrences",
        ));
    }
    Ok(traversed as usize)
}

struct OwnerCommitStats {
    stored: usize,
    accumulator_capacity: usize,
    accumulator_entries: usize,
    reused_input_keys: usize,
    foreign_entry_visits: usize,
}

#[allow(clippy::too_many_arguments)]
fn commit_one_owner<H: HashBuild>(
    owner_i: usize,
    owner: &mut Owner<H>,
    mut routes: Vec<Route<H>>,
    corpus: &[AtomicU32],
    lengths: &[u32],
    selected: &HashSet<u64, H>,
    fresh_start: u32,
    minimum: u64,
    policy: HeapPolicy,
    workers: usize,
    reuse_largest: bool,
) -> Result<OwnerCommitStats> {
    let largest = if reuse_largest {
        routes
            .iter()
            .enumerate()
            .max_by_key(|(_, route)| route.delta.len())
            .map(|(i, _)| i)
    } else {
        None
    };
    let mut combined = if let Some(i) = largest {
        // Only the map header/allocation moves. Route i keeps its born Vec;
        // combined heads still index precisely that producer's born Vec.
        std::mem::take(&mut routes[i].delta)
    } else {
        HashMap::<u64, Delta, H>::with_hasher(H::default())
    };
    let reused_input_keys = combined.len();
    let mut foreign_entry_visits = 0;
    for (i, route) in routes.iter().enumerate() {
        if Some(i) == largest {
            continue;
        }
        for (&pair, &delta) in &route.delta {
            // accumulate_delta touches weight/count only. Existing local
            // heads in combined remain intact; foreign-only keys enter with
            // the MAX head, never a foreign born-Vec index.
            accumulate_delta(&mut combined, pair, delta)?;
            foreign_entry_visits += 1;
        }
    }
    let accumulator_capacity = combined.capacity();
    let accumulator_entries = combined.len();
    let mut expected = Vec::new();
    for (&pair, &delta) in &combined {
        if selected.contains(&pair) {
            continue;
        }
        if is_new_pair(pair, fresh_start) {
            if owner.entries.contains_key(&pair) {
                return Err(TrainError::InternalInvariant("fresh pair already in owner"));
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

    let mut stored = 0;
    if let Some(i) = largest {
        for (&pair, &delta) in &combined {
            if !is_new_pair(pair, fresh_start) {
                if delta.head != u32::MAX {
                    return Err(TrainError::InternalInvariant("old delta owns birth chain"));
                }
                continue;
            }
            if owner_for(pair, workers) != owner_i {
                return Err(TrainError::InternalInvariant("birth owner route mismatch"));
            }
            if owner.entries.contains_key(&pair) {
                stored += fill_chain(
                    owner,
                    pair,
                    delta.head,
                    &routes[i].born,
                    None,
                    delta.occurrences,
                    corpus,
                    lengths,
                )?;
            }
        }
    }
    for (i, route) in routes.iter().enumerate() {
        if Some(i) == largest {
            continue;
        }
        for (&pair, &delta) in &route.delta {
            if !is_new_pair(pair, fresh_start) {
                if delta.head != u32::MAX {
                    return Err(TrainError::InternalInvariant("old delta owns birth chain"));
                }
                continue;
            }
            if owner_for(pair, workers) != owner_i {
                return Err(TrainError::InternalInvariant("birth owner route mismatch"));
            }
            if owner.entries.contains_key(&pair) {
                let total = combined
                    .get(&pair)
                    .ok_or(TrainError::InternalInvariant(
                        "foreign fresh pair absent in accumulator",
                    ))?
                    .occurrences;
                stored += fill_chain(
                    owner,
                    pair,
                    delta.head,
                    &route.born,
                    Some(delta.occurrences),
                    total,
                    corpus,
                    lengths,
                )?;
            }
        }
    }
    for (pair, count) in expected {
        let entry = owner
            .entries
            .get(&pair)
            .ok_or(TrainError::InternalInvariant(
                "fresh eligible posting disappeared",
            ))?;
        if entry.positions.len() != count as usize {
            return Err(TrainError::InternalInvariant(
                "birth count differs from total occurrences",
            ));
        }
    }
    Ok(OwnerCommitStats {
        stored,
        accumulator_capacity,
        accumulator_entries,
        reused_input_keys,
        foreign_entry_visits,
    })
}

#[allow(clippy::too_many_arguments)]
fn commit_routes_fused<H: HashBuild>(
    pool: &ThreadPool,
    owners: &mut [Owner<H>],
    outputs: &mut [WorkerOutput<H>],
    corpus: &[AtomicU32],
    lengths: &[u32],
    selected: &HashSet<u64, H>,
    fresh_start: u32,
    minimum: u64,
    policy: HeapPolicy,
    reuse_largest: bool,
    metrics: &mut Metrics,
) -> Result<()> {
    record_route_metrics(outputs, metrics);
    let started = Instant::now();
    let workers = owners.len();
    let original_header_bytes: usize = outputs
        .iter()
        .map(|output| output.routes.capacity() * std::mem::size_of::<Route<H>>())
        .sum();
    let mut owner_routes = (0..workers)
        .map(|_| Vec::<Route<H>>::with_capacity(outputs.len()))
        .collect::<Vec<_>>();
    for output in outputs {
        if output.routes.len() != workers {
            return Err(TrainError::InternalInvariant(
                "producer route count differs from owners",
            ));
        }
        for (owner_i, route) in std::mem::take(&mut output.routes).into_iter().enumerate() {
            owner_routes[owner_i].push(route);
        }
    }
    let transposed_header_bytes: usize = owner_routes
        .iter()
        .map(|routes| routes.capacity() * std::mem::size_of::<Route<H>>())
        .sum();
    // This is a conservative overlap proxy, not a measured heap peak.
    metrics.transposed_route_header_bytes_peak = metrics
        .transposed_route_header_bytes_peak
        .max(original_header_bytes + transposed_header_bytes);
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .zip(owner_routes.into_par_iter())
            .map(|((owner_i, owner), routes)| {
                commit_one_owner(
                    owner_i,
                    owner,
                    routes,
                    corpus,
                    lengths,
                    selected,
                    fresh_start,
                    minimum,
                    policy,
                    workers,
                    reuse_largest,
                )
            })
            .collect::<Vec<_>>()
    });
    let checks = checks.into_iter().collect::<Result<Vec<_>>>()?;
    metrics.stored_born_postings += checks.iter().map(|check| check.stored).sum::<usize>();
    metrics.accumulator_capacity_sum_peak = metrics
        .accumulator_capacity_sum_peak
        .max(checks.iter().map(|check| check.accumulator_capacity).sum());
    metrics.accumulator_entries_sum_peak = metrics
        .accumulator_entries_sum_peak
        .max(checks.iter().map(|check| check.accumulator_entries).sum());
    metrics.accumulator_reused_input_keys += checks
        .iter()
        .map(|check| check.reused_input_keys)
        .sum::<usize>();
    metrics.accumulator_foreign_entry_visits += checks
        .iter()
        .map(|check| check.foreign_entry_visits)
        .sum::<usize>();
    metrics.fused_commit_seconds += started.elapsed().as_secs_f64();
    Ok(())
}

/// Validate the shared input contract inside the timed call.
pub fn train(input: Prepared, options: TrainOptions, config: Config) -> Result<Output> {
    match config.integer_hash {
        IntegerHash::Std => train_impl::<StdRandomState>(input, options, config),
        IntegerHash::AHash => train_impl::<ahash::RandomState>(input, options, config),
    }
}

fn train_impl<H: HashBuild>(
    input: Prepared,
    options: TrainOptions,
    config: Config,
) -> Result<Output> {
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
        corpus,
        mut initial_lengths,
        pivots,
        weights,
    } = input;
    let corpus: Vec<AtomicU32> = corpus.into_iter().map(AtomicU32::new).collect();
    let started = Instant::now();
    let mut owners = initial_index::<H>(
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
        let mut heads = HashSet::<u32, H>::with_hasher(H::default());
        let mut tails = HashSet::<u32, H>::with_hasher(H::default());
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
        let mut selected =
            HashMap::<u64, u32, H>::with_capacity_and_hasher(chosen.len(), H::default());
        let mut selected_keys =
            HashSet::<u64, H>::with_capacity_and_hasher(chosen.len(), H::default());
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
            let outputs = route_aa::<H>(
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
        match config.owner_commit {
            OwnerCommit::Staged => commit_routes(
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
            )?,
            OwnerCommit::FusedFresh | OwnerCommit::FusedReuse => commit_routes_fused(
                &pool,
                &mut owners,
                &mut outputs,
                &corpus,
                &initial_lengths,
                &selected_keys,
                batch[0].new_id,
                options.min_frequency,
                config.heap_policy,
                config.owner_commit == OwnerCommit::FusedReuse,
                &mut metrics,
            )?,
        }
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
    use std::hash::{BuildHasherDefault, Hasher};

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
        for integer_hash in [IntegerHash::Std, IntegerHash::AHash] {
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
                                integer_hash,
                                owner_commit: OwnerCommit::Staged,
                            },
                        )
                        .unwrap();
                        assert_eq!(
                            actual.rules, expected.merges,
                            "rules: hash={integer_hash:?}, policy={heap_policy:?}, workers={workers}, chunk={chunk_size}"
                        );
                        assert_eq!(
                            actual.final_tokens, expected.final_tokens,
                            "tokens: hash={integer_hash:?}, policy={heap_policy:?}, workers={workers}, chunk={chunk_size}"
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
                integer_hash: IntegerHash::AHash,
                owner_commit: OwnerCommit::Staged,
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
                integer_hash: IntegerHash::AHash,
                owner_commit: OwnerCommit::Staged,
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

    fn compare_commit_modes(input: Prepared, merges: usize, minimum: u64) {
        let options = TrainOptions {
            max_merges: merges,
            min_frequency: minimum,
            bounds: Bounds::Checked,
        };
        let expected = reference(input.clone(), options).unwrap();
        for integer_hash in [IntegerHash::Std, IntegerHash::AHash] {
            for workers in [1, 4] {
                let mut control = None;
                for owner_commit in [
                    OwnerCommit::Staged,
                    OwnerCommit::FusedFresh,
                    OwnerCommit::FusedReuse,
                ] {
                    let actual = train(
                        input.clone(),
                        options,
                        Config {
                            workers,
                            chunk_size: 2,
                            heap_policy: HeapPolicy::Lazy,
                            integer_hash,
                            owner_commit,
                        },
                    )
                    .unwrap();
                    assert_eq!(actual.rules, expected.merges, "{owner_commit:?}");
                    assert_eq!(
                        actual.final_tokens, expected.final_tokens,
                        "{owner_commit:?}"
                    );
                    let work = (
                        actual.metrics.actual_merges,
                        actual.metrics.posting_visits,
                        actual.metrics.stale_visits,
                        actual.metrics.generated_birth_records,
                        actual.metrics.stored_born_postings,
                        actual.metrics.batch_rounds,
                        actual.metrics.max_batch_width,
                    );
                    if let Some(control) = control {
                        assert_eq!(work, control, "{owner_commit:?}");
                    } else {
                        control = Some(work);
                    }
                }
            }
        }
    }

    #[test]
    fn commit_modes_match_weighted_aa_adjacent_and_threshold_traces() {
        compare_commit_modes(
            prepared(&[(vec![1; 17], 3), (vec![1, 2, 1, 2, 1], 4)], 2),
            12,
            1,
        );
        compare_commit_modes(
            prepared(
                &[(vec![1, 2, 3, 4], 7), (vec![1, 2], 3), (vec![3, 4], 2)],
                4,
            ),
            12,
            2,
        );
        compare_commit_modes(
            prepared(&[(vec![1, 2, 1, 2, 3, 4, 3, 4], 2), (vec![5, 6], 1)], 6),
            14,
            3,
        );
    }

    #[derive(Clone, Default)]
    struct CollidingHasher;

    impl Hasher for CollidingHasher {
        fn finish(&self) -> u64 {
            0
        }

        fn write(&mut self, _bytes: &[u8]) {}
    }

    type CollidingBuild = BuildHasherDefault<CollidingHasher>;

    fn collision_routes() -> Vec<Route<CollidingBuild>> {
        let mut routes = (0..3).map(|_| Route::default()).collect::<Vec<_>>();
        let pair_45 = key(4, 5);
        let pair_54 = key(5, 4);
        let pair_46 = key(4, 6);
        let old = key(1, 2);
        routes[0].born = vec![
            BirthNode {
                pos: 1,
                next: u32::MAX,
            },
            BirthNode {
                pos: 2,
                next: u32::MAX,
            },
        ];
        routes[0].delta.insert(
            pair_45,
            Delta {
                weight: 2,
                occurrences: 1,
                head: 0,
            },
        );
        routes[0].delta.insert(
            pair_54,
            Delta {
                weight: 1,
                occurrences: 1,
                head: 1,
            },
        );
        routes[0].delta.insert(
            old,
            Delta {
                weight: 2,
                occurrences: 2,
                head: u32::MAX,
            },
        );
        routes[1].born = vec![BirthNode {
            pos: 3,
            next: u32::MAX,
        }];
        routes[1].delta.insert(
            pair_45,
            Delta {
                weight: 3,
                occurrences: 1,
                head: 0,
            },
        );
        routes[1].delta.insert(
            old,
            Delta {
                weight: 1,
                occurrences: 1,
                head: u32::MAX,
            },
        );
        routes[2].born = vec![BirthNode {
            pos: 5,
            next: u32::MAX,
        }];
        routes[2].delta.insert(
            pair_46,
            Delta {
                weight: 3,
                occurrences: 1,
                head: 0,
            },
        );
        routes
    }

    fn commit_collision_routes(
        routes: Vec<Route<CollidingBuild>>,
        reuse_largest: bool,
    ) -> Result<(Owner<CollidingBuild>, OwnerCommitStats)> {
        let corpus = [0, 4, 5, 4, 5, 4, 6, 0].map(AtomicU32::new);
        let lengths = vec![1; 7];
        let selected = HashSet::<u64, CollidingBuild>::with_hasher(CollidingBuild::default());
        let mut owner = Owner::<CollidingBuild>::default();
        owner.entries.insert(
            key(1, 2),
            Entry {
                frequency: 9,
                positions: SmallPosting::default(),
            },
        );
        let stats = commit_one_owner(
            0,
            &mut owner,
            routes,
            &corpus,
            &lengths,
            &selected,
            4,
            2,
            HeapPolicy::Lazy,
            1,
            reuse_largest,
        )?;
        Ok((owner, stats))
    }

    #[test]
    fn reused_head_survives_collisions_and_foreign_only_birth() {
        for reuse in [false, true] {
            let (owner, stats) = commit_collision_routes(collision_routes(), reuse).unwrap();
            assert_eq!(owner.entries[&key(1, 2)].frequency, 6);
            assert_eq!(owner.entries[&key(4, 5)].frequency, 5);
            assert_eq!(owner.entries[&key(4, 5)].positions.len(), 2);
            assert_eq!(owner.entries[&key(4, 6)].frequency, 3);
            assert_eq!(owner.entries[&key(4, 6)].positions.len(), 1);
            assert!(!owner.entries.contains_key(&key(5, 4)));
            assert_eq!(stats.stored, 3);
            assert_eq!(stats.reused_input_keys, if reuse { 3 } else { 0 });
            assert_eq!(stats.foreign_entry_visits, if reuse { 3 } else { 6 });
        }
    }

    #[test]
    fn malformed_foreign_count_and_reused_cycle_are_rejected() {
        let mut routes = collision_routes();
        routes[1].delta.get_mut(&key(4, 5)).unwrap().occurrences = 2;
        assert!(commit_collision_routes(routes, true).is_err());
        let mut routes = collision_routes();
        routes[0].born[0].next = 0;
        assert!(commit_collision_routes(routes, true).is_err());
    }
}

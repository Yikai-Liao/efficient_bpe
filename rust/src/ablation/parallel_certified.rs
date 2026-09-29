//! Exact certified-prefix training with worker-owned occurrence fragments.
//!
//! Workers retain only their own historical positions. The coordinator owns
//! scalar global frequencies and the heap; it never walks an occurrence list.
//! A batch has an immutable read/plan phase, then a disjoint endpoint-write
//! phase. Self pairs use a conservative global left-to-right selection path.

use super::{
    CoreStats, HeapEntry, add_delta, apply_delta, core_result, initial_heap, metric, pair_key,
    pop_best,
};
use crate::ablation::{Options, Result};
use crate::{Prepared, Rule, TrainError};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

type TrainResult<T> = std::result::Result<T, TrainError>;

#[derive(Clone, Copy)]
struct RuleSpec {
    key: u64,
    a: u32,
    b: u32,
    new_id: u32,
    new_length: u32,
    a_length: usize,
    b_length: usize,
    frequency: u64,
}

#[derive(Clone, Copy)]
struct Plan {
    pos: u32,
    right: u32,
    after: u32,
    new_id: u32,
}

const _: [(); 16] = [(); std::mem::size_of::<Plan>()];

#[derive(Clone, Copy)]
struct NewEdge {
    left: u32,
    right: u32,
    pos: u32,
}

impl NewEdge {
    fn new(key: u64, pos: u32) -> Self {
        Self {
            left: (key >> 32) as u32,
            right: key as u32,
            pos,
        }
    }
    fn key(self) -> u64 {
        pair_key(self.left, self.right)
    }
}

const _: [(); 12] = [(); std::mem::size_of::<NewEdge>()];

struct InitialCounts {
    counts: HashMap<u64, u64>,
    occurrences: usize,
}

#[derive(Default, Clone, Copy)]
struct IndexMemory {
    position_len: usize,
    position_capacity: usize,
    map_len: usize,
    map_capacity: usize,
}

struct PreparedReply {
    delta: HashMap<u64, i128>,
    visited: usize,
    stale: usize,
    valid: usize,
    plan_capacity: usize,
    edge_capacity: usize,
    work_seconds: f64,
}

struct GatheredReply {
    valid_positions: Vec<u32>,
    valid_capacity: usize,
    visited: usize,
    stale: usize,
    work_seconds: f64,
}

struct AppliedReply {
    merges: usize,
    work_seconds: f64,
}

struct FinalReply {
    memory: IndexMemory,
    peak_memory: IndexMemory,
    plan_peak_bytes: usize,
    edge_peak_bytes: usize,
    worker_plan_seconds: f64,
    worker_apply_seconds: f64,
}

enum Command {
    BuildIndex(Arc<HashSet<u64>>),
    PrepareBatch {
        rules: Arc<Vec<RuleSpec>>,
        selected: Arc<HashMap<u64, u32>>,
    },
    GatherSelf(RuleSpec),
    PrepareSelf {
        rule: RuleSpec,
        positions: Arc<Vec<u32>>,
        start: usize,
        end: usize,
    },
    Apply(Arc<HashSet<u64>>),
    Finish,
}

enum Reply {
    Initial(InitialCounts),
    Built(IndexMemory),
    Prepared(PreparedReply),
    Gathered(GatheredReply),
    Applied(AppliedReply),
    Final(FinalReply),
    Error(TrainError),
}

struct Worker {
    commands: Sender<Command>,
    replies: Receiver<Reply>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Finish);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[inline(always)]
fn load<const UNCHECKED: bool>(corpus: &[AtomicU32], pos: usize) -> u32 {
    if UNCHECKED {
        // SAFETY: initial ranges are derived from validated corpus length;
        // historical positions and all derived neighbors are checked before
        // loading. No corpus writes occur before every plan reply is received.
        unsafe { corpus.get_unchecked(pos) }.load(Ordering::Relaxed)
    } else {
        corpus[pos].load(Ordering::Relaxed)
    }
}

#[inline(always)]
fn store<const UNCHECKED: bool>(corpus: &[AtomicU32], pos: usize, value: u32) {
    if UNCHECKED {
        // SAFETY: a Plan was built from an in-range live pair. Certified
        // nonself matches are token-disjoint; AA positions are selected left
        // to right without overlap. Atomic cells also prevent a data race.
        unsafe { corpus.get_unchecked(pos) }.store(value, Ordering::Relaxed);
    } else {
        corpus[pos].store(value, Ordering::Relaxed);
    }
}

fn index_memory(index: &HashMap<u64, Vec<u32>>) -> IndexMemory {
    IndexMemory {
        position_len: index.values().map(Vec::len).sum(),
        position_capacity: index.values().map(Vec::capacity).sum(),
        map_len: index.len(),
        map_capacity: index.capacity(),
    }
}

fn take_positions(
    index: &mut HashMap<u64, Vec<u32>>,
    memory: &mut IndexMemory,
    key: u64,
) -> Vec<u32> {
    let positions = index.remove(&key).unwrap_or_default();
    memory.position_len -= positions.len();
    memory.position_capacity -= positions.capacity();
    positions
}

fn append_position(
    index: &mut HashMap<u64, Vec<u32>>,
    memory: &mut IndexMemory,
    key: u64,
    pos: u32,
) {
    let positions = index.entry(key).or_default();
    let before = positions.capacity();
    positions.push(pos);
    memory.position_len += 1;
    memory.position_capacity += positions.capacity() - before;
}

fn sample_memory(index: &HashMap<u64, Vec<u32>>, memory: &mut IndexMemory, peak: &mut IndexMemory) {
    memory.map_len = index.len();
    memory.map_capacity = index.capacity();
    peak.position_len = peak.position_len.max(memory.position_len);
    peak.position_capacity = peak.position_capacity.max(memory.position_capacity);
    peak.map_len = peak.map_len.max(memory.map_len);
    peak.map_capacity = peak.map_capacity.max(memory.map_capacity);
}

fn scan_initial<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    pivots: &[u32],
    weights: &[u64],
    start: usize,
    end: usize,
    eligible: Option<&HashSet<u64>>,
    index: &mut HashMap<u64, Vec<u32>>,
) -> TrainResult<InitialCounts> {
    let mut counts = HashMap::new();
    let mut occurrences = 0;
    let mut wi = pivots.partition_point(|&pivot| pivot as usize <= start) - 1;
    for pos in start..end {
        while wi + 1 < pivots.len() && pos >= pivots[wi + 1] as usize {
            wi += 1;
        }
        let a = load::<UNCHECKED>(corpus, pos);
        let b = load::<UNCHECKED>(corpus, pos + 1);
        if a == 0 || b == 0 {
            continue;
        }
        let key = pair_key(a, b);
        if let Some(eligible) = eligible {
            if eligible.contains(&key) {
                index.entry(key).or_default().push(pos as u32);
            }
        } else {
            let value = counts.entry(key).or_insert(0_u64);
            *value = value.checked_add(weights[wi]).ok_or(TrainError::Overflow(
                "initial local pair frequency exceeds u64",
            ))?;
            occurrences += 1;
        }
    }
    Ok(InitialCounts {
        counts,
        occurrences,
    })
}

#[inline]
fn valid_pair<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    pos: usize,
    rule: RuleSpec,
) -> Option<(usize, usize)> {
    let last = corpus.len() - 1;
    if pos == 0 || pos >= last || load::<UNCHECKED>(corpus, pos) != rule.a {
        return None;
    }
    let right = pos.checked_add(rule.a_length)?;
    if right >= last || load::<UNCHECKED>(corpus, right) != rule.b {
        return None;
    }
    let after = right.checked_add(rule.b_length)?;
    (after <= last).then_some((right, after))
}

/// From the stable batch snapshot, emit only the final edges. The left match
/// owns a boundary shared by two selected spans; the right match omits it.
#[allow(clippy::too_many_arguments)] // Explicit immutable snapshot and worker-local output buffers.
fn add_plan<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    rule: RuleSpec,
    pos: usize,
    right: usize,
    after: usize,
    left_selected: bool,
    right_selected_id: Option<u32>,
    plans: &mut Vec<Plan>,
    new_edges: &mut Vec<NewEdge>,
    delta: &mut HashMap<u64, i128>,
) -> TrainResult<()> {
    let last = corpus.len() - 1;
    if after > last || pos == 0 {
        return Err(TrainError::InternalInvariant("batch plan outside corpus"));
    }
    let wi = pivots.partition_point(|&pivot| pivot as usize <= pos) - 1;
    let weight = i128::from(weights[wi]);
    add_delta(delta, rule.key, -weight);
    let left_id = load::<UNCHECKED>(corpus, pos - 1);
    if left_id != 0 && !left_selected {
        let before = pos
            .checked_sub(lengths[left_id as usize] as usize)
            .ok_or(TrainError::InternalInvariant("left boundary underflow"))?;
        add_delta(delta, pair_key(left_id, rule.a), -weight);
        let new_key = pair_key(left_id, rule.new_id);
        add_delta(delta, new_key, weight);
        new_edges.push(NewEdge::new(new_key, before as u32));
    }
    let right_id = load::<UNCHECKED>(corpus, after);
    if right_id != 0 {
        add_delta(delta, pair_key(rule.b, right_id), -weight);
        let new_key = pair_key(rule.new_id, right_selected_id.unwrap_or(right_id));
        add_delta(delta, new_key, weight);
        new_edges.push(NewEdge::new(new_key, pos as u32));
    }
    plans.push(Plan {
        pos: pos as u32,
        right: right as u32,
        after: after as u32,
        new_id: rule.new_id,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)] // The same phase-local buffers serve both planning paths.
fn prepare_batch<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    lengths: &mut Vec<u32>,
    pivots: &[u32],
    weights: &[u64],
    index: &mut HashMap<u64, Vec<u32>>,
    memory: &mut IndexMemory,
    rules: &[RuleSpec],
    selected: &HashMap<u64, u32>,
    plans: &mut Vec<Plan>,
    new_edges: &mut Vec<NewEdge>,
) -> TrainResult<PreparedReply> {
    let started = Instant::now();
    plans.clear();
    new_edges.clear();
    let mut delta = HashMap::new();
    let mut visited = 0;
    let mut stale = 0;
    for &rule in rules {
        if rule.new_id as usize != lengths.len() {
            return Err(TrainError::InternalInvariant(
                "worker token lengths out of order",
            ));
        }
        lengths.push(rule.new_length);
        for raw in take_positions(index, memory, rule.key) {
            visited += 1;
            let pos = raw as usize;
            let Some((right, after)) = valid_pair::<UNCHECKED>(corpus, pos, rule) else {
                stale += 1;
                continue;
            };
            let left_id = load::<UNCHECKED>(corpus, pos - 1);
            let left_selected = if left_id == 0 {
                false
            } else {
                let before = pos
                    .checked_sub(lengths[left_id as usize] as usize)
                    .ok_or(TrainError::InternalInvariant("left boundary underflow"))?;
                if before == 0 {
                    false
                } else {
                    let x = load::<UNCHECKED>(corpus, before - 1);
                    x != 0 && selected.contains_key(&pair_key(x, left_id))
                }
            };
            let r = load::<UNCHECKED>(corpus, after);
            let right_selected_id = if r == 0 {
                None
            } else {
                let y_pos = after
                    .checked_add(lengths[r as usize] as usize)
                    .ok_or(TrainError::InternalInvariant("right neighbor overflow"))?;
                if y_pos > corpus.len() - 1 {
                    return Err(TrainError::InternalInvariant(
                        "right neighbor outside corpus",
                    ));
                }
                let y = load::<UNCHECKED>(corpus, y_pos);
                selected.get(&pair_key(r, y)).copied()
            };
            add_plan::<UNCHECKED>(
                corpus,
                lengths,
                pivots,
                weights,
                rule,
                pos,
                right,
                after,
                left_selected,
                right_selected_id,
                plans,
                new_edges,
                &mut delta,
            )?;
        }
    }
    Ok(PreparedReply {
        delta,
        visited,
        stale,
        valid: plans.len(),
        plan_capacity: plans.capacity(),
        edge_capacity: new_edges.capacity(),
        work_seconds: started.elapsed().as_secs_f64(),
    })
}

#[allow(clippy::too_many_arguments)] // AA supplies globally selected starts to the shared planner.
fn prepare_self<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    lengths: &mut Vec<u32>,
    pivots: &[u32],
    weights: &[u64],
    rule: RuleSpec,
    selected_positions: &[u32],
    start: usize,
    end: usize,
    plans: &mut Vec<Plan>,
    new_edges: &mut Vec<NewEdge>,
) -> TrainResult<PreparedReply> {
    let started = Instant::now();
    plans.clear();
    new_edges.clear();
    if rule.new_id as usize != lengths.len() {
        return Err(TrainError::InternalInvariant(
            "worker token lengths out of order",
        ));
    }
    lengths.push(rule.new_length);
    let mut delta = HashMap::new();
    let span = rule.a_length * 2;
    for &raw in &selected_positions[start..end] {
        let pos = raw as usize;
        let Some((right, after)) = valid_pair::<UNCHECKED>(corpus, pos, rule) else {
            return Err(TrainError::InternalInvariant(
                "selected AA match disappeared",
            ));
        };
        let left_selected = pos
            .checked_sub(span)
            .is_some_and(|before| selected_positions.binary_search(&(before as u32)).is_ok());
        let right_selected_id = selected_positions
            .binary_search(&(after as u32))
            .ok()
            .map(|_| rule.new_id);
        add_plan::<UNCHECKED>(
            corpus,
            lengths,
            pivots,
            weights,
            rule,
            pos,
            right,
            after,
            left_selected,
            right_selected_id,
            plans,
            new_edges,
            &mut delta,
        )?;
    }
    Ok(PreparedReply {
        delta,
        visited: 0, // Historical visits were counted by GatherSelf.
        stale: 0,
        valid: plans.len(),
        plan_capacity: plans.capacity(),
        edge_capacity: new_edges.capacity(),
        work_seconds: started.elapsed().as_secs_f64(),
    })
}

fn apply_plans<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    index: &mut HashMap<u64, Vec<u32>>,
    memory: &mut IndexMemory,
    plans: &mut Vec<Plan>,
    new_edges: &mut Vec<NewEdge>,
    drop_keys: &HashSet<u64>,
) -> AppliedReply {
    let started = Instant::now();
    let merged = plans.len();
    for plan in plans.drain(..) {
        store::<UNCHECKED>(corpus, plan.pos as usize, plan.new_id);
        if plan.after - plan.right == 1 {
            store::<UNCHECKED>(corpus, plan.right as usize, plan.new_id);
        } else {
            store::<UNCHECKED>(corpus, plan.right as usize, 0);
            store::<UNCHECKED>(corpus, (plan.after - 1) as usize, plan.new_id);
        }
    }
    for key in drop_keys {
        let _ = take_positions(index, memory, *key);
    }
    for edge in new_edges.drain(..) {
        if !drop_keys.contains(&edge.key()) {
            append_position(index, memory, edge.key(), edge.pos);
        }
    }
    AppliedReply {
        merges: merged,
        work_seconds: started.elapsed().as_secs_f64(),
    }
}

fn spawn_worker<const UNCHECKED: bool>(
    corpus: Arc<Vec<AtomicU32>>,
    pivots: Arc<Vec<u32>>,
    weights: Arc<Vec<u64>>,
    initial_lengths: Arc<Vec<u32>>,
    start: usize,
    end: usize,
) -> Worker {
    let (commands, rx) = mpsc::channel();
    let (tx, replies) = mpsc::channel();
    let handle = thread::spawn(move || {
        let mut index: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut memory = IndexMemory::default();
        let mut peak_memory = IndexMemory::default();
        let mut lengths = (*initial_lengths).clone();
        let first =
            scan_initial::<UNCHECKED>(&corpus, &pivots, &weights, start, end, None, &mut index);
        match first {
            Ok(counts) => {
                if tx.send(Reply::Initial(counts)).is_err() {
                    return;
                }
            }
            Err(error) => {
                let _ = tx.send(Reply::Error(error));
                return;
            }
        }
        let mut plans = Vec::new();
        let mut new_edges = Vec::new();
        let mut plan_peak_bytes = 0;
        let mut edge_peak_bytes = 0;
        let mut worker_plan_seconds = 0.0;
        let mut worker_apply_seconds = 0.0;
        while let Ok(command) = rx.recv() {
            let reply = match command {
                Command::BuildIndex(eligible) => {
                    let built = scan_initial::<UNCHECKED>(
                        &corpus,
                        &pivots,
                        &weights,
                        start,
                        end,
                        Some(&eligible),
                        &mut index,
                    );
                    built.map(|_| {
                        memory = index_memory(&index);
                        peak_memory = memory;
                        Reply::Built(memory)
                    })
                }
                Command::PrepareBatch { rules, selected } => prepare_batch::<UNCHECKED>(
                    &corpus,
                    &mut lengths,
                    &pivots,
                    &weights,
                    &mut index,
                    &mut memory,
                    &rules,
                    &selected,
                    &mut plans,
                    &mut new_edges,
                )
                .map(|reply| {
                    plan_peak_bytes =
                        plan_peak_bytes.max(reply.plan_capacity * std::mem::size_of::<Plan>());
                    edge_peak_bytes =
                        edge_peak_bytes.max(reply.edge_capacity * std::mem::size_of::<NewEdge>());
                    worker_plan_seconds += reply.work_seconds;
                    Reply::Prepared(reply)
                }),
                Command::GatherSelf(rule) => {
                    let started = Instant::now();
                    let mut valid_positions = Vec::new();
                    let mut visited = 0;
                    let mut stale = 0;
                    for raw in take_positions(&mut index, &mut memory, rule.key) {
                        visited += 1;
                        if valid_pair::<UNCHECKED>(&corpus, raw as usize, rule).is_some() {
                            valid_positions.push(raw);
                        } else {
                            stale += 1;
                        }
                    }
                    Ok(Reply::Gathered(GatheredReply {
                        valid_capacity: valid_positions.capacity(),
                        valid_positions,
                        visited,
                        stale,
                        work_seconds: started.elapsed().as_secs_f64(),
                    }))
                }
                Command::PrepareSelf {
                    rule,
                    positions,
                    start,
                    end,
                } => prepare_self::<UNCHECKED>(
                    &corpus,
                    &mut lengths,
                    &pivots,
                    &weights,
                    rule,
                    &positions,
                    start,
                    end,
                    &mut plans,
                    &mut new_edges,
                )
                .map(|reply| {
                    plan_peak_bytes =
                        plan_peak_bytes.max(reply.plan_capacity * std::mem::size_of::<Plan>());
                    edge_peak_bytes =
                        edge_peak_bytes.max(reply.edge_capacity * std::mem::size_of::<NewEdge>());
                    worker_plan_seconds += reply.work_seconds;
                    Reply::Prepared(reply)
                }),
                Command::Apply(drop_keys) => {
                    let reply = apply_plans::<UNCHECKED>(
                        &corpus,
                        &mut index,
                        &mut memory,
                        &mut plans,
                        &mut new_edges,
                        &drop_keys,
                    );
                    sample_memory(&index, &mut memory, &mut peak_memory);
                    worker_apply_seconds += reply.work_seconds;
                    Ok(Reply::Applied(reply))
                }
                Command::Finish => {
                    let _ = tx.send(Reply::Final(FinalReply {
                        memory,
                        peak_memory,
                        plan_peak_bytes,
                        edge_peak_bytes,
                        worker_plan_seconds,
                        worker_apply_seconds,
                    }));
                    return;
                }
            };
            match reply {
                Ok(reply) => {
                    if tx.send(reply).is_err() {
                        return;
                    }
                }
                Err(error) => {
                    let _ = tx.send(Reply::Error(error));
                    return;
                }
            }
        }
    });
    Worker {
        commands,
        replies,
        handle: Some(handle),
    }
}

fn recv(worker: &Worker) -> TrainResult<Reply> {
    match worker
        .replies
        .recv()
        .map_err(|_| TrainError::InternalInvariant("certified worker disconnected"))?
    {
        Reply::Error(error) => Err(error),
        other => Ok(other),
    }
}

#[allow(clippy::too_many_arguments)] // Selection state and diagnostic counters stay separate.
fn choose_prefix<const RELAXED: bool>(
    heap: &mut BinaryHeap<HeapEntry>,
    frequencies: &HashMap<u64, u64>,
    minimum: u64,
    remaining: usize,
    cap: usize,
    heap_pops: &mut usize,
    stop_self: &mut usize,
    stop_conflict: &mut usize,
) -> Vec<(u64, u64)> {
    let mut pending = Vec::new();
    let mut heads = HashSet::new();
    let mut tails = HashSet::new();
    let mut reserved = HashSet::new();
    let mut deferred = Vec::new();
    let limit = remaining.min(cap.max(1));
    let mut scanned = 0;
    while pending.len() < limit {
        if RELAXED && scanned >= limit.saturating_mul(8) {
            break;
        }
        let Some((key, frequency)) = pop_best(heap, frequencies, minimum, heap_pops) else {
            break;
        };
        if !reserved.insert(key) {
            continue; // duplicate lazy entry for a pending key
        }
        scanned += 1;
        let a = (key >> 32) as u32;
        let b = key as u32;
        if a == b {
            *stop_self += 1;
            if pending.is_empty() {
                pending.push((key, frequency));
            } else {
                deferred.push(HeapEntry { key, frequency });
                if RELAXED {
                    continue;
                }
            }
            break;
        }
        if tails.contains(&a) || heads.contains(&b) {
            *stop_conflict += 1;
            deferred.push(HeapEntry { key, frequency });
            if RELAXED {
                continue;
            }
            break;
        }
        heads.insert(a);
        tails.insert(b);
        pending.push((key, frequency));
    }
    heap.extend(deferred);
    pending
}

/// Called only after `parallel::train` has validated the public Prepared input.
pub(super) fn train<const UNCHECKED: bool>(
    input: Prepared,
    options: Options,
    cap: usize,
) -> TrainResult<Result> {
    run::<UNCHECKED, false>(input, options, cap)
}

/// Experimental heuristic: skip conflicting candidates within a bounded
/// window. It preserves replacement semantics but may change greedy choices.
pub(super) fn train_relaxed<const UNCHECKED: bool>(
    input: Prepared,
    options: Options,
    cap: usize,
) -> TrainResult<Result> {
    run::<UNCHECKED, true>(input, options, cap)
}

fn run<const UNCHECKED: bool, const RELAXED: bool>(
    input: Prepared,
    options: Options,
    cap: usize,
) -> TrainResult<Result> {
    let started = Instant::now();
    let Prepared {
        corpus,
        initial_lengths,
        pivots,
        weights,
    } = input;
    let corpus_positions = corpus.len();
    let last = corpus_positions - 1;
    let worker_count = options.workers.min((last - 1).max(1));
    let mut lengths = initial_lengths;
    let worker_initial_lengths = Arc::new(lengths.clone());
    let pivots = Arc::new(pivots);
    let weights = Arc::new(weights);
    let corpus: Arc<Vec<AtomicU32>> = Arc::new(corpus.into_iter().map(AtomicU32::new).collect());
    let workers: Vec<Worker> = (0..worker_count)
        .map(|i| {
            let start = 1 + i * (last - 1) / worker_count;
            let end = 1 + (i + 1) * (last - 1) / worker_count;
            spawn_worker::<UNCHECKED>(
                Arc::clone(&corpus),
                Arc::clone(&pivots),
                Arc::clone(&weights),
                Arc::clone(&worker_initial_lengths),
                start,
                end,
            )
        })
        .collect();
    let mut frequencies = HashMap::<u64, u64>::new();
    let mut initial_occurrences = 0;
    for worker in &workers {
        let Reply::Initial(initial) = recv(worker)? else {
            return Err(TrainError::InternalInvariant(
                "expected initial count reply",
            ));
        };
        initial_occurrences += initial.occurrences;
        for (key, count) in initial.counts {
            let value = frequencies.entry(key).or_insert(0);
            *value = value.checked_add(count).ok_or(TrainError::Overflow(
                "global initial pair frequency exceeds u64",
            ))?;
        }
    }
    let initial_distinct_keys = frequencies.len();
    frequencies.retain(|_, count| *count >= options.min_frequency);
    let eligible = Arc::new(frequencies.keys().copied().collect::<HashSet<_>>());
    for worker in &workers {
        worker
            .commands
            .send(Command::BuildIndex(Arc::clone(&eligible)))
            .map_err(|_| TrainError::InternalInvariant("certified build-index dispatch failed"))?;
    }
    let mut initial_index_memory = IndexMemory::default();
    for worker in &workers {
        let Reply::Built(memory) = recv(worker)? else {
            return Err(TrainError::InternalInvariant("expected built-index reply"));
        };
        initial_index_memory.position_len += memory.position_len;
        initial_index_memory.position_capacity += memory.position_capacity;
        initial_index_memory.map_len += memory.map_len;
        initial_index_memory.map_capacity += memory.map_capacity;
    }
    drop(eligible);
    let mut heap = initial_heap(&frequencies, options.min_frequency);
    let init_seconds = started.elapsed().as_secs_f64();

    let merge_started = Instant::now();
    let mut merges = Vec::new();
    let mut actual_merges = 0;
    let mut position_visits = 0;
    let mut stale_visits = 0;
    let mut heap_pops = 0;
    let mut epochs = 0;
    let mut width_sum = 0;
    let mut max_width = 0;
    let mut singleton_epochs = 0;
    let mut hit_cap = 0;
    let mut stop_self = 0;
    let mut stop_conflict = 0;
    let mut aa_epochs = 0;
    let mut aa_gather_positions = 0;
    let mut aa_gather_worker_capacity_peak_bytes = 0;
    let mut aa_coordinator_capacity_peak_bytes = 0;
    let mut select_seconds = 0.0;
    let mut plan_seconds = 0.0;
    let mut apply_seconds = 0.0;
    let mut reduce_seconds = 0.0;
    let mut delta_collect_seconds = 0.0;
    let mut worker_plan_seconds = 0.0;
    let mut worker_apply_seconds = 0.0;
    let mut delta_keys_total = 0;
    let mut worker_delta_keys_total = 0;
    let mut new_edge_capacity_peak_bytes = 0;
    let mut plan_capacity_peak_bytes = 0;
    let mut plan_items_peak = 0;
    let mut messages = 2 * worker_count; // build-index send/reply
    while merges.len() < options.max_merges {
        let select_started = Instant::now();
        let prefix = choose_prefix::<RELAXED>(
            &mut heap,
            &frequencies,
            options.min_frequency,
            options.max_merges - merges.len(),
            cap,
            &mut heap_pops,
            &mut stop_self,
            &mut stop_conflict,
        );
        select_seconds += select_started.elapsed().as_secs_f64();
        if prefix.is_empty() {
            break;
        }
        epochs += 1;
        width_sum += prefix.len();
        max_width = max_width.max(prefix.len());
        singleton_epochs += usize::from(prefix.len() == 1);
        hit_cap += usize::from(prefix.len() == cap.min(options.max_merges - merges.len()));
        let first_new_id = u32::try_from(lengths.len())
            .map_err(|_| TrainError::Overflow("fresh token ID exceeds u32"))?;
        let mut specs = Vec::with_capacity(prefix.len());
        for (key, frequency) in prefix {
            let a = (key >> 32) as u32;
            let b = key as u32;
            let new_id = u32::try_from(lengths.len())
                .map_err(|_| TrainError::Overflow("fresh token ID exceeds u32"))?;
            let new_length = lengths[a as usize]
                .checked_add(lengths[b as usize])
                .ok_or(TrainError::Overflow("merged token length exceeds u32"))?;
            let spec = RuleSpec {
                key,
                a,
                b,
                new_id,
                new_length,
                a_length: lengths[a as usize] as usize,
                b_length: lengths[b as usize] as usize,
                frequency,
            };
            lengths.push(new_length);
            specs.push(spec);
        }
        let rules = Arc::new(specs);
        let plan_started = Instant::now();
        let mut gathered_visits = 0;
        let mut gathered_stale = 0;
        if rules[0].a == rules[0].b {
            aa_epochs += 1;
            let rule = rules[0];
            for worker in &workers {
                worker
                    .commands
                    .send(Command::GatherSelf(rule))
                    .map_err(|_| {
                        TrainError::InternalInvariant("certified AA gather dispatch failed")
                    })?;
            }
            messages += 2 * worker_count;
            let mut valid = Vec::new();
            let mut gathered_capacity = 0;
            for worker in &workers {
                let Reply::Gathered(reply) = recv(worker)? else {
                    return Err(TrainError::InternalInvariant("expected AA gather reply"));
                };
                gathered_visits += reply.visited;
                gathered_stale += reply.stale;
                worker_plan_seconds += reply.work_seconds;
                gathered_capacity += reply.valid_capacity * std::mem::size_of::<u32>();
                valid.extend(reply.valid_positions);
            }
            aa_gather_worker_capacity_peak_bytes =
                aa_gather_worker_capacity_peak_bytes.max(gathered_capacity);
            aa_coordinator_capacity_peak_bytes = aa_coordinator_capacity_peak_bytes
                .max(valid.capacity() * std::mem::size_of::<u32>());
            aa_gather_positions += valid.len();
            valid.sort_unstable();
            if valid.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(TrainError::InternalInvariant(
                    "duplicate live AA occurrence",
                ));
            }
            let mut selected = Vec::new();
            let valid_capacity = valid.capacity();
            let mut previous_after = 0;
            for raw in valid {
                let pos = raw as usize;
                if pos < previous_after {
                    gathered_stale += 1;
                    continue;
                }
                previous_after = pos + rule.a_length + rule.b_length;
                selected.push(raw);
            }
            let selected = Arc::new(selected);
            aa_coordinator_capacity_peak_bytes = aa_coordinator_capacity_peak_bytes
                .max((valid_capacity + selected.capacity()) * std::mem::size_of::<u32>());
            for (i, worker) in workers.iter().enumerate() {
                let start = i * selected.len() / worker_count;
                let end = (i + 1) * selected.len() / worker_count;
                worker
                    .commands
                    .send(Command::PrepareSelf {
                        rule,
                        positions: Arc::clone(&selected),
                        start,
                        end,
                    })
                    .map_err(|_| {
                        TrainError::InternalInvariant("certified AA plan dispatch failed")
                    })?;
            }
            messages += 2 * worker_count;
        } else {
            let selected = Arc::new(
                rules
                    .iter()
                    .map(|rule| (rule.key, rule.new_id))
                    .collect::<HashMap<_, _>>(),
            );
            for worker in &workers {
                worker
                    .commands
                    .send(Command::PrepareBatch {
                        rules: Arc::clone(&rules),
                        selected: Arc::clone(&selected),
                    })
                    .map_err(|_| TrainError::InternalInvariant("certified plan dispatch failed"))?;
            }
            messages += 2 * worker_count;
        }
        let mut delta = HashMap::<u64, i128>::new();
        let mut epoch_plans = 0;
        let mut epoch_plan_capacity = 0;
        let mut epoch_edge_capacity = 0;
        position_visits += gathered_visits;
        stale_visits += gathered_stale;
        for worker in &workers {
            let Reply::Prepared(reply) = recv(worker)? else {
                return Err(TrainError::InternalInvariant("expected batch plan reply"));
            };
            position_visits += reply.visited;
            stale_visits += reply.stale;
            epoch_plans += reply.valid;
            epoch_plan_capacity += reply.plan_capacity * std::mem::size_of::<Plan>();
            epoch_edge_capacity += reply.edge_capacity * std::mem::size_of::<NewEdge>();
            worker_delta_keys_total += reply.delta.len();
            worker_plan_seconds += reply.work_seconds;
            let collect_started = Instant::now();
            for (key, change) in reply.delta {
                add_delta(&mut delta, key, change);
            }
            delta_collect_seconds += collect_started.elapsed().as_secs_f64();
        }
        plan_items_peak = plan_items_peak.max(epoch_plans);
        plan_capacity_peak_bytes = plan_capacity_peak_bytes.max(epoch_plan_capacity);
        new_edge_capacity_peak_bytes = new_edge_capacity_peak_bytes.max(epoch_edge_capacity);
        plan_seconds += plan_started.elapsed().as_secs_f64();
        let reduce_started = Instant::now();
        delta_keys_total += delta.len();
        let mut drop_keys = HashSet::new();
        let mut fresh_keys = Vec::new();
        for (key, change) in delta {
            let fresh = (key >> 32) as u32 >= first_new_id || key as u32 >= first_new_id;
            // Old pair frequencies can only fall. Once below min_frequency,
            // their positions and scalar count were discarded permanently;
            // workers still report their deleted physical edges in a batch.
            if !fresh && !frequencies.contains_key(&key) {
                if change > 0 {
                    return Err(TrainError::InternalInvariant(
                        "discarded old pair increased",
                    ));
                }
                continue;
            }
            apply_delta(&mut frequencies, key, change)?;
            if frequencies[&key] < options.min_frequency {
                drop_keys.insert(key);
            } else if fresh {
                fresh_keys.push(key);
            }
        }
        for rule in rules.iter() {
            if frequencies.get(&rule.key).copied().unwrap_or(0) != 0 {
                return Err(TrainError::InternalInvariant(
                    "selected pair remained after batch",
                ));
            }
            frequencies.remove(&rule.key);
            drop_keys.insert(rule.key);
            merges.push(Rule {
                left: rule.a,
                right: rule.b,
                frequency: rule.frequency,
            });
        }
        for key in fresh_keys {
            heap.push(HeapEntry {
                frequency: frequencies[&key],
                key,
            });
        }
        for &key in &drop_keys {
            if frequencies
                .get(&key)
                .is_some_and(|&f| f < options.min_frequency)
            {
                frequencies.remove(&key);
            }
        }
        reduce_seconds += reduce_started.elapsed().as_secs_f64();
        let apply_started = Instant::now();
        let drop_keys = Arc::new(drop_keys);
        for worker in &workers {
            worker
                .commands
                .send(Command::Apply(Arc::clone(&drop_keys)))
                .map_err(|_| TrainError::InternalInvariant("certified apply dispatch failed"))?;
        }
        messages += 2 * worker_count;
        let mut applied = 0;
        for worker in &workers {
            let Reply::Applied(reply) = recv(worker)? else {
                return Err(TrainError::InternalInvariant("expected batch apply reply"));
            };
            applied += reply.merges;
            worker_apply_seconds += reply.work_seconds;
        }
        if applied != epoch_plans {
            return Err(TrainError::InternalInvariant(
                "batch plan/apply count mismatch",
            ));
        }
        actual_merges += applied;
        apply_seconds += apply_started.elapsed().as_secs_f64();
    }
    let merge_seconds = merge_started.elapsed().as_secs_f64();
    let mut final_memory = IndexMemory::default();
    let mut peak_memory_sum = IndexMemory::default();
    let mut worker_plan_peak_bytes = 0;
    let mut worker_edge_peak_bytes = 0;
    for worker in &workers {
        worker
            .commands
            .send(Command::Finish)
            .map_err(|_| TrainError::InternalInvariant("certified finish dispatch failed"))?;
    }
    for worker in &workers {
        let Reply::Final(reply) = recv(worker)? else {
            return Err(TrainError::InternalInvariant("expected worker final reply"));
        };
        final_memory.position_len += reply.memory.position_len;
        final_memory.position_capacity += reply.memory.position_capacity;
        final_memory.map_len += reply.memory.map_len;
        final_memory.map_capacity += reply.memory.map_capacity;
        peak_memory_sum.position_len += reply.peak_memory.position_len;
        peak_memory_sum.position_capacity += reply.peak_memory.position_capacity;
        peak_memory_sum.map_len += reply.peak_memory.map_len;
        peak_memory_sum.map_capacity += reply.peak_memory.map_capacity;
        worker_plan_peak_bytes += reply.plan_peak_bytes;
        worker_edge_peak_bytes += reply.edge_peak_bytes;
        let _ = (reply.worker_plan_seconds, reply.worker_apply_seconds);
    }
    messages += 2 * worker_count;
    drop(workers);
    let mut final_tokens = Vec::new();
    let mut pos = 0;
    loop {
        let token = load::<UNCHECKED>(&corpus, pos);
        final_tokens.push(token);
        if pos == last {
            break;
        }
        pos = pos.checked_add(lengths[token as usize] as usize).ok_or(
            TrainError::InternalInvariant("final token boundary overflow"),
        )?;
        if pos > last {
            return Err(TrainError::InternalInvariant("final token outside corpus"));
        }
    }
    let max_token_length = lengths.iter().copied().max().unwrap_or(1);
    let core = core_result(
        merges,
        final_tokens,
        CoreStats {
            init_seconds,
            merge_seconds,
            actual_merges,
            position_visits,
            stale_visits,
            heap_pops,
            backend_buffer_bytes: corpus_positions * std::mem::size_of::<AtomicU32>(),
            initial_occurrence_bytes: initial_index_memory.position_len * 4,
            max_token_length,
            corpus_positions,
        },
    );
    let mut metrics = BTreeMap::new();
    metric(
        &mut metrics,
        "relaxed_batch_selection",
        if RELAXED { 1.0 } else { 0.0 },
    );
    metric(
        &mut metrics,
        "plan_record_bytes",
        std::mem::size_of::<Plan>() as f64,
    );
    metric(
        &mut metrics,
        "new_edge_record_bytes",
        std::mem::size_of::<NewEdge>() as f64,
    );
    metric(
        &mut metrics,
        "aa_gather_worker_capacity_peak_bytes",
        aa_gather_worker_capacity_peak_bytes as f64,
    );
    metric(
        &mut metrics,
        "aa_coordinator_capacity_peak_bytes",
        aa_coordinator_capacity_peak_bytes as f64,
    );
    metric(&mut metrics, "workers_requested", options.workers as f64);
    metric(&mut metrics, "workers_actual", worker_count as f64);
    metric(
        &mut metrics,
        "shared_corpus_bytes",
        (corpus_positions * 4) as f64,
    );
    metric(
        &mut metrics,
        "initial_distinct_keys",
        initial_distinct_keys as f64,
    );
    metric(
        &mut metrics,
        "initial_unfiltered_occurrences",
        initial_occurrences as f64,
    );
    metric(
        &mut metrics,
        "initial_index_position_len",
        initial_index_memory.position_len as f64,
    );
    metric(
        &mut metrics,
        "initial_index_position_capacity_bytes",
        (initial_index_memory.position_capacity * 4) as f64,
    );
    metric(
        &mut metrics,
        "initial_index_map_entries_sum",
        initial_index_memory.map_len as f64,
    );
    metric(
        &mut metrics,
        "initial_index_map_capacity_sum",
        initial_index_memory.map_capacity as f64,
    );
    metric(
        &mut metrics,
        "final_index_position_len",
        final_memory.position_len as f64,
    );
    metric(
        &mut metrics,
        "final_index_position_capacity_bytes",
        (final_memory.position_capacity * 4) as f64,
    );
    metric(
        &mut metrics,
        "final_index_map_entries_sum",
        final_memory.map_len as f64,
    );
    metric(
        &mut metrics,
        "final_index_map_capacity_sum",
        final_memory.map_capacity as f64,
    );
    metric(
        &mut metrics,
        "index_position_len_peak_sum",
        peak_memory_sum.position_len as f64,
    );
    metric(
        &mut metrics,
        "index_position_capacity_peak_bytes_sum",
        (peak_memory_sum.position_capacity * 4) as f64,
    );
    metric(
        &mut metrics,
        "index_map_entries_peak_sum",
        peak_memory_sum.map_len as f64,
    );
    metric(
        &mut metrics,
        "index_map_capacity_peak_sum",
        peak_memory_sum.map_capacity as f64,
    );
    metric(
        &mut metrics,
        "plan_capacity_peak_bytes",
        plan_capacity_peak_bytes as f64,
    );
    metric(
        &mut metrics,
        "new_edge_capacity_peak_bytes",
        new_edge_capacity_peak_bytes as f64,
    );
    metric(
        &mut metrics,
        "worker_plan_peak_bytes_sum",
        worker_plan_peak_bytes as f64,
    );
    metric(
        &mut metrics,
        "worker_edge_peak_bytes_sum",
        worker_edge_peak_bytes as f64,
    );
    metric(&mut metrics, "plan_items_peak", plan_items_peak as f64);
    metric(&mut metrics, "certificate_epochs", epochs as f64);
    metric(&mut metrics, "certificate_rules", width_sum as f64);
    metric(&mut metrics, "certificate_max_width", max_width as f64);
    metric(
        &mut metrics,
        "certificate_singleton_epochs",
        singleton_epochs as f64,
    );
    metric(&mut metrics, "certificate_hit_cap", hit_cap as f64);
    metric(&mut metrics, "certificate_stop_self", stop_self as f64);
    metric(
        &mut metrics,
        "certificate_stop_conflict",
        stop_conflict as f64,
    );
    metric(&mut metrics, "aa_epochs", aa_epochs as f64);
    metric(
        &mut metrics,
        "aa_gather_positions",
        aa_gather_positions as f64,
    );
    metric(&mut metrics, "select_seconds", select_seconds);
    metric(&mut metrics, "plan_seconds", plan_seconds);
    metric(&mut metrics, "apply_seconds", apply_seconds);
    metric(&mut metrics, "reduce_seconds", reduce_seconds);
    metric(&mut metrics, "delta_collect_seconds", delta_collect_seconds);
    metric(&mut metrics, "worker_plan_seconds_sum", worker_plan_seconds);
    metric(
        &mut metrics,
        "worker_apply_seconds_sum",
        worker_apply_seconds,
    );
    metric(&mut metrics, "delta_keys_total", delta_keys_total as f64);
    metric(
        &mut metrics,
        "worker_delta_keys_total",
        worker_delta_keys_total as f64,
    );
    metric(&mut metrics, "round_messages", messages as f64);
    metric(
        &mut metrics,
        "messages_per_epoch",
        if epochs == 0 {
            0.0
        } else {
            messages as f64 / epochs as f64
        },
    );
    metric(
        &mut metrics,
        "unchecked_corpus_access",
        if UNCHECKED { 1.0 } else { 0.0 },
    );
    Ok(Result { core, metrics })
}

//! Spatially certified batch training with worker-owned occurrence fragments and pair scalars.
//!
//! The same persistent workers alternate between position and pair-owner
//! phases. The coordinator merges only small, sorted candidate frontiers.

use super::{
    CoreStats, HeapEntry, add_delta, apply_delta, core_result, metric, pair_key, pop_best,
};
use crate::ablation::{Options, Result};
use crate::{Prepared, Rule, TrainError};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
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
struct PlanGroup {
    end: u32,
    new_id: u32,
    a_length: u32,
    b_length: u32,
}

const _: [(); 16] = [(); std::mem::size_of::<PlanGroup>()];

#[derive(Default)]
struct PlanBuffer<const COMPACT: bool> {
    wide: Vec<Plan>,
    starts: Vec<u32>,
    groups: Vec<PlanGroup>,
}

impl<const COMPACT: bool> PlanBuffer<COMPACT> {
    fn clear(&mut self) {
        if COMPACT {
            self.starts.clear();
            self.groups.clear();
        } else {
            self.wide.clear();
        }
    }
    fn len(&self) -> usize {
        if COMPACT {
            self.starts.len()
        } else {
            self.wide.len()
        }
    }
    fn push(&mut self, rule: RuleSpec, pos: usize, right: usize, after: usize) {
        if COMPACT {
            self.starts.push(pos as u32);
        } else {
            self.wide.push(Plan {
                pos: pos as u32,
                right: right as u32,
                after: after as u32,
                new_id: rule.new_id,
            });
        }
    }
    fn finish_rule(&mut self, rule: RuleSpec, start: usize) {
        if COMPACT && self.starts.len() > start {
            self.groups.push(PlanGroup {
                end: self.starts.len() as u32,
                new_id: rule.new_id,
                a_length: rule.a_length as u32,
                b_length: rule.b_length as u32,
            });
        }
    }
    fn capacity_bytes(&self) -> usize {
        if COMPACT {
            self.starts.capacity() * std::mem::size_of::<u32>()
        } else {
            self.wide.capacity() * std::mem::size_of::<Plan>()
        }
    }
    fn group_capacity_bytes(&self) -> usize {
        self.groups.capacity() * std::mem::size_of::<PlanGroup>()
    }
}

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
    delta_keys: usize,
    visited: usize,
    stale: usize,
    valid: usize,
    plan_capacity_bytes: usize,
    group_capacity_bytes: usize,
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

struct ProbeReply {
    cutoff: usize,
    visited: usize,
    stale: usize,
    work_seconds: f64,
}

#[derive(Default)]
struct SpatialStats {
    base_width_sum: usize,
    final_width_sum: usize,
    widened_epochs: usize,
    conflict_epochs: usize,
    budget_epochs: usize,
    metadata_seconds: f64,
    probe_seconds: f64,
    probe_worker_seconds_sum: f64,
    base_history_sum: usize,
    probe_visits: usize,
    probe_stale: usize,
    metadata_messages: usize,
    probe_messages: usize,
}

struct AppliedReply {
    merges: usize,
    work_seconds: f64,
}

struct FinalReply {
    memory: IndexMemory,
    peak_memory: IndexMemory,
    plan_peak_bytes: usize,
    group_peak_bytes: usize,
    edge_peak_bytes: usize,
    scalar_keys: usize,
    scalar_capacity: usize,
    heap_capacity: usize,
    worker_plan_seconds: f64,
    worker_apply_seconds: f64,
}

enum Command {
    BuildScalars,
    BuildIndex,
    Prefetch(usize),
    MeasureWindow(Arc<Vec<u64>>),
    ProbeWindow(Arc<Vec<u64>>),
    ReturnCandidates(Arc<HashSet<u64>>),
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
    Reduce {
        first_new_id: u32,
        rules: Arc<Vec<RuleSpec>>,
        drops: Arc<Vec<OnceLock<Arc<HashSet<u64>>>>>,
    },
    Apply(Arc<Vec<OnceLock<Arc<HashSet<u64>>>>>),
    Finish,
}

enum Reply {
    Initial {
        occurrences: usize,
        count_keys: usize,
        count_capacity: usize,
    },
    Scalars {
        keys: usize,
        capacity: usize,
        heap_capacity: usize,
        eligible_capacity: usize,
    },
    Built(IndexMemory),
    Candidates {
        entries: Vec<HeapEntry>,
        pops: usize,
    },
    WindowLengths(Vec<usize>),
    Probed(ProbeReply),
    Returned,
    Reduced {
        delta_keys: usize,
        scalar_keys: usize,
        scalar_capacity: usize,
        heap_capacity: usize,
        dropped: usize,
        seconds: f64,
    },
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
    eligible: Option<&[OnceLock<Arc<HashSet<u64>>>]>,
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
            if eligible[owner_pair(key, eligible.len())]
                .get()
                .is_some_and(|set| set.contains(&key))
            {
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

/// Every overlapping pair has a left occurrence whose right neighbor is the
/// other pair. Read only the stable corpus; selected lists remain in the index.
fn probe_window<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    index: &HashMap<u64, Vec<u32>>,
    lengths: &[u32],
    keys: &[u64],
) -> ProbeReply {
    let started = Instant::now();
    let ranks: HashMap<u64, usize> = keys.iter().enumerate().map(|(i, &key)| (key, i)).collect();
    let last = corpus.len() - 1;
    let mut cutoff = keys.len();
    let mut visited = 0;
    let mut stale = 0;
    for (rank, &key) in keys.iter().enumerate() {
        let a = (key >> 32) as u32;
        let b = key as u32;
        let a_length = lengths[a as usize] as usize;
        let b_length = lengths[b as usize] as usize;
        if let Some(positions) = index.get(&key) {
            for &raw in positions {
                visited += 1;
                let pos = raw as usize;
                if pos == 0 || pos >= last || load::<UNCHECKED>(corpus, pos) != a {
                    stale += 1;
                    continue;
                }
                let Some(right) = pos.checked_add(a_length) else {
                    stale += 1;
                    continue;
                };
                if right >= last || load::<UNCHECKED>(corpus, right) != b {
                    stale += 1;
                    continue;
                }
                let Some(after) = right.checked_add(b_length) else {
                    stale += 1;
                    continue;
                };
                if after > last {
                    stale += 1;
                    continue;
                }
                let c = load::<UNCHECKED>(corpus, after);
                if c != 0
                    && let Some(&next_rank) = ranks.get(&pair_key(b, c))
                {
                    cutoff = cutoff.min(rank.max(next_rank));
                }
            }
        }
    }
    ProbeReply {
        cutoff,
        visited,
        stale,
        work_seconds: started.elapsed().as_secs_f64(),
    }
}

/// From the stable batch snapshot, emit only the final edges. The left match
/// owns a boundary shared by two selected spans; the right match omits it.
#[allow(clippy::too_many_arguments)] // Explicit immutable snapshot and worker-local output buffers.
fn add_plan<const UNCHECKED: bool, const COMPACT: bool>(
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
    plans: &mut PlanBuffer<COMPACT>,
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
    plans.push(rule, pos, right, after);
    Ok(())
}

#[allow(clippy::too_many_arguments)] // The same phase-local buffers serve both planning paths.
fn prepare_batch<const UNCHECKED: bool, const COMPACT: bool>(
    corpus: &[AtomicU32],
    lengths: &mut Vec<u32>,
    pivots: &[u32],
    weights: &[u64],
    index: &mut HashMap<u64, Vec<u32>>,
    memory: &mut IndexMemory,
    rules: &[RuleSpec],
    selected: &HashMap<u64, u32>,
    plans: &mut PlanBuffer<COMPACT>,
    new_edges: &mut Vec<NewEdge>,
) -> TrainResult<PreparedReply> {
    let started = Instant::now();
    plans.clear();
    new_edges.clear();
    let mut delta = HashMap::new();
    let mut visited = 0;
    let mut stale = 0;
    for &rule in rules {
        let rule_start = plans.len();
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
            add_plan::<UNCHECKED, COMPACT>(
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
        plans.finish_rule(rule, rule_start);
    }
    let delta_keys = delta.len();
    Ok(PreparedReply {
        delta,
        delta_keys,
        visited,
        stale,
        valid: plans.len(),
        plan_capacity_bytes: plans.capacity_bytes(),
        group_capacity_bytes: plans.group_capacity_bytes(),
        edge_capacity: new_edges.capacity(),
        work_seconds: started.elapsed().as_secs_f64(),
    })
}

#[allow(clippy::too_many_arguments)] // AA supplies globally selected starts to the shared planner.
fn prepare_self<const UNCHECKED: bool, const COMPACT: bool>(
    corpus: &[AtomicU32],
    lengths: &mut Vec<u32>,
    pivots: &[u32],
    weights: &[u64],
    rule: RuleSpec,
    selected_positions: &[u32],
    start: usize,
    end: usize,
    plans: &mut PlanBuffer<COMPACT>,
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
        add_plan::<UNCHECKED, COMPACT>(
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
    plans.finish_rule(rule, 0);
    let delta_keys = delta.len();
    Ok(PreparedReply {
        delta,
        delta_keys,
        visited: 0, // Historical visits were counted by GatherSelf.
        stale: 0,
        valid: plans.len(),
        plan_capacity_bytes: plans.capacity_bytes(),
        group_capacity_bytes: plans.group_capacity_bytes(),
        edge_capacity: new_edges.capacity(),
        work_seconds: started.elapsed().as_secs_f64(),
    })
}

#[inline]
fn write_plan<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    pos: usize,
    right: usize,
    after: usize,
    new_id: u32,
) {
    store::<UNCHECKED>(corpus, pos, new_id);
    if after - right == 1 {
        store::<UNCHECKED>(corpus, right, new_id);
    } else {
        store::<UNCHECKED>(corpus, right, 0);
        store::<UNCHECKED>(corpus, after - 1, new_id);
    }
}

fn apply_plans<const UNCHECKED: bool, const COMPACT: bool>(
    corpus: &[AtomicU32],
    index: &mut HashMap<u64, Vec<u32>>,
    memory: &mut IndexMemory,
    plans: &mut PlanBuffer<COMPACT>,
    new_edges: &mut Vec<NewEdge>,
    drop_sets: &[OnceLock<Arc<HashSet<u64>>>],
) -> AppliedReply {
    let started = Instant::now();
    let merged = plans.len();
    if COMPACT {
        let mut start = 0;
        for group in plans.groups.drain(..) {
            let end = group.end as usize;
            for &raw in &plans.starts[start..end] {
                let pos = raw as usize;
                // Every compact start came from valid_pair in the stable plan
                // snapshot. Certified matches are disjoint until this barrier.
                let right = pos + group.a_length as usize;
                let after = right + group.b_length as usize;
                write_plan::<UNCHECKED>(corpus, pos, right, after, group.new_id);
            }
            start = end;
        }
        debug_assert_eq!(start, plans.starts.len());
        plans.starts.clear();
    } else {
        for plan in plans.wide.drain(..) {
            write_plan::<UNCHECKED>(
                corpus,
                plan.pos as usize,
                plan.right as usize,
                plan.after as usize,
                plan.new_id,
            );
        }
    }
    for set in drop_sets {
        for key in set.get().expect("owner reduction barrier").iter() {
            let _ = take_positions(index, memory, *key);
        }
    }
    for edge in new_edges.drain(..) {
        let key = edge.key();
        if !drop_sets[owner_pair(key, drop_sets.len())]
            .get()
            .expect("owner reduction barrier")
            .contains(&key)
        {
            append_position(index, memory, key, edge.pos);
        }
    }
    AppliedReply {
        merges: merged,
        work_seconds: started.elapsed().as_secs_f64(),
    }
}

// A stable two-half mix avoids sending all fresh-ID pairs to one owner.
#[inline]
fn owner_pair(key: u64, workers: usize) -> usize {
    let mut x = key ^ (key >> 30);
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    ((x ^ (x >> 31)) % workers as u64) as usize
}

type Mailboxes = Arc<Vec<Mutex<Vec<(u64, i128)>>>>;
type Sets = Arc<Vec<OnceLock<Arc<HashSet<u64>>>>>;

fn route(mailboxes: &Mailboxes, values: impl IntoIterator<Item = (u64, i128)>) {
    let mut buckets: Vec<Vec<(u64, i128)>> = (0..mailboxes.len()).map(|_| Vec::new()).collect();
    for (key, value) in values {
        buckets[owner_pair(key, mailboxes.len())].push((key, value));
    }
    for (owner, mut bucket) in buckets.into_iter().enumerate() {
        if !bucket.is_empty() {
            mailboxes[owner]
                .lock()
                .expect("owner mailbox poisoned")
                .append(&mut bucket);
        }
    }
}

fn mailbox_capacity(mailboxes: &Mailboxes) -> usize {
    mailboxes
        .iter()
        .map(|m| m.lock().expect("owner mailbox poisoned").capacity())
        .sum()
}

#[derive(Default)]
struct PairState {
    frequencies: HashMap<u64, u64>,
    heap: BinaryHeap<HeapEntry>,
    held: Vec<HeapEntry>,
}

impl PairState {
    fn initialize(&mut self, inbox: &Mutex<Vec<(u64, i128)>>, minimum: u64) -> TrainResult<()> {
        let mut inbox = inbox.lock().expect("owner mailbox poisoned");
        for (key, value) in inbox.drain(..) {
            let value = u64::try_from(value)
                .map_err(|_| TrainError::InternalInvariant("negative initial count"))?;
            let count = self.frequencies.entry(key).or_default();
            *count = count
                .checked_add(value)
                .ok_or(TrainError::Overflow("initial owner frequency exceeds u64"))?;
        }
        inbox.shrink_to_fit();
        self.frequencies.retain(|_, count| *count >= minimum);
        self.heap = self
            .frequencies
            .iter()
            .map(|(&key, &frequency)| HeapEntry { key, frequency })
            .collect();
        Ok(())
    }

    fn prefetch(&mut self, count: usize, minimum: u64) -> (Vec<HeapEntry>, usize) {
        let mut entries = Vec::with_capacity(count);
        let mut pops = 0;
        let mut already: HashSet<u64> = self.held.iter().map(|e| e.key).collect();
        while entries.len() < count {
            let Some((key, frequency)) =
                pop_best(&mut self.heap, &self.frequencies, minimum, &mut pops)
            else {
                break;
            };
            if already.insert(key) {
                let entry = HeapEntry { key, frequency };
                self.held.push(entry);
                entries.push(entry);
            }
        }
        (entries, pops)
    }

    fn return_candidates(&mut self, selected: &HashSet<u64>) {
        for entry in self.held.drain(..) {
            if !selected.contains(&entry.key) {
                self.heap.push(entry);
            }
        }
    }

    fn reduce(
        &mut self,
        inbox: &Mutex<Vec<(u64, i128)>>,
        first_new_id: u32,
        rules: &[RuleSpec],
        minimum: u64,
    ) -> TrainResult<(HashSet<u64>, usize)> {
        let mut pending = HashMap::<u64, i128>::new();
        let mut inbox = inbox.lock().expect("owner mailbox poisoned");
        for (key, value) in inbox.drain(..) {
            add_delta(&mut pending, key, value);
        }
        drop(inbox);
        let delta_keys = pending.len();
        let mut drops = HashSet::new();
        for (key, change) in pending {
            let fresh = (key >> 32) as u32 >= first_new_id || key as u32 >= first_new_id;
            if !fresh && !self.frequencies.contains_key(&key) {
                if change > 0 {
                    return Err(TrainError::InternalInvariant(
                        "discarded old pair increased",
                    ));
                }
                continue;
            }
            apply_delta(&mut self.frequencies, key, change)?;
            let frequency = self.frequencies[&key];
            if frequency < minimum {
                drops.insert(key);
            } else if fresh {
                self.heap.push(HeapEntry { key, frequency });
            }
        }
        for rule in rules {
            // The caller passes only rules owned by this scalar shard.
            if self.frequencies.get(&rule.key).copied().unwrap_or(0) != 0 {
                return Err(TrainError::InternalInvariant(
                    "selected pair remained after batch",
                ));
            }
            drops.insert(rule.key);
        }
        for key in &drops {
            self.frequencies.remove(key);
        }
        Ok((drops, delta_keys))
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_worker<const UNCHECKED: bool, const COMPACT: bool>(
    worker_id: usize,
    corpus: Arc<Vec<AtomicU32>>,
    pivots: Arc<Vec<u32>>,
    weights: Arc<Vec<u64>>,
    initial_lengths: Arc<Vec<u32>>,
    initial_mailboxes: Mailboxes,
    delta_mailboxes: Mailboxes,
    eligibles: Sets,
    start: usize,
    end: usize,
    minimum: u64,
) -> Worker {
    let (commands, rx) = mpsc::channel();
    let (tx, replies) = mpsc::channel();
    let handle = thread::spawn(move || {
        let mut index = HashMap::<u64, Vec<u32>>::new();
        let mut memory = IndexMemory::default();
        let mut peak_memory = IndexMemory::default();
        let mut lengths = (*initial_lengths).clone();
        let mut scalars = PairState::default();
        let first =
            scan_initial::<UNCHECKED>(&corpus, &pivots, &weights, start, end, None, &mut index);
        match first {
            Ok(first) => {
                let count_keys = first.counts.len();
                let count_capacity = first.counts.capacity();
                route(
                    &initial_mailboxes,
                    first.counts.into_iter().map(|(k, v)| (k, i128::from(v))),
                );
                if tx
                    .send(Reply::Initial {
                        occurrences: first.occurrences,
                        count_keys,
                        count_capacity,
                    })
                    .is_err()
                {
                    return;
                }
            }
            Err(error) => {
                let _ = tx.send(Reply::Error(error));
                return;
            }
        }
        let mut initial_mailboxes = Some(initial_mailboxes);
        let mut eligibles = Some(eligibles);
        let mut plans = PlanBuffer::<COMPACT>::default();
        let mut new_edges = Vec::new();
        let mut plan_peak_bytes = 0;
        let mut group_peak_bytes = 0;
        let mut edge_peak_bytes = 0;
        let mut worker_plan_seconds = 0.0;
        let mut worker_apply_seconds = 0.0;
        while let Ok(command) = rx.recv() {
            let reply: TrainResult<Reply> = match command {
                Command::BuildScalars => {
                    let inboxes = initial_mailboxes.take().expect("initial owner phase once");
                    scalars.initialize(&inboxes[worker_id], minimum).map(|_| {
                        let eligible: Arc<HashSet<u64>> =
                            Arc::new(scalars.frequencies.keys().copied().collect());
                        let eligible_capacity = eligible.capacity();
                        let _ = eligibles.as_ref().expect("eligible set exists")[worker_id]
                            .set(eligible);
                        Reply::Scalars {
                            keys: scalars.frequencies.len(),
                            capacity: scalars.frequencies.capacity(),
                            heap_capacity: scalars.heap.capacity(),
                            eligible_capacity,
                        }
                    })
                }
                Command::BuildIndex => {
                    let sets = eligibles.take().expect("initial index phase once");
                    scan_initial::<UNCHECKED>(
                        &corpus,
                        &pivots,
                        &weights,
                        start,
                        end,
                        Some(&sets),
                        &mut index,
                    )
                    .map(|_| {
                        memory = index_memory(&index);
                        peak_memory = memory;
                        Reply::Built(memory)
                    })
                }
                Command::Prefetch(count) => {
                    let (entries, pops) = scalars.prefetch(count, minimum);
                    Ok(Reply::Candidates { entries, pops })
                }
                Command::MeasureWindow(keys) => Ok(Reply::WindowLengths(
                    keys.iter()
                        .map(|key| index.get(key).map_or(0, Vec::len))
                        .collect(),
                )),
                Command::ProbeWindow(keys) => Ok(Reply::Probed(probe_window::<UNCHECKED>(
                    &corpus, &index, &lengths, &keys,
                ))),
                Command::ReturnCandidates(selected) => {
                    scalars.return_candidates(&selected);
                    Ok(Reply::Returned)
                }
                Command::PrepareBatch { rules, selected } => prepare_batch::<UNCHECKED, COMPACT>(
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
                .map(|mut reply| {
                    route(&delta_mailboxes, std::mem::take(&mut reply.delta));
                    plan_peak_bytes = plan_peak_bytes.max(reply.plan_capacity_bytes);
                    group_peak_bytes = group_peak_bytes.max(reply.group_capacity_bytes);
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
                } => prepare_self::<UNCHECKED, COMPACT>(
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
                .map(|mut reply| {
                    route(&delta_mailboxes, std::mem::take(&mut reply.delta));
                    plan_peak_bytes = plan_peak_bytes.max(reply.plan_capacity_bytes);
                    group_peak_bytes = group_peak_bytes.max(reply.group_capacity_bytes);
                    edge_peak_bytes =
                        edge_peak_bytes.max(reply.edge_capacity * std::mem::size_of::<NewEdge>());
                    worker_plan_seconds += reply.work_seconds;
                    Reply::Prepared(reply)
                }),
                Command::Reduce {
                    first_new_id,
                    rules,
                    drops,
                } => {
                    let started = Instant::now();
                    let owned: Vec<RuleSpec> = rules
                        .iter()
                        .copied()
                        .filter(|r| owner_pair(r.key, delta_mailboxes.len()) == worker_id)
                        .collect();
                    scalars
                        .reduce(&delta_mailboxes[worker_id], first_new_id, &owned, minimum)
                        .map(|(drop_keys, delta_keys)| {
                            let dropped = drop_keys.len();
                            let _ = drops[worker_id].set(Arc::new(drop_keys));
                            Reply::Reduced {
                                delta_keys,
                                scalar_keys: scalars.frequencies.len(),
                                scalar_capacity: scalars.frequencies.capacity(),
                                heap_capacity: scalars.heap.capacity(),
                                dropped,
                                seconds: started.elapsed().as_secs_f64(),
                            }
                        })
                }
                Command::Apply(drops) => {
                    let reply = apply_plans::<UNCHECKED, COMPACT>(
                        &corpus,
                        &mut index,
                        &mut memory,
                        &mut plans,
                        &mut new_edges,
                        &drops,
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
                        group_peak_bytes,
                        edge_peak_bytes,
                        scalar_keys: scalars.frequencies.len(),
                        scalar_capacity: scalars.frequencies.capacity(),
                        heap_capacity: scalars.heap.capacity(),
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
        .map_err(|_| TrainError::InternalInvariant("pair-owner worker disconnected"))?
    {
        Reply::Error(error) => Err(error),
        other => Ok(other),
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Frontier {
    entry: HeapEntry,
    owner: usize,
}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.entry.cmp(&other.entry)
    }
}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn fetch(
    worker: &Worker,
    page_size: usize,
    refill_messages: &mut usize,
    heap_pops: &mut usize,
) -> TrainResult<Vec<HeapEntry>> {
    worker
        .commands
        .send(Command::Prefetch(page_size))
        .map_err(|_| TrainError::InternalInvariant("candidate prefetch send failed"))?;
    *refill_messages += 2;
    match recv(worker)? {
        Reply::Candidates { entries, pops } => {
            *heap_pops += pops;
            Ok(entries)
        }
        _ => Err(TrainError::InternalInvariant("expected candidate reply")),
    }
}

#[allow(clippy::too_many_arguments)]
fn select_prefix(
    workers: &[Worker],
    remaining: usize,
    cap: usize,
    heap_pops: &mut usize,
    refill_messages: &mut usize,
    stop_self: &mut usize,
    stop_conflict: &mut usize,
    spatial: &mut SpatialStats,
) -> TrainResult<Vec<(u64, u64)>> {
    let mut queues: Vec<VecDeque<HeapEntry>> = Vec::with_capacity(workers.len());
    let mut frontier = BinaryHeap::new();
    // Preserve the caller's type-only cap; rent at most 64 further keys once
    // that certificate stops, so speculative heap work stays bounded.
    let limit = remaining.min(cap.max(1));
    let page_size = limit.min(32);
    for worker in workers {
        worker
            .commands
            .send(Command::Prefetch(page_size))
            .map_err(|_| TrainError::InternalInvariant("candidate prefetch send failed"))?;
    }
    *refill_messages += 2 * workers.len();
    for (owner, worker) in workers.iter().enumerate() {
        let queue: VecDeque<_> = match recv(worker)? {
            Reply::Candidates { entries, pops } => {
                *heap_pops += pops;
                entries.into()
            }
            _ => return Err(TrainError::InternalInvariant("expected candidate reply")),
        };
        if let Some(&entry) = queue.front() {
            frontier.push(Frontier { entry, owner });
        }
        queues.push(queue);
    }
    let mut pending = Vec::new();
    let mut heads = HashSet::new();
    let mut tails = HashSet::new();
    let mut base_width = None;
    while pending.len() < limit {
        let Some(Frontier { entry, owner }) = frontier.pop() else {
            break;
        };
        let a = (entry.key >> 32) as u32;
        let b = entry.key as u32;
        if a == b {
            *stop_self += 1;
            if !pending.is_empty() {
                break;
            }
            pending.push((entry.key, entry.frequency));
            break;
        }
        if base_width.is_none() {
            if tails.contains(&a) || heads.contains(&b) {
                *stop_conflict += 1;
                base_width = Some(pending.len());
            } else {
                heads.insert(a);
                tails.insert(b);
            }
        }
        pending.push((entry.key, entry.frequency));
        queues[owner].pop_front();
        if base_width.is_some_and(|base| pending.len() >= base.saturating_add(64)) {
            break;
        }
        if pending.len() < limit {
            if queues[owner].is_empty() {
                queues[owner] =
                    fetch(&workers[owner], page_size, refill_messages, heap_pops)?.into();
            }
            if let Some(&next) = queues[owner].front() {
                frontier.push(Frontier { entry: next, owner });
            }
        }
    }
    let base_width = base_width.unwrap_or(pending.len());
    spatial.base_width_sum += base_width;
    if base_width < pending.len() {
        let keys: Arc<Vec<u64>> = Arc::new(pending.iter().map(|(key, _)| *key).collect());
        let metadata_started = Instant::now();
        for worker in workers {
            worker
                .commands
                .send(Command::MeasureWindow(Arc::clone(&keys)))
                .map_err(|_| TrainError::InternalInvariant("spatial length send failed"))?;
        }
        spatial.metadata_messages += 2 * workers.len();
        let mut histories = vec![0_usize; keys.len()];
        for worker in workers {
            let Reply::WindowLengths(local) = recv(worker)? else {
                return Err(TrainError::InternalInvariant("expected spatial lengths"));
            };
            if local.len() != histories.len() {
                return Err(TrainError::InternalInvariant(
                    "spatial length count mismatch",
                ));
            }
            for (total, length) in histories.iter_mut().zip(local) {
                *total = total.saturating_add(length);
            }
        }
        spatial.metadata_seconds += metadata_started.elapsed().as_secs_f64();
        let base_history = histories[..base_width]
            .iter()
            .fold(0_usize, |sum, &length| sum.saturating_add(length));
        spatial.base_history_sum = spatial.base_history_sum.saturating_add(base_history);
        let budget = base_history.saturating_mul(2);
        let mut extra = 0_usize;
        let mut scan_width = base_width;
        for &length in &histories[base_width..] {
            if length > budget.saturating_sub(extra) {
                break;
            }
            extra += length;
            scan_width += 1;
        }
        spatial.budget_epochs += usize::from(scan_width < pending.len());
        if scan_width > base_width {
            let probe_keys = Arc::new(keys[..scan_width].to_vec());
            let expected_visits = histories[..scan_width]
                .iter()
                .fold(0_usize, |sum, &length| sum.saturating_add(length));
            let visits_before = spatial.probe_visits;
            let probe_started = Instant::now();
            for worker in workers {
                worker
                    .commands
                    .send(Command::ProbeWindow(Arc::clone(&probe_keys)))
                    .map_err(|_| TrainError::InternalInvariant("spatial probe send failed"))?;
            }
            spatial.probe_messages += 2 * workers.len();
            let mut cutoff = scan_width;
            for worker in workers {
                let Reply::Probed(reply) = recv(worker)? else {
                    return Err(TrainError::InternalInvariant("expected spatial probe"));
                };
                cutoff = cutoff.min(reply.cutoff);
                spatial.probe_visits += reply.visited;
                spatial.probe_stale += reply.stale;
                spatial.probe_worker_seconds_sum += reply.work_seconds;
            }
            spatial.probe_seconds += probe_started.elapsed().as_secs_f64();
            if spatial.probe_visits - visits_before != expected_visits {
                return Err(TrainError::InternalInvariant(
                    "spatial index changed during read-only probe",
                ));
            }
            if cutoff < base_width {
                return Err(TrainError::InternalInvariant(
                    "spatial conflict invalidated type certificate",
                ));
            }
            spatial.conflict_epochs += usize::from(cutoff < scan_width);
            pending.truncate(cutoff);
        } else {
            pending.truncate(base_width);
        }
    }
    spatial.widened_epochs += usize::from(pending.len() > base_width);
    spatial.final_width_sum += pending.len();
    let selected = Arc::new(pending.iter().map(|(key, _)| *key).collect::<HashSet<_>>());
    for worker in workers {
        worker
            .commands
            .send(Command::ReturnCandidates(Arc::clone(&selected)))
            .map_err(|_| TrainError::InternalInvariant("candidate restore send failed"))?;
    }
    *refill_messages += 2 * workers.len();
    for worker in workers {
        if !matches!(recv(worker)?, Reply::Returned) {
            return Err(TrainError::InternalInvariant(
                "expected candidate restore reply",
            ));
        }
    }
    Ok(pending)
}

pub(super) fn train<const UNCHECKED: bool>(
    input: Prepared,
    options: Options,
    cap: usize,
) -> TrainResult<Result> {
    run::<UNCHECKED, false>(input, options, cap)
}

fn run<const UNCHECKED: bool, const COMPACT: bool>(
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
    let initial_mailboxes: Mailboxes =
        Arc::new((0..worker_count).map(|_| Mutex::new(Vec::new())).collect());
    let delta_mailboxes: Mailboxes =
        Arc::new((0..worker_count).map(|_| Mutex::new(Vec::new())).collect());
    let eligibles: Sets = Arc::new((0..worker_count).map(|_| OnceLock::new()).collect());
    let workers: Vec<Worker> = (0..worker_count)
        .map(|id| {
            let start = 1 + id * (last - 1) / worker_count;
            let end = 1 + (id + 1) * (last - 1) / worker_count;
            spawn_worker::<UNCHECKED, COMPACT>(
                id,
                Arc::clone(&corpus),
                Arc::clone(&pivots),
                Arc::clone(&weights),
                Arc::clone(&worker_initial_lengths),
                Arc::clone(&initial_mailboxes),
                Arc::clone(&delta_mailboxes),
                Arc::clone(&eligibles),
                start,
                end,
                options.min_frequency,
            )
        })
        .collect();
    let mut initial_occurrences = 0;
    let mut initial_count_keys_sum = 0;
    let mut initial_count_capacity_sum = 0;
    for worker in &workers {
        match recv(worker)? {
            Reply::Initial {
                occurrences,
                count_keys,
                count_capacity,
            } => {
                initial_occurrences += occurrences;
                initial_count_keys_sum += count_keys;
                initial_count_capacity_sum += count_capacity;
            }
            _ => {
                return Err(TrainError::InternalInvariant(
                    "expected initial count reply",
                ));
            }
        }
    }
    let initial_mailbox_capacity = mailbox_capacity(&initial_mailboxes);
    for worker in &workers {
        worker
            .commands
            .send(Command::BuildScalars)
            .map_err(|_| TrainError::InternalInvariant("owner init send failed"))?;
    }
    let mut initial_scalar_keys = 0;
    let mut initial_scalar_capacity = 0;
    let mut initial_heap_capacity = 0;
    let mut initial_eligible_capacity = 0;
    for worker in &workers {
        match recv(worker)? {
            Reply::Scalars {
                keys,
                capacity,
                heap_capacity,
                eligible_capacity,
            } => {
                initial_scalar_keys += keys;
                initial_scalar_capacity += capacity;
                initial_heap_capacity += heap_capacity;
                initial_eligible_capacity += eligible_capacity;
            }
            _ => return Err(TrainError::InternalInvariant("expected owner init reply")),
        }
    }
    drop(initial_mailboxes);
    for worker in &workers {
        worker
            .commands
            .send(Command::BuildIndex)
            .map_err(|_| TrainError::InternalInvariant("index init send failed"))?;
    }
    let mut initial_index_memory = IndexMemory::default();
    for worker in &workers {
        let Reply::Built(memory) = recv(worker)? else {
            return Err(TrainError::InternalInvariant("expected index init reply"));
        };
        initial_index_memory.position_len += memory.position_len;
        initial_index_memory.position_capacity += memory.position_capacity;
        initial_index_memory.map_len += memory.map_len;
        initial_index_memory.map_capacity += memory.map_capacity;
    }
    drop(eligibles);
    let init_seconds = started.elapsed().as_secs_f64();
    let merge_started = Instant::now();
    let mut merges = Vec::new();
    let mut actual_merges = 0;
    let mut position_visits = 0;
    let mut stale_visits = 0;
    let mut heap_pops = 0;
    let mut epochs = 0;
    let mut max_width = 0;
    let mut stop_self = 0;
    let mut stop_conflict = 0;
    let mut aa_epochs = 0;
    let mut aa_gather_positions = 0;
    let mut select_seconds = 0.0;
    let mut plan_seconds = 0.0;
    let mut apply_seconds = 0.0;
    let mut owner_reduce_seconds = 0.0;
    let mut owner_reduce_work_seconds_sum = 0.0;
    let mut worker_plan_seconds_sum = 0.0;
    let mut worker_apply_seconds_sum = 0.0;
    let mut worker_delta_keys_total = 0;
    let mut owner_delta_keys_total = 0;
    let mut owner_drop_keys_total = 0;
    let mut refill_messages = 0;
    let mut messages = 6 * worker_count; // init scalar/index send+reply, plus initial replies
    let mut mailbox_capacity_peak = initial_mailbox_capacity;
    let mut plan_capacity_peak = 0;
    let mut group_capacity_peak = 0;
    let mut edge_capacity_peak = 0;
    let mut owner_scalar_capacity_peak = initial_scalar_capacity;
    let mut owner_heap_capacity_peak = initial_heap_capacity;
    let mut spatial = SpatialStats::default();
    while merges.len() < options.max_merges {
        let selection_started = Instant::now();
        let prefix = select_prefix(
            &workers,
            options.max_merges - merges.len(),
            cap,
            &mut heap_pops,
            &mut refill_messages,
            &mut stop_self,
            &mut stop_conflict,
            &mut spatial,
        )?;
        select_seconds += selection_started.elapsed().as_secs_f64();
        if prefix.is_empty() {
            break;
        }
        epochs += 1;
        max_width = max_width.max(prefix.len());
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
            specs.push(RuleSpec {
                key,
                a,
                b,
                new_id,
                new_length,
                a_length: lengths[a as usize] as usize,
                b_length: lengths[b as usize] as usize,
                frequency,
            });
            lengths.push(new_length);
        }
        let rules = Arc::new(specs);
        let plan_started = Instant::now();
        if rules[0].a == rules[0].b {
            aa_epochs += 1;
            let rule = rules[0];
            for worker in &workers {
                worker
                    .commands
                    .send(Command::GatherSelf(rule))
                    .map_err(|_| TrainError::InternalInvariant("AA gather send failed"))?;
            }
            messages += 2 * worker_count;
            let mut valid = Vec::new();
            for worker in &workers {
                let Reply::Gathered(reply) = recv(worker)? else {
                    return Err(TrainError::InternalInvariant("expected AA gather reply"));
                };
                position_visits += reply.visited;
                stale_visits += reply.stale;
                worker_plan_seconds_sum += reply.work_seconds;
                let _ = reply.valid_capacity;
                valid.extend(reply.valid_positions);
            }
            aa_gather_positions += valid.len();
            valid.sort_unstable();
            if valid.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(TrainError::InternalInvariant(
                    "duplicate live AA occurrence",
                ));
            }
            let mut selected = Vec::new();
            let mut previous_after = 0;
            for raw in valid {
                let pos = raw as usize;
                if pos < previous_after {
                    stale_visits += 1;
                    continue;
                }
                previous_after = pos + rule.a_length + rule.b_length;
                selected.push(raw);
            }
            let selected = Arc::new(selected);
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
                    .map_err(|_| TrainError::InternalInvariant("AA prepare send failed"))?;
            }
            messages += 2 * worker_count;
        } else {
            let selected = Arc::new(rules.iter().map(|r| (r.key, r.new_id)).collect());
            for worker in &workers {
                worker
                    .commands
                    .send(Command::PrepareBatch {
                        rules: Arc::clone(&rules),
                        selected: Arc::clone(&selected),
                    })
                    .map_err(|_| TrainError::InternalInvariant("prepare send failed"))?;
            }
            messages += 2 * worker_count;
        }
        let mut planned = 0;
        let mut round_plan_capacity = 0;
        let mut round_group_capacity = 0;
        let mut round_edge_capacity = 0;
        for worker in &workers {
            let Reply::Prepared(reply) = recv(worker)? else {
                return Err(TrainError::InternalInvariant("expected prepared reply"));
            };
            position_visits += reply.visited;
            stale_visits += reply.stale;
            planned += reply.valid;
            round_plan_capacity += reply.plan_capacity_bytes;
            round_group_capacity += reply.group_capacity_bytes;
            round_edge_capacity += reply.edge_capacity * std::mem::size_of::<NewEdge>();
            worker_delta_keys_total += reply.delta_keys;
            worker_plan_seconds_sum += reply.work_seconds;
        }
        plan_capacity_peak = plan_capacity_peak.max(round_plan_capacity);
        group_capacity_peak = group_capacity_peak.max(round_group_capacity);
        edge_capacity_peak = edge_capacity_peak.max(round_edge_capacity);
        mailbox_capacity_peak = mailbox_capacity_peak.max(mailbox_capacity(&delta_mailboxes));
        plan_seconds += plan_started.elapsed().as_secs_f64();
        let reduce_started = Instant::now();
        let drops: Sets = Arc::new((0..worker_count).map(|_| OnceLock::new()).collect());
        for worker in &workers {
            worker
                .commands
                .send(Command::Reduce {
                    first_new_id,
                    rules: Arc::clone(&rules),
                    drops: Arc::clone(&drops),
                })
                .map_err(|_| TrainError::InternalInvariant("owner reduce send failed"))?;
        }
        messages += 2 * worker_count;
        let mut scalar_capacity = 0;
        let mut heap_capacity = 0;
        for worker in &workers {
            match recv(worker)? {
                Reply::Reduced {
                    delta_keys,
                    scalar_keys,
                    scalar_capacity: sc,
                    heap_capacity: hc,
                    dropped,
                    seconds,
                } => {
                    owner_delta_keys_total += delta_keys;
                    owner_drop_keys_total += dropped;
                    scalar_capacity += sc;
                    heap_capacity += hc;
                    owner_reduce_work_seconds_sum += seconds;
                    let _ = scalar_keys;
                }
                _ => return Err(TrainError::InternalInvariant("expected owner reduce reply")),
            }
        }
        owner_scalar_capacity_peak = owner_scalar_capacity_peak.max(scalar_capacity);
        owner_heap_capacity_peak = owner_heap_capacity_peak.max(heap_capacity);
        for rule in rules.iter() {
            merges.push(Rule {
                left: rule.a,
                right: rule.b,
                frequency: rule.frequency,
            });
        }
        owner_reduce_seconds += reduce_started.elapsed().as_secs_f64();
        let apply_started = Instant::now();
        for worker in &workers {
            worker
                .commands
                .send(Command::Apply(Arc::clone(&drops)))
                .map_err(|_| TrainError::InternalInvariant("apply send failed"))?;
        }
        messages += 2 * worker_count;
        let mut applied = 0;
        for worker in &workers {
            let Reply::Applied(reply) = recv(worker)? else {
                return Err(TrainError::InternalInvariant("expected applied reply"));
            };
            applied += reply.merges;
            worker_apply_seconds_sum += reply.work_seconds;
        }
        if applied != planned {
            return Err(TrainError::InternalInvariant(
                "batch plan/apply count mismatch",
            ));
        }
        actual_merges += applied;
        apply_seconds += apply_started.elapsed().as_secs_f64();
    }
    let merge_seconds = merge_started.elapsed().as_secs_f64();
    for worker in &workers {
        worker
            .commands
            .send(Command::Finish)
            .map_err(|_| TrainError::InternalInvariant("finish send failed"))?;
    }
    let mut final_memory = IndexMemory::default();
    let mut index_peak_sum = IndexMemory::default();
    let mut final_scalar_keys = 0;
    let mut final_scalar_capacity = 0;
    let mut final_heap_capacity = 0;
    let mut worker_plan_peak_bytes_sum = 0;
    let mut worker_group_peak_bytes_sum = 0;
    let mut worker_edge_peak_bytes_sum = 0;
    for worker in &workers {
        let Reply::Final(reply) = recv(worker)? else {
            return Err(TrainError::InternalInvariant("expected final reply"));
        };
        final_memory.position_len += reply.memory.position_len;
        final_memory.position_capacity += reply.memory.position_capacity;
        final_memory.map_len += reply.memory.map_len;
        final_memory.map_capacity += reply.memory.map_capacity;
        index_peak_sum.position_len += reply.peak_memory.position_len;
        index_peak_sum.position_capacity += reply.peak_memory.position_capacity;
        index_peak_sum.map_len += reply.peak_memory.map_len;
        index_peak_sum.map_capacity += reply.peak_memory.map_capacity;
        final_scalar_keys += reply.scalar_keys;
        final_scalar_capacity += reply.scalar_capacity;
        final_heap_capacity += reply.heap_capacity;
        worker_plan_peak_bytes_sum += reply.plan_peak_bytes;
        worker_group_peak_bytes_sum += reply.group_peak_bytes;
        worker_edge_peak_bytes_sum += reply.edge_peak_bytes;
        let _ = (reply.worker_plan_seconds, reply.worker_apply_seconds);
    }
    messages += 2 * worker_count + refill_messages;
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
            backend_buffer_bytes: corpus_positions * 4,
            initial_occurrence_bytes: initial_index_memory.position_len * 4,
            max_token_length,
            corpus_positions,
        },
    );
    let mut metrics = BTreeMap::new();
    metric(&mut metrics, "workers_actual", worker_count as f64);
    metric(
        &mut metrics,
        "shared_corpus_bytes",
        (corpus_positions * 4) as f64,
    );
    metric(
        &mut metrics,
        "initial_unfiltered_occurrences",
        initial_occurrences as f64,
    );
    metric(
        &mut metrics,
        "initial_count_keys_sum",
        initial_count_keys_sum as f64,
    );
    metric(
        &mut metrics,
        "initial_count_map_capacity_sum",
        initial_count_capacity_sum as f64,
    );
    metric(
        &mut metrics,
        "initial_mailbox_capacity_entries",
        initial_mailbox_capacity as f64,
    );
    metric(
        &mut metrics,
        "initial_scalar_keys",
        initial_scalar_keys as f64,
    );
    metric(
        &mut metrics,
        "initial_scalar_map_capacity_sum",
        initial_scalar_capacity as f64,
    );
    metric(
        &mut metrics,
        "initial_heap_capacity_sum",
        initial_heap_capacity as f64,
    );
    metric(
        &mut metrics,
        "initial_eligible_capacity_sum",
        initial_eligible_capacity as f64,
    );
    metric(
        &mut metrics,
        "initial_index_position_capacity_bytes",
        (initial_index_memory.position_capacity * 4) as f64,
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
        "index_position_capacity_peak_bytes_sum",
        (index_peak_sum.position_capacity * 4) as f64,
    );
    metric(
        &mut metrics,
        "index_map_capacity_peak_sum",
        index_peak_sum.map_capacity as f64,
    );
    metric(
        &mut metrics,
        "final_owner_scalar_keys",
        final_scalar_keys as f64,
    );
    metric(
        &mut metrics,
        "final_owner_scalar_capacity_sum",
        final_scalar_capacity as f64,
    );
    metric(
        &mut metrics,
        "final_owner_heap_capacity_sum",
        final_heap_capacity as f64,
    );
    metric(
        &mut metrics,
        "owner_scalar_capacity_peak_sum",
        owner_scalar_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "owner_heap_capacity_peak_sum",
        owner_heap_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "mailbox_capacity_peak_entries",
        mailbox_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "plan_capacity_peak_bytes",
        plan_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "plan_group_capacity_peak_bytes",
        group_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "plan_start_record_bytes",
        if COMPACT { 4.0 } else { 16.0 },
    );
    metric(
        &mut metrics,
        "plan_group_record_bytes",
        if COMPACT { 16.0 } else { 0.0 },
    );
    metric(
        &mut metrics,
        "new_edge_capacity_peak_bytes",
        edge_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "worker_plan_peak_bytes_sum",
        worker_plan_peak_bytes_sum as f64,
    );
    metric(
        &mut metrics,
        "worker_plan_group_peak_bytes_sum",
        worker_group_peak_bytes_sum as f64,
    );
    metric(
        &mut metrics,
        "worker_edge_peak_bytes_sum",
        worker_edge_peak_bytes_sum as f64,
    );
    metric(&mut metrics, "certificate_epochs", epochs as f64);
    metric(&mut metrics, "certificate_max_width", max_width as f64);
    metric(&mut metrics, "certificate_stop_self", stop_self as f64);
    metric(
        &mut metrics,
        "certificate_stop_conflict",
        stop_conflict as f64,
    );
    metric(
        &mut metrics,
        "spatial_base_width_sum",
        spatial.base_width_sum as f64,
    );
    metric(
        &mut metrics,
        "spatial_final_width_sum",
        spatial.final_width_sum as f64,
    );
    metric(
        &mut metrics,
        "spatial_widened_epochs",
        spatial.widened_epochs as f64,
    );
    metric(
        &mut metrics,
        "spatial_conflict_epochs",
        spatial.conflict_epochs as f64,
    );
    metric(
        &mut metrics,
        "spatial_budget_epochs",
        spatial.budget_epochs as f64,
    );
    metric(
        &mut metrics,
        "spatial_metadata_seconds",
        spatial.metadata_seconds,
    );
    metric(&mut metrics, "spatial_probe_seconds", spatial.probe_seconds);
    metric(
        &mut metrics,
        "spatial_probe_worker_seconds_sum",
        spatial.probe_worker_seconds_sum,
    );
    metric(
        &mut metrics,
        "spatial_base_history_sum",
        spatial.base_history_sum as f64,
    );
    metric(
        &mut metrics,
        "spatial_probe_visits",
        spatial.probe_visits as f64,
    );
    metric(
        &mut metrics,
        "spatial_probe_stale",
        spatial.probe_stale as f64,
    );
    metric(
        &mut metrics,
        "spatial_metadata_messages",
        spatial.metadata_messages as f64,
    );
    metric(
        &mut metrics,
        "spatial_probe_messages",
        spatial.probe_messages as f64,
    );
    metric(&mut metrics, "aa_epochs", aa_epochs as f64);
    metric(
        &mut metrics,
        "aa_gather_positions",
        aa_gather_positions as f64,
    );
    metric(&mut metrics, "select_seconds", select_seconds);
    metric(&mut metrics, "plan_seconds", plan_seconds);
    metric(&mut metrics, "owner_reduce_seconds", owner_reduce_seconds);
    metric(
        &mut metrics,
        "owner_reduce_work_seconds_sum",
        owner_reduce_work_seconds_sum,
    );
    metric(&mut metrics, "apply_seconds", apply_seconds);
    metric(
        &mut metrics,
        "worker_plan_seconds_sum",
        worker_plan_seconds_sum,
    );
    metric(
        &mut metrics,
        "worker_apply_seconds_sum",
        worker_apply_seconds_sum,
    );
    metric(
        &mut metrics,
        "worker_delta_keys_total",
        worker_delta_keys_total as f64,
    );
    metric(
        &mut metrics,
        "owner_delta_keys_total",
        owner_delta_keys_total as f64,
    );
    metric(
        &mut metrics,
        "owner_drop_keys_total",
        owner_drop_keys_total as f64,
    );
    metric(
        &mut metrics,
        "candidate_refill_messages",
        refill_messages as f64,
    );
    metric(
        &mut metrics,
        "round_messages",
        (messages + spatial.metadata_messages + spatial.probe_messages) as f64,
    );
    metric(
        &mut metrics,
        "unchecked_corpus_access",
        if UNCHECKED { 1.0 } else { 0.0 },
    );
    Ok(Result { core, metrics })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_overlap_distinguishes_separate_pieces_from_abc() {
        let keys = [pair_key(1, 2), pair_key(2, 3)];
        let lengths = [0, 1, 1, 1];
        let separate: Vec<_> = [0, 1, 2, 0, 2, 3, 0]
            .into_iter()
            .map(AtomicU32::new)
            .collect();
        let separate_index = HashMap::from([(keys[0], vec![1]), (keys[1], vec![4])]);
        assert_eq!(
            probe_window::<false>(&separate, &separate_index, &lengths, &keys).cutoff,
            2
        );

        let chain: Vec<_> = [0, 1, 2, 3, 0].into_iter().map(AtomicU32::new).collect();
        let chain_index = HashMap::from([(keys[0], vec![1]), (keys[1], vec![2])]);
        assert_eq!(
            probe_window::<false>(&chain, &chain_index, &lengths, &keys).cutoff,
            1
        );
    }
}

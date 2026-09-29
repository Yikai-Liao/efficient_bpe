//! Exact single-rule BPE with sparse position and scalar owner activation.
//!
//! A pair's historical position holders are fixed at birth. The coordinator
//! caches only one exact head per owner and wakes the holders of the winner.

#[path = "sparse_frontier.rs"]
mod sparse_frontier;
#[path = "worker_groups.rs"]
mod worker_groups;

use sparse_frontier::{Frontier, Head};
use worker_groups::WorkerGroups;

use super::{CoreStats, HeapEntry, add_delta, core_result, metric, pair_key};
use crate::ablation::{Options, Result};
use crate::{Prepared, Rule, TrainError};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
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
    edge_capacity: usize,
    work_seconds: f64,
}

struct GatheredReply {
    valid_positions: Vec<u32>,
    visited: usize,
    stale: usize,
    work_seconds: f64,
}

struct AppliedReply {
    merges: usize,
    work_seconds: f64,
    pruned: usize,
}

struct FinalReply {
    memory: IndexMemory,
    peak_memory: IndexMemory,
    plan_peak_bytes: usize,
    edge_peak_bytes: usize,
    scalar_keys: usize,
    scalar_capacity: usize,
    heap_capacity: usize,
    router_capacity: usize,
    heap_pops: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Delta {
    amount: i128,
    key: u64,
    holders: u64,
}
const _: [(); 32] = [(); std::mem::size_of::<Delta>()];

#[derive(Clone, Copy)]
struct PairScalar {
    frequency: u64,
    holders: u64,
}
const _: [(); 16] = [(); std::mem::size_of::<PairScalar>()];

enum Command {
    BuildScalars,
    BuildIndex,
    Prepare {
        epoch: u64,
        rule: RuleSpec,
    },
    GatherSelf {
        epoch: u64,
        rule: RuleSpec,
    },
    PrepareSelf {
        epoch: u64,
        rule: RuleSpec,
        positions: Arc<Vec<u32>>,
        start: usize,
        end: usize,
    },
    Reduce {
        epoch: u64,
        first_new_id: u32,
        selected: Option<RuleSpec>,
    },
    Apply {
        epoch: u64,
    },
    Finish,
}

enum Reply {
    Initial {
        occurrences: usize,
        count_keys: usize,
        count_capacity: usize,
    },
    Scalars {
        top: Option<Head>,
        keys: usize,
        capacity: usize,
        heap_capacity: usize,
        eligible_capacity: usize,
    },
    Built(IndexMemory),
    Prepared(PreparedReply),
    Gathered(GatheredReply),
    Reduced {
        top: Option<Head>,
        recipients: u64,
        delta_keys: usize,
        dropped: usize,
        prune_entries: usize,
        seconds: f64,
        scalar_capacity: usize,
        heap_capacity: usize,
    },
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
    lengths: &[u32],
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
        if lengths.get(rule.new_id as usize) != Some(&rule.new_length) {
            return Err(TrainError::InternalInvariant(
                "shared token lengths out of order",
            ));
        }
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
        edge_capacity: new_edges.capacity(),
        work_seconds: started.elapsed().as_secs_f64(),
    })
}

#[allow(clippy::too_many_arguments)] // AA supplies globally selected starts to the shared planner.
fn prepare_self<const UNCHECKED: bool, const COMPACT: bool>(
    corpus: &[AtomicU32],
    lengths: &[u32],
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
    if lengths.get(rule.new_id as usize) != Some(&rule.new_length) {
        return Err(TrainError::InternalInvariant(
            "shared token lengths out of order",
        ));
    }
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
    drop_keys: &HashSet<u64>,
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
    for key in drop_keys {
        let _ = take_positions(index, memory, *key);
    }
    for edge in new_edges.drain(..) {
        let key = edge.key();
        if !drop_keys.contains(&key) {
            append_position(index, memory, key, edge.pos);
        }
    }
    AppliedReply {
        merges: merged,
        work_seconds: started.elapsed().as_secs_f64(),
        pruned: drop_keys.len(),
    }
}

// Mix both token IDs; owner and holder grouping are independent.
#[inline]
fn owner_pair(key: u64, workers: usize) -> usize {
    let mut x = key ^ (key >> 30);
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    ((x ^ (x >> 31)) % workers as u64) as usize
}

#[derive(Default)]
struct EpochInbox {
    epoch: u64,
    records: Vec<Delta>,
}
type Mailboxes = Arc<Vec<Mutex<EpochInbox>>>;
type PruneMailboxes = Arc<Vec<Mutex<Vec<(u64, u64)>>>>;
type EligibleSets = Arc<Vec<OnceLock<Arc<HashSet<u64>>>>>;

struct Router {
    buckets: Vec<Vec<Delta>>,
    touched: Vec<usize>,
}
impl Router {
    fn new(workers: usize) -> Self {
        Self {
            buckets: (0..workers).map(|_| Vec::new()).collect(),
            touched: Vec::new(),
        }
    }
    fn capacity(&self) -> usize {
        self.buckets.iter().map(Vec::capacity).sum()
    }
    fn route(
        &mut self,
        epoch: u64,
        source: usize,
        values: impl IntoIterator<Item = (u64, i128)>,
        mailboxes: &Mailboxes,
        groups: &WorkerGroups,
        touched_owners: Option<&AtomicU64>,
    ) {
        let holder = groups.bit(source);
        for (key, amount) in values {
            let owner = owner_pair(key, self.buckets.len());
            if self.buckets[owner].is_empty() {
                self.touched.push(owner);
            }
            self.buckets[owner].push(Delta {
                amount,
                key,
                holders: holder,
            });
        }
        for owner in self.touched.drain(..) {
            let mut inbox = mailboxes[owner].lock().expect("delta mailbox poisoned");
            assert!(
                inbox.records.is_empty() || inbox.epoch == epoch,
                "mixed delta epochs"
            );
            inbox.epoch = epoch;
            inbox.records.append(&mut self.buckets[owner]);
            if let Some(bits) = touched_owners {
                bits.fetch_or(groups.bit(owner), Ordering::Relaxed);
            }
        }
    }
}

#[derive(Default)]
struct PairState {
    frequencies: HashMap<u64, PairScalar>,
    heap: BinaryHeap<HeapEntry>,
    heap_pops: usize,
}
impl PairState {
    fn initialize(&mut self, inbox: &Mutex<EpochInbox>, minimum: u64) -> TrainResult<()> {
        let mut inbox = inbox.lock().expect("initial mailbox poisoned");
        if inbox.epoch != 0 {
            return Err(TrainError::InternalInvariant(
                "initial mailbox epoch mismatch",
            ));
        }
        for delta in inbox.records.drain(..) {
            let value = u64::try_from(delta.amount)
                .map_err(|_| TrainError::InternalInvariant("negative initial count"))?;
            let scalar = self.frequencies.entry(delta.key).or_insert(PairScalar {
                frequency: 0,
                holders: 0,
            });
            scalar.frequency = scalar
                .frequency
                .checked_add(value)
                .ok_or(TrainError::Overflow("initial owner frequency exceeds u64"))?;
            scalar.holders |= delta.holders;
        }
        inbox.records.shrink_to_fit();
        self.frequencies.retain(|_, v| v.frequency >= minimum);
        self.heap = self
            .frequencies
            .iter()
            .map(|(&key, v)| HeapEntry {
                key,
                frequency: v.frequency,
            })
            .collect();
        Ok(())
    }
    fn top(&mut self, minimum: u64) -> Option<Head> {
        loop {
            let entry = *self.heap.peek()?;
            let scalar = self.frequencies.get(&entry.key).copied();
            match scalar {
                Some(value) if value.frequency >= minimum && value.frequency == entry.frequency => {
                    return Some(Head {
                        key: entry.key,
                        frequency: value.frequency,
                        holders: value.holders,
                    });
                }
                Some(value) if value.frequency >= minimum => {
                    self.heap.pop();
                    self.heap_pops += 1;
                    self.heap.push(HeapEntry {
                        key: entry.key,
                        frequency: value.frequency,
                    });
                }
                _ => {
                    self.heap.pop();
                    self.heap_pops += 1;
                }
            }
        }
    }
    #[allow(clippy::too_many_arguments)] // Epoch, owner state, and recipient mailboxes are one reduction phase.
    fn reduce(
        &mut self,
        inbox: &Mutex<EpochInbox>,
        epoch: u64,
        first_new_id: u32,
        selected: Option<RuleSpec>,
        minimum: u64,
        prune: &PruneMailboxes,
        groups: &WorkerGroups,
    ) -> TrainResult<(u64, usize, usize, usize)> {
        let mut pending = HashMap::<u64, (i128, u64)>::new();
        let mut inbox = inbox.lock().expect("delta mailbox poisoned");
        if !inbox.records.is_empty() && inbox.epoch != epoch {
            return Err(TrainError::InternalInvariant(
                "delta mailbox epoch mismatch",
            ));
        }
        for delta in inbox.records.drain(..) {
            let entry = pending.entry(delta.key).or_default();
            entry.0 += delta.amount;
            if delta.amount > 0 {
                entry.1 |= delta.holders;
            }
        }
        drop(inbox);
        let delta_keys = pending.len();
        let mut retired = HashMap::<u64, u64>::new();
        for (key, (change, birth_holders)) in pending {
            let fresh = (key >> 32) as u32 >= first_new_id || key as u32 >= first_new_id;
            if !fresh && !self.frequencies.contains_key(&key) {
                if change > 0 {
                    return Err(TrainError::InternalInvariant(
                        "discarded old pair increased",
                    ));
                }
                continue;
            }
            let scalar = self.frequencies.entry(key).or_insert(PairScalar {
                frequency: 0,
                holders: birth_holders,
            });
            if fresh {
                scalar.holders |= birth_holders;
            }
            let magnitude = change.unsigned_abs();
            let magnitude = u64::try_from(magnitude)
                .map_err(|_| TrainError::Overflow("pair delta exceeds u64"))?;
            scalar.frequency = if change >= 0 {
                scalar.frequency.checked_add(magnitude)
            } else {
                scalar.frequency.checked_sub(magnitude)
            }
            .ok_or(TrainError::InternalInvariant(
                "owner pair frequency underflow/overflow",
            ))?;
            if scalar.frequency < minimum {
                retired.insert(key, scalar.holders);
            } else if fresh {
                self.heap.push(HeapEntry {
                    key,
                    frequency: scalar.frequency,
                });
            }
        }
        if let Some(rule) = selected {
            let scalar = self
                .frequencies
                .get(&rule.key)
                .ok_or(TrainError::InternalInvariant(
                    "selected pair missing at owner",
                ))?;
            if scalar.frequency != 0 {
                return Err(TrainError::InternalInvariant(
                    "selected pair remained after replacement",
                ));
            }
            retired.insert(rule.key, scalar.holders);
        }
        let mut recipients = 0;
        let mut prune_entries = 0;
        for (key, mask) in &retired {
            for worker in groups.members(*mask) {
                prune[worker]
                    .lock()
                    .expect("prune mailbox poisoned")
                    .push((epoch, *key));
                prune_entries += 1;
                recipients |= groups.bit(worker);
            }
        }
        for key in retired.keys() {
            self.frequencies.remove(key);
        }
        Ok((recipients, delta_keys, retired.len(), prune_entries))
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_worker<const UNCHECKED: bool>(
    worker_id: usize,
    corpus: Arc<Vec<AtomicU32>>,
    pivots: Arc<Vec<u32>>,
    weights: Arc<Vec<u64>>,
    lengths: Arc<RwLock<Vec<u32>>>,
    groups: Arc<WorkerGroups>,
    initial_mailboxes: Mailboxes,
    delta_mailboxes: Mailboxes,
    prune_mailboxes: PruneMailboxes,
    eligibles: EligibleSets,
    touched_owners: Arc<AtomicU64>,
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
        let mut scalars = PairState::default();
        let mut router = Router::new(delta_mailboxes.len());
        match scan_initial::<UNCHECKED>(&corpus, &pivots, &weights, start, end, None, &mut index) {
            Ok(first) => {
                let count_keys = first.counts.len();
                let count_capacity = first.counts.capacity();
                router.route(
                    0,
                    worker_id,
                    first.counts.into_iter().map(|(k, v)| (k, i128::from(v))),
                    &initial_mailboxes,
                    &groups,
                    None,
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
        let mut plans = PlanBuffer::<false>::default();
        let mut new_edges = Vec::new();
        let mut planned_epoch = None;
        let mut gathered_epoch = None;
        let mut last_applied_epoch = 0;
        let mut plan_peak_bytes = 0;
        let mut edge_peak_bytes = 0;
        while let Ok(command) = rx.recv() {
            let reply: TrainResult<Reply> = match command {
                Command::BuildScalars => {
                    let inboxes = initial_mailboxes.take().expect("initial owner phase once");
                    scalars.initialize(&inboxes[worker_id], minimum).map(|_| {
                        let eligible: Arc<HashSet<u64>> =
                            Arc::new(scalars.frequencies.keys().copied().collect());
                        let eligible_capacity = eligible.capacity();
                        let _ = eligibles.as_ref().expect("eligible phase exists")[worker_id]
                            .set(eligible);
                        Reply::Scalars {
                            top: scalars.top(minimum),
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
                Command::Prepare { epoch, rule } => {
                    if epoch <= last_applied_epoch
                        || planned_epoch.is_some()
                        || plans.len() != 0
                        || !new_edges.is_empty()
                    {
                        Err(TrainError::InternalInvariant("position plan epoch overlap"))
                    } else {
                        let selected = HashMap::from([(rule.key, rule.new_id)]);
                        let mut result = {
                            let guard = lengths.read().expect("length table poisoned");
                            prepare_batch::<UNCHECKED, false>(
                                &corpus,
                                &guard,
                                &pivots,
                                &weights,
                                &mut index,
                                &mut memory,
                                std::slice::from_ref(&rule),
                                &selected,
                                &mut plans,
                                &mut new_edges,
                            )
                        };
                        if let Ok(reply) = &mut result {
                            router.route(
                                epoch,
                                worker_id,
                                std::mem::take(&mut reply.delta),
                                &delta_mailboxes,
                                &groups,
                                Some(&touched_owners),
                            );
                            planned_epoch = Some(epoch);
                            plan_peak_bytes = plan_peak_bytes.max(reply.plan_capacity_bytes);
                            edge_peak_bytes = edge_peak_bytes
                                .max(reply.edge_capacity * std::mem::size_of::<NewEdge>());
                        }
                        result.map(Reply::Prepared)
                    }
                }
                Command::GatherSelf { epoch, rule } => {
                    if epoch <= last_applied_epoch
                        || gathered_epoch.is_some()
                        || planned_epoch.is_some()
                    {
                        Err(TrainError::InternalInvariant("AA gather epoch overlap"))
                    } else {
                        let started = Instant::now();
                        let mut valid_positions = Vec::new();
                        let mut visited = 0;
                        let mut stale = 0;
                        for raw in take_positions(&mut index, &mut memory, rule.key) {
                            visited += 1;
                            if valid_pair::<UNCHECKED>(&corpus, raw as usize, rule).is_some() {
                                valid_positions.push(raw)
                            } else {
                                stale += 1
                            }
                        }
                        gathered_epoch = Some(epoch);
                        Ok(Reply::Gathered(GatheredReply {
                            valid_positions,
                            visited,
                            stale,
                            work_seconds: started.elapsed().as_secs_f64(),
                        }))
                    }
                }
                Command::PrepareSelf {
                    epoch,
                    rule,
                    positions,
                    start,
                    end,
                } => {
                    if gathered_epoch != Some(epoch) || planned_epoch.is_some() {
                        Err(TrainError::InternalInvariant("AA prepare without gather"))
                    } else {
                        let mut result = {
                            let guard = lengths.read().expect("length table poisoned");
                            prepare_self::<UNCHECKED, false>(
                                &corpus,
                                &guard,
                                &pivots,
                                &weights,
                                rule,
                                &positions,
                                start,
                                end,
                                &mut plans,
                                &mut new_edges,
                            )
                        };
                        if let Ok(reply) = &mut result {
                            router.route(
                                epoch,
                                worker_id,
                                std::mem::take(&mut reply.delta),
                                &delta_mailboxes,
                                &groups,
                                Some(&touched_owners),
                            );
                            planned_epoch = Some(epoch);
                            gathered_epoch = None;
                            plan_peak_bytes = plan_peak_bytes.max(reply.plan_capacity_bytes);
                            edge_peak_bytes = edge_peak_bytes
                                .max(reply.edge_capacity * std::mem::size_of::<NewEdge>());
                        }
                        result.map(Reply::Prepared)
                    }
                }
                Command::Reduce {
                    epoch,
                    first_new_id,
                    selected,
                } => {
                    let started = Instant::now();
                    scalars
                        .reduce(
                            &delta_mailboxes[worker_id],
                            epoch,
                            first_new_id,
                            selected,
                            minimum,
                            &prune_mailboxes,
                            &groups,
                        )
                        .map(
                            |(recipients, delta_keys, dropped, prune_entries)| Reply::Reduced {
                                top: scalars.top(minimum),
                                recipients,
                                delta_keys,
                                dropped,
                                prune_entries,
                                seconds: started.elapsed().as_secs_f64(),
                                scalar_capacity: scalars.frequencies.capacity(),
                                heap_capacity: scalars.heap.capacity(),
                            },
                        )
                }
                Command::Apply { epoch } => {
                    if epoch <= last_applied_epoch
                        || planned_epoch.is_some_and(|e| e != epoch)
                        || gathered_epoch.is_some()
                    {
                        Err(TrainError::InternalInvariant(
                            "position apply epoch mismatch",
                        ))
                    } else {
                        let mut keys = HashSet::new();
                        let mut mailbox = prune_mailboxes[worker_id]
                            .lock()
                            .expect("prune mailbox poisoned");
                        let mut bad_epoch = false;
                        for (from, key) in mailbox.drain(..) {
                            if from != epoch {
                                bad_epoch = true;
                            }
                            keys.insert(key);
                        }
                        drop(mailbox);
                        if bad_epoch {
                            Err(TrainError::InternalInvariant("stale prune epoch"))
                        } else {
                            let reply = apply_plans::<UNCHECKED, false>(
                                &corpus,
                                &mut index,
                                &mut memory,
                                &mut plans,
                                &mut new_edges,
                                &keys,
                            );
                            sample_memory(&index, &mut memory, &mut peak_memory);
                            planned_epoch = None;
                            last_applied_epoch = epoch;
                            Ok(Reply::Applied(reply))
                        }
                    }
                }
                Command::Finish => {
                    let _ = tx.send(Reply::Final(FinalReply {
                        memory,
                        peak_memory,
                        plan_peak_bytes,
                        edge_peak_bytes,
                        scalar_keys: scalars.frequencies.len(),
                        scalar_capacity: scalars.frequencies.capacity(),
                        heap_capacity: scalars.heap.capacity(),
                        router_capacity: router.capacity(),
                        heap_pops: scalars.heap_pops,
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
        .map_err(|_| TrainError::InternalInvariant("sparse worker disconnected"))?
    {
        Reply::Error(e) => Err(e),
        reply => Ok(reply),
    }
}

pub(super) fn train<const UNCHECKED: bool>(
    input: Prepared,
    options: Options,
) -> TrainResult<Result> {
    run::<UNCHECKED, false>(input, options)
}

pub(super) fn train_all<const UNCHECKED: bool>(
    input: Prepared,
    options: Options,
) -> TrainResult<Result> {
    run::<UNCHECKED, true>(input, options)
}

fn run<const UNCHECKED: bool, const ALL: bool>(
    input: Prepared,
    options: Options,
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
    let groups = Arc::new(WorkerGroups::new(worker_count));
    let all_groups = groups.all();
    let lengths = Arc::new(RwLock::new(initial_lengths));
    let pivots = Arc::new(pivots);
    let weights = Arc::new(weights);
    let corpus: Arc<Vec<AtomicU32>> = Arc::new(corpus.into_iter().map(AtomicU32::new).collect());
    let initial_mailboxes: Mailboxes = Arc::new(
        (0..worker_count)
            .map(|_| Mutex::new(EpochInbox::default()))
            .collect(),
    );
    let delta_mailboxes: Mailboxes = Arc::new(
        (0..worker_count)
            .map(|_| Mutex::new(EpochInbox::default()))
            .collect(),
    );
    let prune_mailboxes: PruneMailboxes =
        Arc::new((0..worker_count).map(|_| Mutex::new(Vec::new())).collect());
    let eligibles: EligibleSets = Arc::new((0..worker_count).map(|_| OnceLock::new()).collect());
    let touched_owners = Arc::new(AtomicU64::new(0));
    let workers: Vec<Worker> = (0..worker_count)
        .map(|id| {
            let start = 1 + id * (last - 1) / worker_count;
            let end = 1 + (id + 1) * (last - 1) / worker_count;
            spawn_worker::<UNCHECKED>(
                id,
                Arc::clone(&corpus),
                Arc::clone(&pivots),
                Arc::clone(&weights),
                Arc::clone(&lengths),
                Arc::clone(&groups),
                Arc::clone(&initial_mailboxes),
                Arc::clone(&delta_mailboxes),
                Arc::clone(&prune_mailboxes),
                Arc::clone(&eligibles),
                Arc::clone(&touched_owners),
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
    let initial_mailbox_capacity: usize = initial_mailboxes
        .iter()
        .map(|m| {
            m.lock()
                .expect("initial mailbox poisoned")
                .records
                .capacity()
        })
        .sum();
    for worker in &workers {
        worker
            .commands
            .send(Command::BuildScalars)
            .map_err(|_| TrainError::InternalInvariant("scalar init send failed"))?;
    }
    let mut frontier = Frontier::new(worker_count);
    let mut initial_scalar_keys = 0;
    let mut initial_scalar_capacity = 0;
    let mut initial_heap_capacity = 0;
    let mut initial_eligible_capacity = 0;
    let mut owner_scalar_capacities = vec![0; worker_count];
    let mut owner_heap_capacities = vec![0; worker_count];
    for (id, worker) in workers.iter().enumerate() {
        match recv(worker)? {
            Reply::Scalars {
                top,
                keys,
                capacity,
                heap_capacity,
                eligible_capacity,
            } => {
                frontier.replace(id, top);
                initial_scalar_keys += keys;
                initial_scalar_capacity += capacity;
                initial_heap_capacity += heap_capacity;
                initial_eligible_capacity += eligible_capacity;
                owner_scalar_capacities[id] = capacity;
                owner_heap_capacities[id] = heap_capacity;
            }
            _ => return Err(TrainError::InternalInvariant("expected scalar init reply")),
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
        let Reply::Built(m) = recv(worker)? else {
            return Err(TrainError::InternalInvariant("expected index init reply"));
        };
        initial_index_memory.position_len += m.position_len;
        initial_index_memory.position_capacity += m.position_capacity;
        initial_index_memory.map_len += m.map_len;
        initial_index_memory.map_capacity += m.map_capacity;
    }
    drop(eligibles);
    let init_seconds = started.elapsed().as_secs_f64();
    let merge_started = Instant::now();
    let mut merges = Vec::new();
    let mut actual_merges = 0;
    let mut position_visits = 0;
    let mut stale_visits = 0;
    let mut aa_epochs = 0;
    let mut aa_gather_positions = 0;
    let mut select_seconds = 0.0;
    let mut plan_seconds = 0.0;
    let mut owner_reduce_seconds = 0.0;
    let mut apply_seconds = 0.0;
    let mut active_plan_workers_total = 0;
    let mut active_owner_workers_total = 0;
    let mut active_apply_workers_total = 0;
    let mut max_active_plan_workers = 0;
    let mut max_active_owner_workers = 0;
    let mut owner_delta_keys_total = 0;
    let mut worker_delta_keys_total = 0;
    let mut retired_keys_total = 0;
    let mut prune_entries_total = 0;
    let mut plan_capacity_peak = 0;
    let mut edge_capacity_peak = 0;
    let mut active_plan_capacity_peak = 0;
    let mut active_edge_capacity_peak = 0;
    let mut plan_capacities = vec![0; worker_count];
    let mut edge_capacities = vec![0; worker_count];
    let mut plan_capacity_total = 0;
    let mut edge_capacity_total = 0;
    let mut owner_scalar_capacity_total = initial_scalar_capacity;
    let mut owner_heap_capacity_total = initial_heap_capacity;
    let mut owner_scalar_capacity_peak = owner_scalar_capacity_total;
    let mut owner_heap_capacity_peak = owner_heap_capacity_total;
    let mut messages = 6 * worker_count;
    while merges.len() < options.max_merges {
        let select_started = Instant::now();
        let Some((selected_owner, head)) = frontier.best() else {
            break;
        };
        if head.frequency < options.min_frequency {
            return Err(TrainError::InternalInvariant(
                "cached owner top below minimum",
            ));
        }
        if owner_pair(head.key, worker_count) != selected_owner || head.holders == 0 {
            return Err(TrainError::InternalInvariant("cached owner head malformed"));
        }
        let epoch = (merges.len() + 1) as u64;
        let a = (head.key >> 32) as u32;
        let b = head.key as u32;
        let rule = {
            let mut table = lengths.write().expect("length table poisoned");
            let new_id = u32::try_from(table.len())
                .map_err(|_| TrainError::Overflow("fresh token ID exceeds u32"))?;
            let a_length = table[a as usize] as usize;
            let b_length = table[b as usize] as usize;
            let new_length = table[a as usize]
                .checked_add(table[b as usize])
                .ok_or(TrainError::Overflow("merged token length exceeds u32"))?;
            table.push(new_length);
            RuleSpec {
                key: head.key,
                a,
                b,
                new_id,
                new_length,
                a_length,
                b_length,
                frequency: head.frequency,
            }
        };
        select_seconds += select_started.elapsed().as_secs_f64();
        let plan_started = Instant::now();
        let holder_workers: Vec<usize> = groups.members(head.holders).collect();
        let plan_mask = if ALL { all_groups } else { head.holders };
        let plan_workers: Vec<usize> = groups.members(plan_mask).collect();
        if plan_workers.is_empty() {
            return Err(TrainError::InternalInvariant(
                "selected pair has no holders",
            ));
        }
        active_plan_workers_total += plan_workers.len();
        max_active_plan_workers = max_active_plan_workers.max(plan_workers.len());
        let mut gathered_visits = 0;
        let mut gathered_stale = 0;
        if a == b {
            aa_epochs += 1;
            for &id in &plan_workers {
                workers[id]
                    .commands
                    .send(Command::GatherSelf { epoch, rule })
                    .map_err(|_| TrainError::InternalInvariant("AA gather send failed"))?;
            }
            messages += 2 * plan_workers.len();
            let mut valid = Vec::new();
            for &id in &plan_workers {
                let Reply::Gathered(reply) = recv(&workers[id])? else {
                    return Err(TrainError::InternalInvariant("expected AA gather reply"));
                };
                gathered_visits += reply.visited;
                gathered_stale += reply.stale;
                valid.extend(reply.valid_positions);
                let _ = reply.work_seconds;
            }
            aa_gather_positions += valid.len();
            valid.sort_unstable();
            if valid.windows(2).any(|x| x[0] == x[1]) {
                return Err(TrainError::InternalInvariant(
                    "duplicate AA live occurrence",
                ));
            }
            let mut selected = Vec::new();
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
            for &id in &plan_workers {
                let (start, end) = if let Ok(slot) = holder_workers.binary_search(&id) {
                    (
                        slot * selected.len() / holder_workers.len(),
                        (slot + 1) * selected.len() / holder_workers.len(),
                    )
                } else {
                    (0, 0)
                };
                workers[id]
                    .commands
                    .send(Command::PrepareSelf {
                        epoch,
                        rule,
                        positions: Arc::clone(&selected),
                        start,
                        end,
                    })
                    .map_err(|_| TrainError::InternalInvariant("AA prepare send failed"))?;
            }
        } else {
            for &id in &plan_workers {
                workers[id]
                    .commands
                    .send(Command::Prepare { epoch, rule })
                    .map_err(|_| TrainError::InternalInvariant("prepare send failed"))?;
            }
        }
        messages += 2 * plan_workers.len();
        position_visits += gathered_visits;
        stale_visits += gathered_stale;
        let mut planned = 0;
        let mut plan_capacity = 0;
        let mut edge_capacity = 0;
        for &id in &plan_workers {
            let Reply::Prepared(reply) = recv(&workers[id])? else {
                return Err(TrainError::InternalInvariant("expected prepared reply"));
            };
            position_visits += reply.visited;
            stale_visits += reply.stale;
            planned += reply.valid;
            worker_delta_keys_total += reply.delta_keys;
            let new_plan_capacity = reply.plan_capacity_bytes;
            let new_edge_capacity = reply.edge_capacity * std::mem::size_of::<NewEdge>();
            plan_capacity += new_plan_capacity;
            edge_capacity += new_edge_capacity;
            plan_capacity_total = plan_capacity_total - plan_capacities[id] + new_plan_capacity;
            edge_capacity_total = edge_capacity_total - edge_capacities[id] + new_edge_capacity;
            plan_capacities[id] = new_plan_capacity;
            edge_capacities[id] = new_edge_capacity;
            let _ = reply.work_seconds;
        }
        if planned == 0 {
            return Err(TrainError::InternalInvariant(
                "selected pair had no valid matches",
            ));
        }
        plan_capacity_peak = plan_capacity_peak.max(plan_capacity_total);
        edge_capacity_peak = edge_capacity_peak.max(edge_capacity_total);
        active_plan_capacity_peak = active_plan_capacity_peak.max(plan_capacity);
        active_edge_capacity_peak = active_edge_capacity_peak.max(edge_capacity);
        plan_seconds += plan_started.elapsed().as_secs_f64();
        let reduce_started = Instant::now();
        let touched = touched_owners.swap(0, Ordering::SeqCst) | groups.bit(selected_owner);
        let owner_mask = if ALL { all_groups } else { touched };
        let owner_workers: Vec<usize> = groups.members(owner_mask).collect();
        active_owner_workers_total += owner_workers.len();
        max_active_owner_workers = max_active_owner_workers.max(owner_workers.len());
        for &id in &owner_workers {
            workers[id]
                .commands
                .send(Command::Reduce {
                    epoch,
                    first_new_id: rule.new_id,
                    selected: (id == selected_owner).then_some(rule),
                })
                .map_err(|_| TrainError::InternalInvariant("owner reduce send failed"))?;
        }
        messages += 2 * owner_workers.len();
        let mut recipients = 0;
        for &id in &owner_workers {
            let Reply::Reduced {
                top,
                recipients: mask,
                delta_keys,
                dropped,
                prune_entries,
                seconds,
                scalar_capacity,
                heap_capacity,
            } = recv(&workers[id])?
            else {
                return Err(TrainError::InternalInvariant(
                    "expected owner reduction reply",
                ));
            };
            frontier.replace(id, top);
            recipients |= mask;
            owner_delta_keys_total += delta_keys;
            retired_keys_total += dropped;
            prune_entries_total += prune_entries;
            owner_scalar_capacity_total =
                owner_scalar_capacity_total - owner_scalar_capacities[id] + scalar_capacity;
            owner_heap_capacity_total =
                owner_heap_capacity_total - owner_heap_capacities[id] + heap_capacity;
            owner_scalar_capacities[id] = scalar_capacity;
            owner_heap_capacities[id] = heap_capacity;
            let _ = seconds;
        }
        owner_scalar_capacity_peak = owner_scalar_capacity_peak.max(owner_scalar_capacity_total);
        owner_heap_capacity_peak = owner_heap_capacity_peak.max(owner_heap_capacity_total);
        owner_reduce_seconds += reduce_started.elapsed().as_secs_f64();
        let apply_started = Instant::now();
        let apply_mask = if ALL {
            all_groups
        } else {
            head.holders | recipients
        };
        let apply_workers: Vec<usize> = groups.members(apply_mask).collect();
        active_apply_workers_total += apply_workers.len();
        for &id in &apply_workers {
            workers[id]
                .commands
                .send(Command::Apply { epoch })
                .map_err(|_| TrainError::InternalInvariant("apply send failed"))?;
        }
        messages += 2 * apply_workers.len();
        let mut applied = 0;
        for &id in &apply_workers {
            let Reply::Applied(reply) = recv(&workers[id])? else {
                return Err(TrainError::InternalInvariant("expected apply reply"));
            };
            applied += reply.merges;
            let _ = (reply.work_seconds, reply.pruned);
        }
        if applied != planned {
            return Err(TrainError::InternalInvariant("plan/apply merge mismatch"));
        }
        actual_merges += applied;
        merges.push(Rule {
            left: a,
            right: b,
            frequency: rule.frequency,
        });
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
    let mut peak_memory = IndexMemory::default();
    let mut final_scalar_keys = 0;
    let mut final_scalar_capacity = 0;
    let mut final_heap_capacity = 0;
    let mut router_capacity = 0;
    let mut worker_plan_peak = 0;
    let mut worker_edge_peak = 0;
    let mut heap_pops = 0;
    for worker in &workers {
        let Reply::Final(reply) = recv(worker)? else {
            return Err(TrainError::InternalInvariant("expected final reply"));
        };
        final_memory.position_len += reply.memory.position_len;
        final_memory.position_capacity += reply.memory.position_capacity;
        final_memory.map_len += reply.memory.map_len;
        final_memory.map_capacity += reply.memory.map_capacity;
        peak_memory.position_len += reply.peak_memory.position_len;
        peak_memory.position_capacity += reply.peak_memory.position_capacity;
        peak_memory.map_len += reply.peak_memory.map_len;
        peak_memory.map_capacity += reply.peak_memory.map_capacity;
        final_scalar_keys += reply.scalar_keys;
        final_scalar_capacity += reply.scalar_capacity;
        final_heap_capacity += reply.heap_capacity;
        router_capacity += reply.router_capacity;
        worker_plan_peak += reply.plan_peak_bytes;
        worker_edge_peak += reply.edge_peak_bytes;
        heap_pops += reply.heap_pops;
    }
    messages += 2 * worker_count;
    drop(workers);
    let mut final_tokens = Vec::new();
    let mut pos = 0;
    let table = lengths.read().expect("length table poisoned");
    loop {
        let token = load::<UNCHECKED>(&corpus, pos);
        final_tokens.push(token);
        if pos == last {
            break;
        }
        pos = pos
            .checked_add(table[token as usize] as usize)
            .ok_or(TrainError::InternalInvariant("final boundary overflow"))?;
        if pos > last {
            return Err(TrainError::InternalInvariant(
                "final boundary outside corpus",
            ));
        }
    }
    let max_token_length = table.iter().copied().max().unwrap_or(1);
    let shared_length_capacity_bytes = table.capacity() * std::mem::size_of::<u32>();
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
        "force_all_dispatch",
        if ALL { 1.0 } else { 0.0 },
    );
    metric(
        &mut metrics,
        "shared_corpus_bytes",
        (corpus_positions * 4) as f64,
    );
    metric(
        &mut metrics,
        "shared_length_capacity_bytes",
        shared_length_capacity_bytes as f64,
    );
    metric(
        &mut metrics,
        "owner_scalar_record_bytes",
        std::mem::size_of::<PairScalar>() as f64,
    );
    metric(
        &mut metrics,
        "delta_record_bytes",
        std::mem::size_of::<Delta>() as f64,
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
        "initial_scalar_capacity_sum",
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
        "final_index_map_capacity_sum",
        final_memory.map_capacity as f64,
    );
    metric(
        &mut metrics,
        "index_position_capacity_peak_bytes_sum",
        (peak_memory.position_capacity * 4) as f64,
    );
    metric(&mut metrics, "final_scalar_keys", final_scalar_keys as f64);
    metric(
        &mut metrics,
        "final_scalar_capacity_sum",
        final_scalar_capacity as f64,
    );
    metric(
        &mut metrics,
        "final_heap_capacity_sum",
        final_heap_capacity as f64,
    );
    metric(
        &mut metrics,
        "router_capacity_entries_sum",
        router_capacity as f64,
    );
    metric(
        &mut metrics,
        "plan_capacity_peak_bytes",
        plan_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "active_plan_capacity_peak_bytes",
        active_plan_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "edge_capacity_peak_bytes",
        edge_capacity_peak as f64,
    );
    metric(
        &mut metrics,
        "active_edge_capacity_peak_bytes",
        active_edge_capacity_peak as f64,
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
        "worker_plan_peak_bytes_sum",
        worker_plan_peak as f64,
    );
    metric(
        &mut metrics,
        "worker_edge_peak_bytes_sum",
        worker_edge_peak as f64,
    );
    metric(
        &mut metrics,
        "frontier_capacity_bytes",
        frontier.capacity_bytes() as f64,
    );
    metric(&mut metrics, "rules", core.rules as f64);
    metric(&mut metrics, "aa_epochs", aa_epochs as f64);
    metric(
        &mut metrics,
        "aa_gather_positions",
        aa_gather_positions as f64,
    );
    metric(
        &mut metrics,
        "active_plan_workers_total",
        active_plan_workers_total as f64,
    );
    metric(
        &mut metrics,
        "active_owner_workers_total",
        active_owner_workers_total as f64,
    );
    metric(
        &mut metrics,
        "active_apply_workers_total",
        active_apply_workers_total as f64,
    );
    metric(
        &mut metrics,
        "max_active_plan_workers",
        max_active_plan_workers as f64,
    );
    metric(
        &mut metrics,
        "max_active_owner_workers",
        max_active_owner_workers as f64,
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
        "retired_keys_total",
        retired_keys_total as f64,
    );
    metric(&mut metrics, "prune_entries", prune_entries_total as f64);
    metric(
        &mut metrics,
        "prune_mailbox_locks",
        prune_entries_total as f64,
    );
    metric(&mut metrics, "select_seconds", select_seconds);
    metric(&mut metrics, "plan_seconds", plan_seconds);
    metric(&mut metrics, "owner_reduce_seconds", owner_reduce_seconds);
    metric(&mut metrics, "apply_seconds", apply_seconds);
    metric(&mut metrics, "round_messages", messages as f64);
    metric(
        &mut metrics,
        "unchecked_corpus_access",
        if UNCHECKED { 1.0 } else { 0.0 },
    );
    metric(
        &mut metrics,
        "grouped_holder_mask",
        if worker_count > 64 { 1.0 } else { 0.0 },
    );
    let _ = (delta_mailboxes, prune_mailboxes, touched_owners);
    Ok(Result { core, metrics })
}

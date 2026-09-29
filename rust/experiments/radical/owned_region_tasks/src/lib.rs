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
mod snapshot;

use small_posting::SmallPosting;
use snapshot::{BoundaryState, DeferredStore, RegionAccess};

type Result<T> = std::result::Result<T, TrainError>;
const HEAD: u32 = 1 << 31;
const ID_MASK: u32 = HEAD - 1;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub workers: usize,
    pub regions_per_worker: usize,
    pub chunk_size: usize,
    pub heap_policy: HeapPolicy,
    pub integer_hash: IntegerHash,
    pub endpoint_plan: EndpointPlan,
    pub region_mode: RegionMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionMode {
    Dynamic,
    Region,
    Snapshot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndpointPlan {
    TwoPass,
    TaggedTwoPass,
    TaggedFused,
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
    pub fused_non_aa_merges: usize,
    pub fused_non_aa_batches: usize,
    pub non_aa_start_positions_peak: usize,
    pub non_aa_start_bytes_peak_proxy: usize,
    pub decoder_zero_rereads: usize,
    pub endpoint_domain_fallback: bool,
    pub region_partition_searches: usize,
    pub region_partition_worker_seconds: f64,
    pub region_posting_visits: usize,
    pub region_valid_merges: usize,
    pub region_non_aa_posting_visits: usize,
    pub region_non_aa_valid_merges: usize,
    pub region_cross_births: usize,
    pub region_peak_route_delta_capacity: usize,
    pub region_peak_route_born_capacity: usize,
    pub region_aa_regroup_seconds: f64,
    pub region_aa_regroup_capacity_upper_bytes: usize,
    pub region_max_visits_per_batch: usize,
    pub region_max_merges_per_batch: usize,
    pub region_sum_max_visits: usize,
    pub region_sum_max_merges: usize,
    pub region_count_effective: usize,
    pub region_peak_route_header_capacity_bytes: usize,
    pub region_sum_visit_makespan_lower_bound: usize,
    pub region_sum_merge_makespan_lower_bound: usize,
    pub snapshot_build_seconds: f64,
    pub snapshot_refresh_seconds: f64,
    pub snapshot_cut_count: usize,
    pub snapshot_peak_descriptor_capacity_bytes: usize,
    pub snapshot_local_reads: usize,
    pub snapshot_local_writes: usize,
    pub snapshot_boundary_queries: usize,
    pub snapshot_deferred_stores: usize,
    pub snapshot_peak_deferred_len: usize,
    pub snapshot_peak_deferred_capacity_upper: usize,
    pub snapshot_deferred_apply_seconds: f64,
}

#[derive(Debug)]
pub struct Output {
    pub rules: Vec<Rule>,
    pub final_tokens: Vec<u32>,
    pub metrics: Metrics,
    pub effective_endpoint_plan: EndpointPlan,
    pub effective_region_mode: RegionMode,
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
    zero_rereads: usize,
}

struct RegionCuts {
    cuts: Vec<usize>,
}

impl RegionCuts {
    fn new(positions: usize, regions: usize) -> Result<Self> {
        let capacity = regions
            .checked_add(1)
            .ok_or(TrainError::Overflow("region boundary count exceeds usize"))?;
        let mut cuts = Vec::new();
        cuts.try_reserve_exact(capacity)
            .map_err(|_| TrainError::InvalidInput("cannot allocate region boundaries"))?;
        for region in 0..=regions {
            cuts.push(((positions as u128 * region as u128) / regions as u128) as usize);
        }
        Ok(Self { cuts })
    }

    fn count(&self) -> usize {
        self.cuts.len() - 1
    }

    fn of(&self, pos: usize) -> usize {
        debug_assert!(pos < *self.cuts.last().unwrap());
        self.cuts.partition_point(|&cut| cut <= pos) - 1
    }

    fn bounds(&self, region: usize) -> (usize, usize) {
        (self.cuts[region], self.cuts[region + 1])
    }
}

#[derive(Clone, Copy)]
struct CrossBirth {
    target_region: usize,
    pair: u64,
    pos: u32,
    weight: u64,
}

struct RegionWork<H: HashBuild> {
    output: WorkerOutput<H>,
    cross_births: Vec<CrossBirth>,
    partition_seconds: f64,
    searches: usize,
}

struct SnapshotRegionWork<H: HashBuild> {
    output: WorkerOutput<H>,
    cross_births: Vec<CrossBirth>,
    deferred: Vec<DeferredStore>,
    partition_seconds: f64,
    searches: usize,
    local_reads: usize,
    local_writes: usize,
    boundary_queries: usize,
}

#[inline]
fn key(a: u32, b: u32) -> u64 {
    (u64::from(a) << 32) | u64::from(b)
}

#[inline]
fn read<const TAGGED: bool>(corpus: &[AtomicU32], pos: usize) -> u32 {
    let raw = corpus[pos].load(Ordering::Relaxed);
    if TAGGED { raw & ID_MASK } else { raw }
}

fn inspect<const TAGGED: bool>(
    corpus: &[AtomicU32],
    lengths: &[u32],
    pos: usize,
    a: u32,
    b: u32,
) -> Option<Plan> {
    let last = corpus.len() - 1;
    if pos == 0 || pos >= last || read::<TAGGED>(corpus, pos) != a {
        return None;
    }
    let right = pos.checked_add(lengths[a as usize] as usize)?;
    if right >= last || read::<TAGGED>(corpus, right) != b {
        return None;
    }
    let after = right.checked_add(lengths[b as usize] as usize)?;
    if after > last {
        return None;
    }
    let prior_length = lengths[read::<TAGGED>(corpus, pos - 1) as usize] as usize;
    let before = pos.checked_sub(prior_length)?;
    Some(Plan {
        pos: pos as u32,
        right: right as u32,
        after: after as u32,
        before: before as u32,
        left_id: read::<TAGGED>(corpus, before),
        right_id: read::<TAGGED>(corpus, after),
        weight: 0,
    })
}

struct FusedPlan {
    plan: Plan,
    left_selected: bool,
    final_right: u32,
    zero_rereads: usize,
}

#[inline]
fn batch_rule_for_raw(raw: u32, batch: &[BatchRule], fresh_begin: u32) -> Result<&BatchRule> {
    let rank = (raw & ID_MASK)
        .checked_sub(fresh_begin)
        .ok_or(TrainError::InternalInvariant("fresh ID before batch start"))?
        as usize;
    batch
        .get(rank)
        .ok_or(TrainError::InternalInvariant("fresh ID outside batch"))
}

/// `raw` is at the end of a token that existed at batch start. Such a cell
/// cannot be cleared. A current-batch head can occur there only if old a was
/// length one; a current-batch bare tail represents old b.
#[inline]
fn old_end_id(raw: u32, batch: &[BatchRule], fresh_begin: u32) -> Result<u32> {
    let id = raw & ID_MASK;
    if id < fresh_begin {
        return Ok(id);
    }
    let rule = batch_rule_for_raw(raw, batch, fresh_begin)?;
    if raw & HEAD != 0 {
        if rule.a_length != 1 {
            return Err(TrainError::InternalInvariant(
                "fresh head at long old token end",
            ));
        }
        Ok(rule.a)
    } else {
        Ok(rule.b)
    }
}

/// `u` is the known batch-start successor of old C. Fresh HEAD means D is
/// the left constituent of another merge; fresh bare tail means (C,D) merged
/// and D had length one. This is intentionally not a general-position reader.
#[inline]
fn old_next_id(raw: u32, c: u32, batch: &[BatchRule], fresh_begin: u32) -> Result<u32> {
    let id = raw & ID_MASK;
    if id < fresh_begin {
        return Ok(id);
    }
    let rule = batch_rule_for_raw(raw, batch, fresh_begin)?;
    if raw & HEAD != 0 {
        Ok(rule.a)
    } else if rule.a == c && rule.b_length == 1 {
        Ok(rule.b)
    } else {
        Err(TrainError::InternalInvariant(
            "unexpected fresh tail at right successor",
        ))
    }
}

/// Only for a certified non-AA batch. Every actual selected match owns
/// disjoint old token spans; other tasks can change boundary cells but cannot
/// change this occurrence's old a/b starts before its own publication.
fn inspect_fused<H: HashBuild>(
    corpus: &[AtomicU32],
    lengths: &[u32],
    pos: usize,
    rule: &BatchRule,
    batch: &[BatchRule],
    selected: &HashMap<u64, u32, H>,
) -> Result<Option<FusedPlan>> {
    let last = corpus.len() - 1;
    if pos == 0 || pos >= last || (corpus[pos].load(Ordering::Acquire) & ID_MASK) != rule.a {
        return Ok(None);
    }
    let Some(right) = pos.checked_add(rule.a_length) else {
        return Ok(None);
    };
    if right >= last || (corpus[right].load(Ordering::Acquire) & ID_MASK) != rule.b {
        return Ok(None);
    }
    let Some(after) = right.checked_add(rule.b_length) else {
        return Ok(None);
    };
    if after > last {
        return Ok(None);
    }
    let fresh_begin = batch[0].new_id;
    let left_id = old_end_id(corpus[pos - 1].load(Ordering::Acquire), batch, fresh_begin)?;
    let before = pos.checked_sub(lengths[left_id as usize] as usize).ok_or(
        TrainError::InternalInvariant("left old token starts before corpus"),
    )?;
    let left_selected = if left_id == 0 {
        false
    } else {
        let prior_end = before.checked_sub(1).ok_or(TrainError::InternalInvariant(
            "left old token crosses sentinel",
        ))?;
        let prior_id = old_end_id(
            corpus[prior_end].load(Ordering::Acquire),
            batch,
            fresh_begin,
        )?;
        prior_id != 0 && selected.contains_key(&key(prior_id, left_id))
    };

    let t_raw = corpus[after].load(Ordering::Acquire);
    let t_id = t_raw & ID_MASK;
    let mut zero_rereads = 0;
    let (right_id, final_right) = if t_id >= fresh_begin {
        if t_raw & HEAD == 0 {
            return Err(TrainError::InternalInvariant(
                "fresh tail at right token start",
            ));
        }
        let next_rule = batch_rule_for_raw(t_raw, batch, fresh_begin)?;
        (next_rule.a, next_rule.new_id)
    } else if t_id == 0 {
        (0, 0)
    } else {
        let c = t_id;
        let u = after.checked_add(lengths[c as usize] as usize).ok_or(
            TrainError::InternalInvariant("right old token boundary overflows"),
        )?;
        if u > last {
            return Err(TrainError::InternalInvariant(
                "right old token crosses corpus end",
            ));
        }
        let u_raw = corpus[u].load(Ordering::Acquire);
        if u_raw != 0 {
            let d = old_next_id(u_raw, c, batch, fresh_begin)?;
            (c, selected.get(&key(c, d)).copied().unwrap_or(c))
        } else {
            // A zero may be the original sentinel after C or step 2 of a
            // concurrent (C,D) merge. Acquire of that clear synchronizes
            // with its Release store, so the earlier head must be visible
            // to this Acquire reread of t (which has only one batch writer).
            zero_rereads = 1;
            let reread = corpus[after].load(Ordering::Acquire);
            let reread_id = reread & ID_MASK;
            if reread_id >= fresh_begin {
                if reread & HEAD == 0 {
                    return Err(TrainError::InternalInvariant("fresh tail after right zero"));
                }
                let next_rule = batch_rule_for_raw(reread, batch, fresh_begin)?;
                if next_rule.a != c {
                    return Err(TrainError::InternalInvariant("right old ID changed"));
                }
                (c, next_rule.new_id)
            } else if reread_id == c {
                // u is a sentinel after C; C remains the final right neighbor.
                (c, c)
            } else {
                return Err(TrainError::InternalInvariant("right old ID disappeared"));
            }
        }
    };
    Ok(Some(FusedPlan {
        plan: Plan {
            pos: pos as u32,
            right: right as u32,
            after: after as u32,
            before: before as u32,
            left_id,
            right_id,
            weight: 0,
        },
        left_selected,
        final_right,
        zero_rereads,
    }))
}

/// Exclusive-region counterpart to inspect_fused. All local cells are read
/// through get_mut; every remote old head/tail must be present in the
/// batch-start window adjacent to this region.
fn inspect_fused_snapshot<H: HashBuild>(
    access: &mut RegionAccess<'_>,
    corpus_len: usize,
    lengths: &[u32],
    pos: usize,
    rule: &BatchRule,
    batch: &[BatchRule],
    selected: &HashMap<u64, u32, H>,
) -> Result<Option<FusedPlan>> {
    let last = corpus_len - 1;
    if pos == 0 || pos >= last || (access.head_raw(pos)? & ID_MASK) != rule.a {
        return Ok(None);
    }
    let Some(right) = pos.checked_add(rule.a_length) else {
        return Ok(None);
    };
    if right >= last || (access.head_raw(right)? & ID_MASK) != rule.b {
        return Ok(None);
    }
    let Some(after) = right.checked_add(rule.b_length) else {
        return Ok(None);
    };
    if after > last {
        return Ok(None);
    }
    let fresh_begin = batch[0].new_id;
    let left_id = old_end_id(access.tail_raw(pos - 1)?, batch, fresh_begin)?;
    let before = pos.checked_sub(lengths[left_id as usize] as usize).ok_or(
        TrainError::InternalInvariant("left old token starts before corpus"),
    )?;
    let left_selected = if left_id == 0 {
        false
    } else {
        let prior_end = before.checked_sub(1).ok_or(TrainError::InternalInvariant(
            "left old token crosses sentinel",
        ))?;
        let prior_id = old_end_id(access.tail_raw(prior_end)?, batch, fresh_begin)?;
        prior_id != 0 && selected.contains_key(&key(prior_id, left_id))
    };

    let t_raw = access.head_raw(after)?;
    let t_id = t_raw & ID_MASK;
    let (right_id, final_right) = if t_id >= fresh_begin {
        if t_raw & HEAD == 0 {
            return Err(TrainError::InternalInvariant(
                "fresh tail at snapshot right token start",
            ));
        }
        let next_rule = batch_rule_for_raw(t_raw, batch, fresh_begin)?;
        (next_rule.a, next_rule.new_id)
    } else if t_id == 0 {
        (0, 0)
    } else {
        let c = t_id;
        let u = after.checked_add(lengths[c as usize] as usize).ok_or(
            TrainError::InternalInvariant("right old token boundary overflows"),
        )?;
        if u > last {
            return Err(TrainError::InternalInvariant(
                "right old token crosses corpus end",
            ));
        }
        let u_raw = access.head_raw(u)?;
        if u_raw != 0 {
            let d = old_next_id(u_raw, c, batch, fresh_begin)?;
            (c, selected.get(&key(c, d)).copied().unwrap_or(c))
        } else {
            // The one region writer cannot change t/u between reads. A
            // remote u is the batch-start snapshot. Therefore an old t
            // followed by zero u denotes a stable piece sentinel.
            (c, c)
        }
    };
    Ok(Some(FusedPlan {
        plan: Plan {
            pos: pos as u32,
            right: right as u32,
            after: after as u32,
            before: before as u32,
            left_id,
            right_id,
            weight: 0,
        },
        left_selected,
        final_right,
        zero_rereads: 0,
    }))
}

fn write_merge_snapshot(access: &mut RegionAccess<'_>, plan: Plan, b_length: usize, new_id: u32) {
    access.put(plan.pos as usize, new_id | HEAD);
    if b_length == 1 {
        access.put(plan.right as usize, new_id);
    } else {
        access.put(plan.right as usize, 0);
        access.put(plan.after as usize - 1, new_id);
    }
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

fn write_merge<const TAGGED: bool>(corpus: &[AtomicU32], plan: Plan, b_length: usize, new_id: u32) {
    let pos = plan.pos as usize;
    let right = plan.right as usize;
    let after = plan.after as usize;
    let order = if TAGGED {
        Ordering::Release
    } else {
        Ordering::Relaxed
    };
    let head = if TAGGED { new_id | HEAD } else { new_id };
    corpus[pos].store(head, order);
    if b_length == 1 {
        corpus[right].store(new_id, order);
    } else {
        corpus[right].store(0, order);
        corpus[after - 1].store(new_id, order);
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
        zero_rereads: 0,
    }
}

fn debug_assert_region_order(posting: &[u32], cuts: &RegionCuts) {
    debug_assert!(
        posting
            .windows(2)
            .all(|edge| cuts.of(edge[0] as usize) <= cuts.of(edge[1] as usize))
    );
}

fn record_region_load<H: HashBuild>(
    outputs: &[WorkerOutput<H>],
    workers: usize,
    non_aa_region_tasks: bool,
    metrics: &mut Metrics,
) {
    let total_visits: usize = outputs.iter().map(|output| output.visits).sum();
    let total_merges: usize = outputs.iter().map(|output| output.merges).sum();
    let max_visits = outputs
        .iter()
        .map(|output| output.visits)
        .max()
        .unwrap_or(0);
    let max_merges = outputs
        .iter()
        .map(|output| output.merges)
        .max()
        .unwrap_or(0);
    metrics.region_max_visits_per_batch = metrics.region_max_visits_per_batch.max(max_visits);
    metrics.region_max_merges_per_batch = metrics.region_max_merges_per_batch.max(max_merges);
    metrics.region_sum_max_visits += max_visits;
    metrics.region_sum_max_merges += max_merges;
    if non_aa_region_tasks {
        metrics.region_non_aa_posting_visits += total_visits;
        metrics.region_non_aa_valid_merges += total_merges;
        metrics.region_sum_visit_makespan_lower_bound +=
            total_visits.div_ceil(workers).max(max_visits);
        metrics.region_sum_merge_makespan_lower_bound +=
            total_merges.div_ceil(workers).max(max_merges);
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
fn initial_index<H: HashBuild, const TAGGED: bool>(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    pivots: &[u32],
    weights: &[u64],
    workers: usize,
    chunk_size: usize,
    minimum: u64,
    regions: Option<&RegionCuts>,
    metrics: &mut Metrics,
) -> Result<Vec<Owner<H>>> {
    let started = Instant::now();
    let last = corpus.len() - 1;
    let positions = last.saturating_sub(1);
    let task_count = positions.div_ceil(chunk_size);
    let cursor = AtomicUsize::new(0);
    let outputs = if let Some(regions) = regions {
        pool.install(|| {
            (0..regions.count())
                .into_par_iter()
                .map(|region| {
                    let mut positions_by_owner = vec![Vec::<u32>::new(); workers];
                    let (lower, upper) = regions.bounds(region);
                    for pos in lower.max(1)..upper.min(last) {
                        let a = read::<TAGGED>(corpus, pos);
                        let b = read::<TAGGED>(corpus, pos + 1);
                        if a != 0 && b != 0 {
                            positions_by_owner[owner_for(key(a, b), workers)].push(pos as u32);
                        }
                    }
                    positions_by_owner
                })
                .collect::<Vec<_>>()
        })
    } else {
        pool.install(|| {
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
                            let a = read::<TAGGED>(corpus, pos);
                            let b = read::<TAGGED>(corpus, pos + 1);
                            if a != 0 && b != 0 {
                                positions_by_owner[owner_for(key(a, b), workers)].push(pos as u32);
                            }
                        }
                    }
                    positions_by_owner
                })
                .collect::<Vec<_>>()
        })
    };
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
                        let a = read::<TAGGED>(corpus, pos as usize);
                        let b = read::<TAGGED>(corpus, pos as usize + 1);
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
                if let Some(cuts) = regions {
                    for entry in owner.entries.values() {
                        debug_assert_region_order(entry.positions.as_slice(), cuts);
                    }
                }
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

fn aa_plans_by_region(chunks: Vec<Vec<Plan>>, cuts: &RegionCuts) -> Vec<Vec<Plan>> {
    let mut regions = vec![Vec::new(); cuts.count()];
    for chunk in chunks {
        for plan in chunk {
            regions[cuts.of(plan.pos as usize)].push(plan);
        }
    }
    regions
}

#[allow(clippy::too_many_arguments)]
fn route_aa_region<H: HashBuild>(
    pool: &ThreadPool,
    chunks: &[Vec<Plan>],
    cuts: &RegionCuts,
    a: u32,
    b: u32,
    new_id: u32,
    workers: usize,
    metrics: &mut Metrics,
) -> Result<(Vec<WorkerOutput<H>>, Vec<CrossBirth>)> {
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
    let results = pool.install(|| {
        chunks
            .par_iter()
            .enumerate()
            .map(
                |(region, chunk)| -> Result<(WorkerOutput<H>, Vec<CrossBirth>)> {
                    let mut output = empty_worker(workers);
                    let mut cross_births = Vec::new();
                    let (lower, _) = cuts.bounds(region);
                    for (local_i, &plan) in chunk.iter().enumerate() {
                        let previous_selected = if local_i > 0 {
                            chunk[local_i - 1].right == plan.before
                        } else {
                            previous_right[region] == Some(plan.before)
                        };
                        let next_selected = if local_i + 1 < chunk.len() {
                            chunk[local_i + 1].pos == plan.after
                        } else {
                            next_pos[region] == Some(plan.after)
                        };
                        let weight = plan.weight;
                        if plan.left_id != 0 && !previous_selected {
                            let old_key = key(plan.left_id, a);
                            if old_key != key(a, b) {
                                route_delta(&mut output, workers, old_key, weight)?;
                            }
                            let pair = key(plan.left_id, new_id);
                            if (plan.before as usize) < lower {
                                cross_births.push(CrossBirth {
                                    target_region: cuts.of(plan.before as usize),
                                    pair,
                                    pos: plan.before,
                                    weight,
                                });
                            } else {
                                route_birth(&mut output, workers, pair, plan.before, weight)?;
                            }
                        }
                        if plan.right_id != 0 {
                            let old_key = key(b, plan.right_id);
                            if old_key != key(a, b) {
                                route_delta(&mut output, workers, old_key, weight)?;
                            }
                            let final_right = if next_selected { new_id } else { plan.right_id };
                            route_birth(
                                &mut output,
                                workers,
                                key(new_id, final_right),
                                plan.pos,
                                weight,
                            )?;
                        }
                        output.merges += 1;
                    }
                    Ok((output, cross_births))
                },
            )
            .collect::<Vec<_>>()
    });
    let mut outputs = Vec::with_capacity(cuts.count());
    let mut exceptions = Vec::new();
    for result in results {
        let (output, cross_births) = result?;
        outputs.push(output);
        exceptions.extend(cross_births);
    }
    if exceptions.len() > cuts.count().saturating_sub(1) {
        return Err(TrainError::InternalInvariant(
            "too many crossed AA region birth anchors",
        ));
    }
    Ok((outputs, exceptions))
}

#[allow(clippy::too_many_arguments)]
fn prepare_batch<H: HashBuild, const TAGGED: bool>(
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
                        let Some(plan) =
                            inspect::<TAGGED>(corpus, lengths, pos as usize, rule.a, rule.b)
                        else {
                            continue;
                        };
                        valid.push(pos);
                        output.merges += 1;
                        let w = weight_at(pivots, weights, pos);
                        if plan.left_id != 0 {
                            let before = plan.before as usize;
                            let prior_id = read::<TAGGED>(corpus, before - 1);
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
                            let next_id = read::<TAGGED>(corpus, next);
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

/// Produce the same non-AA routes as `prepare_batch`, then publish this
/// occurrence's tagged endpoints. No valid-start list or ordered apply pass
/// is constructed. The selected postings remain borrowed until every worker
/// has joined, after which the coordinator may drop them.
#[allow(clippy::too_many_arguments)]
fn prepare_batch_fused<H: HashBuild>(
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
                    for &pos in &rule.posting.as_slice()[task.start..task.end] {
                        let Some(found) =
                            inspect_fused(corpus, lengths, pos as usize, rule, batch, selected)?
                        else {
                            continue;
                        };
                        let plan = found.plan;
                        let w = weight_at(pivots, weights, pos);
                        if plan.left_id != 0 && !found.left_selected {
                            route_delta(&mut output, workers, key(plan.left_id, rule.a), w)?;
                            route_birth(
                                &mut output,
                                workers,
                                key(plan.left_id, rule.new_id),
                                plan.before,
                                w,
                            )?;
                        }
                        if plan.right_id != 0 {
                            route_delta(&mut output, workers, key(rule.b, plan.right_id), w)?;
                            route_birth(
                                &mut output,
                                workers,
                                key(rule.new_id, found.final_right),
                                plan.pos,
                                w,
                            )?;
                        }
                        output.zero_rereads += found.zero_rereads;
                        output.merges += 1;
                        // All route records for this occurrence now contain
                        // old-token IDs. Its own cells are disjoint from every
                        // other certified match, so publishing cannot
                        // invalidate another worker's p/q validation.
                        write_merge::<true>(corpus, plan, rule.b_length, rule.new_id);
                    }
                }
                Ok(output)
            })
            .collect::<Vec<_>>()
    });
    results.into_iter().collect()
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn prepare_batch_region_fused<H: HashBuild>(
    pool: &ThreadPool,
    corpus: &[AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    batch: &[BatchRule],
    selected: &HashMap<u64, u32, H>,
    cuts: &RegionCuts,
    workers: usize,
) -> Result<(Vec<WorkerOutput<H>>, Vec<CrossBirth>, f64, usize)> {
    let results = pool.install(|| {
        (0..cuts.count())
            .into_par_iter()
            .map(|region| -> Result<RegionWork<H>> {
                let mut output = empty_worker(workers);
                let mut cross_births = Vec::new();
                let mut partition_seconds = 0.0;
                let mut searches = 0;
                let (lower, upper) = cuts.bounds(region);
                for rule in batch {
                    let started = Instant::now();
                    let posting = rule.posting.as_slice();
                    let first = posting.partition_point(|&pos| (pos as usize) < lower);
                    let end = posting.partition_point(|&pos| (pos as usize) < upper);
                    partition_seconds += started.elapsed().as_secs_f64();
                    searches += 2;
                    if first > end {
                        return Err(TrainError::InternalInvariant(
                            "region posting search bounds are reversed",
                        ));
                    }
                    output.visits += end - first;
                    for &pos in &posting[first..end] {
                        debug_assert_eq!(cuts.of(pos as usize), region);
                        let Some(found) =
                            inspect_fused(corpus, lengths, pos as usize, rule, batch, selected)?
                        else {
                            continue;
                        };
                        let plan = found.plan;
                        let weight = weight_at(pivots, weights, pos);
                        if plan.left_id != 0 && !found.left_selected {
                            route_delta(&mut output, workers, key(plan.left_id, rule.a), weight)?;
                            let pair = key(plan.left_id, rule.new_id);
                            if (plan.before as usize) < lower {
                                cross_births.push(CrossBirth {
                                    target_region: cuts.of(plan.before as usize),
                                    pair,
                                    pos: plan.before,
                                    weight,
                                });
                            } else {
                                route_birth(&mut output, workers, pair, plan.before, weight)?;
                            }
                        }
                        if plan.right_id != 0 {
                            route_delta(&mut output, workers, key(rule.b, plan.right_id), weight)?;
                            route_birth(
                                &mut output,
                                workers,
                                key(rule.new_id, found.final_right),
                                pos,
                                weight,
                            )?;
                        }
                        output.zero_rereads += found.zero_rereads;
                        output.merges += 1;
                        write_merge::<true>(corpus, plan, rule.b_length, rule.new_id);
                    }
                }
                Ok(RegionWork {
                    output,
                    cross_births,
                    partition_seconds,
                    searches,
                })
            })
            .collect::<Vec<_>>()
    });
    let mut outputs = Vec::with_capacity(cuts.count());
    let mut exceptions = Vec::new();
    let mut seconds = 0.0;
    let mut searches = 0;
    for result in results {
        let work = result?;
        outputs.push(work.output);
        exceptions.extend(work.cross_births);
        seconds += work.partition_seconds;
        searches += work.searches;
    }
    if exceptions.len() > cuts.count().saturating_sub(1) {
        return Err(TrainError::InternalInvariant(
            "too many crossed region birth anchors",
        ));
    }
    Ok((outputs, exceptions, seconds, searches))
}

fn split_region_slices<'a>(
    corpus: &'a mut [AtomicU32],
    cuts: &RegionCuts,
) -> Vec<&'a mut [AtomicU32]> {
    let mut regions = Vec::with_capacity(cuts.count());
    let mut rest = corpus;
    let mut prior = 0;
    for &upper in &cuts.cuts[1..] {
        let (slice, remaining) = rest.split_at_mut(upper - prior);
        regions.push(slice);
        rest = remaining;
        prior = upper;
    }
    debug_assert!(rest.is_empty());
    regions
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare_batch_region_snapshot<H: HashBuild>(
    pool: &ThreadPool,
    corpus: &mut [AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    batch: &[BatchRule],
    selected: &HashMap<u64, u32, H>,
    cuts: &RegionCuts,
    snapshots: &BoundaryState,
    workers: usize,
) -> Result<(
    Vec<WorkerOutput<H>>,
    Vec<CrossBirth>,
    Vec<DeferredStore>,
    f64,
    usize,
    usize,
    usize,
    usize,
    usize,
)> {
    let corpus_len = corpus.len();
    let regions = split_region_slices(corpus, cuts);
    let results = pool.install(|| {
        regions
            .into_par_iter()
            .enumerate()
            .map(|(region, local)| -> Result<SnapshotRegionWork<H>> {
                let mut access = snapshots.accessor(region, cuts, local);
                let mut output = empty_worker(workers);
                let mut cross_births = Vec::new();
                let mut partition_seconds = 0.0;
                let mut searches = 0;
                let (lower, upper) = cuts.bounds(region);
                for rule in batch {
                    let started = Instant::now();
                    let posting = rule.posting.as_slice();
                    let first = posting.partition_point(|&pos| (pos as usize) < lower);
                    let end = posting.partition_point(|&pos| (pos as usize) < upper);
                    partition_seconds += started.elapsed().as_secs_f64();
                    searches += 2;
                    if first > end {
                        return Err(TrainError::InternalInvariant(
                            "snapshot region posting search bounds are reversed",
                        ));
                    }
                    output.visits += end - first;
                    for &pos in &posting[first..end] {
                        let Some(found) = inspect_fused_snapshot(
                            &mut access,
                            corpus_len,
                            lengths,
                            pos as usize,
                            rule,
                            batch,
                            selected,
                        )?
                        else {
                            continue;
                        };
                        let plan = found.plan;
                        let weight = weight_at(pivots, weights, pos);
                        if plan.left_id != 0 && !found.left_selected {
                            route_delta(&mut output, workers, key(plan.left_id, rule.a), weight)?;
                            let pair = key(plan.left_id, rule.new_id);
                            if (plan.before as usize) < lower {
                                cross_births.push(CrossBirth {
                                    target_region: cuts.of(plan.before as usize),
                                    pair,
                                    pos: plan.before,
                                    weight,
                                });
                            } else {
                                route_birth(&mut output, workers, pair, plan.before, weight)?;
                            }
                        }
                        if plan.right_id != 0 {
                            route_delta(&mut output, workers, key(rule.b, plan.right_id), weight)?;
                            route_birth(
                                &mut output,
                                workers,
                                key(rule.new_id, found.final_right),
                                pos,
                                weight,
                            )?;
                        }
                        output.merges += 1;
                        write_merge_snapshot(&mut access, plan, rule.b_length, rule.new_id);
                    }
                }
                Ok(SnapshotRegionWork {
                    output,
                    cross_births,
                    deferred: std::mem::take(&mut access.deferred),
                    partition_seconds,
                    searches,
                    local_reads: access.local_reads,
                    local_writes: access.local_writes,
                    boundary_queries: access.boundary_queries,
                })
            })
            .collect::<Vec<_>>()
    });
    let worker_deferred_capacity: usize = results
        .iter()
        .filter_map(|result| result.as_ref().ok())
        .map(|work| work.deferred.capacity())
        .sum();
    let mut outputs = Vec::with_capacity(cuts.count());
    let mut exceptions = Vec::new();
    let mut deferred = Vec::new();
    let mut seconds = 0.0;
    let mut searches = 0;
    let mut local_reads = 0;
    let mut local_writes = 0;
    let mut boundary_queries = 0;
    for result in results {
        let work = result?;
        outputs.push(work.output);
        exceptions.extend(work.cross_births);
        deferred.extend(work.deferred);
        seconds += work.partition_seconds;
        searches += work.searches;
        local_reads += work.local_reads;
        local_writes += work.local_writes;
        boundary_queries += work.boundary_queries;
    }
    if exceptions.len() > cuts.count().saturating_sub(1) {
        return Err(TrainError::InternalInvariant(
            "too many crossed snapshot region birth anchors",
        ));
    }
    if deferred.len() > cuts.count().saturating_sub(1).saturating_mul(2) {
        return Err(TrainError::InternalInvariant(
            "too many deferred cross-region endpoint stores",
        ));
    }
    let capacity_upper = worker_deferred_capacity.saturating_add(deferred.capacity());
    Ok((
        outputs,
        exceptions,
        deferred,
        seconds,
        searches,
        local_reads,
        local_writes,
        boundary_queries,
        capacity_upper,
    ))
}

fn inject_region_births<H: HashBuild>(
    outputs: &mut [WorkerOutput<H>],
    cross_births: Vec<CrossBirth>,
    workers: usize,
) -> Result<usize> {
    let count = cross_births.len();
    for birth in cross_births {
        let output = outputs
            .get_mut(birth.target_region)
            .ok_or(TrainError::InternalInvariant(
                "cross birth target region outside outputs",
            ))?;
        route_birth(output, workers, birth.pair, birth.pos, birth.weight)?;
    }
    Ok(count)
}

fn apply_batch<H: HashBuild, const TAGGED: bool>(
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
                    write_merge::<TAGGED>(
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
fn commit_routes<H: HashBuild, const TAGGED: bool>(
    pool: &ThreadPool,
    owners: &mut [Owner<H>],
    outputs: &[WorkerOutput<H>],
    corpus: &[AtomicU32],
    lengths: &[u32],
    selected: &HashSet<u64, H>,
    fresh_start: u32,
    minimum: u64,
    policy: HeapPolicy,
    region_cuts: Option<&RegionCuts>,
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
                                    let a = read::<TAGGED>(corpus, p);
                                    let next = p + lengths[a as usize] as usize;
                                    key(a, read::<TAGGED>(corpus, next))
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
                    if let Some(cuts) = region_cuts {
                        debug_assert_region_order(entry.positions.as_slice(), cuts);
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

/// Validate the shared input contract inside the timed call.
fn tagged_domain(initial_id_count: usize, max_merges: usize) -> bool {
    initial_id_count
        .checked_add(max_merges)
        .is_some_and(|end| end <= HEAD as usize)
}

pub fn train(input: Prepared, options: TrainOptions, config: Config) -> Result<Output> {
    if config.regions_per_worker == 0 {
        return Err(TrainError::InvalidInput(
            "regions_per_worker must be positive",
        ));
    }
    if config.region_mode == RegionMode::Dynamic && config.regions_per_worker != 1 {
        return Err(TrainError::InvalidInput(
            "dynamic mode requires regions_per_worker=1",
        ));
    }
    config
        .workers
        .checked_mul(config.regions_per_worker)
        .ok_or(TrainError::Overflow(
            "worker times region factor exceeds usize",
        ))?;
    if config.region_mode != RegionMode::Dynamic
        && config.endpoint_plan != EndpointPlan::TaggedFused
    {
        return Err(TrainError::InvalidInput(
            "region and snapshot modes require tagged-fused endpoint plan",
        ));
    }
    // The maximal new ID is initial_lengths.len() + max_merges - 1. If it
    // could touch the head bit, retain the full-width original representation
    // for this entire call; never switch layouts between epochs.
    let domain_ok = tagged_domain(input.initial_lengths.len(), options.max_merges);
    let fallback = config.endpoint_plan != EndpointPlan::TwoPass && !domain_ok;
    let effective = if fallback {
        EndpointPlan::TwoPass
    } else {
        config.endpoint_plan
    };
    let region_active =
        config.region_mode != RegionMode::Dynamic && effective == EndpointPlan::TaggedFused;
    match (config.integer_hash, effective, region_active) {
        (IntegerHash::Std, EndpointPlan::TwoPass, _) => {
            train_impl::<StdRandomState, false, false, false>(input, options, config, fallback)
        }
        (IntegerHash::AHash, EndpointPlan::TwoPass, _) => {
            train_impl::<ahash::RandomState, false, false, false>(input, options, config, fallback)
        }
        (IntegerHash::Std, EndpointPlan::TaggedTwoPass, _) => {
            train_impl::<StdRandomState, true, false, false>(input, options, config, false)
        }
        (IntegerHash::AHash, EndpointPlan::TaggedTwoPass, _) => {
            train_impl::<ahash::RandomState, true, false, false>(input, options, config, false)
        }
        (IntegerHash::Std, EndpointPlan::TaggedFused, false) => {
            train_impl::<StdRandomState, true, true, false>(input, options, config, false)
        }
        (IntegerHash::AHash, EndpointPlan::TaggedFused, false) => {
            train_impl::<ahash::RandomState, true, true, false>(input, options, config, false)
        }
        (IntegerHash::Std, EndpointPlan::TaggedFused, true) => {
            train_impl::<StdRandomState, true, true, true>(input, options, config, false)
        }
        (IntegerHash::AHash, EndpointPlan::TaggedFused, true) => {
            train_impl::<ahash::RandomState, true, true, true>(input, options, config, false)
        }
    }
}

fn train_impl<H: HashBuild, const TAGGED: bool, const FUSED: bool, const REGION: bool>(
    input: Prepared,
    options: TrainOptions,
    config: Config,
    domain_fallback: bool,
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
    metrics.endpoint_domain_fallback = domain_fallback;
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
    let mut corpus: Vec<AtomicU32> = corpus
        .into_iter()
        .map(|id| AtomicU32::new(if TAGGED && id != 0 { id | HEAD } else { id }))
        .collect();
    let region_cuts = if REGION {
        let requested = config
            .workers
            .checked_mul(config.regions_per_worker)
            .ok_or(TrainError::Overflow(
                "worker times region factor exceeds usize",
            ))?;
        Some(RegionCuts::new(corpus.len(), requested.min(corpus.len()))?)
    } else {
        None
    };
    metrics.region_count_effective = region_cuts.as_ref().map_or(0, RegionCuts::count);
    let started = Instant::now();
    let mut owners = initial_index::<H, TAGGED>(
        &pool,
        &corpus,
        &pivots,
        &weights,
        config.workers,
        config.chunk_size,
        options.min_frequency,
        region_cuts.as_ref(),
        &mut metrics,
    )?;
    metrics.init_seconds = started.elapsed().as_secs_f64();
    let mut snapshots = if REGION && config.region_mode == RegionMode::Snapshot {
        let started = Instant::now();
        let state = BoundaryState::new(&corpus, &initial_lengths, region_cuts.as_ref().unwrap())?;
        metrics.snapshot_build_seconds += started.elapsed().as_secs_f64();
        metrics.snapshot_cut_count = state.windows_len();
        metrics.snapshot_peak_descriptor_capacity_bytes = state.capacity_bytes();
        Some(state)
    } else {
        None
    };
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
                                inspect::<TAGGED>(
                                    &corpus,
                                    &initial_lengths,
                                    pos as usize,
                                    rule.a,
                                    rule.b,
                                )
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
                            let mut plan = inspect::<TAGGED>(
                                &corpus,
                                &initial_lengths,
                                pos as usize,
                                rule.a,
                                rule.b,
                            )
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
            metrics.peak_plan_len = metrics.peak_plan_len.max(planned);
            metrics.peak_task_starts = metrics.peak_task_starts.max(planned);
            if REGION {
                let cuts = region_cuts.as_ref().unwrap();
                let old_plan_capacity_bytes = plan_chunks.iter().fold(0_usize, |total, chunk| {
                    total.saturating_add(
                        chunk.capacity().saturating_mul(std::mem::size_of::<Plan>()),
                    )
                });
                let regroup_started = Instant::now();
                let region_plans = aa_plans_by_region(plan_chunks, cuts);
                let new_plan_capacity_bytes = region_plans.iter().fold(0_usize, |total, chunk| {
                    total.saturating_add(
                        chunk.capacity().saturating_mul(std::mem::size_of::<Plan>()),
                    )
                });
                metrics.region_aa_regroup_capacity_upper_bytes = metrics
                    .region_aa_regroup_capacity_upper_bytes
                    .max(old_plan_capacity_bytes.saturating_add(new_plan_capacity_bytes));
                metrics.region_aa_regroup_seconds += regroup_started.elapsed().as_secs_f64();
                metrics.flat_tasks += cuts.count();
                metrics.peak_flat_tasks = metrics.peak_flat_tasks.max(cuts.count());
                metrics.region_posting_visits += rule.posting.len();
                metrics.region_valid_merges += planned;
                let posting = rule.posting.as_slice();
                let mut region_visits = Vec::with_capacity(cuts.count());
                for region in 0..cuts.count() {
                    let (lower, upper) = cuts.bounds(region);
                    let first = posting.partition_point(|&pos| (pos as usize) < lower);
                    let end = posting.partition_point(|&pos| (pos as usize) < upper);
                    region_visits.push(end - first);
                }
                debug_assert_eq!(region_visits.iter().sum::<usize>(), posting.len());
                metrics.region_partition_searches += 2 * cuts.count();
                let (mut outputs, exceptions) = route_aa_region::<H>(
                    &pool,
                    &region_plans,
                    cuts,
                    rule.a,
                    rule.b,
                    rule.new_id,
                    config.workers,
                    &mut metrics,
                )?;
                for (output, visits) in outputs.iter_mut().zip(region_visits) {
                    output.visits = visits;
                }
                record_region_load(&outputs, config.workers, false, &mut metrics);
                metrics.plan_seconds += started.elapsed().as_secs_f64();
                drop(rule.posting.take());
                let apply_started = Instant::now();
                pool.install(|| {
                    region_plans.par_iter().for_each(|chunk| {
                        for &plan in chunk {
                            write_merge::<TAGGED>(&corpus, plan, rule.b_length, rule.new_id);
                        }
                    })
                });
                metrics.apply_seconds += apply_started.elapsed().as_secs_f64();
                metrics.region_cross_births +=
                    inject_region_births(&mut outputs, exceptions, config.workers)?;
                outputs
            } else {
                metrics.flat_tasks += plan_chunks.len();
                metrics.peak_flat_tasks = metrics.peak_flat_tasks.max(plan_chunks.len());
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
                let apply_started = Instant::now();
                pool.install(|| {
                    plan_chunks.par_iter().for_each(|chunk| {
                        for &plan in chunk {
                            write_merge::<TAGGED>(&corpus, plan, rule.b_length, rule.new_id);
                        }
                    })
                });
                metrics.apply_seconds += apply_started.elapsed().as_secs_f64();
                outputs
            }
        } else {
            let mut tasks = Vec::<FlatTask>::new();
            if !REGION {
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
            }
            let task_count = if REGION {
                region_cuts.as_ref().unwrap().count()
            } else {
                tasks.len()
            };
            metrics.flat_tasks += task_count;
            metrics.peak_flat_tasks = metrics.peak_flat_tasks.max(task_count);
            let started = Instant::now();
            let mut outputs = if REGION && config.region_mode == RegionMode::Snapshot {
                let cuts = region_cuts.as_ref().unwrap();
                let state = snapshots.as_ref().unwrap();
                let (
                    mut outputs,
                    exceptions,
                    deferred,
                    seconds,
                    searches,
                    local_reads,
                    local_writes,
                    boundary_queries,
                    deferred_capacity_upper,
                ) = prepare_batch_region_snapshot(
                    &pool,
                    &mut corpus,
                    &initial_lengths,
                    &pivots,
                    &weights,
                    &batch,
                    &selected,
                    cuts,
                    state,
                    config.workers,
                )?;
                metrics.region_partition_worker_seconds += seconds;
                metrics.region_partition_searches += searches;
                metrics.snapshot_local_reads += local_reads;
                metrics.snapshot_local_writes += local_writes;
                metrics.snapshot_boundary_queries += boundary_queries;
                metrics.snapshot_deferred_stores += deferred.len();
                metrics.snapshot_peak_deferred_len =
                    metrics.snapshot_peak_deferred_len.max(deferred.len());
                metrics.snapshot_peak_deferred_capacity_upper = metrics
                    .snapshot_peak_deferred_capacity_upper
                    .max(deferred_capacity_upper);
                let replay_started = Instant::now();
                for store in deferred {
                    corpus[store.pos].store(store.value, Ordering::Relaxed);
                }
                metrics.snapshot_deferred_apply_seconds += replay_started.elapsed().as_secs_f64();
                metrics.region_cross_births +=
                    inject_region_births(&mut outputs, exceptions, config.workers)?;
                outputs
            } else if REGION {
                let cuts = region_cuts.as_ref().unwrap();
                let (mut outputs, exceptions, seconds, searches) = prepare_batch_region_fused(
                    &pool,
                    &corpus,
                    &initial_lengths,
                    &pivots,
                    &weights,
                    &batch,
                    &selected,
                    cuts,
                    config.workers,
                )?;
                metrics.region_partition_worker_seconds += seconds;
                metrics.region_partition_searches += searches;
                metrics.region_cross_births +=
                    inject_region_births(&mut outputs, exceptions, config.workers)?;
                outputs
            } else if FUSED {
                prepare_batch_fused::<H>(
                    &pool,
                    &corpus,
                    &initial_lengths,
                    &pivots,
                    &weights,
                    &batch,
                    &tasks,
                    &selected,
                    config.workers,
                )?
            } else {
                prepare_batch::<H, TAGGED>(
                    &pool,
                    &corpus,
                    &initial_lengths,
                    &pivots,
                    &weights,
                    &batch,
                    &tasks,
                    &selected,
                    config.workers,
                )?
            };
            let visited: usize = outputs.iter().map(|output| output.visits).sum();
            let planned: usize = outputs.iter().map(|output| output.merges).sum();
            metrics.posting_visits += visited;
            metrics.stale_visits += visited - planned;
            metrics.planned_positions += planned;
            if REGION {
                metrics.region_posting_visits += visited;
                metrics.region_valid_merges += planned;
                record_region_load(&outputs, config.workers, true, &mut metrics);
            }
            metrics.decoder_zero_rereads += outputs
                .iter()
                .map(|output| output.zero_rereads)
                .sum::<usize>();
            if FUSED {
                metrics.fused_non_aa_batches += 1;
                metrics.fused_non_aa_merges += planned;
            } else {
                metrics.peak_plan_len = metrics.peak_plan_len.max(planned);
                metrics.peak_task_starts = metrics.peak_task_starts.max(planned);
                metrics.non_aa_start_positions_peak =
                    metrics.non_aa_start_positions_peak.max(planned);
                metrics.non_aa_start_bytes_peak_proxy =
                    metrics.non_aa_start_positions_peak * std::mem::size_of::<u32>();
            }
            metrics.plan_seconds += started.elapsed().as_secs_f64();
            // In fused mode all workers have joined before selected postings
            // are released. In two-pass mode apply reads only rule metadata
            // and the valid starts in outputs.
            for rule in &mut batch {
                drop(rule.posting.take());
            }
            if !FUSED {
                let started = Instant::now();
                apply_batch::<H, TAGGED>(&pool, &corpus, &batch, &tasks, &mut outputs);
                metrics.apply_seconds += started.elapsed().as_secs_f64();
            }
            outputs
        };
        if REGION {
            let delta_capacity: usize = outputs
                .iter()
                .flat_map(|output| &output.routes)
                .map(|route| route.delta.capacity())
                .sum();
            let born_capacity: usize = outputs
                .iter()
                .flat_map(|output| &output.routes)
                .map(|route| route.born.capacity())
                .sum();
            let route_header_capacity_bytes = outputs.iter().fold(0_usize, |total, output| {
                total.saturating_add(
                    output
                        .routes
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Route<H>>()),
                )
            });
            metrics.region_peak_route_delta_capacity =
                metrics.region_peak_route_delta_capacity.max(delta_capacity);
            metrics.region_peak_route_born_capacity =
                metrics.region_peak_route_born_capacity.max(born_capacity);
            metrics.region_peak_route_header_capacity_bytes = metrics
                .region_peak_route_header_capacity_bytes
                .max(route_header_capacity_bytes);
        }
        if let Some(state) = snapshots.as_mut() {
            let started = Instant::now();
            state.refresh(&corpus, &initial_lengths)?;
            metrics.snapshot_refresh_seconds += started.elapsed().as_secs_f64();
            metrics.snapshot_peak_descriptor_capacity_bytes = metrics
                .snapshot_peak_descriptor_capacity_bytes
                .max(state.capacity_bytes());
        }
        commit_routes::<H, TAGGED>(
            &pool,
            &mut owners,
            &outputs,
            &corpus,
            &initial_lengths,
            &selected_keys,
            batch[0].new_id,
            options.min_frequency,
            config.heap_policy,
            region_cuts.as_ref(),
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
        let id = read::<TAGGED>(&corpus, pos);
        final_tokens.push(id);
        if pos == corpus.len() - 1 {
            break;
        }
        let next = pos + initial_lengths[id as usize] as usize;
        if next >= corpus.len() {
            return Err(TrainError::InternalInvariant("invalid final boundary"));
        }
        if id != 0 && read::<TAGGED>(&corpus, next) != 0 {
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
        effective_endpoint_plan: if FUSED {
            EndpointPlan::TaggedFused
        } else if TAGGED {
            EndpointPlan::TaggedTwoPass
        } else {
            EndpointPlan::TwoPass
        },
        effective_region_mode: if REGION {
            config.region_mode
        } else {
            RegionMode::Dynamic
        },
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
        for integer_hash in [IntegerHash::Std, IntegerHash::AHash] {
            for heap_policy in [HeapPolicy::Lazy, HeapPolicy::Eager] {
                for workers in [1, 2, 4] {
                    for chunk_size in [1, 5, 32] {
                        let actual = train(
                            input.clone(),
                            options,
                            Config {
                                workers,
                                regions_per_worker: 1,
                                chunk_size,
                                heap_policy,
                                integer_hash,
                                endpoint_plan: EndpointPlan::TwoPass,
                                region_mode: RegionMode::Dynamic,
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
                regions_per_worker: 1,
                chunk_size: 1,
                heap_policy: HeapPolicy::Lazy,
                integer_hash: IntegerHash::AHash,
                endpoint_plan: EndpointPlan::TwoPass,
                region_mode: RegionMode::Dynamic,
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
                regions_per_worker: 1,
                chunk_size: usize::MAX,
                heap_policy: HeapPolicy::Lazy,
                integer_hash: IntegerHash::AHash,
                endpoint_plan: EndpointPlan::TwoPass,
                region_mode: RegionMode::Dynamic,
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
    fn tagged_domain_boundary_without_large_allocation() {
        assert!(tagged_domain(HEAD as usize, 0));
        assert!(tagged_domain(HEAD as usize - 1, 1));
        assert!(!tagged_domain(HEAD as usize - 1, 2));
        assert!(!tagged_domain(HEAD as usize + 1, 0));
        assert!(!tagged_domain(usize::MAX, 1));
        for endpoint_plan in [EndpointPlan::TaggedTwoPass, EndpointPlan::TaggedFused] {
            for region_mode in if endpoint_plan == EndpointPlan::TaggedFused {
                &[RegionMode::Region, RegionMode::Snapshot][..]
            } else {
                &[RegionMode::Dynamic][..]
            } {
                let regions_per_worker = if *region_mode == RegionMode::Dynamic {
                    1
                } else {
                    4
                };
                let output = train(
                    prepared(&[(vec![1, 2], 1)], 2),
                    TrainOptions {
                        max_merges: HEAD as usize,
                        min_frequency: 1,
                        bounds: Bounds::Checked,
                    },
                    Config {
                        workers: 1,
                        regions_per_worker,
                        chunk_size: 8,
                        heap_policy: HeapPolicy::Lazy,
                        integer_hash: IntegerHash::Std,
                        endpoint_plan,
                        region_mode: *region_mode,
                    },
                )
                .unwrap();
                assert_eq!(output.effective_endpoint_plan, EndpointPlan::TwoPass);
                assert_eq!(output.effective_region_mode, RegionMode::Dynamic);
                assert!(output.metrics.endpoint_domain_fallback);
                assert_eq!(output.metrics.region_count_effective, 0);
                assert_eq!(output.rules.len(), 1);
            }
        }
    }

    fn compare_endpoint_modes(input: Prepared, max_merges: usize, minimum: u64) {
        let options = TrainOptions {
            max_merges,
            min_frequency: minimum,
            bounds: Bounds::Checked,
        };
        let expected = reference(input.clone(), options).unwrap();
        for endpoint_plan in [
            EndpointPlan::TwoPass,
            EndpointPlan::TaggedTwoPass,
            EndpointPlan::TaggedFused,
        ] {
            for integer_hash in [IntegerHash::Std, IntegerHash::AHash] {
                for workers in [1, 4] {
                    for region_mode in [
                        RegionMode::Dynamic,
                        RegionMode::Region,
                        RegionMode::Snapshot,
                    ] {
                        if endpoint_plan != EndpointPlan::TaggedFused
                            && region_mode != RegionMode::Dynamic
                        {
                            continue;
                        }
                        let factors: &[usize] = if region_mode == RegionMode::Dynamic {
                            &[1]
                        } else {
                            &[1, 4]
                        };
                        for &regions_per_worker in factors {
                            let output = train(
                                input.clone(),
                                options,
                                Config {
                                    workers,
                                    regions_per_worker,
                                    chunk_size: 17,
                                    heap_policy: HeapPolicy::Lazy,
                                    integer_hash,
                                    endpoint_plan,
                                    region_mode,
                                },
                            )
                            .unwrap();
                            assert_eq!(
                                output.rules, expected.merges,
                                "{endpoint_plan:?} {region_mode:?} {integer_hash:?} W{workers} k{regions_per_worker}"
                            );
                            assert_eq!(
                                output.final_tokens, expected.final_tokens,
                                "{endpoint_plan:?} {region_mode:?} {integer_hash:?} W{workers} k{regions_per_worker}"
                            );
                            assert_eq!(output.effective_endpoint_plan, endpoint_plan);
                            assert_eq!(output.effective_region_mode, region_mode);
                            assert!(!output.metrics.endpoint_domain_fallback);
                            if region_mode != RegionMode::Dynamic {
                                assert_eq!(
                                    output.metrics.region_count_effective,
                                    (workers * regions_per_worker).min(input.corpus.len())
                                );
                            }
                            if endpoint_plan != EndpointPlan::TaggedFused {
                                assert_eq!(output.metrics.fused_non_aa_merges, 0);
                            } else {
                                assert_eq!(output.metrics.non_aa_start_positions_peak, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tagged_endpoint_complete_traces() {
        // Same rule at adjacent occurrences, and a separate AA epoch.
        compare_endpoint_modes(
            prepared(&[(vec![1, 2, 1, 2, 1, 2, 1, 2], 7), (vec![3; 129], 2)], 3),
            16,
            1,
        );
        // Distinct neighboring rules, piece sentinels, and weighted ties.
        compare_endpoint_modes(
            prepared(
                &[(vec![1, 2, 3, 4], 9), (vec![1, 2], 3), (vec![3, 4], 3)],
                4,
            ),
            16,
            1,
        );
        // Selected matches separated by old tokens, which may merge outward.
        compare_endpoint_modes(
            prepared(&[(vec![1, 2, 3, 4, 5, 6], 5), (vec![3, 4, 3, 4], 2)], 6),
            20,
            1,
        );
        // Repeated AA creates long old token lengths and historical stale starts.
        compare_endpoint_modes(prepared(&[(vec![1; 1025], 1)], 1), 12, 1);
        compare_endpoint_modes(
            prepared(
                &[(vec![1, 2, 1, 2, 3, 2, 3, 2, 3], 3), (vec![3, 2, 1, 2], 8)],
                3,
            ),
            20,
            2,
        );
    }

    #[test]
    fn region_projection_and_cross_cut_births() {
        let cuts = RegionCuts::new(7, 7).unwrap();
        assert_eq!(cuts.count(), 7);
        assert!(cuts.cuts.windows(2).all(|edge| edge[0] < edge[1]));
        for pos in 0..7 {
            let region = cuts.of(pos);
            let (lower, upper) = cuts.bounds(region);
            assert!(lower <= pos && pos < upper);
        }
        debug_assert_region_order(&[1, 0, 2, 5], &RegionCuts::new(8, 2).unwrap());

        // Eight AA epochs leave two 256-position tokens before (2,3).
        // The later (2,3) merge has a left birth anchored two regions back.
        let input = prepared(&[([vec![1; 512], vec![2, 3]].concat(), 3)], 3);
        let options = TrainOptions {
            max_merges: 9,
            min_frequency: 1,
            bounds: Bounds::Checked,
        };
        let expected = reference(input.clone(), options).unwrap();
        let long_cuts = RegionCuts::new(516, 4).unwrap();
        assert!(long_cuts.of(257) + 1 < long_cuts.of(513));
        for region_mode in [RegionMode::Region, RegionMode::Snapshot] {
            for regions_per_worker in [1, 4] {
                let actual = train(
                    input.clone(),
                    options,
                    Config {
                        workers: 4,
                        regions_per_worker,
                        chunk_size: 17,
                        heap_policy: HeapPolicy::Lazy,
                        integer_hash: IntegerHash::Std,
                        endpoint_plan: EndpointPlan::TaggedFused,
                        region_mode,
                    },
                )
                .unwrap();
                assert_eq!(actual.rules, expected.merges);
                assert_eq!(actual.final_tokens, expected.final_tokens);
                assert_eq!((actual.rules[8].left, actual.rules[8].right), (2, 3));
                assert!(actual.metrics.region_cross_births > 0);
                assert!(actual.metrics.region_max_visits_per_batch > 0);
                assert!(actual.metrics.region_max_merges_per_batch > 0);
                assert!(actual.metrics.region_aa_regroup_capacity_upper_bytes > 0);
                assert_eq!(
                    actual.metrics.region_count_effective,
                    4 * regions_per_worker
                );
                assert!(actual.metrics.region_sum_visit_makespan_lower_bound > 0);
                assert!(actual.metrics.region_peak_route_header_capacity_bytes > 0);
            }
        }
    }

    #[test]
    fn snapshot_defers_non_aa_cross_cut_endpoints() {
        let input = prepared(&[(vec![1, 2, 3, 4, 5, 6], 3)], 6);
        let options = TrainOptions {
            max_merges: 5,
            min_frequency: 1,
            bounds: Bounds::Checked,
        };
        let expected = reference(input.clone(), options).unwrap();
        for regions_per_worker in [1, 4] {
            let actual = train(
                input.clone(),
                options,
                Config {
                    workers: 4,
                    regions_per_worker,
                    chunk_size: 2,
                    heap_policy: HeapPolicy::Lazy,
                    integer_hash: IntegerHash::Std,
                    endpoint_plan: EndpointPlan::TaggedFused,
                    region_mode: RegionMode::Snapshot,
                },
            )
            .unwrap();
            assert_eq!(actual.rules, expected.merges);
            assert_eq!(actual.final_tokens, expected.final_tokens);
            assert!(actual.metrics.snapshot_boundary_queries > 0);
            assert!(actual.metrics.snapshot_deferred_stores > 0);
            assert_eq!(
                actual.metrics.region_count_effective,
                (4 * regions_per_worker).min(input.corpus.len())
            );
        }
    }

    #[test]
    fn snapshot_adjacent_matches_agree_in_both_execution_orders() {
        let make_rule = |a, b, new_id| BatchRule {
            a,
            b,
            new_id,
            frequency: 1,
            a_length: 1,
            b_length: 1,
            posting: SmallPosting::default(),
        };
        let batch = [make_rule(1, 2, 5), make_rule(3, 4, 6)];
        let selected = HashMap::<u64, u32>::from([(key(1, 2), 5), (key(3, 4), 6)]);
        let lengths = vec![1, 1, 1, 1, 1, 2, 2];
        let cuts = RegionCuts {
            cuts: vec![0, 2, 6],
        };
        for order in [[0_usize, 1], [1, 0]] {
            let mut corpus = [0, 1 | HEAD, 2 | HEAD, 3 | HEAD, 4 | HEAD, 0].map(AtomicU32::new);
            let mut state = BoundaryState::new(&corpus, &lengths, &cuts).unwrap();
            let mut deferred = Vec::new();
            for rank in order {
                let pos = if rank == 0 { 1 } else { 3 };
                let region = cuts.of(pos);
                let (lower, upper) = cuts.bounds(region);
                let corpus_len = corpus.len();
                let mut access = state.accessor(region, &cuts, &mut corpus[lower..upper]);
                let found = inspect_fused_snapshot(
                    &mut access,
                    corpus_len,
                    &lengths,
                    pos,
                    &batch[rank],
                    &batch,
                    &selected,
                )
                .unwrap()
                .unwrap();
                if rank == 0 {
                    assert_eq!((found.plan.right_id, found.final_right), (3, 6));
                } else {
                    assert_eq!(found.plan.left_id, 2);
                    assert!(found.left_selected);
                }
                write_merge_snapshot(&mut access, found.plan, 1, batch[rank].new_id);
                deferred.extend(std::mem::take(&mut access.deferred));
            }
            assert_eq!(deferred, vec![DeferredStore { pos: 2, value: 5 }]);
            for store in deferred {
                corpus[store.pos].store(store.value, Ordering::Relaxed);
            }
            state.refresh(&corpus, &lengths).unwrap();
            assert_eq!(
                corpus
                    .iter()
                    .map(|cell| cell.load(Ordering::Relaxed))
                    .collect::<Vec<_>>(),
                vec![0, 5 | HEAD, 5, 6 | HEAD, 6, 0]
            );
            assert_eq!(state.windows_len(), 1);
        }
    }

    #[test]
    fn region_more_workers_than_positions_and_weighted_aa() {
        compare_endpoint_modes(
            prepared(&[(vec![1; 9], 7), (vec![2, 3, 2, 3], 2)], 3),
            12,
            1,
        );
        let input = prepared(&[(vec![1; 9], 7), (vec![2, 3], 2)], 3);
        let options = TrainOptions {
            max_merges: 12,
            min_frequency: 1,
            bounds: Bounds::Checked,
        };
        let expected = reference(input.clone(), options).unwrap();
        for region_mode in [RegionMode::Region, RegionMode::Snapshot] {
            for regions_per_worker in [1, 4] {
                let actual = train(
                    input.clone(),
                    options,
                    Config {
                        workers: 16,
                        regions_per_worker,
                        chunk_size: 3,
                        heap_policy: HeapPolicy::Lazy,
                        integer_hash: IntegerHash::AHash,
                        endpoint_plan: EndpointPlan::TaggedFused,
                        region_mode,
                    },
                )
                .unwrap();
                assert_eq!(actual.rules, expected.merges);
                assert_eq!(actual.final_tokens, expected.final_tokens);
                assert_eq!(actual.metrics.region_count_effective, input.corpus.len());
            }
        }

        let empty = Prepared {
            corpus: vec![0],
            initial_lengths: vec![1],
            pivots: vec![],
            weights: vec![],
        };
        let expected_empty = reference(empty.clone(), options).unwrap();
        for region_mode in [RegionMode::Region, RegionMode::Snapshot] {
            for regions_per_worker in [1, 4] {
                let actual = train(
                    empty.clone(),
                    options,
                    Config {
                        workers: 8,
                        regions_per_worker,
                        chunk_size: 1,
                        heap_policy: HeapPolicy::Lazy,
                        integer_hash: IntegerHash::Std,
                        endpoint_plan: EndpointPlan::TaggedFused,
                        region_mode,
                    },
                )
                .unwrap();
                assert_eq!(actual.rules, expected_empty.merges);
                assert_eq!(actual.final_tokens, expected_empty.final_tokens);
                assert_eq!(actual.metrics.region_count_effective, 1);
            }
        }
    }

    #[test]
    fn microregions_outnumber_workers_and_validate_factor() {
        let input = prepared(&[(vec![1, 2, 3, 4, 1, 2, 3, 4], 7)], 4);
        let options = TrainOptions {
            max_merges: 8,
            min_frequency: 1,
            bounds: Bounds::Checked,
        };
        let expected = reference(input.clone(), options).unwrap();
        let mut config = Config {
            workers: 2,
            regions_per_worker: 4,
            chunk_size: 2,
            heap_policy: HeapPolicy::Lazy,
            integer_hash: IntegerHash::AHash,
            endpoint_plan: EndpointPlan::TaggedFused,
            region_mode: RegionMode::Region,
        };
        for mode in [RegionMode::Region, RegionMode::Snapshot] {
            config.region_mode = mode;
            let actual = train(input.clone(), options, config).unwrap();
            assert_eq!(actual.rules, expected.merges);
            assert_eq!(actual.final_tokens, expected.final_tokens);
            assert_eq!(actual.metrics.region_count_effective, 8);
            assert!(actual.metrics.region_sum_visit_makespan_lower_bound > 0);
            assert!(actual.metrics.region_peak_route_header_capacity_bytes > 0);
        }

        config.regions_per_worker = 0;
        assert!(matches!(
            train(input.clone(), options, config),
            Err(TrainError::InvalidInput(_))
        ));
        config.regions_per_worker = usize::MAX;
        assert!(matches!(
            train(input.clone(), options, config),
            Err(TrainError::Overflow(_))
        ));
        config.region_mode = RegionMode::Dynamic;
        config.regions_per_worker = 4;
        assert!(matches!(
            train(input, options, config),
            Err(TrainError::InvalidInput(_))
        ));
    }

    #[test]
    fn fused_reader_handles_partial_neighbor_publication() {
        let make_rule = |a, b, new_id, b_length| BatchRule {
            a,
            b,
            new_id,
            frequency: 1,
            a_length: 1,
            b_length,
            posting: SmallPosting::default(),
        };
        let batch = [make_rule(1, 2, 5, 1), make_rule(3, 4, 6, 2)];
        let mut lengths = vec![1; 7];
        lengths[4] = 2;
        lengths[5] = 2;
        lengths[6] = 3;
        let selected = HashMap::<u64, u32>::from([(key(1, 2), 5), (key(3, 4), 6)]);
        let corpus = [0, 1 | HEAD, 2 | HEAD, 3 | HEAD, 4 | HEAD, 4, 0].map(AtomicU32::new);
        let own = inspect_fused(&corpus, &lengths, 1, &batch[0], &batch, &selected)
            .unwrap()
            .unwrap();
        assert_eq!((own.plan.right_id, own.final_right), (3, 6));

        // Right (3,4) has published head, then cleared the long right start;
        // the old C read may precede these writes, but zero demands reread.
        corpus[3].store(6 | HEAD, Ordering::Release);
        corpus[4].store(0, Ordering::Release);
        let own = inspect_fused(&corpus, &lengths, 1, &batch[0], &batch, &selected)
            .unwrap()
            .unwrap();
        assert_eq!((own.plan.right_id, own.final_right), (3, 6));
        corpus[5].store(6, Ordering::Release);

        // On a separate snapshot, left (1,2) publishes while right (3,4)
        // has not. Its bare tail and fresh head must decode as old 2 and 1,
        // suppressing the right occurrence's left birth.
        let left_corpus = [0, 1 | HEAD, 2 | HEAD, 3 | HEAD, 4 | HEAD, 4, 0].map(AtomicU32::new);
        left_corpus[1].store(5 | HEAD, Ordering::Release);
        left_corpus[2].store(5, Ordering::Release);
        let right = inspect_fused(&left_corpus, &lengths, 3, &batch[1], &batch, &selected)
            .unwrap()
            .unwrap();
        assert_eq!(right.plan.left_id, 2);
        assert!(right.left_selected);

        let separate = [0, 1 | HEAD, 2 | HEAD, 3 | HEAD, 0].map(AtomicU32::new);
        let single = [make_rule(1, 2, 4, 1)];
        let lengths = vec![1, 1, 1, 1, 2];
        let selected = HashMap::<u64, u32>::from([(key(1, 2), 4)]);
        let boundary = inspect_fused(&separate, &lengths, 1, &single[0], &single, &selected)
            .unwrap()
            .unwrap();
        assert_eq!((boundary.plan.right_id, boundary.final_right), (3, 3));
        assert_eq!(boundary.zero_rereads, 1);
    }

    #[test]
    fn fused_reader_old_tag_long_left_and_stale_start() {
        let batch = [
            BatchRule {
                a: 3,
                b: 4,
                new_id: 5,
                frequency: 1,
                a_length: 2,
                b_length: 257,
                posting: SmallPosting::default(),
            },
            BatchRule {
                a: 1,
                b: 2,
                new_id: 6,
                frequency: 1,
                a_length: 1,
                b_length: 1,
                posting: SmallPosting::default(),
            },
        ];
        let mut lengths = vec![1; 7];
        lengths[3] = 2;
        lengths[4] = 257;
        lengths[5] = 259;
        lengths[6] = 2;
        let corpus = (0..263).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
        corpus[1].store(3 | HEAD, Ordering::Relaxed);
        corpus[2].store(3, Ordering::Relaxed);
        corpus[3].store(4 | HEAD, Ordering::Relaxed);
        corpus[259].store(4, Ordering::Relaxed);
        corpus[260].store(1 | HEAD, Ordering::Relaxed);
        corpus[261].store(2 | HEAD, Ordering::Relaxed);
        let selected = HashMap::<u64, u32>::from([(key(3, 4), 5), (key(1, 2), 6)]);
        assert_eq!(old_end_id(4 | HEAD, &batch, 5).unwrap(), 4);
        assert_eq!(old_next_id(4 | HEAD, 1, &batch, 5).unwrap(), 4);
        let first = inspect_fused(&corpus, &lengths, 260, &batch[1], &batch, &selected)
            .unwrap()
            .unwrap();
        assert_eq!((first.plan.before, first.plan.left_id), (3, 4));
        assert!(first.left_selected);

        corpus[1].store(5 | HEAD, Ordering::Release);
        corpus[3].store(0, Ordering::Release);
        corpus[259].store(5, Ordering::Release);
        let after_left = inspect_fused(&corpus, &lengths, 260, &batch[1], &batch, &selected)
            .unwrap()
            .unwrap();
        assert_eq!((after_left.plan.before, after_left.plan.left_id), (3, 4));
        assert!(after_left.left_selected);

        corpus[260].store(6 | HEAD, Ordering::Release);
        corpus[261].store(6, Ordering::Release);
        assert!(
            inspect_fused(&corpus, &lengths, 260, &batch[1], &batch, &selected)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn fused_reader_decodes_right_bare_tail_and_outward_head() {
        let rule = |a, b, new_id| BatchRule {
            a,
            b,
            new_id,
            frequency: 1,
            a_length: 1,
            b_length: 1,
            posting: SmallPosting::default(),
        };
        let batch = [rule(1, 2, 5), rule(3, 4, 6)];
        let lengths = vec![1, 1, 1, 1, 1, 2, 2];
        let selected = HashMap::<u64, u32>::from([(key(1, 2), 5), (key(3, 4), 6)]);
        // A reader may observe old C at t before another worker publishes
        // its head, then see the newly written bare tail at u.
        let bare_tail_view = [0, 1 | HEAD, 2 | HEAD, 3 | HEAD, 6, 0].map(AtomicU32::new);
        let found = inspect_fused(&bare_tail_view, &lengths, 1, &batch[0], &batch, &selected)
            .unwrap()
            .unwrap();
        assert_eq!((found.plan.right_id, found.final_right), (3, 6));

        // Another selected pair starts at D, outside old C. Its HEAD at u
        // decodes old D, but the final right neighbor of our merge is C.
        let outward = [rule(1, 2, 6), rule(4, 5, 7)];
        let lengths = vec![1, 1, 1, 1, 1, 1, 2, 2];
        let selected = HashMap::<u64, u32>::from([(key(1, 2), 6), (key(4, 5), 7)]);
        let outward_view =
            [0, 1 | HEAD, 2 | HEAD, 3 | HEAD, 7 | HEAD, 5 | HEAD, 0].map(AtomicU32::new);
        let found = inspect_fused(&outward_view, &lengths, 1, &outward[0], &outward, &selected)
            .unwrap()
            .unwrap();
        assert_eq!((found.plan.right_id, found.final_right), (3, 3));
    }
}

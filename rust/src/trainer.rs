//! Exact weighted BPE trainer matching the packed-key Python experiment.

use crate::backend::Endpoints;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Prepared {
    pub corpus: Vec<u32>,
    pub initial_lengths: Vec<u32>,
    pub pivots: Vec<u32>,
    pub weights: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bounds {
    Checked,
    Unchecked,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainOptions {
    pub max_merges: usize,
    pub min_frequency: u64,
    pub bounds: Bounds,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            max_merges: 1000,
            min_frequency: 2,
            bounds: Bounds::Checked,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    pub left: u32,
    pub right: u32,
    pub frequency: u64,
}

#[derive(Debug, Clone)]
pub struct TrainResult {
    pub merges: Vec<Rule>,
    pub final_tokens: Vec<u32>,
    pub init_seconds: f64,
    pub merge_seconds: f64,
    pub train_seconds: f64,
    pub rules: usize,
    pub actual_merges: usize,
    pub position_visits: usize,
    pub stale_visits: usize,
    pub heap_pops: usize,
    pub backend_buffer_bytes: usize,
    pub initial_occurrence_bytes: usize,
    pub max_token_length: u32,
    pub corpus_positions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    InvalidInput(&'static str),
    Overflow(&'static str),
    InternalInvariant(&'static str),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid prepared input: {message}"),
            Self::Overflow(message) => write!(f, "numeric overflow: {message}"),
            Self::InternalInvariant(message) => write!(f, "internal invariant: {message}"),
        }
    }
}

impl Error for TrainError {}

#[derive(Clone, Copy, Eq, PartialEq)]
struct HeapEntry {
    frequency: u64,
    key: u64,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.key.cmp(&self.key))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

#[cfg_attr(feature = "profiling", hotpath::measure)]
pub(crate) fn validate(prepared: &Prepared, options: TrainOptions) -> Result<(), TrainError> {
    let corpus = &prepared.corpus;
    let lengths = &prepared.initial_lengths;
    let pivots = &prepared.pivots;
    let weights = &prepared.weights;
    if options.min_frequency == 0 {
        return Err(TrainError::InvalidInput("min_frequency must be positive"));
    }
    if corpus.is_empty() || u64::try_from(corpus.len()).map_or(true, |n| n >= 1_u64 << 32) {
        return Err(TrainError::InvalidInput(
            "corpus must contain 1..2^32-1 positions",
        ));
    }
    if corpus[0] != 0 || *corpus.last().unwrap() != 0 {
        return Err(TrainError::InvalidInput(
            "corpus needs leading and trailing zero separators",
        ));
    }
    if lengths.is_empty() || lengths.iter().any(|&len| len != 1) {
        return Err(TrainError::InvalidInput(
            "all initial token lengths must equal one",
        ));
    }
    if pivots.len() != weights.len() || weights.contains(&0) {
        return Err(TrainError::InvalidInput(
            "pivots and positive weights must align",
        ));
    }
    if corpus.len() == 1 {
        if lengths.len() != 1 || !pivots.is_empty() {
            return Err(TrainError::InvalidInput(
                "empty corpus requires only ID zero and no weights",
            ));
        }
        return Ok(());
    }
    if corpus.len() < 3 || corpus[1] == 0 || corpus[corpus.len() - 2] == 0 {
        return Err(TrainError::InvalidInput(
            "each piece must contain a nonzero token",
        ));
    }
    let initial_count = u64::try_from(lengths.len())
        .map_err(|_| TrainError::Overflow("initial token count exceeds u64"))?;
    let merge_limit = u64::try_from(options.max_merges)
        .map_err(|_| TrainError::Overflow("merge limit exceeds u64"))?;
    if initial_count
        .checked_add(merge_limit)
        .is_none_or(|n| n > 1_u64 << 32)
    {
        return Err(TrainError::Overflow("fresh token IDs exceed u32"));
    }
    if pivots.first() != Some(&1) {
        return Err(TrainError::InvalidInput("first weight pivot must be one"));
    }
    let last = corpus.len() - 1;
    let mut previous_pivot = 0_usize;
    for &pivot in pivots {
        let pos = pivot as usize;
        if pos <= previous_pivot || pos >= last || corpus[pos - 1] != 0 || corpus[pos] == 0 {
            return Err(TrainError::InvalidInput(
                "pivots must be increasing piece starts",
            ));
        }
        previous_pivot = pos;
    }
    let mut seen = vec![false; lengths.len()];
    let mut weight_i = 0;
    let mut total_adjacency = 0_u64;
    for pos in 1..last {
        while weight_i + 1 < pivots.len() && pos >= pivots[weight_i + 1] as usize {
            weight_i += 1;
        }
        let id = corpus[pos] as usize;
        if id >= lengths.len() {
            return Err(TrainError::InvalidInput(
                "initial ID is outside the dense alphabet",
            ));
        }
        if id == 0 {
            if corpus[pos - 1] == 0 || corpus[pos + 1] == 0 {
                return Err(TrainError::InvalidInput(
                    "empty pieces or duplicate separators",
                ));
            }
        } else {
            seen[id] = true;
            if corpus[pos + 1] != 0 {
                total_adjacency = total_adjacency
                    .checked_add(weights[weight_i])
                    .ok_or(TrainError::Overflow("total weighted adjacency exceeds u64"))?;
            }
        }
    }
    if seen[1..].iter().any(|&present| !present) {
        return Err(TrainError::InvalidInput(
            "initial alphabet IDs must be dense and used",
        ));
    }
    Ok(())
}

fn subtract_frequency(
    frequencies: &mut HashMap<u64, u64>,
    key: u64,
    weight: u64,
) -> Result<(), TrainError> {
    let current = frequencies
        .get_mut(&key)
        .ok_or(TrainError::InternalInvariant(
            "affected pair has no frequency",
        ))?;
    *current = current
        .checked_sub(weight)
        .ok_or(TrainError::InternalInvariant(
            "affected pair frequency underflow",
        ))?;
    Ok(())
}

fn add_frequency(
    frequencies: &mut HashMap<u64, u64>,
    key: u64,
    weight: u64,
) -> Result<(), TrainError> {
    let current = frequencies.entry(key).or_default();
    *current = current
        .checked_add(weight)
        .ok_or(TrainError::Overflow("pair frequency exceeds u64"))?;
    Ok(())
}

struct Core<const UNCHECKED: bool> {
    backend: Endpoints<UNCHECKED>,
    lengths: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
    pair_pos: HashMap<u64, Vec<u32>>,
    frequencies: HashMap<u64, u64>,
    heap: BinaryHeap<HeapEntry>,
    initial_occurrence_bytes: usize,
    actual_merges: usize,
    position_visits: usize,
    stale_visits: usize,
    heap_pops: usize,
}

#[cfg_attr(feature = "profiling", hotpath::measure)]
pub fn train(prepared: Prepared, options: TrainOptions) -> Result<TrainResult, TrainError> {
    validate(&prepared, options)?;
    match options.bounds {
        Bounds::Checked => train_impl::<false>(prepared, options),
        Bounds::Unchecked => train_impl::<true>(prepared, options),
    }
}

#[cfg_attr(feature = "profiling", hotpath::measure)]
fn initialize<const UNCHECKED: bool>(
    prepared: Prepared,
    min_frequency: u64,
) -> Result<Core<UNCHECKED>, TrainError> {
    let Prepared {
        corpus,
        initial_lengths,
        pivots,
        weights,
    } = prepared;
    let backend = Endpoints::<UNCHECKED>::new(corpus);
    let mut pair_pos: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut frequencies: HashMap<u64, u64> = HashMap::new();
    let mut weight_i = 0;
    let mut initial_occurrences = 0_usize;
    for pos in 1..backend.len() - 1 {
        while weight_i + 1 < pivots.len() && pos >= pivots[weight_i + 1] as usize {
            weight_i += 1;
        }
        let a = backend.initial_token(pos);
        let b = backend.initial_token(pos + 1);
        if a != 0 && b != 0 {
            let key = pair_key(a, b);
            pair_pos.entry(key).or_default().push(pos as u32);
            add_frequency(&mut frequencies, key, weights[weight_i])?;
            initial_occurrences += 1;
        }
    }
    // Match Python's heapify: construct the initial heap in linear time.
    let candidates: Vec<_> = frequencies
        .iter()
        .filter_map(|(&key, &frequency)| {
            (frequency >= min_frequency).then_some(HeapEntry { frequency, key })
        })
        .collect();
    let heap = BinaryHeap::from(candidates);
    Ok(Core {
        backend,
        lengths: initial_lengths,
        pivots,
        weights,
        pair_pos,
        frequencies,
        heap,
        initial_occurrence_bytes: initial_occurrences * std::mem::size_of::<u32>(),
        actual_merges: 0,
        position_visits: 0,
        stale_visits: 0,
        heap_pops: 0,
    })
}

#[cfg_attr(feature = "profiling", hotpath::measure)]
fn pop_best<const UNCHECKED: bool>(
    core: &mut Core<UNCHECKED>,
    min_frequency: u64,
) -> Option<(u64, u64)> {
    loop {
        let entry = core.heap.pop()?;
        core.heap_pops += 1;
        let current = core.frequencies.get(&entry.key).copied().unwrap_or(0);
        if current < min_frequency {
            core.pair_pos.remove(&entry.key);
            continue;
        }
        if current != entry.frequency {
            core.heap.push(HeapEntry {
                frequency: current,
                key: entry.key,
            });
            continue;
        }
        return Some((entry.key, current));
    }
}

#[cfg_attr(feature = "profiling", hotpath::measure)]
fn apply_rule<const UNCHECKED: bool>(
    core: &mut Core<UNCHECKED>,
    key: u64,
    frequency: u64,
    min_frequency: u64,
) -> Result<Rule, TrainError> {
    let a = (key >> 32) as u32;
    let b = key as u32;
    let new_id = u32::try_from(core.lengths.len())
        .map_err(|_| TrainError::Overflow("fresh token ID exceeds u32"))?;
    let new_length = core.lengths[a as usize]
        .checked_add(core.lengths[b as usize])
        .ok_or(TrainError::Overflow("merged token length exceeds u32"))?;
    core.lengths.push(new_length);
    let positions = core
        .pair_pos
        .remove(&key)
        .ok_or(TrainError::InternalInvariant(
            "selected pair has no historical occurrence list",
        ))?;
    let mut new_pairs = HashSet::new();
    for pos in positions {
        core.position_visits += 1;
        let pos = pos as usize;
        let Some(context) = core.backend.inspect_pair(pos, a, b, &core.lengths) else {
            core.stale_visits += 1;
            continue;
        };
        let wi = core.pivots.partition_point(|&pivot| pivot as usize <= pos) - 1;
        let weight = core.weights[wi];
        subtract_frequency(&mut core.frequencies, key, weight)?;
        if context.left_id != 0 {
            subtract_frequency(&mut core.frequencies, pair_key(context.left_id, a), weight)?;
        }
        if context.right_id != 0 {
            subtract_frequency(&mut core.frequencies, pair_key(b, context.right_id), weight)?;
        }
        core.backend.merge_known(pos, context, new_id);
        core.actual_merges += 1;
        if context.left_id != 0 {
            let new_key = pair_key(context.left_id, new_id);
            add_frequency(&mut core.frequencies, new_key, weight)?;
            let before = context.before.ok_or(TrainError::InternalInvariant(
                "left neighbor lacks a boundary",
            ))?;
            core.pair_pos
                .entry(new_key)
                .or_default()
                .push(before as u32);
            new_pairs.insert(new_key);
        }
        if context.right_id != 0 {
            let new_key = pair_key(new_id, context.right_id);
            add_frequency(&mut core.frequencies, new_key, weight)?;
            core.pair_pos.entry(new_key).or_default().push(pos as u32);
            new_pairs.insert(new_key);
        }
    }
    for new_key in new_pairs {
        let new_frequency = core.frequencies[&new_key];
        if new_frequency >= min_frequency {
            core.heap.push(HeapEntry {
                frequency: new_frequency,
                key: new_key,
            });
        } else {
            core.pair_pos.remove(&new_key);
        }
    }
    core.frequencies.remove(&key);
    Ok(Rule {
        left: a,
        right: b,
        frequency,
    })
}

fn train_impl<const UNCHECKED: bool>(
    prepared: Prepared,
    options: TrainOptions,
) -> Result<TrainResult, TrainError> {
    let corpus_positions = prepared.corpus.len();
    if corpus_positions == 1 {
        return Ok(TrainResult {
            merges: Vec::new(),
            final_tokens: vec![0],
            init_seconds: 0.0,
            merge_seconds: 0.0,
            train_seconds: 0.0,
            rules: 0,
            actual_merges: 0,
            position_visits: 0,
            stale_visits: 0,
            heap_pops: 0,
            backend_buffer_bytes: 0,
            initial_occurrence_bytes: 0,
            max_token_length: 1,
            corpus_positions,
        });
    }
    let started = Instant::now();
    let mut core = initialize::<UNCHECKED>(prepared, options.min_frequency)?;
    let init_seconds = started.elapsed().as_secs_f64();
    let merge_started = Instant::now();
    let mut merges = Vec::new();
    for _ in 0..options.max_merges {
        let Some((key, frequency)) = pop_best(&mut core, options.min_frequency) else {
            break;
        };
        merges.push(apply_rule(
            &mut core,
            key,
            frequency,
            options.min_frequency,
        )?);
    }
    let merge_seconds = merge_started.elapsed().as_secs_f64();
    let train_seconds = started.elapsed().as_secs_f64();
    let final_tokens = core.backend.final_tokens(&core.lengths);
    let max_token_length = core.lengths.iter().copied().max().unwrap_or(1);
    Ok(TrainResult {
        rules: merges.len(),
        merges,
        final_tokens,
        init_seconds,
        merge_seconds,
        train_seconds,
        actual_merges: core.actual_merges,
        position_visits: core.position_visits,
        stale_visits: core.stale_visits,
        heap_pops: core.heap_pops,
        backend_buffer_bytes: core.backend.len() * std::mem::size_of::<u32>(),
        initial_occurrence_bytes: core.initial_occurrence_bytes,
        max_token_length,
        corpus_positions,
    })
}

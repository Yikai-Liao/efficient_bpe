//! Rule-local occurrence parallelism on one continuous, unsplit corpus.
//!
//! Persistent workers first validate historical positions against one stable
//! snapshot. The coordinator selects AA run matches left-to-right and corrects
//! the left context of directly adjacent matches (ABAB) without an overlay.
//! A validation barrier precedes the disjoint-write apply barrier. The two
//! barriers and the remaining serial selection are reported separately.

use super::{
    CoreStats, HeapEntry, add_delta, apply_delta, core_result, initial_heap, metric, pair_key,
    pop_best,
};
use crate::ablation::{Options, Result};
use crate::{Bounds, Prepared, Rule, TrainError};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Clone, Copy)]
struct Plan {
    pos: u32,
    before: u32,
    left_id: u32,
    right_id: u32,
    weight: u64,
}

#[derive(Clone, Copy)]
struct Valid {
    pos: u32,
}

#[inline(always)]
fn load<const UNCHECKED: bool>(corpus: &[AtomicU32], pos: usize) -> u32 {
    if UNCHECKED {
        // SAFETY: The public parallel entry validates corpus length/IDs and
        // separators. Historical starts are created only from validated
        // in-range positions; each derived boundary is checked before load.
        // No worker writes during validation or selection (phase barrier).
        unsafe { corpus.get_unchecked(pos) }.load(Ordering::Relaxed)
    } else {
        corpus[pos].load(Ordering::Relaxed)
    }
}

#[inline(always)]
fn store<const UNCHECKED: bool>(corpus: &[AtomicU32], pos: usize, value: u32) {
    if UNCHECKED {
        // SAFETY: A Plan is emitted only after validating pos, right and
        // after within the corpus. The task carries the same token lengths;
        // writes stay inside its accepted disjoint token span. Atomic stores
        // also prevent a data race if this ownership invariant regresses.
        unsafe { corpus.get_unchecked(pos) }.store(value, Ordering::Relaxed);
    } else {
        corpus[pos].store(value, Ordering::Relaxed);
    }
}

/// The input is in historical append order, which is physical position order.
/// For AA, snapshot-valid edges form runs; greedily taking an edge then
/// skipping any edge inside its span picks even edges in each run. For AB,
/// valid matches cannot overlap. A later match directly adjacent to an
/// earlier one sees the earlier new token as its left neighbor, while its
/// right context is still the stable snapshot.
fn select_plans<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    valid: &[Valid],
    key: u64,
    new_id: u32,
) -> std::result::Result<(Vec<Plan>, usize), TrainError> {
    let a = (key >> 32) as u32;
    let b = key as u32;
    let mut plans = Vec::new();
    let mut overlap_stale = 0;
    let mut previous_after = 0;
    let mut previous_pos = 0;
    for item in valid {
        let pos = item.pos as usize;
        let right =
            pos.checked_add(lengths[a as usize] as usize)
                .ok_or(TrainError::InternalInvariant(
                    "snapshot right boundary overflow",
                ))?;
        let after = right.checked_add(lengths[b as usize] as usize).ok_or(
            TrainError::InternalInvariant("snapshot end boundary overflow"),
        )?;
        if after >= corpus.len() {
            return Err(TrainError::InternalInvariant(
                "validated end exceeds corpus",
            ));
        }
        if pos < previous_after {
            if a != b {
                return Err(TrainError::InternalInvariant(
                    "distinct pair matches overlap",
                ));
            }
            overlap_stale += 1;
            continue;
        }
        let (before, left_id) = if pos == previous_after {
            (previous_pos, new_id)
        } else {
            let predecessor = load::<UNCHECKED>(corpus, pos - 1);
            let before = pos
                .checked_sub(lengths[predecessor as usize] as usize)
                .ok_or(TrainError::InternalInvariant(
                    "snapshot predecessor underflow",
                ))?;
            (before, load::<UNCHECKED>(corpus, before))
        };
        let right_id = load::<UNCHECKED>(corpus, after);
        let wi = pivots.partition_point(|&pivot| pivot as usize <= pos) - 1;
        plans.push(Plan {
            pos: item.pos,
            before: before as u32,
            left_id,
            right_id,
            weight: weights[wi],
        });
        previous_pos = pos;
        previous_after = after;
    }
    Ok((plans, overlap_stale))
}

struct ValidateTask {
    positions: Arc<Vec<u32>>,
    start: usize,
    end: usize,
    a: u32,
    b: u32,
    a_length: usize,
    b_length: usize,
}

struct ApplyTask {
    plans: Arc<Vec<Plan>>,
    start: usize,
    end: usize,
    key: u64,
    new_id: u32,
    a_length: usize,
    b_length: usize,
}

struct WorkResult {
    deltas: HashMap<u64, i128>,
    new_occurrences: Vec<(u64, u32)>,
    merged: usize,
    work_seconds: f64,
}

struct ValidatedResult {
    valid: Vec<Valid>,
    ordered: bool,
    first: u32,
    last: u32,
    visits: usize,
    work_seconds: f64,
}

enum Command {
    Validate(ValidateTask),
    Apply(ApplyTask),
    Finish,
}

enum Reply {
    Validated(ValidatedResult),
    Applied(WorkResult),
}

struct Worker {
    commands: Sender<Command>,
    replies: Receiver<Reply>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Error paths drop the pool too; closing the round and joining here
        // prevents worker threads from outliving the training call.
        let _ = self.commands.send(Command::Finish);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn spawn_worker<const UNCHECKED: bool>(corpus: Arc<Vec<AtomicU32>>) -> Worker {
    let (commands, command_rx) = mpsc::channel();
    let (reply_tx, replies) = mpsc::channel();
    let handle = thread::spawn(move || {
        while let Ok(command) = command_rx.recv() {
            match command {
                Command::Validate(task) => {
                    let started = Instant::now();
                    let positions = &task.positions[task.start..task.end];
                    let mut valid = Vec::new();
                    let mut ordered = true;
                    let mut prior = positions[0];
                    let last = corpus.len() - 1;
                    for &raw_pos in positions {
                        if raw_pos < prior {
                            ordered = false;
                        }
                        prior = raw_pos;
                        let pos = raw_pos as usize;
                        if pos == 0 || pos >= last || load::<UNCHECKED>(&corpus, pos) != task.a {
                            continue;
                        }
                        let Some(right) = pos.checked_add(task.a_length) else {
                            continue;
                        };
                        if right >= last || load::<UNCHECKED>(&corpus, right) != task.b {
                            continue;
                        }
                        let Some(after) = right.checked_add(task.b_length) else {
                            continue;
                        };
                        if after <= last {
                            valid.push(Valid { pos: raw_pos });
                        }
                    }
                    let result = ValidatedResult {
                        valid,
                        ordered,
                        first: positions[0],
                        last: *positions.last().unwrap(),
                        visits: positions.len(),
                        work_seconds: started.elapsed().as_secs_f64(),
                    };
                    drop(task); // release the position Arc before the barrier reply
                    if reply_tx.send(Reply::Validated(result)).is_err() {
                        return;
                    }
                }
                Command::Apply(task) => {
                    let started = Instant::now();
                    let a = (task.key >> 32) as u32;
                    let b = task.key as u32;
                    let mut deltas = HashMap::new();
                    let mut new_occurrences = Vec::new();
                    for plan in &task.plans[task.start..task.end] {
                        let pos = plan.pos as usize;
                        let right = pos + task.a_length;
                        let after = right + task.b_length;
                        let weight = i128::from(plan.weight);
                        add_delta(&mut deltas, task.key, -weight);
                        if plan.left_id != 0 {
                            add_delta(&mut deltas, pair_key(plan.left_id, a), -weight);
                        }
                        if plan.right_id != 0 {
                            add_delta(&mut deltas, pair_key(b, plan.right_id), -weight);
                        }
                        store::<UNCHECKED>(&corpus, pos, task.new_id);
                        if after - right == 1 {
                            store::<UNCHECKED>(&corpus, right, task.new_id);
                        } else {
                            store::<UNCHECKED>(&corpus, right, 0);
                            store::<UNCHECKED>(&corpus, after - 1, task.new_id);
                        }
                        if plan.left_id != 0 {
                            let new_key = pair_key(plan.left_id, task.new_id);
                            add_delta(&mut deltas, new_key, weight);
                            new_occurrences.push((new_key, plan.before));
                        }
                        if plan.right_id != 0 {
                            let new_key = pair_key(task.new_id, plan.right_id);
                            add_delta(&mut deltas, new_key, weight);
                            new_occurrences.push((new_key, plan.pos));
                        }
                    }
                    let result = WorkResult {
                        deltas,
                        new_occurrences,
                        merged: task.end - task.start,
                        work_seconds: started.elapsed().as_secs_f64(),
                    };
                    drop(task); // release the plan Arc before the barrier reply
                    if reply_tx.send(Reply::Applied(result)).is_err() {
                        return;
                    }
                }
                Command::Finish => return,
            }
        }
    });
    Worker {
        commands,
        replies,
        handle: Some(handle),
    }
}

/// A small rule can be completed by the coordinator in physical occurrence
/// order. Reading the already updated endpoints gives exactly the same AA
/// overlap and adjacent ABAB context as the serial reference implementation.
fn apply_local<const UNCHECKED: bool>(
    corpus: &[AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    positions: &[u32],
    key: u64,
    new_id: u32,
) -> std::result::Result<(WorkResult, usize), TrainError> {
    let started = Instant::now();
    let a = (key >> 32) as u32;
    let b = key as u32;
    let a_length = lengths[a as usize] as usize;
    let b_length = lengths[b as usize] as usize;
    let last = corpus.len() - 1;
    let mut deltas = HashMap::new();
    let mut new_occurrences = Vec::new();
    let mut merged = 0;
    let mut stale = 0;
    let mut prior = 0;
    for &raw_pos in positions {
        if raw_pos < prior {
            return Err(TrainError::InternalInvariant(
                "historical positions are not physically ordered",
            ));
        }
        prior = raw_pos;
        let pos = raw_pos as usize;
        if pos == 0 || pos >= last || load::<UNCHECKED>(corpus, pos) != a {
            stale += 1;
            continue;
        }
        let right = pos
            .checked_add(a_length)
            .ok_or(TrainError::InternalInvariant(
                "local right boundary overflow",
            ))?;
        if right >= last || load::<UNCHECKED>(corpus, right) != b {
            stale += 1;
            continue;
        }
        let after = right
            .checked_add(b_length)
            .ok_or(TrainError::InternalInvariant("local end boundary overflow"))?;
        if after > last {
            return Err(TrainError::InternalInvariant("local end exceeds corpus"));
        }
        let predecessor = load::<UNCHECKED>(corpus, pos - 1);
        let before = pos
            .checked_sub(lengths[predecessor as usize] as usize)
            .ok_or(TrainError::InternalInvariant("local predecessor underflow"))?;
        let left_id = load::<UNCHECKED>(corpus, before);
        let right_id = load::<UNCHECKED>(corpus, after);
        let wi = pivots.partition_point(|&pivot| pivot as usize <= pos) - 1;
        let weight = i128::from(weights[wi]);
        add_delta(&mut deltas, key, -weight);
        if left_id != 0 {
            add_delta(&mut deltas, pair_key(left_id, a), -weight);
        }
        if right_id != 0 {
            add_delta(&mut deltas, pair_key(b, right_id), -weight);
        }
        store::<UNCHECKED>(corpus, pos, new_id);
        if b_length == 1 {
            store::<UNCHECKED>(corpus, right, new_id);
        } else {
            store::<UNCHECKED>(corpus, right, 0);
            store::<UNCHECKED>(corpus, after - 1, new_id);
        }
        if left_id != 0 {
            let new_key = pair_key(left_id, new_id);
            add_delta(&mut deltas, new_key, weight);
            new_occurrences.push((new_key, before as u32));
        }
        if right_id != 0 {
            let new_key = pair_key(new_id, right_id);
            add_delta(&mut deltas, new_key, weight);
            new_occurrences.push((new_key, raw_pos));
        }
        merged += 1;
    }
    Ok((
        WorkResult {
            deltas,
            new_occurrences,
            merged,
            work_seconds: started.elapsed().as_secs_f64(),
        },
        stale,
    ))
}

pub(super) fn train(input: Prepared, options: Options) -> std::result::Result<Result, TrainError> {
    match options.bounds {
        Bounds::Checked => train_impl::<false>(input, options, None),
        Bounds::Unchecked => train_impl::<true>(input, options, None),
    }
}

pub(super) fn train_adaptive(
    input: Prepared,
    options: Options,
    grain: usize,
) -> std::result::Result<Result, TrainError> {
    match options.bounds {
        Bounds::Checked => train_impl::<false>(input, options, Some(grain)),
        Bounds::Unchecked => train_impl::<true>(input, options, Some(grain)),
    }
}

fn train_impl<const UNCHECKED: bool>(
    input: Prepared,
    options: Options,
    adaptive_grain: Option<usize>,
) -> std::result::Result<Result, TrainError> {
    let started = Instant::now();
    let Prepared {
        corpus,
        initial_lengths,
        pivots,
        weights,
    } = input;
    let corpus_positions = corpus.len();
    let mut lengths = initial_lengths;
    // Consumes the owned u32 corpus once. No per-rule corpus clone is made.
    let corpus: Arc<Vec<AtomicU32>> = Arc::new(corpus.into_iter().map(AtomicU32::new).collect());
    let last = corpus.len() - 1;
    let mut pair_pos: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut frequencies: HashMap<u64, u64> = HashMap::new();
    let mut weight_i = 0;
    let mut initial_occurrences = 0;
    for pos in 1..last {
        while weight_i + 1 < pivots.len() && pos >= pivots[weight_i + 1] as usize {
            weight_i += 1;
        }
        let a = load::<UNCHECKED>(&corpus, pos);
        let b = load::<UNCHECKED>(&corpus, pos + 1);
        if a != 0 && b != 0 {
            let key = pair_key(a, b);
            pair_pos.entry(key).or_default().push(pos as u32);
            let value = frequencies.entry(key).or_insert(0_u64);
            *value = value
                .checked_add(weights[weight_i])
                .ok_or(TrainError::Overflow("initial pair frequency exceeds u64"))?;
            initial_occurrences += 1;
        }
    }
    let mut heap = initial_heap(&frequencies, options.min_frequency);
    let worker_count = options.workers.min(initial_occurrences.max(1));
    let worker_threads = if adaptive_grain.is_some() && worker_count == 1 {
        0
    } else {
        worker_count
    };
    let workers: Vec<Worker> = (0..worker_threads)
        .map(|_| spawn_worker::<UNCHECKED>(Arc::clone(&corpus)))
        .collect();
    let init_seconds = started.elapsed().as_secs_f64();

    let merge_started = Instant::now();
    let mut merges = Vec::new();
    let mut actual_merges = 0;
    let mut position_visits = 0;
    let mut stale_visits = 0;
    let mut heap_pops = 0;
    let mut plan_seconds = 0.0;
    let mut validation_seconds = 0.0;
    let mut serial_select_seconds = 0.0;
    let mut apply_seconds = 0.0;
    let mut reduce_seconds = 0.0;
    let mut sync_seconds = 0.0;
    let mut serial_seconds = 0.0;
    let mut serial_rounds = 0;
    let mut serial_visits = 0;
    let mut parallel_rounds = 0;
    let mut plan_items_peak = 0;
    let mut plan_capacity_peak_bytes = 0;
    let mut valid_items_peak = 0;
    let mut valid_capacity_peak_bytes = 0;
    let mut selected_position_capacity_peak_bytes = 0;
    let mut dispatches = vec![0_usize; worker_count];
    let mut validation_dispatches = vec![0_usize; worker_count];
    let mut worker_validated = vec![0_usize; worker_count];
    let mut worker_validation_work = vec![0.0; worker_count];
    let mut worker_merges = vec![0_usize; worker_count];
    let mut worker_work = vec![0.0; worker_count];
    for _ in 0..options.max_merges {
        let Some((key, frequency)) = pop_best(
            &mut heap,
            &frequencies,
            options.min_frequency,
            &mut heap_pops,
        ) else {
            break;
        };
        let a = (key >> 32) as u32;
        let b = key as u32;
        let new_id = u32::try_from(lengths.len())
            .map_err(|_| TrainError::Overflow("fresh token ID exceeds u32"))?;
        let new_length = lengths[a as usize]
            .checked_add(lengths[b as usize])
            .ok_or(TrainError::Overflow("merged token length exceeds u32"))?;
        lengths.push(new_length);
        merges.push(Rule {
            left: a,
            right: b,
            frequency,
        });
        let positions = pair_pos.remove(&key).ok_or(TrainError::InternalInvariant(
            "selected pair has no historical positions",
        ))?;
        selected_position_capacity_peak_bytes = selected_position_capacity_peak_bytes
            .max(positions.capacity() * std::mem::size_of::<u32>());
        position_visits += positions.len();
        if positions.is_empty() {
            return Err(TrainError::InternalInvariant(
                "eligible rule has no historical position",
            ));
        }
        let local_round = adaptive_grain.is_some_and(|grain| {
            worker_count == 1 || positions.len() < grain.saturating_mul(worker_count)
        });
        let results = if local_round {
            let (result, stale) = apply_local::<UNCHECKED>(
                &corpus, &lengths, &pivots, &weights, &positions, key, new_id,
            )?;
            serial_seconds += result.work_seconds;
            serial_rounds += 1;
            serial_visits += positions.len();
            stale_visits += stale;
            vec![result]
        } else {
            parallel_rounds += 1;
            let plan_started = Instant::now();
            let validate_started = Instant::now();
            let positions = Arc::new(positions);
            let validate_count = worker_count.min(positions.len());
            for i in 0..validate_count {
                let start = i * positions.len() / validate_count;
                let end = (i + 1) * positions.len() / validate_count;
                workers[i]
                    .commands
                    .send(Command::Validate(ValidateTask {
                        positions: Arc::clone(&positions),
                        start,
                        end,
                        a,
                        b,
                        a_length: lengths[a as usize] as usize,
                        b_length: lengths[b as usize] as usize,
                    }))
                    .map_err(|_| {
                        TrainError::InternalInvariant("snapshot validator disconnected")
                    })?;
                validation_dispatches[i] += 1;
            }
            let validation_wait = Instant::now();
            let mut validated = Vec::with_capacity(positions.len());
            let mut previous_raw = None;
            for (i, worker) in workers.iter().take(validate_count).enumerate() {
                let result = match worker.replies.recv().map_err(|_| {
                    TrainError::InternalInvariant("snapshot validator dropped result")
                })? {
                    Reply::Validated(result) => result,
                    Reply::Applied(_) => {
                        return Err(TrainError::InternalInvariant(
                            "unexpected apply reply in validation phase",
                        ));
                    }
                };
                if !result.ordered || previous_raw.is_some_and(|last| last > result.first) {
                    return Err(TrainError::InternalInvariant(
                        "historical positions are not physically ordered",
                    ));
                }
                previous_raw = Some(result.last);
                worker_validated[i] += result.visits;
                worker_validation_work[i] += result.work_seconds;
                validated.extend(result.valid);
            }
            sync_seconds += validation_wait.elapsed().as_secs_f64();
            validation_seconds += validate_started.elapsed().as_secs_f64();
            stale_visits += positions.len() - validated.len();
            drop(positions);
            valid_items_peak = valid_items_peak.max(validated.len());
            valid_capacity_peak_bytes =
                valid_capacity_peak_bytes.max(validated.capacity() * std::mem::size_of::<Valid>());
            let select_started = Instant::now();
            let (plans, overlap_stale) = select_plans::<UNCHECKED>(
                &corpus, &lengths, &pivots, &weights, &validated, key, new_id,
            )?;
            drop(validated);
            serial_select_seconds += select_started.elapsed().as_secs_f64();
            stale_visits += overlap_stale;
            plan_seconds += plan_started.elapsed().as_secs_f64();
            plan_items_peak = plan_items_peak.max(plans.len());
            plan_capacity_peak_bytes =
                plan_capacity_peak_bytes.max(plans.capacity() * std::mem::size_of::<Plan>());
            let target_count = worker_count.min(plans.len());
            if target_count == 0 {
                return Err(TrainError::InternalInvariant(
                    "eligible rule has no live occurrence",
                ));
            }
            let plans = Arc::new(plans);
            let apply_started = Instant::now();
            for i in 0..target_count {
                let start = i * plans.len() / target_count;
                let end = (i + 1) * plans.len() / target_count;
                workers[i]
                    .commands
                    .send(Command::Apply(ApplyTask {
                        plans: Arc::clone(&plans),
                        start,
                        end,
                        key,
                        new_id,
                        a_length: lengths[a as usize] as usize,
                        b_length: lengths[b as usize] as usize,
                    }))
                    .map_err(|_| TrainError::InternalInvariant("occurrence worker disconnected"))?;
                dispatches[i] += 1;
            }
            drop(plans);
            let wait_started = Instant::now();
            let mut results = Vec::with_capacity(target_count);
            for worker in workers.iter().take(target_count) {
                let result = worker.replies.recv().map_err(|_| {
                    TrainError::InternalInvariant("snapshot apply worker dropped result")
                })?;
                match result {
                    Reply::Applied(result) => results.push(result),
                    Reply::Validated(_) => {
                        return Err(TrainError::InternalInvariant(
                            "unexpected validation reply in apply phase",
                        ));
                    }
                }
            }
            sync_seconds += wait_started.elapsed().as_secs_f64();
            apply_seconds += apply_started.elapsed().as_secs_f64();
            results
        };
        let reduce_started = Instant::now();
        let mut fresh = HashSet::new();
        // Results are reduced in contiguous plan-range order, preserving the
        // exact historical append order for each new pair's occurrence list.
        for (i, result) in results.into_iter().enumerate() {
            actual_merges += result.merged;
            if !local_round {
                worker_merges[i] += result.merged;
                worker_work[i] += result.work_seconds;
            }
            for (pair, delta) in result.deltas {
                apply_delta(&mut frequencies, pair, delta)?;
            }
            for (pair, pos) in result.new_occurrences {
                pair_pos.entry(pair).or_default().push(pos);
                fresh.insert(pair);
            }
        }
        for pair in fresh {
            let current = frequencies[&pair];
            if current >= options.min_frequency {
                heap.push(HeapEntry {
                    frequency: current,
                    key: pair,
                });
            } else {
                pair_pos.remove(&pair);
            }
        }
        frequencies.remove(&key);
        reduce_seconds += reduce_started.elapsed().as_secs_f64();
    }
    let merge_seconds = merge_started.elapsed().as_secs_f64();
    drop(workers);
    let mut final_tokens = Vec::new();
    let mut pos = 0;
    loop {
        let token = load::<UNCHECKED>(&corpus, pos);
        final_tokens.push(token);
        if pos == last {
            break;
        }
        pos += lengths[token as usize] as usize;
        if pos > last {
            return Err(TrainError::InternalInvariant(
                "final corpus boundary overflow",
            ));
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
            initial_occurrence_bytes: initial_occurrences * std::mem::size_of::<u32>(),
            max_token_length,
            corpus_positions,
        },
    );
    let mut metrics = BTreeMap::new();
    metric(&mut metrics, "workers_requested", options.workers as f64);
    metric(&mut metrics, "workers_actual", worker_count as f64);
    metric(
        &mut metrics,
        "worker_threads_spawned",
        worker_threads as f64,
    );
    metric(
        &mut metrics,
        "shared_corpus_bytes",
        corpus_positions as f64 * 4.0,
    );
    metric(&mut metrics, "plan_items_peak", plan_items_peak as f64);
    metric(&mut metrics, "valid_items_peak", valid_items_peak as f64);
    metric(
        &mut metrics,
        "plan_item_bytes",
        std::mem::size_of::<Plan>() as f64,
    );
    metric(
        &mut metrics,
        "valid_item_bytes",
        std::mem::size_of::<Valid>() as f64,
    );
    metric(
        &mut metrics,
        "plan_capacity_peak_bytes",
        plan_capacity_peak_bytes as f64,
    );
    metric(
        &mut metrics,
        "valid_capacity_peak_bytes",
        valid_capacity_peak_bytes as f64,
    );
    metric(
        &mut metrics,
        "selected_position_capacity_peak_bytes",
        selected_position_capacity_peak_bytes as f64,
    );
    metric(
        &mut metrics,
        "plan_capacity_peak_bytes_lower_bound",
        plan_items_peak as f64 * std::mem::size_of::<Plan>() as f64,
    );
    metric(&mut metrics, "plan_seconds", plan_seconds);
    metric(&mut metrics, "validation_seconds", validation_seconds);
    metric(&mut metrics, "serial_select_seconds", serial_select_seconds);
    metric(&mut metrics, "apply_seconds", apply_seconds);
    metric(&mut metrics, "reduce_seconds", reduce_seconds);
    metric(&mut metrics, "sync_seconds", sync_seconds);
    metric(&mut metrics, "serial_seconds", serial_seconds);
    metric(&mut metrics, "serial_rounds", serial_rounds as f64);
    metric(&mut metrics, "serial_visits", serial_visits as f64);
    metric(&mut metrics, "parallel_rounds", parallel_rounds as f64);
    metric(
        &mut metrics,
        "adaptive_grain",
        adaptive_grain.unwrap_or(0) as f64,
    );
    metric(
        &mut metrics,
        "round_messages",
        (2 * (validation_dispatches.iter().sum::<usize>() + dispatches.iter().sum::<usize>()))
            as f64,
    );
    metric(
        &mut metrics,
        "messages_per_round",
        if core.rules == 0 {
            0.0
        } else {
            2.0 * (validation_dispatches.iter().sum::<usize>() + dispatches.iter().sum::<usize>())
                as f64
                / core.rules as f64
        },
    );
    metric(&mut metrics, "barriers", (2 * parallel_rounds) as f64);
    metric(&mut metrics, "actualbarriers", (2 * parallel_rounds) as f64);
    metric(
        &mut metrics,
        "unchecked_corpus_access",
        if UNCHECKED { 1.0 } else { 0.0 },
    );
    for i in 0..worker_count {
        metric(
            &mut metrics,
            &format!("worker_{i}_validation_dispatches"),
            validation_dispatches[i] as f64,
        );
        metric(
            &mut metrics,
            &format!("worker_{i}_validated"),
            worker_validated[i] as f64,
        );
        metric(
            &mut metrics,
            &format!("worker_{i}_validation_work_seconds"),
            worker_validation_work[i],
        );
        metric(
            &mut metrics,
            &format!("worker_{i}_dispatches"),
            dispatches[i] as f64,
        );
        metric(
            &mut metrics,
            &format!("worker_{i}_skips"),
            (core.rules - dispatches[i]) as f64,
        );
        metric(
            &mut metrics,
            &format!("worker_{i}_merges"),
            worker_merges[i] as f64,
        );
        metric(
            &mut metrics,
            &format!("worker_{i}_work_seconds"),
            worker_work[i],
        );
    }
    Ok(Result { core, metrics })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_reports_unordered_historical_positions() {
        let corpus = Arc::new(
            vec![0, 1, 2, 1, 2, 0]
                .into_iter()
                .map(AtomicU32::new)
                .collect(),
        );
        let worker = spawn_worker::<false>(corpus);
        worker
            .commands
            .send(Command::Validate(ValidateTask {
                positions: Arc::new(vec![3, 1]),
                start: 0,
                end: 2,
                a: 1,
                b: 2,
                a_length: 1,
                b_length: 1,
            }))
            .unwrap();
        match worker.replies.recv().unwrap() {
            Reply::Validated(result) => assert!(!result.ordered),
            Reply::Applied(_) => panic!("validation returned apply result"),
        }
    }

    #[test]
    fn compact_snapshot_records() {
        assert_eq!(std::mem::size_of::<Plan>(), 24);
        assert_eq!(std::mem::size_of::<Valid>(), 4);
    }
}

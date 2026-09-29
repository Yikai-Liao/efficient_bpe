//! Rule-local occurrence parallelism on one continuous, unsplit corpus.
//!
//! Planning reads a stable corpus plus a sparse overlay of this rule's earlier
//! endpoint writes. The overlay reproduces left-to-right replacement order,
//! including AA overlap and adjacent ABAB boundary changes. Accepted token
//! spans are disjoint. Only after planning ends do persistent workers apply
//! their disjoint endpoint writes. The coordinator reduces results in plan
//! order before selecting another global rule. Atomics make the shared corpus
//! data-race-free; channel replies are the phase barrier.

use super::{
    CoreStats, HeapEntry, add_delta, apply_delta, core_result, initial_heap, metric, pair_key,
    pop_best,
};
use crate::ablation::{Options, Result};
use crate::{Prepared, Rule, TrainError};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Clone, Copy)]
struct Plan {
    pos: usize,
    before: usize,
    left_id: u32,
    right: usize,
    after: usize,
    right_id: u32,
    weight: u64,
}

#[inline(always)]
fn read(corpus: &[AtomicU32], overlay: &HashMap<usize, u32>, pos: usize) -> u32 {
    overlay
        .get(&pos)
        .copied()
        .unwrap_or_else(|| corpus[pos].load(Ordering::Relaxed))
}

/// Serial conflict resolution over historical positions, without scanning or
/// copying the whole corpus per rule. The overlay has O(actual merges) entries.
fn plan_rule(
    corpus: &[AtomicU32],
    lengths: &[u32],
    pivots: &[u32],
    weights: &[u64],
    positions: &[u32],
    key: u64,
    new_id: u32,
) -> std::result::Result<(Vec<Plan>, usize), TrainError> {
    let a = (key >> 32) as u32;
    let b = key as u32;
    let last = corpus.len() - 1;
    let mut overlay = HashMap::new();
    let mut plans = Vec::new();
    let mut stale = 0;
    let mut previous_after = 0;
    for &raw_pos in positions {
        let pos = raw_pos as usize;
        if pos == 0 || pos >= last || read(corpus, &overlay, pos) != a {
            stale += 1;
            continue;
        }
        let right =
            pos.checked_add(lengths[a as usize] as usize)
                .ok_or(TrainError::InternalInvariant(
                    "planned right boundary overflow",
                ))?;
        if right >= last || read(corpus, &overlay, right) != b {
            stale += 1;
            continue;
        }
        let after = right.checked_add(lengths[b as usize] as usize).ok_or(
            TrainError::InternalInvariant("planned end boundary overflow"),
        )?;
        if after > last || pos < previous_after {
            return Err(TrainError::InternalInvariant("planned merges overlap"));
        }
        let predecessor = read(corpus, &overlay, pos - 1);
        let before = pos
            .checked_sub(lengths[predecessor as usize] as usize)
            .ok_or(TrainError::InternalInvariant(
                "planned predecessor underflow",
            ))?;
        let left_id = read(corpus, &overlay, before);
        let right_id = read(corpus, &overlay, after);
        let wi = pivots.partition_point(|&pivot| pivot as usize <= pos) - 1;
        plans.push(Plan {
            pos,
            before,
            left_id,
            right,
            after,
            right_id,
            weight: weights[wi],
        });
        // Simulate the Lean two/three-write endpoint update for later
        // historical occurrences in this same rule. These sparse writes are
        // materialized in the shared corpus only after planning completes.
        overlay.insert(pos, new_id);
        if after - right == 1 {
            overlay.insert(right, new_id);
        } else {
            overlay.insert(right, 0);
            overlay.insert(after - 1, new_id);
        }
        previous_after = after;
    }
    Ok((plans, stale))
}

struct Task {
    plans: Arc<Vec<Plan>>,
    start: usize,
    end: usize,
    key: u64,
    new_id: u32,
}

struct WorkResult {
    deltas: HashMap<u64, i128>,
    new_occurrences: Vec<(u64, u32)>,
    merged: usize,
    work_seconds: f64,
}

enum Command {
    Apply(Task),
    Finish,
}

struct Worker {
    commands: Sender<Command>,
    replies: Receiver<WorkResult>,
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

fn spawn_worker(corpus: Arc<Vec<AtomicU32>>) -> Worker {
    let (commands, command_rx) = mpsc::channel();
    let (reply_tx, replies) = mpsc::channel();
    let handle = thread::spawn(move || {
        while let Ok(command) = command_rx.recv() {
            match command {
                Command::Apply(task) => {
                    let started = Instant::now();
                    let a = (task.key >> 32) as u32;
                    let b = task.key as u32;
                    let mut deltas = HashMap::new();
                    let mut new_occurrences = Vec::new();
                    for plan in &task.plans[task.start..task.end] {
                        let weight = i128::from(plan.weight);
                        add_delta(&mut deltas, task.key, -weight);
                        if plan.left_id != 0 {
                            add_delta(&mut deltas, pair_key(plan.left_id, a), -weight);
                        }
                        if plan.right_id != 0 {
                            add_delta(&mut deltas, pair_key(b, plan.right_id), -weight);
                        }
                        corpus[plan.pos].store(task.new_id, Ordering::Relaxed);
                        if plan.after - plan.right == 1 {
                            corpus[plan.right].store(task.new_id, Ordering::Relaxed);
                        } else {
                            corpus[plan.right].store(0, Ordering::Relaxed);
                            corpus[plan.after - 1].store(task.new_id, Ordering::Relaxed);
                        }
                        if plan.left_id != 0 {
                            let new_key = pair_key(plan.left_id, task.new_id);
                            add_delta(&mut deltas, new_key, weight);
                            new_occurrences.push((new_key, plan.before as u32));
                        }
                        if plan.right_id != 0 {
                            let new_key = pair_key(task.new_id, plan.right_id);
                            add_delta(&mut deltas, new_key, weight);
                            new_occurrences.push((new_key, plan.pos as u32));
                        }
                    }
                    if reply_tx
                        .send(WorkResult {
                            deltas,
                            new_occurrences,
                            merged: task.end - task.start,
                            work_seconds: started.elapsed().as_secs_f64(),
                        })
                        .is_err()
                    {
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

pub(super) fn train(input: Prepared, options: Options) -> std::result::Result<Result, TrainError> {
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
        let a = corpus[pos].load(Ordering::Relaxed);
        let b = corpus[pos + 1].load(Ordering::Relaxed);
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
    let workers: Vec<Worker> = (0..worker_count)
        .map(|_| spawn_worker(Arc::clone(&corpus)))
        .collect();
    let init_seconds = started.elapsed().as_secs_f64();

    let merge_started = Instant::now();
    let mut merges = Vec::new();
    let mut actual_merges = 0;
    let mut position_visits = 0;
    let mut stale_visits = 0;
    let mut heap_pops = 0;
    let mut plan_seconds = 0.0;
    let mut apply_seconds = 0.0;
    let mut reduce_seconds = 0.0;
    let mut sync_seconds = 0.0;
    let mut plan_items_peak = 0;
    let mut dispatches = vec![0_usize; worker_count];
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
        position_visits += positions.len();
        let plan_started = Instant::now();
        let (plans, stale) = plan_rule(
            &corpus, &lengths, &pivots, &weights, &positions, key, new_id,
        )?;
        plan_seconds += plan_started.elapsed().as_secs_f64();
        stale_visits += stale;
        plan_items_peak = plan_items_peak.max(plans.len());
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
                .send(Command::Apply(Task {
                    plans: Arc::clone(&plans),
                    start,
                    end,
                    key,
                    new_id,
                }))
                .map_err(|_| TrainError::InternalInvariant("occurrence worker disconnected"))?;
            dispatches[i] += 1;
        }
        let wait_started = Instant::now();
        let mut results = Vec::with_capacity(target_count);
        for worker in workers.iter().take(target_count) {
            results.push(
                worker.replies.recv().map_err(|_| {
                    TrainError::InternalInvariant("occurrence worker dropped result")
                })?,
            );
        }
        sync_seconds += wait_started.elapsed().as_secs_f64();
        apply_seconds += apply_started.elapsed().as_secs_f64();
        let reduce_started = Instant::now();
        let mut fresh = HashSet::new();
        // Results are reduced in contiguous plan-range order, preserving the
        // exact historical append order for each new pair's occurrence list.
        for (i, result) in results.into_iter().enumerate() {
            actual_merges += result.merged;
            worker_merges[i] += result.merged;
            worker_work[i] += result.work_seconds;
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
        let token = corpus[pos].load(Ordering::Relaxed);
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
        "shared_corpus_bytes",
        corpus_positions as f64 * 4.0,
    );
    metric(&mut metrics, "plan_items_peak", plan_items_peak as f64);
    metric(
        &mut metrics,
        "plan_capacity_peak_bytes_lower_bound",
        plan_items_peak as f64 * std::mem::size_of::<Plan>() as f64,
    );
    metric(&mut metrics, "plan_seconds", plan_seconds);
    metric(&mut metrics, "apply_seconds", apply_seconds);
    metric(&mut metrics, "reduce_seconds", reduce_seconds);
    metric(&mut metrics, "sync_seconds", sync_seconds);
    metric(
        &mut metrics,
        "round_messages",
        (2 * dispatches.iter().sum::<usize>()) as f64,
    );
    metric(
        &mut metrics,
        "messages_per_round",
        if core.rules == 0 {
            0.0
        } else {
            2.0 * dispatches.iter().sum::<usize>() as f64 / core.rules as f64
        },
    );
    for i in 0..worker_count {
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

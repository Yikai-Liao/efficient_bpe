//! Whole-piece persistent workers. Every pair is globally selected first.

use super::{
    CoreStats, HeapEntry, add_delta, apply_delta, core_result, initial_heap, metric, pair_key,
    pop_best,
};
use crate::ablation::{Options, Result};
use crate::backend::Endpoints;
use crate::{Bounds, Prepared, Rule, TrainError};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

struct Piece {
    start: usize,
    end: usize, // exclusive of the permanent separator
    weight: u64,
}

struct Shard {
    corpus: Vec<u32>,
    lengths: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
    pieces: usize,
}

fn shards(input: &Prepared, requested: usize) -> Vec<Shard> {
    let mut pieces = Vec::new();
    let mut start = 1;
    for pos in 1..input.corpus.len() {
        if input.corpus[pos] == 0 {
            let wi = input
                .pivots
                .partition_point(|&pivot| pivot as usize <= start)
                - 1;
            pieces.push(Piece {
                start,
                end: pos,
                weight: input.weights[wi],
            });
            start = pos + 1;
        }
    }
    let count = requested.min(pieces.len());
    let mut result = Vec::with_capacity(count);
    for worker in 0..count {
        let begin = worker * pieces.len() / count;
        let stop = (worker + 1) * pieces.len() / count;
        let mut corpus = vec![0];
        let mut pivots = Vec::new();
        let mut weights = Vec::new();
        for piece in &pieces[begin..stop] {
            pivots.push(corpus.len() as u32);
            weights.push(piece.weight);
            corpus.extend_from_slice(&input.corpus[piece.start..=piece.end]);
        }
        result.push(Shard {
            corpus,
            lengths: input.initial_lengths.clone(),
            pivots,
            weights,
            pieces: stop - begin,
        });
    }
    result
}

struct Ready {
    counts: HashMap<u64, u64>,
    occurrence_bytes: usize,
    backend_bytes: usize,
    pieces: usize,
}

struct Round {
    deltas: HashMap<u64, i128>,
    new_keys: HashSet<u64>,
    visits: usize,
    stale: usize,
    merged: usize,
    work_seconds: f64,
}

struct Finished {
    tokens: Vec<u32>,
    backend_bytes: usize,
}

struct PieceState<const UNCHECKED: bool> {
    backend: Endpoints<UNCHECKED>,
    lengths: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
    pair_pos: HashMap<u64, Vec<u32>>,
}

impl<const UNCHECKED: bool> PieceState<UNCHECKED> {
    fn new(shard: Shard) -> std::result::Result<(Self, Ready), TrainError> {
        let backend = Endpoints::new(shard.corpus);
        let mut pair_pos: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut counts = HashMap::new();
        let mut weight_i = 0;
        let mut occurrences = 0;
        for pos in 1..backend.len() - 1 {
            while weight_i + 1 < shard.pivots.len() && pos >= shard.pivots[weight_i + 1] as usize {
                weight_i += 1;
            }
            let a = backend.initial_token(pos);
            let b = backend.initial_token(pos + 1);
            if a != 0 && b != 0 {
                let key = pair_key(a, b);
                pair_pos.entry(key).or_default().push(pos as u32);
                let value = counts.entry(key).or_insert(0_u64);
                *value = value
                    .checked_add(shard.weights[weight_i])
                    .ok_or(TrainError::Overflow(
                        "local initial pair frequency exceeds u64",
                    ))?;
                occurrences += 1;
            }
        }
        let ready = Ready {
            counts,
            occurrence_bytes: occurrences * std::mem::size_of::<u32>(),
            backend_bytes: backend.len() * std::mem::size_of::<u32>(),
            pieces: shard.pieces,
        };
        Ok((
            Self {
                backend,
                lengths: shard.lengths,
                pivots: shard.pivots,
                weights: shard.weights,
                pair_pos,
            },
            ready,
        ))
    }

    fn merge_round(
        &mut self,
        key: u64,
        new_id: u32,
        new_length: u32,
        missing: Vec<u32>,
    ) -> std::result::Result<Round, TrainError> {
        let started = Instant::now();
        self.lengths.extend(missing);
        if self.lengths.len() != new_id as usize {
            return Err(TrainError::InternalInvariant(
                "worker token lengths diverged",
            ));
        }
        self.lengths.push(new_length);
        let a = (key >> 32) as u32;
        let b = key as u32;
        let positions = self.pair_pos.remove(&key).unwrap_or_default();
        let mut deltas = HashMap::new();
        let mut new_keys = HashSet::new();
        let mut visits = 0;
        let mut stale = 0;
        let mut merged = 0;
        for pos in positions {
            visits += 1;
            let pos = pos as usize;
            let Some(context) = self.backend.inspect_pair(pos, a, b, &self.lengths) else {
                stale += 1;
                continue;
            };
            let wi = self.pivots.partition_point(|&pivot| pivot as usize <= pos) - 1;
            let weight = i128::from(self.weights[wi]);
            add_delta(&mut deltas, key, -weight);
            if context.left_id != 0 {
                add_delta(&mut deltas, pair_key(context.left_id, a), -weight);
            }
            if context.right_id != 0 {
                add_delta(&mut deltas, pair_key(b, context.right_id), -weight);
            }
            self.backend.merge_known(pos, context, new_id);
            merged += 1;
            if context.left_id != 0 {
                let new_key = pair_key(context.left_id, new_id);
                add_delta(&mut deltas, new_key, weight);
                self.pair_pos
                    .entry(new_key)
                    .or_default()
                    .push(context.before.ok_or(TrainError::InternalInvariant(
                        "left context lacks a position",
                    ))? as u32);
                new_keys.insert(new_key);
            }
            if context.right_id != 0 {
                let new_key = pair_key(new_id, context.right_id);
                add_delta(&mut deltas, new_key, weight);
                self.pair_pos.entry(new_key).or_default().push(pos as u32);
                new_keys.insert(new_key);
            }
        }
        Ok(Round {
            deltas,
            new_keys,
            visits,
            stale,
            merged,
            work_seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn finish(self) -> Finished {
        Finished {
            tokens: self.backend.final_tokens(&self.lengths),
            backend_bytes: self.backend.len() * std::mem::size_of::<u32>(),
        }
    }
}

enum Command {
    Merge {
        key: u64,
        new_id: u32,
        new_length: u32,
        missing: Vec<u32>,
    },
    Finish,
}

enum Reply {
    Ready(std::result::Result<Ready, TrainError>),
    Round(std::result::Result<Round, TrainError>),
    Finished(Finished),
}

struct ThreadWorker {
    commands: Sender<Command>,
    replies: Receiver<Reply>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for ThreadWorker {
    fn drop(&mut self) {
        // Also runs on early Err: finish any pending round, then release the
        // persistent worker before this training call returns.
        let _ = self.commands.send(Command::Finish);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn spawn_worker<const UNCHECKED: bool>(shard: Shard) -> ThreadWorker {
    let (commands, command_rx) = mpsc::channel();
    let (reply_tx, replies) = mpsc::channel();
    let handle = thread::spawn(move || {
        let (mut state, ready) = match PieceState::<UNCHECKED>::new(shard) {
            Ok(result) => result,
            Err(error) => {
                let _ = reply_tx.send(Reply::Ready(Err(error)));
                return;
            }
        };
        if reply_tx.send(Reply::Ready(Ok(ready))).is_err() {
            return;
        }
        while let Ok(command) = command_rx.recv() {
            match command {
                Command::Merge {
                    key,
                    new_id,
                    new_length,
                    missing,
                } => {
                    if reply_tx
                        .send(Reply::Round(
                            state.merge_round(key, new_id, new_length, missing),
                        ))
                        .is_err()
                    {
                        return;
                    }
                }
                Command::Finish => {
                    let _ = reply_tx.send(Reply::Finished(state.finish()));
                    return;
                }
            }
        }
    });
    ThreadWorker {
        commands,
        replies,
        handle: Some(handle),
    }
}

fn receive_ready(worker: &ThreadWorker) -> std::result::Result<Ready, TrainError> {
    match worker.replies.recv().map_err(|_| {
        TrainError::InternalInvariant("piece worker disconnected during initialization")
    })? {
        Reply::Ready(result) => result,
        _ => Err(TrainError::InternalInvariant(
            "unexpected worker initialization reply",
        )),
    }
}

fn receive_round(worker: &ThreadWorker) -> std::result::Result<Round, TrainError> {
    match worker
        .replies
        .recv()
        .map_err(|_| TrainError::InternalInvariant("piece worker disconnected during merge"))?
    {
        Reply::Round(result) => result,
        _ => Err(TrainError::InternalInvariant(
            "unexpected worker merge reply",
        )),
    }
}

fn receive_finish(worker: &ThreadWorker) -> std::result::Result<Finished, TrainError> {
    match worker
        .replies
        .recv()
        .map_err(|_| TrainError::InternalInvariant("piece worker disconnected during finish"))?
    {
        Reply::Finished(result) => Ok(result),
        _ => Err(TrainError::InternalInvariant(
            "unexpected worker finish reply",
        )),
    }
}

pub(super) fn train(
    input: Prepared,
    options: Options,
    owner_only: bool,
    serial: bool,
    owner_fallback_broadcast: bool,
) -> std::result::Result<Result, TrainError> {
    match options.bounds {
        Bounds::Checked => {
            train_impl::<false>(input, options, owner_only, serial, owner_fallback_broadcast)
        }
        Bounds::Unchecked => {
            train_impl::<true>(input, options, owner_only, serial, owner_fallback_broadcast)
        }
    }
}

fn train_impl<const UNCHECKED: bool>(
    input: Prepared,
    options: Options,
    owner_only: bool,
    serial: bool,
    owner_fallback_broadcast: bool,
) -> std::result::Result<Result, TrainError> {
    let started = Instant::now();
    let corpus_positions = input.corpus.len();
    let shard_vec = shards(&input, options.workers);
    let count = shard_vec.len();
    let mut lengths = input.initial_lengths.clone();
    let copied_corpus_bytes: usize = shard_vec.iter().map(|s| s.corpus.len() * 4).sum();
    drop(input);

    let mut serial_states = Vec::new();
    let mut thread_workers = Vec::new();
    let mut ready = Vec::new();
    if serial {
        for shard in shard_vec {
            let (state, info) = PieceState::<UNCHECKED>::new(shard)?;
            serial_states.push(state);
            ready.push(info);
        }
    } else {
        for shard in shard_vec {
            thread_workers.push(spawn_worker::<UNCHECKED>(shard));
        }
        for worker in &thread_workers {
            ready.push(receive_ready(worker)?);
        }
    }
    let mut frequencies = HashMap::new();
    let mut initial_masks = HashMap::<u64, u128>::new();
    let mut initial_occurrence_bytes = 0;
    let mut backend_buffer_bytes = 0;
    let mut shard_pieces = Vec::new();
    for (worker_i, info) in ready.into_iter().enumerate() {
        initial_occurrence_bytes += info.occurrence_bytes;
        backend_buffer_bytes += info.backend_bytes;
        shard_pieces.push(info.pieces);
        for (key, frequency) in info.counts {
            let value = frequencies.entry(key).or_insert(0_u64);
            *value = value.checked_add(frequency).ok_or(TrainError::Overflow(
                "global initial pair frequency exceeds u64",
            ))?;
            if owner_only {
                // Owner mode is entered only for requested workers <= 128.
                *initial_masks.entry(key).or_default() |= 1_u128 << worker_i;
            }
        }
    }
    let mut owners: HashMap<u64, u128> = if owner_only {
        initial_masks
            .into_iter()
            .filter(|(key, _)| frequencies[key] >= options.min_frequency)
            .collect()
    } else {
        HashMap::new()
    };
    let mut owner_peak = owners.len();
    let mut heap = initial_heap(&frequencies, options.min_frequency);
    let init_seconds = started.elapsed().as_secs_f64();

    let merge_started = Instant::now();
    let mut merges = Vec::new();
    let mut known_lengths = vec![lengths.len(); count];
    let mut dispatches = vec![0_usize; count];
    let mut worker_merges = vec![0_usize; count];
    let mut worker_visits = vec![0_usize; count];
    let mut worker_work = vec![0_f64; count];
    let mut actual_merges = 0;
    let mut position_visits = 0;
    let mut stale_visits = 0;
    let mut heap_pops = 0;
    let mut round_messages = 0;
    let mut late_lengths = 0;
    let mut max_late_fill = 0;
    let mut sync_seconds = 0.0;
    let mut apply_seconds = 0.0;
    let mut reduce_seconds = 0.0;
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
        let targets: Vec<usize> = if owner_only {
            let mask = owners
                .remove(&key)
                .ok_or(TrainError::InternalInvariant("eligible pair has no owner"))?;
            (0..count).filter(|&i| mask & (1_u128 << i) != 0).collect()
        } else {
            (0..count).collect()
        };
        let sync_started = Instant::now();
        let mut missing = Vec::with_capacity(targets.len());
        for &i in &targets {
            let fill = lengths[known_lengths[i]..new_id as usize].to_vec();
            late_lengths += fill.len();
            max_late_fill = max_late_fill.max(fill.len());
            known_lengths[i] = new_id as usize + 1;
            dispatches[i] += 1;
            missing.push(fill);
        }
        sync_seconds += sync_started.elapsed().as_secs_f64();
        let apply_started = Instant::now();
        let rounds = if serial {
            targets
                .iter()
                .copied()
                .zip(missing)
                .map(|(i, fill)| serial_states[i].merge_round(key, new_id, new_length, fill))
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            for (&i, fill) in targets.iter().zip(missing) {
                thread_workers[i]
                    .commands
                    .send(Command::Merge {
                        key,
                        new_id,
                        new_length,
                        missing: fill,
                    })
                    .map_err(|_| {
                        TrainError::InternalInvariant("piece worker command channel closed")
                    })?;
            }
            round_messages += 2 * targets.len();
            targets
                .iter()
                .map(|&i| receive_round(&thread_workers[i]))
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        apply_seconds += apply_started.elapsed().as_secs_f64();
        let reduce_started = Instant::now();
        let mut fresh_keys = HashSet::new();
        let mut fresh_masks = HashMap::<u64, u128>::new();
        for (&i, round) in targets.iter().zip(rounds) {
            position_visits += round.visits;
            stale_visits += round.stale;
            actual_merges += round.merged;
            worker_merges[i] += round.merged;
            worker_visits[i] += round.visits;
            worker_work[i] += round.work_seconds;
            for (pair, delta) in round.deltas {
                apply_delta(&mut frequencies, pair, delta)?;
            }
            for new_key in round.new_keys {
                fresh_keys.insert(new_key);
                if owner_only {
                    *fresh_masks.entry(new_key).or_default() |= 1_u128 << i;
                }
            }
        }
        for new_key in fresh_keys {
            if frequencies[&new_key] >= options.min_frequency {
                heap.push(HeapEntry {
                    frequency: frequencies[&new_key],
                    key: new_key,
                });
                if owner_only {
                    owners.insert(new_key, fresh_masks[&new_key]);
                }
            }
        }
        owner_peak = owner_peak.max(owners.len());
        frequencies.remove(&key);
        reduce_seconds += reduce_started.elapsed().as_secs_f64();
    }
    let merge_seconds = merge_started.elapsed().as_secs_f64();
    let finishes = if serial {
        serial_states
            .into_iter()
            .map(PieceState::finish)
            .collect::<Vec<_>>()
    } else {
        for worker in &thread_workers {
            worker
                .commands
                .send(Command::Finish)
                .map_err(|_| TrainError::InternalInvariant("piece worker finish channel closed"))?;
        }
        let finished = thread_workers
            .iter()
            .map(receive_finish)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(thread_workers);
        finished
    };
    let mut final_tokens = vec![0];
    for finish in finishes {
        final_tokens.extend_from_slice(&finish.tokens[1..]);
        debug_assert!(finish.backend_bytes > 0);
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
            backend_buffer_bytes,
            initial_occurrence_bytes,
            max_token_length,
            corpus_positions,
        },
    );
    let mut metrics = BTreeMap::new();
    metric(&mut metrics, "workers_requested", options.workers as f64);
    metric(&mut metrics, "workers_actual", count as f64);
    metric(
        &mut metrics,
        "shard_corpus_bytes",
        copied_corpus_bytes as f64,
    );
    metric(&mut metrics, "owner_directory_peak", owner_peak as f64);
    metric(
        &mut metrics,
        "owner_fallback_broadcast",
        if owner_fallback_broadcast { 1.0 } else { 0.0 },
    );
    metric(&mut metrics, "round_messages", round_messages as f64);
    metric(
        &mut metrics,
        "messages_per_round",
        if core.rules == 0 {
            0.0
        } else {
            round_messages as f64 / core.rules as f64
        },
    );
    metric(&mut metrics, "late_token_lengths_sent", late_lengths as f64);
    metric(&mut metrics, "max_late_token_fill", max_late_fill as f64);
    metric(&mut metrics, "plan_seconds", 0.0);
    metric(&mut metrics, "apply_seconds", apply_seconds);
    metric(&mut metrics, "reduce_seconds", reduce_seconds);
    metric(&mut metrics, "sync_seconds", sync_seconds);
    for i in 0..count {
        metric(
            &mut metrics,
            &format!("worker_{i}_pieces"),
            shard_pieces[i] as f64,
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
            &format!("worker_{i}_visits"),
            worker_visits[i] as f64,
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

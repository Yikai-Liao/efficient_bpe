use super::backends::{
    BitmapU32, Corpus, Endpoint, Halfword, Hybrid, Linked, UnfusedEndpoint, UnfusedHalfword,
    UnfusedHybrid,
};
use super::index::{Arena, Combined, Index, Key, Separate};
use super::queue::Queue;
use super::{Options, Result as AblationResult, validate};
use crate::{Bounds, Prepared, Rule, TrainError, TrainResult};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

pub fn variant_names() -> &'static [&'static str] {
    &[
        "archived",
        "full_clear",
        "endpoints",
        "unfused_endpoints",
        "unfused_halfword",
        "unfused_h3",
        "lean",
        "packed",
        "linked12",
        "linked16",
        "bitmap_u32",
        "halfword",
        "h3",
        "h25",
        "filtered",
        "separate_counted",
        "arena",
        "arena_counted",
        "filtered_h3",
        "bucket",
        "bucket_normalized",
        "combined",
        "combined_filtered",
        "combined_filtered_h3",
        "combined_filtered_halfword",
        "parallel_broadcast",
        "parallel_owner",
        "parallel_occurrence",
        "parallel_occurrence_snapshot",
        "parallel_occurrence_adaptive",
        "parallel_occurrence_adaptive_256",
        "parallel_occurrence_adaptive_4096",
        "parallel_serial",
    ]
}

#[derive(Clone, Copy)]
enum Init {
    OnePass,
    Filtered,
    Counted,
}

pub fn train_variant(
    input: Prepared,
    options: Options,
    variant: &str,
) -> Result<AblationResult, TrainError> {
    if options.workers == 0 {
        return Err(TrainError::InvalidInput("workers must be positive"));
    }
    if let Some(mode) = variant.strip_prefix("parallel_") {
        return super::parallel::train(input, options, mode);
    }
    if variant == "archived" {
        let core = crate::train(
            input,
            crate::TrainOptions {
                max_merges: options.max_merges,
                min_frequency: options.min_frequency,
                bounds: options.bounds,
            },
        )?;
        return Ok(AblationResult {
            core,
            metrics: BTreeMap::new(),
        });
    }
    validate(&input, options)?;
    macro_rules! endpoint {
        ($mode:literal, $key:ty, $index:ty, $init:expr, $queue:expr) => {
            if options.bounds == Bounds::Unchecked {
                run::<Endpoint<$mode, true>, $key, $index>(input, options, $init, $queue)
            } else {
                run::<Endpoint<$mode, false>, $key, $index>(input, options, $init, $queue)
            }
        };
    }
    macro_rules! layout {
        ($backend:ident, $key:ty, $index:ty, $init:expr) => {
            if options.bounds == Bounds::Unchecked {
                run::<$backend<true>, $key, $index>(input, options, $init, 0)
            } else {
                run::<$backend<false>, $key, $index>(input, options, $init, 0)
            }
        };
        ($backend:ident, $kind:literal, $key:ty, $index:ty, $init:expr) => {
            if options.bounds == Bounds::Unchecked {
                run::<$backend<$kind, true>, $key, $index>(input, options, $init, 0)
            } else {
                run::<$backend<$kind, false>, $key, $index>(input, options, $init, 0)
            }
        };
    }
    match variant {
        "full_clear" => endpoint!(0, (u32, u32), Separate<(u32, u32)>, Init::OnePass, 0),
        "endpoints" => endpoint!(1, (u32, u32), Separate<(u32, u32)>, Init::OnePass, 0),
        "lean" => endpoint!(2, (u32, u32), Separate<(u32, u32)>, Init::OnePass, 0),
        "packed" => endpoint!(2, u64, Separate<u64>, Init::OnePass, 0),
        "unfused_endpoints" => layout!(
            UnfusedEndpoint,
            (u32, u32),
            Separate<(u32, u32)>,
            Init::OnePass
        ),
        "unfused_halfword" => layout!(
            UnfusedHalfword,
            (u32, u32),
            Separate<(u32, u32)>,
            Init::OnePass
        ),
        "unfused_h3" => layout!(
            UnfusedHybrid,
            false,
            (u32, u32),
            Separate<(u32, u32)>,
            Init::OnePass
        ),
        "separate_counted" => endpoint!(2, u64, Separate<u64>, Init::Counted, 0),
        "filtered" => endpoint!(2, u64, Separate<u64>, Init::Filtered, 0),
        "arena" => endpoint!(2, u64, Arena<u64>, Init::Filtered, 0),
        "arena_counted" => endpoint!(2, u64, Arena<u64>, Init::Counted, 0),
        "combined" => endpoint!(2, u64, Combined<u64>, Init::OnePass, 0),
        "combined_filtered" => endpoint!(2, u64, Combined<u64>, Init::Counted, 0),
        "bucket" => endpoint!(1, (u32, u32), Separate<(u32, u32)>, Init::OnePass, 1),
        "bucket_normalized" => endpoint!(1, (u32, u32), Separate<(u32, u32)>, Init::OnePass, 2),
        "linked12" => layout!(
            Linked,
            false,
            (u32, u32),
            Separate<(u32, u32)>,
            Init::OnePass
        ),
        "linked16" => layout!(
            Linked,
            true,
            (u32, u32),
            Separate<(u32, u32)>,
            Init::OnePass
        ),
        "bitmap_u32" => layout!(BitmapU32, (u32, u32), Separate<(u32, u32)>, Init::OnePass),
        "halfword" => layout!(Halfword, (u32, u32), Separate<(u32, u32)>, Init::OnePass),
        "h3" => layout!(
            Hybrid,
            false,
            (u32, u32),
            Separate<(u32, u32)>,
            Init::OnePass
        ),
        "h25" => layout!(
            Hybrid,
            true,
            (u32, u32),
            Separate<(u32, u32)>,
            Init::OnePass
        ),
        "filtered_h3" => layout!(Hybrid, false, u64, Separate<u64>, Init::Filtered),
        "combined_filtered_h3" => layout!(Hybrid, false, u64, Combined<u64>, Init::Counted),
        "combined_filtered_halfword" => layout!(Halfword, u64, Combined<u64>, Init::Counted),
        _ => Err(TrainError::InvalidInput("unknown ablation variant")),
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

#[cfg_attr(feature = "profiling", hotpath::measure)]
fn run<B: Corpus, K: Key, I: Index<K>>(
    input: Prepared,
    options: Options,
    initialization: Init,
    queue_mode: u8,
) -> Result<AblationResult, TrainError> {
    let started = Instant::now();
    let Prepared {
        corpus,
        initial_lengths: mut lengths,
        pivots,
        weights,
    } = input;
    let n = corpus.len();
    let mut backend = B::new(corpus, lengths.len() - 1)?;
    debug_assert!(!backend.is_empty() && backend.len() == n);
    let mut index = I::default();
    let mut initial_edges = 0;
    let mut initial_stored = 0;
    let mut weight_i = 0;
    let weight_gcd = if queue_mode == 2 {
        weights.iter().copied().fold(0, gcd)
    } else {
        1
    };
    if matches!(initialization, Init::Counted) {
        let mut counts = HashMap::<K, u64>::new();
        for pos in 1..n.saturating_sub(1) {
            while weight_i + 1 < pivots.len() && pos >= pivots[weight_i + 1] as usize {
                weight_i += 1;
            }
            let (a, b) = (backend.initial_token(pos), backend.initial_token(pos + 1));
            if a != 0 && b != 0 {
                let f = counts.entry(K::pair(a, b)).or_default();
                *f = f
                    .checked_add(weights[weight_i])
                    .ok_or(TrainError::Overflow("initial count exceeds u64"))?;
                initial_edges += 1;
            }
        }
        for (key, frequency) in counts {
            if frequency >= options.min_frequency {
                index.add(key, frequency)?;
            }
        }
    } else {
        for pos in 1..n.saturating_sub(1) {
            while weight_i + 1 < pivots.len() && pos >= pivots[weight_i + 1] as usize {
                weight_i += 1;
            }
            let (a, b) = (backend.initial_token(pos), backend.initial_token(pos + 1));
            if a != 0 && b != 0 {
                let key = K::pair(a, b);
                initial_edges += 1;
                if matches!(initialization, Init::OnePass) {
                    index.record(key, weights[weight_i], pos as u32)?;
                    initial_stored += 1;
                } else {
                    index.add(key, weights[weight_i])?;
                }
            }
        }
        if matches!(initialization, Init::Filtered) {
            for (key, frequency) in index.entries() {
                if frequency < options.min_frequency {
                    index.discard(key, false);
                }
            }
        }
    }
    if !matches!(initialization, Init::OnePass) {
        for pos in 1..n.saturating_sub(1) {
            let (a, b) = (backend.initial_token(pos), backend.initial_token(pos + 1));
            if a != 0 && b != 0 {
                let key = K::pair(a, b);
                if index.append_if_tracked(key, pos as u32)? {
                    initial_stored += 1;
                }
            }
        }
    }
    let initial_occurrence_bytes = initial_stored * I::RECORD_BYTES;
    let entries = index.entries();
    let mass = if queue_mode == 0 {
        0
    } else {
        entries.iter().map(|(_, f)| f).sum()
    };
    let mut queue = Queue::new(
        entries,
        options.min_frequency,
        mass,
        queue_mode,
        weight_gcd,
        n,
    );
    let init_seconds = started.elapsed().as_secs_f64();
    let merge_started = Instant::now();
    let mut merges = Vec::new();
    let mut actual_merges = 0;
    let mut position_visits = 0;
    let mut stale_visits = 0;
    let keep_frequency = matches!(initialization, Init::OnePass);
    for _ in 0..options.max_merges {
        let Some((key, frequency)) = queue.pop(|key| {
            let f = index.frequency(key);
            if f < options.min_frequency {
                index.discard(key, keep_frequency);
            }
            f
        }) else {
            break;
        };
        let (a, b) = key.tokens();
        let new_id = u32::try_from(lengths.len())
            .map_err(|_| TrainError::Overflow("new token ID exceeds u32"))?;
        let length = lengths[a as usize]
            .checked_add(lengths[b as usize])
            .ok_or(TrainError::Overflow("token length exceeds u32"))?;
        lengths.push(length);
        merges.push(Rule {
            left: a,
            right: b,
            frequency,
        });
        let selected = index.selected(key);
        let mut batch = index.detach(key);
        let mut fresh = HashSet::new();
        while let Some(pos) = index.next(&mut batch) {
            position_visits += 1;
            let pos = pos as usize;
            let Some(ctx) = backend.inspect_pair(pos, a, b, &lengths) else {
                stale_visits += 1;
                continue;
            };
            let wi = pivots.partition_point(|&p| p as usize <= pos) - 1;
            let weight = weights[wi];
            index.subtract_selected(selected, weight)?;
            if ctx.left_id != 0 {
                index.subtract(K::pair(ctx.left_id, a), weight)?;
            }
            if ctx.right_id != 0 {
                index.subtract(K::pair(b, ctx.right_id), weight)?;
            }
            backend.merge_with_lengths(pos, ctx, new_id, length, &lengths);
            actual_merges += 1;
            if ctx.left_id != 0 {
                let key = K::pair(ctx.left_id, new_id);
                index.record(
                    key,
                    weight,
                    ctx.before
                        .ok_or(TrainError::InternalInvariant("left boundary absent"))?
                        as u32,
                )?;
                fresh.insert(key);
            }
            if ctx.right_id != 0 {
                let key = K::pair(new_id, ctx.right_id);
                index.record(key, weight, pos as u32)?;
                fresh.insert(key);
            }
        }
        for key in fresh {
            let f = index.frequency(key);
            if f >= options.min_frequency {
                queue.add(key, f);
            } else {
                index.discard(key, keep_frequency);
            }
        }
        index.discard(key, false);
    }
    let merge_seconds = merge_started.elapsed().as_secs_f64();
    let train_seconds = started.elapsed().as_secs_f64();
    let final_tokens = backend.final_tokens(&lengths);
    let mut metrics = index.metrics();
    metrics.extend(queue.metrics());
    metrics.insert(
        "backend_capacity_bytes".into(),
        backend.capacity_bytes() as f64,
    );
    metrics.insert(
        "initial_unfiltered_offset_bytes".into(),
        (initial_edges * 4) as f64,
    );
    metrics.insert(
        "final_occurrence_logical_bytes".into(),
        index.occurrence_bytes() as f64,
    );
    metrics.insert("key_size_bytes".into(), std::mem::size_of::<K>() as f64);
    Ok(AblationResult {
        core: TrainResult {
            rules: merges.len(),
            merges,
            final_tokens,
            init_seconds,
            merge_seconds,
            train_seconds,
            actual_merges,
            position_visits,
            stale_visits,
            heap_pops: queue.pops,
            backend_buffer_bytes: backend.logical_bytes(),
            initial_occurrence_bytes,
            corpus_positions: n,
            max_token_length: lengths.iter().copied().max().unwrap_or(1),
        },
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_scalar_ablations_match_archived_rules() {
        let variants: Vec<_> = variant_names()
            .iter()
            .filter(|v| !v.starts_with("parallel_"))
            .copied()
            .collect();
        let mut seed = 727_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed >> 32
        };
        for case in 0..100 {
            let mut corpus = vec![0];
            let mut pivots = vec![];
            let mut weights = vec![];
            // All three initial IDs appear; varied weights and separators.
            for _ in 0..(1 + next() % 5) {
                pivots.push(corpus.len() as u32);
                weights.push((1 + next() % 7) * if case % 11 == 0 { 1 << 40 } else { 1 });
                corpus.extend([1, 2, 3]);
                for _ in 0..(next() % 40) {
                    corpus.push((1 + next() % 3) as u32);
                }
                corpus.push(0);
            }
            let input = Prepared {
                corpus,
                initial_lengths: vec![1; 4],
                pivots,
                weights,
            };
            let options = Options {
                max_merges: 40,
                min_frequency: 1 + next() % 5,
                ..Options::default()
            };
            let expected = train_variant(input.clone(), options, "archived")
                .unwrap()
                .core;
            for &variant in &variants {
                for bounds in [Bounds::Checked, Bounds::Unchecked] {
                    let output =
                        train_variant(input.clone(), Options { bounds, ..options }, variant)
                            .unwrap()
                            .core;
                    assert_eq!(
                        output.merges, expected.merges,
                        "case={case} variant={variant} bounds={bounds:?}"
                    );
                    assert_eq!(
                        output.final_tokens, expected.final_tokens,
                        "case={case} variant={variant}"
                    );
                }
            }
        }
    }

    #[test]
    fn all_scalar_variants_accept_empty_input() {
        for &variant in variant_names()
            .iter()
            .filter(|v| !v.starts_with("parallel_"))
        {
            let input = Prepared {
                corpus: vec![0],
                initial_lengths: vec![1],
                pivots: vec![],
                weights: vec![],
            };
            let out = train_variant(input, Options::default(), variant).unwrap();
            assert!(out.core.merges.is_empty());
            assert_eq!(out.core.final_tokens, vec![0]);
        }
    }
}

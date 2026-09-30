//! Counted birth routing followed by a second scan of selected historical postings.
//! Final positions are filled through disjoint safe slices of zeroed SmallPostings.

use super::region_counts::{CountList, InlineCounts, VecCounts};
use super::*;

struct StagedBirth<C> {
    pair: u64,
    frequency: u64,
    positions: SmallPosting,
    region_counts: C,
}

#[derive(Default)]
struct CombinedDelta<C> {
    delta: Delta,
    region_counts: C,
}

struct FillSegment<'a> {
    remaining: &'a mut [u32],
}

impl FillSegment<'_> {
    fn write(&mut self, pos: u32) -> Result<()> {
        let remaining = std::mem::take(&mut self.remaining);
        let (slot, rest) = remaining
            .split_first_mut()
            .ok_or(TrainError::InternalInvariant(
                "replay segment exceeds counted births",
            ))?;
        *slot = pos;
        self.remaining = rest;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn commit<H: HashBuild>(
    pool: &ThreadPool,
    owners: &mut [Owner<H>],
    outputs: &mut [WorkerOutput<H>],
    batch: &[BatchRule],
    corpus: &[AtomicU32],
    lengths: &[u32],
    cuts: &RegionCuts,
    cross_births: &[CrossBirth],
    selected: &HashSet<u64, H>,
    fresh_start: u32,
    minimum: u64,
    policy: HeapPolicy,
    inline: bool,
    metrics: &mut Metrics,
) -> Result<()> {
    if inline {
        commit_impl::<H, InlineCounts>(
            pool,
            owners,
            outputs,
            batch,
            corpus,
            lengths,
            cuts,
            cross_births,
            selected,
            fresh_start,
            minimum,
            policy,
            metrics,
        )
    } else {
        commit_impl::<H, VecCounts>(
            pool,
            owners,
            outputs,
            batch,
            corpus,
            lengths,
            cuts,
            cross_births,
            selected,
            fresh_start,
            minimum,
            policy,
            metrics,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn commit_impl<H: HashBuild, C: CountList>(
    pool: &ThreadPool,
    owners: &mut [Owner<H>],
    outputs: &mut [WorkerOutput<H>],
    batch: &[BatchRule],
    corpus: &[AtomicU32],
    lengths: &[u32],
    cuts: &RegionCuts,
    cross_births: &[CrossBirth],
    selected: &HashSet<u64, H>,
    fresh_start: u32,
    minimum: u64,
    policy: HeapPolicy,
    metrics: &mut Metrics,
) -> Result<()> {
    let started = Instant::now();
    let workers = owners.len();
    if outputs.len() != cuts.count() {
        return Err(TrainError::InternalInvariant(
            "replay output count differs from regions",
        ));
    }
    metrics.actual_merges += outputs.iter().map(|output| output.merges).sum::<usize>();
    let selected_len: usize = batch.iter().map(|rule| rule.posting.len()).sum();
    let selected_heap_capacity_bytes: usize = batch
        .iter()
        .map(|rule| rule.posting.allocated_capacity().saturating_mul(4))
        .sum();
    metrics.replay_peak_selected_posting_len =
        metrics.replay_peak_selected_posting_len.max(selected_len);
    metrics.replay_peak_selected_heap_capacity_bytes = metrics
        .replay_peak_selected_heap_capacity_bytes
        .max(selected_heap_capacity_bytes);

    let mut born_count = 0_usize;
    let mut birth_keys = 0_usize;
    let mut route_delta_keys = 0_usize;
    let mut route_delta_capacity = 0_usize;
    let mut route_born_capacity = 0_usize;
    for output in outputs.iter() {
        for route in &output.routes {
            metrics.replay_birth_nodes_actual += route.born.len();
            route_born_capacity += route.born.capacity();
            route_delta_keys += route.delta.len();
            route_delta_capacity += route.delta.capacity();
            for (&pair, delta) in &route.delta {
                if delta.head != u32::MAX {
                    return Err(TrainError::InternalInvariant(
                        "replay delta has a birth chain",
                    ));
                }
                if is_new_pair(pair, fresh_start) {
                    born_count = born_count
                        .checked_add(delta.occurrences as usize)
                        .ok_or(TrainError::Overflow("replay birth count exceeds usize"))?;
                    birth_keys += 1;
                }
            }
        }
    }
    if metrics.replay_birth_nodes_actual != 0 || route_born_capacity != 0 {
        return Err(TrainError::InternalInvariant("replay allocated BirthNodes"));
    }
    metrics.generated_birth_records += born_count;
    metrics.peak_birth_records = metrics.peak_birth_records.max(born_count);
    metrics.grouped_birth_keys += birth_keys;
    metrics.peak_delta_keys = metrics.peak_delta_keys.max(route_delta_keys);
    metrics.peak_route_delta_capacity = metrics.peak_route_delta_capacity.max(route_delta_capacity);
    let route_delta_capacity_scaled_bytes =
        route_delta_capacity.saturating_mul(std::mem::size_of::<(u64, Delta)>());

    // Old keys only decrease; fresh keys are staged outside the owner map.
    // This lets the second scan borrow only newly allocated postings, without
    // scanning every historical owner Entry on every batch.
    let reductions = pool.install(|| {
        owners
            .par_iter_mut()
            .enumerate()
            .map(
                |(owner_i, owner)| -> Result<(Vec<StagedBirth<C>>, usize, usize)> {
                    let mut combined =
                        HashMap::<u64, CombinedDelta<C>, H>::with_hasher(H::default());
                    for (region, output) in outputs.iter().enumerate() {
                        for (&pair, &delta) in &output.routes[owner_i].delta {
                            let item = combined.entry(pair).or_default();
                            item.delta.weight = item.delta.weight.checked_add(delta.weight).ok_or(
                                TrainError::Overflow("combined replay frequency exceeds u64"),
                            )?;
                            item.delta.occurrences = item
                                .delta
                                .occurrences
                                .checked_add(delta.occurrences)
                                .ok_or(TrainError::Overflow(
                                    "combined replay occurrences exceed u32",
                                ))?;
                            if is_new_pair(pair, fresh_start) {
                                item.region_counts.push(region, delta.occurrences)?;
                            }
                        }
                    }
                    let combined_capacity_scaled_bytes = combined
                        .capacity()
                        .saturating_mul(std::mem::size_of::<(u64, CombinedDelta<C>)>())
                        .saturating_add(
                            combined
                                .values()
                                .map(|item| item.region_counts.heap_bytes())
                                .sum::<usize>(),
                        );
                    let mut staged = Vec::new();
                    let mut initialized = 0_usize;
                    for (pair, item) in combined {
                        if selected.contains(&pair) {
                            continue;
                        }
                        if is_new_pair(pair, fresh_start) {
                            if owner.entries.contains_key(&pair) {
                                return Err(TrainError::InternalInvariant(
                                    "fresh replay pair already exists",
                                ));
                            }
                            if item.delta.weight >= minimum {
                                initialized = initialized
                                    .checked_add(item.delta.occurrences as usize)
                                    .ok_or(TrainError::Overflow(
                                        "replay initialized positions exceed usize",
                                    ))?;
                                staged.push(StagedBirth {
                                    pair,
                                    frequency: item.delta.weight,
                                    positions: SmallPosting::zeroed(item.delta.occurrences)?,
                                    region_counts: item.region_counts,
                                });
                            }
                        } else if let Some(entry) = owner.entries.get_mut(&pair) {
                            entry.frequency = entry
                                .frequency
                                .checked_sub(item.delta.weight)
                                .ok_or(TrainError::InternalInvariant(
                                    "negative old replay pair frequency",
                                ))?;
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
                    Ok((staged, combined_capacity_scaled_bytes, initialized))
                },
            )
            .collect::<Vec<_>>()
    });
    let mut staging = Vec::with_capacity(workers);
    let mut combined_capacity_scaled_bytes = 0_usize;
    let mut initialized_positions = 0_usize;
    for reduction in reductions {
        let (staged, combined_bytes, zeros) = reduction?;
        staging.push(staged);
        combined_capacity_scaled_bytes += combined_bytes;
        initialized_positions += zeros;
    }
    metrics.frequency_reduce_seconds += started.elapsed().as_secs_f64();
    metrics.replay_zero_initialized_bytes += initialized_positions.saturating_mul(4);
    let staging_entry_count: usize = staging.iter().map(Vec::len).sum();
    let staging_header_capacity_bytes: usize = staging
        .iter()
        .map(|owner| {
            owner
                .capacity()
                .saturating_mul(std::mem::size_of::<StagedBirth<C>>())
        })
        .sum::<usize>()
        + staging
            .iter()
            .flatten()
            .map(|entry| entry.region_counts.heap_bytes())
            .sum::<usize>();
    let staging_region_counts_capacity_bytes: usize = staging
        .iter()
        .flatten()
        .map(|entry| entry.region_counts.heap_bytes())
        .sum();
    metrics.replay_count_header_bytes = std::mem::size_of::<C>();
    metrics.replay_count_peak_heap_bytes = metrics
        .replay_count_peak_heap_bytes
        .max(staging_region_counts_capacity_bytes);
    for entry in staging.iter().flatten() {
        match entry.region_counts.len() {
            1 => metrics.replay_count_single_segments += 1,
            2 => metrics.replay_count_double_segments += 1,
            _ => metrics.replay_count_many_segments += 1,
        }
    }
    let initialized_posting_capacity_bytes: usize = staging
        .iter()
        .flatten()
        .map(|entry| entry.positions.allocated_capacity().saturating_mul(4))
        .sum();
    metrics.replay_peak_staging_entry_count = metrics
        .replay_peak_staging_entry_count
        .max(staging_entry_count);
    metrics.replay_peak_staging_header_capacity_bytes = metrics
        .replay_peak_staging_header_capacity_bytes
        .max(staging_header_capacity_bytes);
    metrics.replay_peak_initialized_posting_capacity_bytes = metrics
        .replay_peak_initialized_posting_capacity_bytes
        .max(initialized_posting_capacity_bytes);
    // Conservative same-batch capacity proxy for reduction. Region-count Vecs
    // move from combined entries into staging, so count their allocation once.
    let reduce_capacity_proxy = selected_heap_capacity_bytes
        .saturating_add(route_delta_capacity_scaled_bytes)
        .saturating_add(combined_capacity_scaled_bytes)
        .saturating_add(
            staging_header_capacity_bytes.saturating_sub(staging_region_counts_capacity_bytes),
        )
        .saturating_add(initialized_posting_capacity_bytes);
    metrics.replay_peak_simultaneous_capacity_proxy_bytes = metrics
        .replay_peak_simultaneous_capacity_proxy_bytes
        .max(reduce_capacity_proxy);

    // No route records are needed once owner frequencies and per-region counts
    // are complete. Free their maps before allocating writer descriptors.
    for output in outputs.iter_mut() {
        output.routes = Vec::new();
    }
    let fill_started = Instant::now();
    let mut cross_by_region = vec![None; cuts.count()];
    for &birth in cross_births {
        let slot =
            cross_by_region
                .get_mut(birth.target_region)
                .ok_or(TrainError::InternalInvariant(
                    "replay cross birth outside regions",
                ))?;
        if slot.replace(birth).is_some() {
            return Err(TrainError::InternalInvariant(
                "multiple replay cross births enter one region",
            ));
        }
    }

    // Every mutable slice here belongs to a different final posting or to a
    // disjoint region segment of one posting. The borrowed staging vector and
    // its SmallPostings cannot be moved or grown until the descriptors drop.
    let mut descriptors: Vec<HashMap<u64, FillSegment<'_>, H>> = (0..cuts.count())
        .map(|_| HashMap::with_hasher(H::default()))
        .collect();
    let mut cross_slots: Vec<(&mut u32, u32)> = Vec::new();
    for owner_staging in &mut staging {
        for entry in owner_staging {
            let counts = std::mem::take(&mut entry.region_counts);
            let mut remaining = entry.positions.as_mut_slice();
            for (region, count) in counts.into_counts() {
                let count = count as usize;
                if count == 0 || count > remaining.len() {
                    return Err(TrainError::InternalInvariant(
                        "invalid replay region segment count",
                    ));
                }
                let (segment, rest) = remaining.split_at_mut(count);
                let cross = cross_by_region
                    .get(region)
                    .ok_or(TrainError::InternalInvariant(
                        "replay segment region outside cuts",
                    ))?
                    .as_ref()
                    .is_some_and(|birth| birth.pair == entry.pair);
                let local = if cross {
                    let (local, last) = segment.split_at_mut(count - 1);
                    let birth = cross_by_region[region].take().unwrap();
                    cross_slots.push((&mut last[0], birth.pos));
                    local
                } else {
                    segment
                };
                if !local.is_empty()
                    && descriptors[region]
                        .insert(entry.pair, FillSegment { remaining: local })
                        .is_some()
                {
                    return Err(TrainError::InternalInvariant(
                        "duplicate replay writer segment",
                    ));
                }
                remaining = rest;
            }
            if !remaining.is_empty() {
                return Err(TrainError::InternalInvariant(
                    "replay region counts do not cover posting",
                ));
            }
        }
    }
    let descriptor_capacity_scaled_bytes: usize = descriptors
        .iter()
        .map(|region| {
            region
                .capacity()
                .saturating_mul(std::mem::size_of::<(u64, FillSegment<'_>)>())
        })
        .sum::<usize>()
        + cross_slots
            .capacity()
            .saturating_mul(std::mem::size_of::<(&mut u32, u32)>());
    metrics.replay_peak_descriptor_capacity_scaled_bytes = metrics
        .replay_peak_descriptor_capacity_scaled_bytes
        .max(descriptor_capacity_scaled_bytes);
    // Prefix count vectors were consumed and dropped while splitting slices.
    let fill_capacity_proxy = selected_heap_capacity_bytes
        .saturating_add(
            staging_header_capacity_bytes.saturating_sub(staging_region_counts_capacity_bytes),
        )
        .saturating_add(initialized_posting_capacity_bytes)
        .saturating_add(descriptor_capacity_scaled_bytes);
    metrics.replay_peak_simultaneous_capacity_proxy_bytes = metrics
        .replay_peak_simultaneous_capacity_proxy_bytes
        .max(fill_capacity_proxy);

    let last = corpus.len() - 1;
    let checks = pool.install(|| {
        descriptors
            .into_par_iter()
            .enumerate()
            .map(|(region, mut writers)| -> Result<(usize, usize)> {
                let (lower, upper) = cuts.bounds(region);
                let mut visits = 0_usize;
                let mut filled = 0_usize;
                for rule in batch {
                    let posting = rule.posting.as_slice();
                    let first = posting.partition_point(|&pos| (pos as usize) < lower);
                    let end = posting.partition_point(|&pos| (pos as usize) < upper);
                    for &pos in &posting[first..end] {
                        visits += 1;
                        let p = pos as usize;
                        if corpus[p].load(Ordering::Relaxed) != (rule.new_id | HEAD) {
                            continue;
                        }
                        let after = p
                            .checked_add(lengths[rule.new_id as usize] as usize)
                            .ok_or(TrainError::Overflow("replay new token end exceeds usize"))?;
                        if after > last {
                            return Err(TrainError::InternalInvariant(
                                "replay fresh head exceeds final corpus",
                            ));
                        }
                        let right_raw = corpus[after].load(Ordering::Relaxed);
                        let right_id = right_raw & ID_MASK;
                        if right_id != 0 {
                            if right_raw & HEAD == 0 {
                                return Err(TrainError::InternalInvariant(
                                    "replay right neighbor is not a head",
                                ));
                            }
                            if let Some(writer) = writers.get_mut(&key(rule.new_id, right_id)) {
                                writer.write(pos)?;
                                filled += 1;
                            }
                        }
                        let left_raw = corpus[p - 1].load(Ordering::Relaxed);
                        let left_id = left_raw & ID_MASK;
                        if left_id != 0 && left_id < fresh_start {
                            let left_length = *lengths.get(left_id as usize).ok_or(
                                TrainError::InternalInvariant("replay left ID outside lengths"),
                            )? as usize;
                            let before =
                                p.checked_sub(left_length)
                                    .ok_or(TrainError::InternalInvariant(
                                        "replay left neighbor underflows",
                                    ))?;
                            debug_assert_eq!(
                                corpus[before].load(Ordering::Relaxed),
                                left_id | HEAD
                            );
                            // before < p < upper, so this one comparison
                            // decides whether the left head belongs here.
                            if before >= lower
                                && let Some(writer) = writers.get_mut(&key(left_id, rule.new_id))
                            {
                                writer.write(before as u32)?;
                                filled += 1;
                            }
                        }
                    }
                }
                if writers.values().any(|writer| !writer.remaining.is_empty()) {
                    return Err(TrainError::InternalInvariant(
                        "replay writer did not fill counted segment",
                    ));
                }
                Ok((visits, filled))
            })
            .collect::<Vec<_>>()
    });
    let mut replay_visits = 0_usize;
    let mut filled = 0_usize;
    for check in checks {
        let (visits, count) = check?;
        replay_visits += visits;
        filled += count;
    }
    for (slot, pos) in cross_slots {
        *slot = pos;
        filled += 1;
    }
    if filled != initialized_positions {
        return Err(TrainError::InternalInvariant(
            "replay filled positions differ from retained births",
        ));
    }
    metrics.replay_posting_visits += replay_visits;
    metrics.posting_visits += replay_visits;
    metrics.replay_filled_positions += filled;
    metrics.stored_born_postings += filled;

    // All mutable slice borrows ended. Move the completed zeroed postings into
    // their unique owners and publish candidates for the next selection round.
    let checks = pool.install(|| {
        owners
            .par_iter_mut()
            .zip(staging.into_par_iter())
            .map(|(owner, staged)| -> Result<()> {
                for birth in staged {
                    debug_assert_global_order(birth.positions.as_slice());
                    if owner
                        .entries
                        .insert(
                            birth.pair,
                            Entry {
                                frequency: birth.frequency,
                                positions: birth.positions,
                            },
                        )
                        .is_some()
                    {
                        return Err(TrainError::InternalInvariant(
                            "replay fresh posting replaced an owner entry",
                        ));
                    }
                    owner.heap.push(Candidate {
                        key: birth.pair,
                        frequency: birth.frequency,
                    });
                }
                Ok(())
            })
            .collect::<Vec<_>>()
    });
    for check in checks {
        check?;
    }
    metrics.replay_fill_seconds += fill_started.elapsed().as_secs_f64();
    metrics.birth_group_fill_seconds += fill_started.elapsed().as_secs_f64();
    Ok(())
}

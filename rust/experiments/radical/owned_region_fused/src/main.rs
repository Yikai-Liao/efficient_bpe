#![recursion_limit = "1024"]

use efficient_bpe_rust::{Bounds, Prepared, TrainOptions};
use radical_owned_region_fused::{
    Config, EndpointPlan, HeapPolicy, IntegerHash, RegionMode, train,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    corpus: Vec<u32>,
    initial_lengths: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
}

fn vm_hwm_mib() -> Result<f64, Box<dyn Error>> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    Ok(status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .ok_or("VmHWM missing")?
        .split_whitespace()
        .next()
        .ok_or("VmHWM value missing")?
        .parse::<f64>()?
        / 1024.0)
}

fn process_cpu() -> Result<f64, std::io::Error> {
    let mut clock = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: clock points to one writable timespec, read only after success.
    if unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, clock.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: successful clock_gettime initialized both timespec fields.
    let clock = unsafe { clock.assume_init() };
    Ok(clock.tv_sec as f64 + clock.tv_nsec as f64 * 1e-9)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut input_path = None;
    let mut workers = 1;
    let mut chunk_size = 4096;
    let mut rules = 3000;
    let mut min_frequency = 2;
    let mut heap_policy = HeapPolicy::Lazy;
    let mut integer_hash = IntegerHash::Std;
    let mut endpoint_plan = EndpointPlan::TwoPass;
    let mut region_mode = RegionMode::Dynamic;
    let mut trace_path = None;
    while let Some(arg) = args.next() {
        let value = args.next().ok_or(format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--input" => input_path = Some(value),
            "--workers" => workers = value.parse()?,
            "--chunk-size" => chunk_size = value.parse()?,
            "--rules" => rules = value.parse()?,
            "--min-frequency" => min_frequency = value.parse()?,
            "--heap-policy" => {
                heap_policy = match value.as_str() {
                    "lazy" => HeapPolicy::Lazy,
                    "eager" => HeapPolicy::Eager,
                    _ => return Err("heap policy must be lazy or eager".into()),
                }
            }
            "--integer-hash" => {
                integer_hash = match value.as_str() {
                    "std" => IntegerHash::Std,
                    "ahash" => IntegerHash::AHash,
                    _ => return Err("integer hash must be std or ahash".into()),
                }
            }
            "--endpoint-plan" => {
                endpoint_plan = match value.as_str() {
                    "two-pass" => EndpointPlan::TwoPass,
                    "tagged-two-pass" => EndpointPlan::TaggedTwoPass,
                    "tagged-fused" => EndpointPlan::TaggedFused,
                    _ => {
                        return Err(
                            "endpoint plan must be two-pass, tagged-two-pass or tagged-fused"
                                .into(),
                        );
                    }
                }
            }
            "--region-mode" => {
                region_mode = match value.as_str() {
                    "dynamic" => RegionMode::Dynamic,
                    "region" => RegionMode::Region,
                    _ => return Err("region mode must be dynamic or region".into()),
                }
            }
            "--trace" => trace_path = Some(value),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let input_path = input_path.ok_or("--input is required")?;
    let input_bytes = std::fs::read(input_path)?;
    let fixture_sha256 = format!("{:x}", Sha256::digest(&input_bytes));
    let input: Input = serde_json::from_slice(&input_bytes)?;
    drop(input_bytes);
    let corpus_positions = input.corpus.len();
    let prepared = Prepared {
        corpus: input.corpus,
        initial_lengths: input.initial_lengths,
        pivots: input.pivots,
        weights: input.weights,
    };
    let started = Instant::now();
    let cpu_started = process_cpu()?;
    let result = train(
        prepared,
        TrainOptions {
            max_merges: rules,
            min_frequency,
            bounds: Bounds::Checked,
        },
        Config {
            workers,
            chunk_size,
            heap_policy,
            integer_hash,
            endpoint_plan,
            region_mode,
        },
    )?;
    let call_cpu_seconds = process_cpu()? - cpu_started;
    let call_seconds = started.elapsed().as_secs_f64();
    let train_vm_hwm_mib = vm_hwm_mib()?;
    let merges: Vec<[u64; 3]> = result
        .rules
        .iter()
        .map(|r| [r.left as u64, r.right as u64, r.frequency])
        .collect();
    let compact = serde_json::to_string(&(&merges, &result.final_tokens))?;
    let fingerprint = format!(
        "{:x}",
        Sha256::digest(compact.replace(',', ", ").as_bytes())
    );
    if let Some(path) = trace_path {
        std::fs::write(
            path,
            serde_json::to_vec(&json!({"merges": merges, "final": result.final_tokens}))?,
        )?;
    }
    let vm_hwm_mib = vm_hwm_mib()?;
    println!(
        "{}",
        json!({
            "variant": "radical_owned_region_fused", "workers": workers,
            "heap_policy": match heap_policy { HeapPolicy::Lazy => "lazy", HeapPolicy::Eager => "eager" },
            "integer_hash": match integer_hash { IntegerHash::Std => "std", IntegerHash::AHash => "ahash" },
            "endpoint_plan_requested": match endpoint_plan { EndpointPlan::TwoPass => "two-pass", EndpointPlan::TaggedTwoPass => "tagged-two-pass", EndpointPlan::TaggedFused => "tagged-fused" },
            "endpoint_plan_effective": match result.effective_endpoint_plan { EndpointPlan::TwoPass => "two-pass", EndpointPlan::TaggedTwoPass => "tagged-two-pass", EndpointPlan::TaggedFused => "tagged-fused" },
            "region_mode_requested": match region_mode { RegionMode::Dynamic => "dynamic", RegionMode::Region => "region" },
            "region_mode_effective": match result.effective_region_mode { RegionMode::Dynamic => "dynamic", RegionMode::Region => "region" },
            "endpoint_domain_fallback": result.metrics.endpoint_domain_fallback,
            "region_partition_searches": result.metrics.region_partition_searches,
            "region_partition_worker_seconds": result.metrics.region_partition_worker_seconds,
            "region_posting_visits": result.metrics.region_posting_visits,
            "region_valid_merges": result.metrics.region_valid_merges,
            "region_cross_births": result.metrics.region_cross_births,
            "region_peak_route_delta_capacity": result.metrics.region_peak_route_delta_capacity,
            "region_peak_route_born_capacity": result.metrics.region_peak_route_born_capacity,
            "region_aa_regroup_seconds": result.metrics.region_aa_regroup_seconds,
            "region_aa_regroup_capacity_upper_bytes": result.metrics.region_aa_regroup_capacity_upper_bytes,
            "region_max_visits_per_batch": result.metrics.region_max_visits_per_batch,
            "region_max_merges_per_batch": result.metrics.region_max_merges_per_batch,
            "region_sum_max_visits": result.metrics.region_sum_max_visits,
            "region_sum_max_merges": result.metrics.region_sum_max_merges,
            "chunk_size": chunk_size, "rules": result.rules.len(),
            "fixture_sha256": fixture_sha256,
            "fingerprint": fingerprint,
            "corpus_positions": corpus_positions,
            "vm_hwm_mib": vm_hwm_mib,
            "train_vm_hwm_mib": train_vm_hwm_mib,
            "call_seconds": call_seconds,
            "call_cpu_seconds": call_cpu_seconds,
            "validation_seconds": result.metrics.validation_seconds,
            "pool_seconds": result.metrics.pool_seconds,
            "init_seconds": result.metrics.init_seconds,
            "initial_count_seconds": result.metrics.initial_count_seconds,
            "initial_fill_seconds": result.metrics.initial_fill_seconds,
            "select_seconds": result.metrics.select_seconds,
            "plan_seconds": result.metrics.plan_seconds,
            "chunk_summary_seconds": result.metrics.chunk_summary_seconds,
            "apply_seconds": result.metrics.apply_seconds,
            "combine_seconds": result.metrics.combine_seconds,
            "frequency_reduce_seconds": result.metrics.frequency_reduce_seconds,
            "birth_sort_seconds": result.metrics.birth_sort_seconds,
            "birth_append_seconds": result.metrics.birth_append_seconds,
            "final_seconds": result.metrics.final_seconds,
            "final_owner_stats_seconds": result.metrics.final_owner_stats_seconds,
            "posting_visits": result.metrics.posting_visits,
            "stale_visits": result.metrics.stale_visits,
            "actual_merges": result.metrics.actual_merges,
            "posting_arena_len": result.metrics.posting_arena_len,
            "posting_arena_capacity": result.metrics.posting_arena_capacity,
            "retained_entry_posting_len": result.metrics.retained_entry_posting_len,
            "eligible_posting_len": result.metrics.eligible_posting_len,
            "final_live_edges": result.metrics.final_live_edges,
            "stored_born_postings": result.metrics.stored_born_postings,
            "generated_birth_records": result.metrics.generated_birth_records,
            "initial_all_postings": result.metrics.initial_all_postings,
            "initial_eligible_postings": result.metrics.initial_eligible_postings,
            "peak_plan_len": result.metrics.peak_plan_len,
            "peak_birth_records": result.metrics.peak_birth_records,
            "peak_delta_keys": result.metrics.peak_delta_keys,
            "peak_route_delta_capacity": result.metrics.peak_route_delta_capacity,
            "delta_value_bytes": result.metrics.delta_value_bytes,
            "delta_entry_bytes": result.metrics.delta_entry_bytes,
            "birth_node_bytes": result.metrics.birth_node_bytes,
            "grouped_birth_keys": result.metrics.grouped_birth_keys,
            "grouped_birth_nodes": result.metrics.grouped_birth_nodes,
            "batch_rounds": result.metrics.batch_rounds,
            "batch_rules": result.metrics.batch_rules,
            "max_batch_width": result.metrics.max_batch_width,
            "singleton_rounds": result.metrics.singleton_rounds,
            "flat_tasks": result.metrics.flat_tasks,
            "planned_positions": result.metrics.planned_positions,
            "peak_flat_tasks": result.metrics.peak_flat_tasks,
            "peak_task_starts": result.metrics.peak_task_starts,
            "initial_route_seconds": result.metrics.initial_route_seconds,
            "initial_owner_seconds": result.metrics.initial_owner_seconds,
            "aa_sort_seconds": result.metrics.aa_sort_seconds,
            "birth_decode_seconds": result.metrics.birth_decode_seconds,
            "birth_group_fill_seconds": result.metrics.birth_group_fill_seconds,
            "heap_pops": result.metrics.heap_pops,
            "heap_refreshes": result.metrics.heap_refreshes,
            "heap_reinsertions": result.metrics.heap_reinsertions,
            "peak_heap_len": result.metrics.peak_heap_len,
            "peak_heap_capacity": result.metrics.peak_heap_capacity,
            "owned_posting_len": result.metrics.owned_posting_len,
            "owned_posting_capacity": result.metrics.owned_posting_capacity,
            "owner_entry_count": result.metrics.owner_entry_count,
            "inline_posting_keys": result.metrics.inline_posting_keys,
            "inline_posting_positions": result.metrics.inline_posting_positions,
            "heap_posting_keys": result.metrics.heap_posting_keys,
            "peak_route_born_len": result.metrics.peak_route_born_len,
            "peak_route_born_capacity": result.metrics.peak_route_born_capacity,
            "fused_non_aa_merges": result.metrics.fused_non_aa_merges,
            "fused_non_aa_batches": result.metrics.fused_non_aa_batches,
            "non_aa_start_positions_peak": result.metrics.non_aa_start_positions_peak,
            "non_aa_start_bytes_peak_proxy": result.metrics.non_aa_start_bytes_peak_proxy,
            "decoder_zero_rereads": result.metrics.decoder_zero_rereads,
        })
    );
    Ok(())
}

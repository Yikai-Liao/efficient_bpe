#![recursion_limit = "256"]

use efficient_bpe_rust::{Bounds, Prepared, TrainOptions};
use radical_owned_counts::{Config, HeapPolicy, train};
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
    Ok(status.lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .ok_or("VmHWM missing")?
        .split_whitespace().next().ok_or("VmHWM value missing")?
        .parse::<f64>()? / 1024.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut input_path = None;
    let mut workers = 1;
    let mut chunk_size = 4096;
    let mut rules = 3000;
    let mut min_frequency = 2;
    let mut heap_policy = HeapPolicy::Lazy;
    let mut trace_path = None;
    while let Some(arg) = args.next() {
        let value = args.next().ok_or(format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--input" => input_path = Some(value),
            "--workers" => workers = value.parse()?,
            "--chunk-size" => chunk_size = value.parse()?,
            "--rules" => rules = value.parse()?,
            "--min-frequency" => min_frequency = value.parse()?,
            "--heap-policy" => heap_policy = match value.as_str() {
                "lazy" => HeapPolicy::Lazy,
                "eager" => HeapPolicy::Eager,
                _ => return Err("heap policy must be lazy or eager".into()),
            },
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
    let result = train(
        prepared,
        TrainOptions { max_merges: rules, min_frequency, bounds: Bounds::Checked },
        Config { workers, chunk_size, heap_policy },
    )?;
    let call_seconds = started.elapsed().as_secs_f64();
    let train_vm_hwm_mib = vm_hwm_mib()?;
    let merges: Vec<[u64; 3]> = result.rules.iter()
        .map(|r| [r.left as u64, r.right as u64, r.frequency]).collect();
    let compact = serde_json::to_string(&(&merges, &result.final_tokens))?;
    let fingerprint = format!("{:x}", Sha256::digest(compact.replace(',', ", ").as_bytes()));
    if let Some(path) = trace_path {
        std::fs::write(path, serde_json::to_vec(&json!({"merges": merges, "final": result.final_tokens}))?)?;
    }
    let vm_hwm_mib = vm_hwm_mib()?;
    println!("{}", json!({
        "variant": "radical_owned_counts", "workers": workers,
        "heap_policy": match heap_policy { HeapPolicy::Lazy => "lazy", HeapPolicy::Eager => "eager" },
        "chunk_size": chunk_size, "rules": result.rules.len(),
        "fixture_sha256": fixture_sha256,
        "fingerprint": fingerprint,
        "corpus_positions": corpus_positions,
        "vm_hwm_mib": vm_hwm_mib,
        "train_vm_hwm_mib": train_vm_hwm_mib,
        "call_seconds": call_seconds,
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
        "heap_pops": result.metrics.heap_pops,
        "heap_refreshes": result.metrics.heap_refreshes,
        "heap_reinsertions": result.metrics.heap_reinsertions,
        "peak_heap_len": result.metrics.peak_heap_len,
        "peak_heap_capacity": result.metrics.peak_heap_capacity,
        "owned_posting_len": result.metrics.owned_posting_len,
        "owned_posting_capacity": result.metrics.owned_posting_capacity,
        "owner_entry_count": result.metrics.owner_entry_count,
        "peak_route_born_len": result.metrics.peak_route_born_len,
        "peak_route_born_capacity": result.metrics.peak_route_born_capacity,
    }));
    Ok(())
}

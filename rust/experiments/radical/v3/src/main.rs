#![recursion_limit = "256"]

use efficient_bpe_rust::{Bounds, Prepared, TrainOptions};
use radical_birth_postings_v3::{Config, train};
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

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut input_path = None;
    let mut workers = 1;
    let mut chunk_size = 4096;
    let mut rules = 3000;
    let mut min_frequency = 2;
    let mut trace_path = None;
    while let Some(arg) = args.next() {
        let value = args.next().ok_or(format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--input" => input_path = Some(value),
            "--workers" => workers = value.parse()?,
            "--chunk-size" => chunk_size = value.parse()?,
            "--rules" => rules = value.parse()?,
            "--min-frequency" => min_frequency = value.parse()?,
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
        Config { workers, chunk_size },
    )?;
    let call_seconds = started.elapsed().as_secs_f64();
    let merges: Vec<[u64; 3]> = result.rules.iter()
        .map(|r| [r.left as u64, r.right as u64, r.frequency]).collect();
    let compact = serde_json::to_string(&(&merges, &result.final_tokens))?;
    let fingerprint = format!("{:x}", Sha256::digest(compact.replace(',', ", ").as_bytes()));
    if let Some(path) = trace_path {
        std::fs::write(path, serde_json::to_vec(&json!({"merges": merges, "final": result.final_tokens}))?)?;
    }
    let vm_hwm_mib = std::fs::read_to_string("/proc/self/status")?.lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .ok_or("VmHWM missing")?
        .split_whitespace().next().ok_or("VmHWM value missing")?
        .parse::<f64>()? / 1024.0;
    println!("{}", json!({
        "variant": "radical_birth_postings_v3", "workers": workers,
        "chunk_size": chunk_size, "rules": result.rules.len(),
        "fixture_sha256": fixture_sha256,
        "fingerprint": fingerprint,
        "corpus_positions": corpus_positions,
        "vm_hwm_mib": vm_hwm_mib,
        "call_seconds": call_seconds,
        "validation_seconds": result.metrics.validation_seconds,
        "pool_seconds": result.metrics.pool_seconds,
        "init_seconds": result.metrics.init_seconds,
        "initial_count_seconds": result.metrics.initial_count_seconds,
        "initial_aggregate_seconds": result.metrics.initial_aggregate_seconds,
        "initial_prefix_seconds": result.metrics.initial_prefix_seconds,
        "initial_fill_seconds": result.metrics.initial_fill_seconds,
        "plan_seconds": result.metrics.plan_seconds,
        "chunk_summary_seconds": result.metrics.chunk_summary_seconds,
        "apply_seconds": result.metrics.apply_seconds,
        "frequency_reduce_seconds": result.metrics.frequency_reduce_seconds,
        "birth_count_seconds": result.metrics.birth_count_seconds,
        "birth_prefix_seconds": result.metrics.birth_prefix_seconds,
        "birth_scatter_seconds": result.metrics.birth_scatter_seconds,
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
        "initial_local_keys": result.metrics.initial_local_keys,
        "initial_local_capacity": result.metrics.initial_local_capacity,
        "peak_plan_len": result.metrics.peak_plan_len,
        "peak_birth_records": result.metrics.peak_birth_records,
        "peak_delta_keys": result.metrics.peak_delta_keys,
        "peak_birth_count_entries": result.metrics.peak_birth_count_entries,
        "peak_birth_cursor_entries": result.metrics.peak_birth_cursor_entries,
        "peak_birth_cursor_capacity": result.metrics.peak_birth_cursor_capacity,
    }));
    Ok(())
}

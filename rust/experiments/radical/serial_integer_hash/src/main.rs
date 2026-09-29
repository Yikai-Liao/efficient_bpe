#![recursion_limit = "256"]

use efficient_bpe_rust::{Bounds, Prepared, TrainOptions};
use radical_serial_integer_hash::{Backend, Config, IntegerHash, train};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::io::Write;
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
    // SAFETY: clock points to writable timespec storage; success initializes it.
    if unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, clock.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: clock_gettime returned success.
    let clock = unsafe { clock.assume_init() };
    Ok(clock.tv_sec as f64 + clock.tv_nsec as f64 * 1e-9)
}

fn peak_rss_mib() -> Result<f64, std::io::Error> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: usage points to writable rusage storage; success initializes it.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: getrusage returned success.
    Ok(unsafe { usage.assume_init() }.ru_maxrss as f64 / 1024.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut input_path = None;
    let mut trace_path = None;
    let mut backend = Backend::CombinedFiltered;
    let mut integer_hash = IntegerHash::Std;
    let mut bounds = Bounds::Checked;
    let mut workers = 1;
    let mut rules = 3000;
    let mut min_frequency = 2;
    while let Some(arg) = args.next() {
        let value = args.next().ok_or(format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--input" => input_path = Some(value),
            "--trace" => trace_path = Some(value),
            "--backend" | "--variant" => {
                backend = match value.as_str() {
                    "combined_filtered" => Backend::CombinedFiltered,
                    "combined_filtered_halfword" => Backend::CombinedFilteredHalfword,
                    _ => {
                        return Err(
                            "backend must be combined_filtered or combined_filtered_halfword"
                                .into(),
                        );
                    }
                }
            }
            "--integer-hash" => {
                integer_hash = match value.as_str() {
                    "std" => IntegerHash::Std,
                    "ahash" => IntegerHash::AHash,
                    _ => return Err("integer hash must be std or ahash".into()),
                }
            }
            "--bounds" => {
                bounds = match value.as_str() {
                    "checked" => Bounds::Checked,
                    "unchecked" => Bounds::Unchecked,
                    _ => return Err("bounds must be checked or unchecked".into()),
                }
            }
            "--workers" => workers = value.parse()?,
            "--rules" => rules = value.parse()?,
            "--min-frequency" => min_frequency = value.parse()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let input_path = input_path.ok_or("--input is required")?;
    let input_bytes = std::fs::read(input_path)?;
    let fixture_sha256 = format!("{:x}", Sha256::digest(&input_bytes));
    let input: Input = serde_json::from_slice(&input_bytes)?;
    drop(input_bytes);
    let prepared = Prepared {
        corpus: input.corpus,
        initial_lengths: input.initial_lengths,
        pivots: input.pivots,
        weights: input.weights,
    };
    let wall = Instant::now();
    let cpu = process_cpu()?;
    let result = train(
        prepared,
        TrainOptions {
            max_merges: rules,
            min_frequency,
            bounds,
        },
        Config {
            backend,
            integer_hash,
            workers,
        },
    )?;
    let call_cpu_seconds = process_cpu()? - cpu;
    let call_seconds = wall.elapsed().as_secs_f64();
    // Snapshot before fingerprint and trace allocations, like the native CLI.
    let train_vm_hwm_mib = vm_hwm_mib()?;
    let core = result.core;
    let merges: Vec<[u64; 3]> = core
        .merges
        .iter()
        .map(|rule| [u64::from(rule.left), u64::from(rule.right), rule.frequency])
        .collect();
    let compact = serde_json::to_string(&(&merges, &core.final_tokens))?;
    let fingerprint = format!(
        "{:x}",
        Sha256::digest(compact.replace(',', ", ").as_bytes())
    );
    if let Some(path) = trace_path {
        let mut output = std::fs::File::create(path)?;
        serde_json::to_writer(
            &mut output,
            &json!({"merges": merges, "final": core.final_tokens}),
        )?;
        output.write_all(b"\n")?;
    }
    let backend_name = match backend {
        Backend::CombinedFiltered => "combined_filtered",
        Backend::CombinedFilteredHalfword => "combined_filtered_halfword",
    };
    println!(
        "{}",
        json!({
            "variant": backend_name,
            "backend": backend_name,
            "integer_hash": match integer_hash { IntegerHash::Std => "std", IntegerHash::AHash => "ahash" },
            "bounds": match bounds { Bounds::Checked => "checked", Bounds::Unchecked => "unchecked" },
            "workers": workers,
            "fixture_sha256": fixture_sha256,
            "fingerprint": fingerprint,
            "rules": core.rules,
            "actual_merges": core.actual_merges,
            "position_visits": core.position_visits,
            "stale_visits": core.stale_visits,
            "heap_pops": core.heap_pops,
            "corpus_positions": core.corpus_positions,
            "max_token_length": core.max_token_length,
            "backend_buffer_bytes": core.backend_buffer_bytes,
            "initial_occurrence_bytes": core.initial_occurrence_bytes,
            "init_seconds": core.init_seconds,
            "merge_seconds": core.merge_seconds,
            "train_seconds": core.train_seconds,
            "call_seconds": call_seconds,
            "call_cpu_seconds": call_cpu_seconds,
            "metrics": result.metrics,
            "peak_rss_mib": peak_rss_mib()?,
            "train_vm_hwm_mib": train_vm_hwm_mib,
            "vm_hwm_mib": vm_hwm_mib()?,
        })
    );
    Ok(())
}

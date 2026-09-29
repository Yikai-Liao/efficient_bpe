//! CLI consumes a prepared numeric corpus; tokenization is outside this baseline.
use efficient_bpe_rust::ablation::{Options, train_variant, variant_names};
use efficient_bpe_rust::{Bounds, Prepared};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    corpus: Vec<u32>,
    initial_lengths: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
}

fn process_cpu() -> Result<f64, std::io::Error> {
    let mut clock = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: clock points to writable storage for exactly one timespec.
    // It is only read after clock_gettime reports that it initialized it.
    if unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, clock.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: the successful call above initializes both timespec fields.
    let clock = unsafe { clock.assume_init() };
    Ok(clock.tv_sec as f64 + clock.tv_nsec as f64 * 1e-9)
}

fn peak_rss_kib() -> Result<i64, std::io::Error> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: usage is writable rusage storage; success initializes it.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: getrusage returned success.
    Ok(unsafe { usage.assume_init() }.ru_maxrss)
}

fn vm_hwm_mib() -> Result<f64, Box<dyn Error>> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let kib: f64 = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .ok_or("VmHWM missing from /proc/self/status")?
        .split_whitespace()
        .next()
        .ok_or("VmHWM has no value")?
        .parse()?;
    Ok(kib / 1024.0)
}

#[cfg_attr(feature = "profiling", hotpath::main)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut input_path = None;
    let mut trace_path: Option<PathBuf> = None;
    let mut bounds_name = "checked".to_owned();
    let mut variant = "packed".to_owned();
    let mut workers = 1;
    let mut max_merges = 3000;
    let mut min_frequency = 2;
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>| {
            args.next()
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg.as_str() {
            "--variant" => variant = value(&mut args)?,
            "--workers" => workers = value(&mut args)?.parse()?,
            "--list-variants" => {
                println!("{}", serde_json::to_string(variant_names())?);
                return Ok(());
            }
            "--input" => input_path = Some(PathBuf::from(value(&mut args)?)),
            "--trace" => trace_path = Some(PathBuf::from(value(&mut args)?)),
            "--bounds" => bounds_name = value(&mut args)?,
            "--rules" => max_merges = value(&mut args)?.parse()?,
            "--min-frequency" => min_frequency = value(&mut args)?.parse()?,
            "--help" | "-h" => {
                println!(
                    "ablation --input prepared.json --variant NAME [--workers N] [--bounds checked|unchecked] [--rules 3000] [--min-frequency 2] [--trace trace.json]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let path = input_path.ok_or("--input is required")?;
    let input_bytes = std::fs::read(&path)?;
    let fixture_sha256 = format!("{:x}", Sha256::digest(&input_bytes));
    let input: Input = serde_json::from_slice(&input_bytes)?;
    drop(input_bytes);
    let prepared = Prepared {
        corpus: input.corpus,
        initial_lengths: input.initial_lengths,
        pivots: input.pivots,
        weights: input.weights,
    };
    let bounds = match bounds_name.as_str() {
        "checked" => Bounds::Checked,
        "unchecked" => Bounds::Unchecked,
        _ => return Err("--bounds must be checked or unchecked".into()),
    };
    let wall = Instant::now();
    let cpu = process_cpu()?;
    let result = train_variant(
        prepared,
        Options {
            workers,
            max_merges,
            min_frequency,
            bounds,
        },
        &variant,
    )?;
    let call_cpu_seconds = process_cpu()? - cpu;
    let call_seconds = wall.elapsed().as_secs_f64();
    let metrics = result.metrics;
    let result = result.core;
    // Python's old fingerprint uses json.dumps defaults: arrays of unsigned
    // numbers with one ASCII space after each comma. No object key ordering
    // or Unicode escaping is involved in this representation.
    let merges: Vec<[u64; 3]> = result
        .merges
        .iter()
        .map(|r| [u64::from(r.left), u64::from(r.right), r.frequency])
        .collect();
    let compact = serde_json::to_string(&(&merges, &result.final_tokens))?;
    let canonical = compact.replace(',', ", ");
    let fingerprint = format!("{:x}", Sha256::digest(canonical.as_bytes()));
    if let Some(path) = trace_path {
        let mut output = std::fs::File::create(path)?;
        serde_json::to_writer(
            &mut output,
            &json!({"merges":merges,"final":result.final_tokens}),
        )?;
        output.write_all(b"\n")?;
    }
    println!(
        "{}",
        json!({
            "bounds":bounds_name,"fixture_sha256":fixture_sha256,
            "variant":variant,"workers":workers,"metrics":metrics,
            "profiling_enabled":cfg!(feature = "profiling"),
            "fingerprint":fingerprint,"rules":result.merges.len(),
            "actual_merges":result.actual_merges,"position_visits":result.position_visits,
            "stale_visits":result.stale_visits,"heap_pops":result.heap_pops,
            "corpus_positions":result.corpus_positions,
            "max_token_length":result.max_token_length,
            "backend_buffer_bytes":result.backend_buffer_bytes,
            "initial_occurrence_bytes":result.initial_occurrence_bytes,
            "init_seconds":result.init_seconds,"merge_seconds":result.merge_seconds,
            "train_seconds":result.train_seconds,
            "call_seconds":call_seconds,"call_cpu_seconds":call_cpu_seconds,
            "peak_rss_mib":peak_rss_kib()? as f64/1024.0,
            "vm_hwm_mib":vm_hwm_mib()?,
        })
    );
    Ok(())
}

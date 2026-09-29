//! Replay one precomputed merge trajectory through each corpus layout.
//! This times boundary inspection plus merge only: no pair frequencies, heap,
//! occurrence collection, fixture preparation, or final traversal is timed.
use super::backends::{BitmapU32, Corpus, Endpoint, Halfword, Hybrid, Linked};
use super::spans::ByteSpans;
use crate::{Bounds, TrainError};
use serde::Serialize;
use std::hint::black_box;
use std::time::Instant;

#[derive(Serialize)]
pub struct MicroResult {
    pub variant: String,
    pub length: usize,
    pub positions: usize,
    pub pattern: String,
    pub bounds: String,
    pub operations: usize,
    pub checksum: u64,
    pub seconds: f64,
    pub buffer_bytes: usize,
    pub capacity_bytes: usize,
    pub vm_hwm_mib: f64,
    pub component_only: bool,
    pub operation_contract: &'static str,
}

#[derive(Clone, Copy)]
struct Step {
    pos: usize,
    a: u32,
    b: u32,
    new_id: u32,
    new_len: u32,
}

struct Trace {
    initial: Vec<u32>,
    steps: Vec<Step>,
    lengths: Vec<u32>,
    final_ids: Vec<u32>,
    final_starts: Vec<usize>,
    expected_checksum: u64,
}

#[inline(always)]
fn contribution(before: usize, right: usize, after: usize) -> u64 {
    (before as u64)
        .wrapping_add(right as u64)
        .wrapping_add(after as u64)
}

fn make_trace(length: usize, positions: usize, pattern: &str) -> Result<Trace, TrainError> {
    if length == 0 || positions == 0 || length > u32::MAX as usize {
        return Err(TrainError::InvalidInput(
            "length and positions must be positive, length <= u32",
        ));
    }
    if !matches!(pattern, "random" | "balanced" | "chain") {
        return Err(TrainError::InvalidInput(
            "pattern must be random, balanced, or chain",
        ));
    }
    let blocks = (positions / length).max(1);
    let n = blocks
        .checked_mul(
            length
                .checked_add(1)
                .ok_or(TrainError::Overflow("length+1"))?,
        )
        .and_then(|x| x.checked_add(1))
        .ok_or(TrainError::Overflow("position count"))?;
    if (n as u128) >= 1_u128 << 32 {
        return Err(TrainError::Overflow("position count exceeds u32"));
    }
    let operations = blocks
        .checked_mul(length - 1)
        .ok_or(TrainError::Overflow("merge count"))?;
    let mut initial = Vec::with_capacity(n);
    let mut steps = Vec::with_capacity(operations);
    let mut lengths = vec![1_u32, 1];
    let mut final_ids = Vec::with_capacity(blocks * 2 + 1);
    let mut final_starts = Vec::with_capacity(blocks * 2 + 1);
    let mut expected_checksum = 0_u64;
    initial.push(0);
    final_ids.push(0);
    final_starts.push(0);
    let mut seed = 202309_u64;
    for _block in 0..blocks {
        let start = initial.len();
        initial.extend(std::iter::repeat_n(1_u32, length));
        initial.push(0);
        let end = start + length;
        let mut ids = vec![1_u32; length];
        let mut add = |pos: usize,
                       before: usize,
                       right: usize,
                       after: usize,
                       a: u32,
                       b: u32,
                       steps: &mut Vec<Step>,
                       lengths: &mut Vec<u32>|
         -> Result<u32, TrainError> {
            let new_id =
                u32::try_from(lengths.len()).map_err(|_| TrainError::Overflow("new ID"))?;
            let new_len =
                u32::try_from(after - pos).map_err(|_| TrainError::Overflow("new length"))?;
            steps.push(Step {
                pos,
                a,
                b,
                new_id,
                new_len,
            });
            lengths.push(new_len);
            expected_checksum = expected_checksum.wrapping_add(contribution(before, right, after));
            Ok(new_id)
        };
        let final_id = match pattern {
            "chain" => {
                let mut a = 1_u32;
                for offset in 1..length {
                    let right = start + offset;
                    a = add(
                        start,
                        start - 1,
                        right,
                        right + 1,
                        a,
                        1,
                        &mut steps,
                        &mut lengths,
                    )?;
                }
                a
            }
            "balanced" => {
                let mut live: Vec<usize> = (start..end).collect();
                while live.len() > 1 {
                    let mut remaining = Vec::with_capacity(live.len().div_ceil(2));
                    for k in (0..live.len()).step_by(2) {
                        let pos = live[k];
                        if k + 1 < live.len() {
                            let right = live[k + 1];
                            let after = live.get(k + 2).copied().unwrap_or(end);
                            let before = remaining.last().copied().unwrap_or(start - 1);
                            ids[pos - start] = add(
                                pos,
                                before,
                                right,
                                after,
                                ids[pos - start],
                                ids[right - start],
                                &mut steps,
                                &mut lengths,
                            )?;
                        }
                        remaining.push(pos);
                    }
                    live = remaining;
                }
                ids[live[0] - start]
            }
            _ => {
                let mut live: Vec<usize> = (start..end).collect();
                while live.len() > 1 {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let k = (seed as usize) % (live.len() - 1);
                    let pos = live[k];
                    let right = live[k + 1];
                    let after = live.get(k + 2).copied().unwrap_or(end);
                    let before = if k == 0 { start - 1 } else { live[k - 1] };
                    ids[pos - start] = add(
                        pos,
                        before,
                        right,
                        after,
                        ids[pos - start],
                        ids[right - start],
                        &mut steps,
                        &mut lengths,
                    )?;
                    live.remove(k + 1);
                }
                ids[live[0] - start]
            }
        };
        final_ids.extend([final_id, 0]);
        final_starts.extend([start, end]);
    }
    debug_assert_eq!(initial.len(), n);
    Ok(Trace {
        initial,
        steps,
        lengths,
        final_ids,
        final_starts,
        expected_checksum,
    })
}

struct ByteLengths {
    tags: Vec<u8>,
    last: usize,
}
impl ByteLengths {
    fn new(n: usize) -> Self {
        Self {
            tags: vec![1; n],
            last: n - 1,
        }
    }
    fn next(&self, p: usize) -> Option<usize> {
        (p != self.last).then(|| p + self.tags[p] as usize)
    }
    fn prev(&self, p: usize) -> Option<usize> {
        (p != 0).then(|| p - self.tags[p - 1] as usize)
    }
    fn merge_known(&mut self, pos: usize, right: usize, after: usize, new_len: u8) {
        self.tags[pos] = new_len;
        if after - right == 1 {
            self.tags[right] = new_len;
        } else {
            self.tags[right] = 0;
            self.tags[after - 1] = new_len;
        }
    }
}

fn vm_hwm_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse::<f64>().ok())
            })
        })
        .map_or(0.0, |kib| kib / 1024.0)
}

struct Measured {
    seconds: f64,
    checksum: u64,
    buffer_bytes: usize,
    capacity_bytes: usize,
    component_only: bool,
    operation_contract: &'static str,
}

fn finish(
    trace: &Trace,
    variant: &str,
    length: usize,
    pattern: &str,
    bounds: Bounds,
    measured: Measured,
) -> Result<MicroResult, TrainError> {
    if measured.checksum != trace.expected_checksum {
        return Err(TrainError::InternalInvariant("boundary checksum mismatch"));
    }
    Ok(MicroResult {
        variant: variant.into(),
        length,
        positions: trace.initial.len(),
        pattern: pattern.into(),
        bounds: if bounds == Bounds::Unchecked {
            "unchecked"
        } else {
            "checked"
        }
        .into(),
        operations: trace.steps.len(),
        checksum: measured.checksum,
        seconds: measured.seconds,
        buffer_bytes: measured.buffer_bytes,
        capacity_bytes: measured.capacity_bytes,
        vm_hwm_mib: vm_hwm_mib(),
        component_only: measured.component_only,
        operation_contract: measured.operation_contract,
    })
}

fn run_corpus<B: Corpus>(
    trace: &Trace,
    variant: &str,
    length: usize,
    pattern: &str,
    bounds: Bounds,
) -> Result<MicroResult, TrainError> {
    let mut backend = B::new(trace.initial.clone(), 1)?;
    let start = Instant::now();
    let mut checksum = 0_u64;
    for step in &trace.steps {
        let ctx = backend
            .inspect_pair(step.pos, step.a, step.b, &trace.lengths)
            .ok_or(TrainError::InternalInvariant("precomputed pair rejected"))?;
        checksum =
            checksum.wrapping_add(contribution(ctx.before.unwrap_or(0), ctx.right, ctx.after));
        backend.merge_known(step.pos, black_box(ctx), step.new_id, step.new_len);
    }
    black_box(checksum);
    let seconds = start.elapsed().as_secs_f64();
    if backend.final_tokens(&trace.lengths) != trace.final_ids {
        return Err(TrainError::InternalInvariant(
            "final token traversal mismatch",
        ));
    }
    finish(
        trace,
        variant,
        length,
        pattern,
        bounds,
        Measured {
            seconds,
            checksum,
            buffer_bytes: backend.logical_bytes(),
            capacity_bytes: backend.capacity_bytes(),
            component_only: false,
            operation_contract: "inspect_pair+merge_known",
        },
    )
}

fn run_u8(
    trace: &Trace,
    variant: &str,
    length: usize,
    pattern: &str,
    bounds: Bounds,
) -> Result<MicroResult, TrainError> {
    if length > 255 {
        return Err(TrainError::InvalidInput("u8_only supports length <=255"));
    }
    let mut backend = ByteLengths::new(trace.initial.len());
    let start = Instant::now();
    let mut checksum = 0_u64;
    for step in &trace.steps {
        let before = backend
            .prev(step.pos)
            .ok_or(TrainError::InternalInvariant("u8 predecessor"))?;
        let right = backend
            .next(step.pos)
            .ok_or(TrainError::InternalInvariant("u8 right"))?;
        let after = backend
            .next(right)
            .ok_or(TrainError::InternalInvariant("u8 after"))?;
        checksum = checksum.wrapping_add(contribution(before, right, after));
        backend.merge_known(step.pos, right, after, step.new_len as u8);
    }
    black_box(checksum);
    let seconds = start.elapsed().as_secs_f64();
    let mut starts = Vec::new();
    let mut pos = 0;
    loop {
        starts.push(pos);
        let Some(next) = backend.next(pos) else { break };
        pos = next;
    }
    if starts != trace.final_starts {
        return Err(TrainError::InternalInvariant("u8 final traversal mismatch"));
    }
    finish(
        trace,
        variant,
        length,
        pattern,
        bounds,
        Measured {
            seconds,
            checksum,
            buffer_bytes: backend.tags.len(),
            capacity_bytes: backend.tags.capacity(),
            component_only: true,
            operation_contract: "prev+next+next+merge_known_no_id",
        },
    )
}

fn run_spans(
    trace: &Trace,
    variant: &str,
    length: usize,
    pattern: &str,
    bounds: Bounds,
) -> Result<MicroResult, TrainError> {
    let mut backend = ByteSpans::new(trace.initial.len())?;
    let start = Instant::now();
    let mut checksum = 0_u64;
    for step in &trace.steps {
        let before = backend
            .prev(step.pos)
            .ok_or(TrainError::InternalInvariant("span predecessor"))?;
        let right = backend
            .next(step.pos)
            .ok_or(TrainError::InternalInvariant("span right"))?;
        let after = backend
            .next(right)
            .ok_or(TrainError::InternalInvariant("span after"))?;
        checksum = checksum.wrapping_add(contribution(before, right, after));
        backend.merge_known(step.pos, right, after)?;
    }
    black_box(checksum);
    let seconds = start.elapsed().as_secs_f64();
    let mut starts = Vec::new();
    let mut pos = 0;
    loop {
        starts.push(pos);
        let Some(next) = backend.next(pos) else { break };
        pos = next;
    }
    if starts != trace.final_starts {
        return Err(TrainError::InternalInvariant(
            "span final traversal mismatch",
        ));
    }
    finish(
        trace,
        variant,
        length,
        pattern,
        bounds,
        Measured {
            seconds,
            checksum,
            buffer_bytes: backend.logical_bytes(),
            capacity_bytes: backend.capacity_bytes(),
            component_only: true,
            operation_contract: "prev+next+next+checked_merge_no_id",
        },
    )
}

/// Public wrapper: the unchecked Corpus trait and its trusted merge Context
/// remain crate-private; callers only select a validated workload and bounds.
pub fn run_micro(
    variant: &str,
    length: usize,
    positions: usize,
    pattern: &str,
    bounds: Bounds,
) -> Result<MicroResult, TrainError> {
    let trace = make_trace(length, positions, pattern)?;
    match variant {
        "u8_only" => run_u8(&trace, variant, length, pattern, bounds),
        "byte_spans" | "bytespans" => run_spans(&trace, variant, length, pattern, bounds),
        "full_clear" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<Endpoint<0, true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<Endpoint<0, false>>(&trace, variant, length, pattern, bounds)
            }
        }
        "endpoints" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<Endpoint<1, true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<Endpoint<1, false>>(&trace, variant, length, pattern, bounds)
            }
        }
        "lean" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<Endpoint<2, true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<Endpoint<2, false>>(&trace, variant, length, pattern, bounds)
            }
        }
        "linked12" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<Linked<false, true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<Linked<false, false>>(&trace, variant, length, pattern, bounds)
            }
        }
        "linked16" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<Linked<true, true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<Linked<true, false>>(&trace, variant, length, pattern, bounds)
            }
        }
        "bitmap_u32" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<BitmapU32<true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<BitmapU32<false>>(&trace, variant, length, pattern, bounds)
            }
        }
        "halfword" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<Halfword<true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<Halfword<false>>(&trace, variant, length, pattern, bounds)
            }
        }
        "h3" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<Hybrid<false, true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<Hybrid<false, false>>(&trace, variant, length, pattern, bounds)
            }
        }
        "h25" => {
            if bounds == Bounds::Unchecked {
                run_corpus::<Hybrid<true, true>>(&trace, variant, length, pattern, bounds)
            } else {
                run_corpus::<Hybrid<true, false>>(&trace, variant, length, pattern, bounds)
            }
        }
        _ => Err(TrainError::InvalidInput("unknown boundary micro variant")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_variants_share_small_traces() {
        let variants = [
            "u8_only",
            "byte_spans",
            "full_clear",
            "endpoints",
            "lean",
            "linked12",
            "linked16",
            "bitmap_u32",
            "halfword",
            "h3",
            "h25",
        ];
        for pattern in ["random", "balanced", "chain"] {
            for &variant in &variants {
                for bounds in [Bounds::Checked, Bounds::Unchecked] {
                    let result = run_micro(variant, 8, 32, pattern, bounds).unwrap();
                    assert_eq!(result.operations, 28);
                    assert!(result.checksum > 0);
                }
            }
        }
    }
    #[test]
    fn component_limits_and_long_chain() {
        assert!(run_micro("u8_only", 256, 256, "chain", Bounds::Checked).is_err());
        let span = run_micro("byte_spans", 8192, 8192, "chain", Bounds::Checked).unwrap();
        assert_eq!(span.operations, 8191);
        assert_eq!(span.buffer_bytes, 8194);
    }
}

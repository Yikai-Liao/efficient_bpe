//! Exact left-to-right AA selection from ordered chunks of valid pair starts.
//!
//! Each chunk summarizes its run boundaries independently. The coordinator
//! processes one summary per chunk; workers then select their own starts using
//! the returned incoming parity. Empty chunks preserve the previous run state.
//! No corpus-sized marking array or coordinator occurrence scan is needed.
//!
//! Preconditions: starts are globally strictly increasing, are valid AA edges
//! from one stable corpus snapshot, and the token's length is positive. Pieces
//! separated by sentinels must have a physical gap greater than token_length.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunSummary {
    pub first: u32,
    pub last: u32,
    /// Parity of the number of AA edges in the final run of this chunk.
    pub trailing_odd: bool,
    pub all_one_run: bool,
}

#[inline]
fn consecutive(left: u32, right: u32, token_length: u32) -> bool {
    left.checked_add(token_length) == Some(right)
}

/// Run independently on each chunk, after validating and ordering its starts.
pub fn summarize(starts: &[u32], token_length: u32) -> Option<RunSummary> {
    assert!(token_length > 0);
    let &first = starts.first()?;
    let mut last = first;
    let mut trailing_odd = true;
    let mut all_one_run = true;
    for &pos in &starts[1..] {
        debug_assert!(pos > last);
        if consecutive(last, pos, token_length) {
            trailing_odd = !trailing_odd;
        } else {
            trailing_odd = true;
            all_one_run = false;
        }
        last = pos;
    }
    Some(RunSummary {
        first,
        last,
        trailing_odd,
        all_one_run,
    })
}

/// Returns whether each chunk must skip its first valid AA edge.
///
/// This is O(chunks) coordinator work, independent of the occurrence count.
/// Only the leading run uses incoming parity. Once a gap is encountered, the
/// worker restarts at the leftmost edge of that new run.
pub fn incoming_parities(summaries: &[Option<RunSummary>], token_length: u32) -> Vec<bool> {
    assert!(token_length > 0);
    let mut last = None;
    let mut trailing_odd = false;
    summaries
        .iter()
        .map(|summary| {
            let Some(summary) = summary else {
                return false;
            };
            debug_assert!(last.is_none_or(|previous| previous < summary.first));
            let incoming = last.is_some_and(|previous| {
                consecutive(previous, summary.first, token_length) && trailing_odd
            });
            trailing_odd = if summary.all_one_run {
                incoming ^ summary.trailing_odd
            } else {
                summary.trailing_odd
            };
            last = Some(summary.last);
            incoming
        })
        .collect()
}

/// Run on each chunk in parallel after its incoming parity is known.
/// The callback can directly build a worker's plan; no extra selected Vec is
/// required by this module. All callbacks must finish planning before writes.
pub fn for_each_selected(
    starts: &[u32],
    token_length: u32,
    incoming_odd: bool,
    mut emit: impl FnMut(u32),
) {
    assert!(token_length > 0);
    let mut previous = None;
    let mut skip = incoming_odd;
    for &pos in starts {
        if let Some(last) = previous {
            debug_assert!(pos > last);
            if !consecutive(last, pos, token_length) {
                skip = false;
            }
        }
        if !skip {
            emit(pos);
        }
        skip = !skip;
        previous = Some(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(starts: &[u32], length: u32, cuts: &[usize]) {
        let chunks: Vec<_> = cuts.windows(2).map(|w| &starts[w[0]..w[1]]).collect();
        let summaries: Vec<_> = chunks.iter().map(|s| summarize(s, length)).collect();
        let incoming = incoming_parities(&summaries, length);
        let mut actual = Vec::new();
        for (chunk, odd) in chunks.iter().zip(incoming) {
            for_each_selected(chunk, length, odd, |pos| actual.push(pos));
        }
        // Independent sequential greedy non-overlap, using the span of two A
        // tokens rather than the parity recurrence under test.
        let mut expected = Vec::new();
        let mut after = 0_u64;
        for &pos in starts {
            if u64::from(pos) >= after {
                expected.push(pos);
                after = u64::from(pos) + 2 * u64::from(length);
            }
        }
        assert_eq!(actual, expected, "length={length}, cuts={cuts:?}");
    }

    #[test]
    fn preserves_runs_over_empty_chunks_and_restarts_after_gaps() {
        let starts = [10, 11, 12, 20, 21, 22, 23, 24, 30];
        check(&starts, 1, &[0, 1, 1, 2, 5, 5, 8, 9]);
        check(&[], 1, &[0, 0, 0]);
        check(&[1], 512, &[0, 0, 1, 1]);
    }

    #[test]
    fn long_tokens_and_large_positions_do_not_wrap() {
        check(&[1, 513, 1025, 1537, 3000, 3512], 512, &[0, 1, 3, 3, 5, 6]);
        check(
            &[u32::MAX - 20, u32::MAX - 15, u32::MAX - 5],
            5,
            &[0, 1, 1, 2, 3],
        );
        assert!(!consecutive(u32::MAX - 1, 3, 5));
    }

    #[test]
    fn all_partitions_of_small_valid_edge_patterns_match_greedy() {
        // A set bit continues an AA run; otherwise leave a gap. Every possible
        // chunk boundary, including duplicated boundaries for empty chunks,
        // is checked against the sequential greedy oracle.
        for length in [1, 3, 257] {
            for gaps in 0_u32..128 {
                let mut starts = vec![1_u32];
                for bit in 0..7 {
                    let gap = if gaps & (1 << bit) == 0 {
                        length
                    } else {
                        3 * length
                    };
                    starts.push(starts.last().unwrap() + gap);
                }
                for partitions in 0_u32..128 {
                    let mut cuts = vec![0];
                    for i in 1..8 {
                        if partitions & (1 << (i - 1)) != 0 {
                            cuts.extend([i, i]);
                        }
                    }
                    cuts.push(8);
                    check(&starts, length, &cuts);
                }
            }
        }
    }
}

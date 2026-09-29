//! In-place American-flag MSD sorting for physical u32 posting positions.

const SMALL: usize = 64;
const BUCKETS: usize = 256;
const ARRAYS_PER_LEVEL: usize = 3;
const ARRAY_BYTES: usize = BUCKETS * std::mem::size_of::<usize>();

#[derive(Default, Clone, Copy, Debug)]
pub(super) struct RadixStats {
    pub distribution_passes: usize,
    pub swaps: usize,
    pub fallback_calls: usize,
    /// Count/end/cursor arrays only; excludes call-frame bookkeeping.
    pub max_stack_payload_bytes: usize,
}

#[inline]
fn digit(value: u32, shift: u32) -> usize {
    ((value >> shift) & 0xff) as usize
}

/// Sorts without an allocation proportional to `positions.len()`.
pub(super) fn sort(positions: &mut [u32]) -> RadixStats {
    let mut stats = RadixStats::default();
    sort_slice(positions, 1, &mut stats);
    stats
}

fn sort_slice(positions: &mut [u32], depth: usize, stats: &mut RadixStats) {
    if positions.len() <= SMALL {
        if positions.len() > 1 {
            positions.sort_unstable();
            stats.fallback_calls += 1;
        }
        return;
    }

    // The highest differing byte is sufficient for MSD partitioning. A
    // constant high byte (or an all-equal slice) needs no distribution pass.
    let mut min = u32::MAX;
    let mut max = 0;
    for &position in positions.iter() {
        min = min.min(position);
        max = max.max(position);
    }
    let difference = min ^ max;
    if difference == 0 {
        return;
    }
    let shift = ((31 - difference.leading_zeros()) / 8) * 8;

    let mut count = [0_usize; BUCKETS];
    let mut end = [0_usize; BUCKETS];
    let mut cursor = [0_usize; BUCKETS];
    stats.distribution_passes += 1;
    stats.max_stack_payload_bytes = stats
        .max_stack_payload_bytes
        .max(depth * ARRAYS_PER_LEVEL * ARRAY_BYTES);
    for &position in positions.iter() {
        count[digit(position, shift)] += 1;
    }
    let mut boundary = 0;
    for bucket in 0..BUCKETS {
        cursor[bucket] = boundary;
        boundary += count[bucket];
        end[bucket] = boundary;
    }
    debug_assert_eq!(boundary, positions.len());

    // [start[b], cursor[b]) is already in bucket b. Swapping a misplaced
    // element into cursor[target] grows that target's settled prefix; the
    // current slot is retried until it contains its own bucket's element.
    for bucket in 0..BUCKETS {
        while cursor[bucket] < end[bucket] {
            let target = digit(positions[cursor[bucket]], shift);
            if target == bucket {
                cursor[bucket] += 1;
            } else {
                debug_assert!(cursor[target] < end[target]);
                positions.swap(cursor[bucket], cursor[target]);
                cursor[target] += 1;
                stats.swaps += 1;
            }
        }
    }

    let mut start = 0;
    for &length in &count {
        if length > 1 {
            sort_slice(&mut positions[start..start + length], depth + 1, stats);
        }
        start += length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(mut values: Vec<u32>) -> RadixStats {
        let mut expected = values.clone();
        expected.sort_unstable();
        let stats = sort(&mut values);
        assert_eq!(values, expected);
        assert!(stats.max_stack_payload_bytes <= 4 * ARRAYS_PER_LEVEL * ARRAY_BYTES);
        stats
    }

    #[test]
    fn small_and_extreme_values() {
        check(vec![]);
        check(vec![u32::MAX]);
        check(vec![0, u32::MAX, 1 << 31, 1, u32::MAX, 0]);
        check(vec![17; 1024]);
    }

    #[test]
    fn ordered_reverse_and_bucket_boundaries() {
        check((0..4096).collect());
        check((0..4096).rev().collect());
        let values = (0..4096)
            .map(|i| {
                let byte = (i % 256) as u32;
                (byte << 24) | ((255 - byte) << 16) | (i / 256) as u32
            })
            .collect();
        assert!(check(values).distribution_passes >= 1);
    }

    #[test]
    fn skewed_duplicates_and_random_multiset() {
        let mut skewed = vec![0x8000_0001; 8192];
        for index in (0..skewed.len()).step_by(71) {
            skewed[index] = index as u32;
        }
        check(skewed);
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        let random = (0..20_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 32) as u32
            })
            .collect();
        assert!(check(random).swaps > 0);
    }
}

//! A single physical-position bitmap for an already selected dense AA pair.
//!
//! Bits come only from historical posting starts validated against a stable
//! corpus. Long-token endpoint tags must never be discovered by physical scan.

use super::{Result, aa_parity};
use efficient_bpe_rust::TrainError;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

// 4096 physical positions per task gives parallel route/apply even for a
// moderately sized dense AA run, without a corpus-dependent tuning rule.
pub(super) const WORDS_PER_CHUNK: usize = 64;

pub(super) struct Bitmap {
    words: Vec<AtomicU64>,
}

impl Bitmap {
    pub(super) fn try_new(positions: usize) -> Option<Self> {
        let count = positions.div_ceil(64);
        let mut words = Vec::new();
        words.try_reserve_exact(count).ok()?;
        words.resize_with(count, || AtomicU64::new(0));
        Some(Self { words })
    }

    pub(super) fn word_len(&self) -> usize {
        self.words.len()
    }

    pub(super) fn capacity_bytes(&self) -> usize {
        self.words.capacity() * std::mem::size_of::<AtomicU64>()
    }

    pub(super) fn set(&self, pos: u32) {
        let pos = pos as usize;
        debug_assert!(pos / 64 < self.words.len());
        self.words[pos / 64].fetch_or(1_u64 << (pos % 64), Ordering::Relaxed);
    }

    pub(super) fn contains(&self, pos: Option<usize>) -> bool {
        let Some(pos) = pos else { return false };
        self.words
            .get(pos / 64)
            .is_some_and(|word| word.load(Ordering::Relaxed) & (1_u64 << (pos % 64)) != 0)
    }

    pub(super) fn for_each_bit(
        &self,
        words: Range<usize>,
        mut emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<()> {
        for word_i in words {
            let mut bits = self.words[word_i].load(Ordering::Relaxed);
            while bits != 0 {
                let pos = word_i * 64 + bits.trailing_zeros() as usize;
                let pos = u32::try_from(pos)
                    .map_err(|_| TrainError::Overflow("AA bitmap position exceeds u32"))?;
                emit(pos)?;
                bits &= bits - 1;
            }
        }
        Ok(())
    }

    pub(super) fn summarize(
        &self,
        words: Range<usize>,
        length: u32,
    ) -> Result<(Option<aa_parity::RunSummary>, usize)> {
        let mut summary: Option<aa_parity::RunSummary> = None;
        let mut count = 0;
        self.for_each_bit(words, |pos| {
            count += 1;
            match &mut summary {
                None => {
                    summary = Some(aa_parity::RunSummary {
                        first: pos,
                        last: pos,
                        trailing_odd: true,
                        all_one_run: true,
                    });
                }
                Some(current) => {
                    if current.last.checked_add(length) == Some(pos) {
                        current.trailing_odd = !current.trailing_odd;
                    } else {
                        current.trailing_odd = true;
                        current.all_one_run = false;
                    }
                    current.last = pos;
                }
            }
            Ok(())
        })?;
        Ok((summary, count))
    }

    pub(super) fn for_each_selected(
        &self,
        words: Range<usize>,
        length: u32,
        incoming_odd: bool,
        mut emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<()> {
        let mut previous: Option<u32> = None;
        let mut skip = incoming_odd;
        self.for_each_bit(words, |pos| {
            if previous.is_some_and(|last| last.checked_add(length) != Some(pos)) {
                skip = false;
            }
            if !skip {
                emit(pos)?;
            }
            skip = !skip;
            previous = Some(pos);
            Ok(())
        })
    }
}

pub(super) fn word_range(chunk: usize, words: usize) -> Range<usize> {
    let start = chunk * WORDS_PER_CHUNK;
    start..start + WORDS_PER_CHUNK.min(words - start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_parity_matches_slice_across_word_boundaries_and_empty_chunks() {
        let bitmap = Bitmap::try_new(262_200).unwrap();
        let starts = [1, 2, 3, 65, 66, 67, 70_000, 70_257, 70_514, 200_000];
        for &pos in &starts {
            bitmap.set(pos);
        }
        for length in [1, 257] {
            let count = bitmap.word_len().div_ceil(WORDS_PER_CHUNK);
            let summaries: Vec<_> = (0..count)
                .map(|i| {
                    bitmap
                        .summarize(word_range(i, bitmap.word_len()), length)
                        .unwrap()
                        .0
                })
                .collect();
            let incoming = aa_parity::incoming_parities(&summaries, length);
            let mut actual = Vec::new();
            for (i, &odd) in incoming.iter().enumerate() {
                bitmap
                    .for_each_selected(word_range(i, bitmap.word_len()), length, odd, |pos| {
                        actual.push(pos);
                        Ok(())
                    })
                    .unwrap();
            }
            let expected =
                aa_parity::incoming_parities(&[aa_parity::summarize(&starts, length)], length);
            let mut selected = Vec::new();
            aa_parity::for_each_selected(&starts, length, expected[0], |p| selected.push(p));
            assert_eq!(actual, selected);
        }
    }

    #[test]
    fn neighbor_bits_identify_selected_aa_matches() {
        for length in [1_usize, 257] {
            let bitmap = Bitmap::try_new(8 * length + 100).unwrap();
            let starts = [
                1,
                1 + length,
                1 + 2 * length,
                1 + 3 * length,
                1 + 4 * length,
                1 + 5 * length,
            ];
            for &pos in &starts {
                bitmap.set(pos as u32);
            }
            let count = bitmap.word_len().div_ceil(WORDS_PER_CHUNK);
            let summaries = (0..count)
                .map(|i| {
                    bitmap
                        .summarize(word_range(i, bitmap.word_len()), length as u32)
                        .unwrap()
                        .0
                })
                .collect::<Vec<_>>();
            let incoming = aa_parity::incoming_parities(&summaries, length as u32);
            let mut selected = Vec::new();
            for (i, &odd) in incoming.iter().enumerate() {
                bitmap
                    .for_each_selected(word_range(i, bitmap.word_len()), length as u32, odd, |p| {
                        selected.push(p as usize);
                        Ok(())
                    })
                    .unwrap();
            }
            assert_eq!(selected, [1, 1 + 2 * length, 1 + 4 * length]);
            for (i, &p) in selected.iter().enumerate() {
                assert_eq!(bitmap.contains(p.checked_sub(length)), i > 0);
                assert_eq!(
                    bitmap.contains(p.checked_add(2 * length)),
                    i + 1 < selected.len()
                );
            }
        }
    }
}

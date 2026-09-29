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

    #[cfg(test)]
    pub(super) fn set(&self, pos: u32) {
        let pos = pos as usize;
        debug_assert!(pos / 64 < self.words.len());
        self.words[pos / 64].fetch_or(1_u64 << (pos % 64), Ordering::Relaxed);
    }

    pub(super) fn or_word(&self, word: usize, bits: u64) {
        debug_assert!(word < self.words.len() && bits != 0);
        self.words[word].fetch_or(bits, Ordering::Relaxed);
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

/// One pending word per posting task. It is never shared across tasks.
#[derive(Default)]
pub(super) struct PendingWord {
    word: Option<usize>,
    bits: u64,
    flushes: usize,
}

impl PendingWord {
    pub(super) fn push(&mut self, bitmap: &Bitmap, pos: u32) {
        let word = pos as usize / 64;
        if self.word != Some(word) {
            self.flush(bitmap);
            self.word = Some(word);
        }
        self.bits |= 1_u64 << (pos % 64);
    }

    fn flush(&mut self, bitmap: &Bitmap) {
        if let Some(word) = self.word.take() {
            bitmap.or_word(word, self.bits);
            self.bits = 0;
            self.flushes += 1;
        }
    }

    pub(super) fn finish(mut self, bitmap: &Bitmap) -> usize {
        self.flush(bitmap);
        self.flushes
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

    #[test]
    fn word_cache_flushes_switches_and_retains_all_bits() {
        let bitmap = Bitmap::try_new(256).unwrap();
        let mut cache = PendingWord::default();
        for pos in [8, 7, 63, 70, 65, 9, 8, 130] {
            cache.push(&bitmap, pos);
        }
        // Four segments: word 0, word 1, word 0 again, word 2.
        assert_eq!(cache.finish(&bitmap), 4);
        for pos in [7, 8, 9, 63, 65, 70, 130] {
            assert!(bitmap.contains(Some(pos)));
        }
        assert!(!bitmap.contains(Some(64)));
    }

    #[test]
    fn separate_tasks_or_bits_in_the_same_word() {
        let bitmap = Bitmap::try_new(128).unwrap();
        let mut first = PendingWord::default();
        let mut second = PendingWord::default();
        first.push(&bitmap, 10);
        second.push(&bitmap, 11);
        let (left, right) = rayon::join(|| first.finish(&bitmap), || second.finish(&bitmap));
        assert_eq!((left, right), (1, 1));
        assert!(bitmap.contains(Some(10)));
        assert!(bitmap.contains(Some(11)));
    }
}

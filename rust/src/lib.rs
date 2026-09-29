//! Exact weighted BPE core with a movable u32 endpoint corpus.

pub mod ablation;
mod backend;
mod trainer;

pub use trainer::{Bounds, Prepared, Rule, TrainError, TrainOptions, TrainResult, train};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn prepared(words: &[(Vec<u32>, u64)], alphabet: usize) -> Prepared {
        let mut corpus = vec![0];
        let mut pivots = Vec::new();
        let mut weights = Vec::new();
        for (word, weight) in words {
            pivots.push(corpus.len() as u32);
            weights.push(*weight);
            corpus.extend(word);
            corpus.push(0);
        }
        Prepared {
            corpus,
            initial_lengths: vec![1; alphabet + 1],
            pivots,
            weights,
        }
    }

    fn naive(
        words: &[(Vec<u32>, u64)],
        alphabet: usize,
        max_merges: usize,
        min_frequency: u64,
    ) -> (Vec<Rule>, Vec<u32>) {
        let mut pieces: Vec<Vec<u32>> = words.iter().map(|(w, _)| w.clone()).collect();
        let mut merges = Vec::new();
        for _ in 0..max_merges {
            let mut counts = HashMap::<(u32, u32), u64>::new();
            for (piece, (_, weight)) in pieces.iter().zip(words) {
                for pair in piece.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_default() += weight;
                }
            }
            let selected = counts
                .into_iter()
                .filter(|(_, frequency)| *frequency >= min_frequency)
                .max_by(|(a, af), (b, bf)| af.cmp(bf).then_with(|| b.cmp(a)));
            let Some(((left, right), frequency)) = selected else {
                break;
            };
            let new_id = (alphabet + merges.len() + 1) as u32;
            merges.push(Rule {
                left,
                right,
                frequency,
            });
            for piece in &mut pieces {
                let mut out = Vec::new();
                let mut i = 0;
                while i < piece.len() {
                    if i + 1 < piece.len() && piece[i] == left && piece[i + 1] == right {
                        out.push(new_id);
                        i += 2;
                    } else {
                        out.push(piece[i]);
                        i += 1;
                    }
                }
                *piece = out;
            }
        }
        let mut final_tokens = vec![0];
        for piece in pieces {
            final_tokens.extend(piece);
            final_tokens.push(0);
        }
        (merges, final_tokens)
    }

    fn options(bounds: Bounds, max_merges: usize, min_frequency: u64) -> TrainOptions {
        TrainOptions {
            bounds,
            max_merges,
            min_frequency,
        }
    }

    #[test]
    fn random_tiny_matches_naive_and_access_modes() {
        let mut seed = 0x5e17_ba9e_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize
        };
        for _ in 0..120 {
            let count = 1 + next() % 5;
            let mut words = Vec::new();
            let mut seen = [false; 4];
            for _ in 0..count {
                let length = 1 + next() % 9;
                let word = (0..length)
                    .map(|_| {
                        let id = 1 + next() % 3;
                        seen[id] = true;
                        id as u32
                    })
                    .collect();
                words.push((word, (1 + next() % 4) as u64));
            }
            // Keep the initial alphabet dense and fully represented.
            for (id, present) in seen.iter().enumerate().skip(1) {
                if !present {
                    words.push((vec![id as u32], 1));
                }
            }
            let min_frequency = (1 + next() % 4) as u64;
            let expected = naive(&words, 3, 16, min_frequency);
            let input = prepared(&words, 3);
            let checked =
                train(input.clone(), options(Bounds::Checked, 16, min_frequency)).unwrap();
            let unchecked = train(input, options(Bounds::Unchecked, 16, min_frequency)).unwrap();
            assert_eq!((checked.merges, checked.final_tokens), expected);
            assert_eq!((unchecked.merges, unchecked.final_tokens), expected);
            assert_eq!(checked.actual_merges, unchecked.actual_merges);
            assert_eq!(checked.position_visits, unchecked.position_visits);
        }
    }

    #[test]
    fn self_overlap_and_historical_stale_positions() {
        let words = vec![(vec![1; 11], 3), (vec![1; 7], 2)];
        let expected = naive(&words, 1, 12, 1);
        for bounds in [Bounds::Checked, Bounds::Unchecked] {
            let result = train(prepared(&words, 1), options(bounds, 12, 1)).unwrap();
            assert_eq!((result.merges, result.final_tokens), expected);
            assert!(result.stale_visits > 0);
        }
    }

    #[test]
    fn id_and_length_cross_65536() {
        let words: Vec<(Vec<u32>, u64)> = (1..=65536).map(|id| (vec![id], 1)).collect();
        let input = prepared(&words, 65536);
        for bounds in [Bounds::Checked, Bounds::Unchecked] {
            let result = train(input.clone(), options(bounds, 0, 2)).unwrap();
            assert_eq!(result.rules, 0);
            assert!(result.final_tokens.contains(&65536));
        }
        let long = prepared(&[(vec![1; 65536], 1)], 1);
        for bounds in [Bounds::Checked, Bounds::Unchecked] {
            let result = train(long.clone(), options(bounds, 20, 1)).unwrap();
            assert_eq!(result.max_token_length, 65536);
            assert_eq!(result.final_tokens, vec![0, 17, 0]);
        }
    }

    #[test]
    fn invalid_inputs_and_weight_overflow_are_errors() {
        let valid = prepared(&[(vec![1, 1], 1)], 1);
        let mut invalid_cases = Vec::new();
        let mut invalid = valid.clone();
        invalid.corpus[0] = 1;
        invalid_cases.push(invalid);
        let mut invalid = valid.clone();
        *invalid.corpus.last_mut().unwrap() = 1;
        invalid_cases.push(invalid);
        let mut invalid = valid.clone();
        invalid.corpus[1] = 2;
        invalid_cases.push(invalid);
        let mut invalid = valid.clone();
        invalid.pivots[0] = 2;
        invalid_cases.push(invalid);
        let mut invalid = valid.clone();
        invalid.weights[0] = 0;
        invalid_cases.push(invalid);
        let mut invalid = valid.clone();
        invalid.initial_lengths[1] = 2;
        invalid_cases.push(invalid);
        let mut invalid = valid.clone();
        invalid.corpus.insert(3, 0);
        invalid_cases.push(invalid);
        let mut invalid = valid;
        invalid.weights.pop();
        invalid_cases.push(invalid);
        let overflow = prepared(&[(vec![1, 1, 1], u64::MAX)], 1);
        let max_valid = prepared(&[(vec![1, 1], u64::MAX)], 1);
        for bounds in [Bounds::Checked, Bounds::Unchecked] {
            for invalid in &invalid_cases {
                assert!(matches!(
                    train(invalid.clone(), options(bounds, 2, 1)),
                    Err(TrainError::InvalidInput(_))
                ));
            }
            assert!(matches!(
                train(overflow.clone(), options(bounds, 2, 1)),
                Err(TrainError::Overflow(_))
            ));
            let result = train(max_valid.clone(), options(bounds, 1, u64::MAX)).unwrap();
            assert_eq!(
                result.merges,
                vec![Rule {
                    left: 1,
                    right: 1,
                    frequency: u64::MAX,
                }]
            );
            assert_eq!(result.final_tokens, vec![0, 2, 0]);
        }
        let empty = Prepared {
            corpus: vec![0],
            initial_lengths: vec![1],
            pivots: vec![],
            weights: vec![],
        };
        let result = train(empty, TrainOptions::default()).unwrap();
        assert_eq!(result.final_tokens, vec![0]);
    }
}

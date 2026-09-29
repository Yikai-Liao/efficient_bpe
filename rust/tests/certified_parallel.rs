//! Compare observable training semantics, not implementation-dependent stale
//! occurrence counts: a batch deliberately avoids transient adjacency records.
use efficient_bpe_rust::ablation::{Options, train_variant};
use efficient_bpe_rust::{Bounds, Prepared, TrainOptions, train};

fn prepared(pieces: &[(Vec<u32>, u64)], alphabet: usize) -> Prepared {
    let mut corpus = vec![0];
    let mut pivots = Vec::new();
    let mut weights = Vec::new();
    for (piece, weight) in pieces {
        assert!(!piece.is_empty());
        pivots.push(corpus.len() as u32);
        weights.push(*weight);
        corpus.extend(piece);
        corpus.push(0);
    }
    Prepared {
        corpus,
        initial_lengths: vec![1; alphabet + 1],
        pivots,
        weights,
    }
}

fn check(input: Prepared, limit: usize, minimum: u64, workers: &[usize]) {
    let expected = train(
        input.clone(),
        TrainOptions {
            max_merges: limit,
            min_frequency: minimum,
            bounds: Bounds::Checked,
        },
    )
    .unwrap();
    for mode in ["parallel_certified", "parallel_certified_single"] {
        for &workers in workers {
            for bounds in [Bounds::Checked, Bounds::Unchecked] {
                let result = train_variant(
                    input.clone(),
                    Options {
                        max_merges: limit,
                        min_frequency: minimum,
                        bounds,
                        workers,
                    },
                    mode,
                )
                .unwrap_or_else(|e| panic!("{mode}, {workers} workers, {bounds:?}: {e}"));
                assert_eq!(result.core.merges, expected.merges, "{mode}/{workers}");
                assert_eq!(
                    result.core.final_tokens, expected.final_tokens,
                    "{mode}/{workers}"
                );
                assert_eq!(
                    result.core.actual_merges, expected.actual_merges,
                    "{mode}/{workers}"
                );
                assert_eq!(result.core.max_token_length, expected.max_token_length);
            }
        }
    }
}

#[test]
fn adjacent_rules_self_runs_and_physical_shard_boundaries() {
    // The two selected patterns precede their cross-boundary pair in a tie.
    let adjacent = [1, 4, 2, 3].repeat(73);
    check(prepared(&[(adjacent, 7)], 4), 30, 1, &[1, 2, 4, 7]);
    for len in [3, 7, 63, 64, 65, 255, 256, 257] {
        check(prepared(&[(vec![1; len], 3)], 1), 20, 1, &[1, 2, 4]);
        check(prepared(&[([1, 2].repeat(len), 5)], 2), 20, 1, &[1, 2, 4]);
    }
}

#[test]
fn weighted_ties_new_candidate_preemption_and_vocabulary_cutoff() {
    let pieces = vec![
        (vec![1, 2, 3], 3),
        (vec![4, 5], 2),
        (vec![1, 4, 2, 3, 1, 4, 2, 3], 1_u64 << 40),
        (vec![3; 13], 7),
    ];
    for limit in [0, 1, 2, 3, 5, 40] {
        check(prepared(&pieces, 5), limit, 2, &[1, 2, 4]);
    }
    check(prepared(&[(vec![1, 2], u64::MAX)], 2), 2, 1, &[1, 4]);
}

#[test]
fn random_weighted_corpora_match_serial_for_both_access_paths() {
    let mut state = 0x97fd_8176_34aa_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as usize
    };
    for case in 0..80 {
        let alphabet = 2 + next() % 7;
        let mut pieces: Vec<(Vec<u32>, u64)> = (0..1 + next() % 6)
            .map(|_| {
                let len = 1 + next() % 100;
                let ids = (0..len).map(|_| (1 + next() % alphabet) as u32).collect();
                (ids, (1 + next() % 11) as u64)
            })
            .collect();
        // The public prepared-input contract requires a fully represented alphabet.
        pieces.extend((1..=alphabet).map(|id| (vec![id as u32], 1)));
        check(prepared(&pieces, alphabet), 30, 1 + case % 4, &[1, 2, 4]);
    }
}

#[test]
fn long_spans_and_wide_initial_ids() {
    check(prepared(&[(vec![1; 65536], 1)], 1), 20, 1, &[1, 4]);
    let pieces: Vec<_> = (1..=65536).map(|id| (vec![id], 1)).collect();
    check(prepared(&pieces, 65536), 0, 1, &[1, 4]);
    check(prepared(&[], 0), 0, 1, &[1, 4]);
}

#[test]
fn invalid_weights_and_access_contracts_are_rejected_before_workers_start() {
    let mut bad = prepared(&[(vec![1, 2], 1)], 2);
    bad.corpus[0] = 1;
    let overflow = prepared(&[(vec![1, 1, 1], u64::MAX)], 1);
    for input in [bad, overflow] {
        for mode in ["parallel_certified", "parallel_certified_single"] {
            assert!(
                train_variant(
                    input.clone(),
                    Options {
                        workers: 4,
                        bounds: Bounds::Unchecked,
                        ..Options::default()
                    },
                    mode
                )
                .is_err()
            );
        }
    }
}

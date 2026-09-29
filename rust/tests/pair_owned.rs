//! Exercise global ordering and threshold decisions when scalar frequencies
//! and candidate heaps live on different workers from their position lists.
use efficient_bpe_rust::ablation::{Options, train_variant};
use efficient_bpe_rust::{Bounds, Prepared, TrainOptions, train};

fn prepared(pieces: &[(Vec<u32>, u64)], alphabet: usize) -> Prepared {
    let mut result = Prepared {
        corpus: vec![0],
        initial_lengths: vec![1; alphabet + 1],
        pivots: Vec::new(),
        weights: Vec::new(),
    };
    for (piece, weight) in pieces {
        result.pivots.push(result.corpus.len() as u32);
        result.weights.push(*weight);
        result.corpus.extend(piece);
        result.corpus.push(0);
    }
    result
}

fn compare(input: Prepared, limit: usize, minimum: u64, workers: &[usize]) {
    let expected = train(
        input.clone(),
        TrainOptions {
            max_merges: limit,
            min_frequency: minimum,
            bounds: Bounds::Checked,
        },
    )
    .unwrap();
    for &workers in workers {
        for variant in ["parallel_pair_owned", "parallel_pair_owned_compact"] {
            for bounds in [Bounds::Checked, Bounds::Unchecked] {
                let actual = train_variant(
                    input.clone(),
                    Options {
                        workers,
                        max_merges: limit,
                        min_frequency: minimum,
                        bounds,
                    },
                    variant,
                )
                .unwrap_or_else(|e| panic!("{variant}, {workers} workers, {bounds:?}: {e}"))
                .core;
                assert_eq!(actual.merges, expected.merges, "{variant}/{workers}/{bounds:?}");
                assert_eq!(actual.final_tokens, expected.final_tokens, "{variant}/{workers}/{bounds:?}");
                assert_eq!(actual.actual_merges, expected.actual_merges, "{variant}/{workers}/{bounds:?}");
                assert_eq!(actual.max_token_length, expected.max_token_length, "{variant}/{workers}/{bounds:?}");
            }
        }
    }
}

#[test]
fn owner_frontiers_preserve_ties_cutoffs_and_batch_cap() {
    // More equal-frequency independent pairs than one certified batch. Fresh
    // IDs must follow global key order regardless of the owning heap.
    let pairs: Vec<_> = (0..300).map(|i| (vec![2 * i + 1, 2 * i + 2], 7)).collect();
    for limit in [1, 31, 256, 257, 400] {
        compare(prepared(&pairs, 600), limit, 2, &[1, 2, 4]);
    }
    // A new high-frequency pair must preempt a lower-frequency independent
    // old pair even when their scalar owners are different.
    let pieces = [(vec![1, 2, 3], 3), (vec![4, 5], 2)];
    for limit in [1, 2, 8] {
        compare(prepared(&pieces, 5), limit, 1, &[1, 2, 4]);
    }
}

#[test]
fn eligibility_uses_global_counts_for_initial_and_new_pairs() {
    // Initial AB and BC, then the freshly created X-C, exceed the threshold
    // only after all position shards' partial counts are combined.
    let pieces = vec![(vec![1, 2, 3], 1); 16];
    compare(prepared(&pieces, 3), 10, 12, &[1, 2, 4, 7]);
    let weighted = [
        (vec![1, 4, 2, 3, 1, 4, 2, 3], 1_u64 << 40),
        (vec![3, 3, 3, 3, 3], 9),
        (vec![1, 2, 3], 3),
    ];
    compare(prepared(&weighted, 4), 40, 2, &[1, 4]);
    compare(prepared(&[(vec![1, 2], u64::MAX)], 2), 2, 1, &[1, 4]);
}

#[test]
fn cross_shard_adjacencies_aa_parity_and_long_spans() {
    compare(
        prepared(&[([1, 4, 2, 3].repeat(79), 3)], 4),
        40,
        1,
        &[1, 4, 7],
    );
    for len in [3, 63, 65, 257, 4097] {
        compare(prepared(&[(vec![1; len], 1)], 1), 20, 1, &[1, 4]);
    }
    compare(prepared(&[([1, 2].repeat(257), 5)], 2), 20, 1, &[1, 4]);
    compare(prepared(&[], 0), 0, 1, &[1, 4]);
    compare(prepared(&[(vec![1], 1)], 1), 20, 1, &[1, 4]);
}

#[test]
fn random_weighted_traces_across_owner_and_access_modes() {
    let mut state = 0x927b_d81e_u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as usize
    };
    for case in 0..32 {
        let alphabet = 2 + next() % 8;
        let mut pieces: Vec<_> = (0..1 + next() % 5)
            .map(|_| {
                let ids = (0..1 + next() % 70)
                    .map(|_| (1 + next() % alphabet) as u32)
                    .collect();
                (ids, (1 + next() % 7) as u64)
            })
            .collect();
        pieces.push(((1..=alphabet as u32).collect(), 1));
        compare(prepared(&pieces, alphabet), 35, 1 + case % 4, &[1, 4]);
    }
}

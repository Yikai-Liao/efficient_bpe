//! Exact-trace checks for the sparse single-rule owner scheduler.
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
        for variant in [
            "parallel_pair_owned_single",
            "parallel_sparse_owner",
            "parallel_sparse_owner_all",
        ] {
            let actual = train_variant(
                input.clone(),
                Options {
                    workers,
                    max_merges: limit,
                    min_frequency: minimum,
                    bounds: Bounds::Checked,
                },
                variant,
            )
            .unwrap_or_else(|e| panic!("{variant}, {workers} workers: {e}"))
            .core;
            assert_eq!(
                actual.merges, expected.merges,
                "{variant}/{workers} workers"
            );
            assert_eq!(
                actual.final_tokens, expected.final_tokens,
                "{variant}/{workers} workers"
            );
            assert_eq!(
                actual.actual_merges, expected.actual_merges,
                "{variant}/{workers} workers"
            );
            assert_eq!(
                actual.max_token_length, expected.max_token_length,
                "{variant}/{workers} workers"
            );
        }
    }
}

#[test]
fn refreshes_cached_heads_and_drops_a_deleted_winner() {
    // AB is the unique initial maximum. Merging it removes high-ranked old
    // candidates and creates fresh candidates on both sides of the span.
    let pieces: Vec<(Vec<u32>, u64)> = vec![
        (vec![1, 2, 3, 4, 1, 2, 3, 4], 4),
        (vec![1, 2, 8], 2),
        (vec![7, 3, 4], 2),
        (vec![5, 6, 5, 6], 3),
    ];
    compare(prepared(&pieces, 8), 32, 1, &[1, 2, 4]);
}

#[test]
fn filters_fresh_pairs_below_the_global_threshold() {
    // Initial AB is eligible; its boundary-created pairs are initially below
    // threshold and must be indexed only while their exact weighted count can
    // reach the threshold later.
    let pieces = [
        (vec![1, 2, 3, 4], 2),
        (vec![1, 2, 5], 1),
        (vec![6, 1, 2], 1),
        (vec![7, 8, 7, 8], 3),
    ];
    compare(prepared(&pieces, 8), 24, 3, &[1, 2, 4]);
}

#[test]
fn handles_aa_across_active_slices_and_delayed_worker_wakeups() {
    // Long AA runs cross several active slices and alternate with other pairs.
    compare(
        prepared(&[(vec![1; 513], 2), (vec![1, 2, 1, 2, 1, 2], 3)], 2),
        40,
        1,
        &[1, 4, 7],
    );

    // Several merge epochs create longer token IDs. Workers whose local
    // slice is temporarily unaffected must still accept later new positions.
    let pieces: Vec<(Vec<u32>, u64)> = vec![
        (
            "abcdefghabcdefgh"
                .chars()
                .map(|c| c as u32 - 'a' as u32 + 1)
                .collect(),
            3,
        ),
        (
            "xyxyabcdefgh"
                .chars()
                .map(|c| c as u32 - 'a' as u32 + 1)
                .collect(),
            2,
        ),
        (
            "mnopmnopxy"
                .chars()
                .map(|c| c as u32 - 'a' as u32 + 1)
                .collect(),
            2,
        ),
        (
            "qrstqrst"
                .chars()
                .map(|c| c as u32 - 'a' as u32 + 1)
                .collect(),
            1,
        ),
        (
            "abcdefghijklmnopqrstuvwxyz"
                .chars()
                .map(|c| c as u32 - 'a' as u32 + 1)
                .collect(),
            1,
        ),
    ];
    compare(prepared(&pieces, 26), 48, 1, &[1, 2, 4, 7]);
}

#[test]
fn prune_only_epochs_do_not_replay_old_plans() {
    // The winner changes a neighboring live span; stale positions for the
    // previous maximum must be pruned without applying its earlier plan.
    let pieces = [
        (vec![1, 2, 1, 2, 3, 4, 3, 4, 1, 2], 5),
        (vec![3, 4, 5, 6, 3, 4], 4),
        (vec![7, 1, 2, 8], 2),
    ];
    compare(prepared(&pieces, 8), 40, 1, &[1, 2, 4, 7]);
}

#[test]
fn conservative_worker_masks_preserve_winner_at_sixty_five_workers() {
    let mut pieces: Vec<_> = (0..64).map(|i| (vec![2 * i + 1, 2 * i + 2], 1)).collect();
    pieces.push((vec![1, 2], 1));
    // The extra occurrence makes (1, 2) eligible while keeping the corpus
    // large enough to instantiate 65 worker slices.
    compare(prepared(&pieces, 128), 8, 2, &[65]);
}

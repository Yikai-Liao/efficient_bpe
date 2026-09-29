use efficient_bpe_rust::ablation::{Options, train_variant};
use efficient_bpe_rust::{Bounds, Prepared, TrainOptions, train};

fn prepared(pieces: &[(Vec<u32>, u64)], alphabet: usize) -> Prepared {
    let mut input = Prepared {
        corpus: vec![0],
        initial_lengths: vec![1; alphabet + 1],
        pivots: Vec::new(),
        weights: Vec::new(),
    };
    for (piece, weight) in pieces {
        input.pivots.push(input.corpus.len() as u32);
        input.weights.push(*weight);
        input.corpus.extend(piece);
        input.corpus.push(0);
    }
    input
}

fn spatial(
    input: Prepared,
    limit: usize,
    workers: usize,
    bounds: Bounds,
    variant: &str,
) -> (usize, usize) {
    let expected = train(
        input.clone(),
        TrainOptions {
            max_merges: limit,
            min_frequency: 1,
            bounds: Bounds::Checked,
        },
    )
    .unwrap();
    let actual = train_variant(
        input,
        Options {
            workers,
            max_merges: limit,
            min_frequency: 1,
            bounds,
        },
        variant,
    )
    .unwrap();
    assert_eq!(actual.core.merges, expected.merges);
    assert_eq!(actual.core.final_tokens, expected.final_tokens);
    assert_eq!(actual.core.actual_merges, expected.actual_merges);
    (
        actual.metrics["spatial_base_width_sum"] as usize,
        actual.metrics["spatial_final_width_sum"] as usize,
    )
}

#[test]
fn separated_ab_bc_share_a_spatial_batch() {
    let mut pieces = Vec::new();
    pieces.extend((0..10).map(|_| (vec![1, 2], 1)));
    pieces.extend((0..9).map(|_| (vec![2, 3], 1)));
    for workers in [1, 4] {
        for bounds in [Bounds::Checked, Bounds::Unchecked] {
            for variant in [
                "parallel_pair_owned_spatial",
                "parallel_pair_owned_spatial_extra",
            ] {
                assert_eq!(
                    spatial(prepared(&pieces, 3), 2, workers, bounds, variant),
                    (1, 2)
                );
            }
        }
    }
}

#[test]
fn actual_abc_overlap_keeps_serial_order() {
    let pieces = vec![(vec![1, 2, 3], 3); 8];
    for workers in [1, 4] {
        for bounds in [Bounds::Checked, Bounds::Unchecked] {
            for variant in [
                "parallel_pair_owned_spatial",
                "parallel_pair_owned_spatial_extra",
            ] {
                assert_eq!(
                    spatial(prepared(&pieces, 3), 2, workers, bounds, variant),
                    (2, 2)
                );
            }
        }
    }
}

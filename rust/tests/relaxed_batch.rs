//! The relaxed experiment may change the greedy order. Each reported merge
//! must still have the true weighted count and apply exactly left-to-right.
use efficient_bpe_rust::ablation::{Options, train_variant};
use efficient_bpe_rust::{Bounds, Prepared, Rule};

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

fn replay(pieces: &[(Vec<u32>, u64)], alphabet: usize, rules: &[Rule]) -> Vec<u32> {
    let mut pieces = pieces.to_vec();
    for (i, rule) in rules.iter().enumerate() {
        let observed = pieces
            .iter()
            .map(|(ids, weight)| {
                ids.windows(2)
                    .filter(|pair| pair == &[rule.left, rule.right])
                    .count() as u64
                    * weight
            })
            .sum::<u64>();
        assert_eq!(observed, rule.frequency);
        assert!(observed > 0);
        for (ids, _) in &mut pieces {
            let mut out = Vec::new();
            let mut pos = 0;
            while pos < ids.len() {
                if pos + 1 < ids.len() && (ids[pos], ids[pos + 1]) == (rule.left, rule.right) {
                    out.push((alphabet + i + 1) as u32);
                    pos += 2;
                } else {
                    out.push(ids[pos]);
                    pos += 1;
                }
            }
            *ids = out;
        }
    }
    let mut final_tokens = vec![0];
    for (ids, _) in pieces {
        final_tokens.extend(ids);
        final_tokens.push(0);
    }
    final_tokens
}

#[test]
fn relaxed_batch_is_explicitly_different_from_exact_greedy() {
    let pieces = vec![(vec![1, 2, 3], 3), (vec![4, 5], 2)];
    let input = prepared(&pieces, 5);
    let options = Options {
        max_merges: 2,
        min_frequency: 1,
        workers: 4,
        bounds: Bounds::Checked,
    };
    let exact = train_variant(input.clone(), options, "parallel_certified").unwrap();
    let relaxed = train_variant(input, options, "parallel_batch_relaxed").unwrap();
    assert_ne!(relaxed.core.merges, exact.core.merges);
    assert_eq!(
        (relaxed.core.merges[1].left, relaxed.core.merges[1].right),
        (4, 5)
    );
    assert_eq!(
        replay(&pieces, 5, &relaxed.core.merges),
        relaxed.core.final_tokens
    );
}

#[test]
fn relaxed_trace_is_valid_and_deterministic_across_worker_counts() {
    let mut seed = 0x1684_3351_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 32) as usize
    };
    for _ in 0..60 {
        let mut pieces: Vec<_> = (0..3)
            .map(|_| {
                let ids = (0..30 + next() % 80)
                    .map(|_| (1 + next() % 6) as u32)
                    .collect();
                (ids, (1 + next() % 9) as u64)
            })
            .collect();
        pieces.push(((1..=6).collect(), 1));
        let input = prepared(&pieces, 6);
        let mut baseline = None;
        for workers in [1, 2, 4] {
            for bounds in [Bounds::Checked, Bounds::Unchecked] {
                let result = train_variant(
                    input.clone(),
                    Options {
                        max_merges: 40,
                        min_frequency: 2,
                        workers,
                        bounds,
                    },
                    "parallel_batch_relaxed",
                )
                .unwrap()
                .core;
                assert_eq!(replay(&pieces, 6, &result.merges), result.final_tokens);
                let trace = (result.merges, result.final_tokens);
                if let Some(expected) = &baseline {
                    assert_eq!(&trace, expected);
                } else {
                    baseline = Some(trace);
                }
            }
        }
    }
}

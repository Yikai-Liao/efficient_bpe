use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Piece {
    ids: Vec<u32>,
    weight: u64,
}

type Pair = (u32, u32);
#[derive(Clone, Debug, PartialEq, Eq)]
struct Rule {
    pair: Pair,
    freq: u64,
    new_id: u32,
}

fn recount(pieces: &[Piece]) -> BTreeMap<Pair, u64> {
    let mut out = BTreeMap::new();
    for piece in pieces {
        for edge in piece.ids.windows(2) {
            *out.entry((edge[0], edge[1])).or_insert(0) += piece.weight;
        }
    }
    out
}

fn ordered_candidates(freq: &BTreeMap<Pair, u64>, min_frequency: u64) -> Vec<(Pair, u64)> {
    let mut out: Vec<_> = freq
        .iter()
        .filter(|(_, f)| **f >= min_frequency)
        .map(|(p, f)| (*p, *f))
        .collect();
    out.sort_by(|(pa, fa), (pb, fb)| fb.cmp(fa).then_with(|| pa.cmp(pb)));
    out
}

fn next_id(pieces: &[Piece]) -> u32 {
    pieces
        .iter()
        .flat_map(|p| p.ids.iter())
        .copied()
        .max()
        .unwrap_or(0)
        + 1
}

fn replace_one(pieces: &mut [Piece], pair: Pair, fresh: u32) -> u64 {
    let mut replacements = 0;
    for piece in pieces {
        let mut out = Vec::with_capacity(piece.ids.len());
        let mut i = 0;
        while i < piece.ids.len() {
            if i + 1 < piece.ids.len() && (piece.ids[i], piece.ids[i + 1]) == pair {
                out.push(fresh);
                replacements += piece.weight;
                i += 2;
            } else {
                out.push(piece.ids[i]);
                i += 1;
            }
        }
        piece.ids = out;
    }
    replacements
}

fn serial(mut pieces: Vec<Piece>, min_frequency: u64, max_rules: usize) -> (Vec<Rule>, Vec<Piece>) {
    let mut rules = Vec::new();
    while rules.len() < max_rules {
        let freq = recount(&pieces);
        let Some((&pair, &count)) = freq
            .iter()
            .filter(|(_, f)| **f >= min_frequency)
            .max_by(|(pa, fa), (pb, fb)| fa.cmp(fb).then_with(|| pb.cmp(pa)))
        else {
            break;
        };
        let id = next_id(&pieces);
        let actual = replace_one(&mut pieces, pair, id);
        assert!(
            actual > 0,
            "selected pair must have an actual non-overlapping replacement"
        );
        rules.push(Rule {
            pair,
            freq: count,
            new_id: id,
        });
    }
    (rules, pieces)
}

fn crosses(a: Pair, b: Pair) -> bool {
    a.1 == b.0 || a.0 == b.1
}

fn choose_batch(freq: &BTreeMap<Pair, u64>, min_frequency: u64) -> Vec<(Pair, u64)> {
    let ordered = ordered_candidates(freq, min_frequency);
    let mut batch = Vec::new();
    for candidate in ordered {
        let pair = candidate.0;
        if pair.0 == pair.1 {
            if batch.is_empty() {
                batch.push(candidate);
            }
            break;
        }
        if batch.iter().any(|(prior, _)| crosses(*prior, pair)) {
            break;
        }
        batch.push(candidate);
    }
    batch
}

fn replace_batch(pieces: &mut [Piece], batch: &[(Pair, u64)], first_id: u32) {
    let ids: BTreeMap<Pair, u32> = batch
        .iter()
        .enumerate()
        .map(|(i, (p, _))| (*p, first_id + i as u32))
        .collect();
    for piece in pieces {
        let mut out = Vec::with_capacity(piece.ids.len());
        let mut i = 0;
        while i < piece.ids.len() {
            if i + 1 < piece.ids.len()
                && let Some(&fresh) = ids.get(&(piece.ids[i], piece.ids[i + 1]))
            {
                out.push(fresh);
                i += 2;
                continue;
            }
            out.push(piece.ids[i]);
            i += 1;
        }
        piece.ids = out;
    }
}

fn certified(
    mut pieces: Vec<Piece>,
    min_frequency: u64,
    max_rules: usize,
) -> (Vec<Rule>, Vec<Piece>, Vec<usize>) {
    let mut rules = Vec::new();
    let mut widths = Vec::new();
    while rules.len() < max_rules {
        let freq = recount(&pieces);
        let batch = choose_batch(&freq, min_frequency);
        if batch.is_empty() {
            break;
        }
        let batch: Vec<_> = batch.into_iter().take(max_rules - rules.len()).collect();
        widths.push(batch.len());
        let first = next_id(&pieces);
        for (i, (pair, count)) in batch.iter().enumerate() {
            rules.push(Rule {
                pair: *pair,
                freq: *count,
                new_id: first + i as u32,
            });
        }
        replace_batch(&mut pieces, &batch, first);
    }
    (rules, pieces, widths)
}

fn check_case(pieces: Vec<Piece>, min_frequency: u64) -> Vec<usize> {
    let (sr, sf) = serial(pieces.clone(), min_frequency, 1000);
    let (br, bf, widths) = certified(pieces, min_frequency, 1000);
    assert_eq!(br, sr, "rule trace mismatch");
    assert_eq!(bf, sf, "final piece mismatch");
    widths
}

fn unweighted(ids: &[u32]) -> Vec<Piece> {
    vec![Piece {
        ids: ids.to_vec(),
        weight: 1,
    }]
}

#[test]
fn exhaustive_all_ternary_strings_through_length_seven() {
    let mut cases = 0usize;
    let mut batches = 0usize;
    let mut multi_batches = 0usize;
    let mut max_width = 0usize;
    for len in 1..=7usize {
        for code in 0..3usize.pow(len as u32) {
            let mut x = code;
            let mut ids = vec![0; len];
            for id in &mut ids {
                *id = (x % 3) as u32;
                x /= 3;
            }
            let widths = check_case(unweighted(&ids), 2);
            cases += 1;
            batches += widths.len();
            multi_batches += widths.iter().filter(|w| **w > 1).count();
            max_width = max_width.max(widths.iter().copied().max().unwrap_or(0));
        }
    }
    eprintln!(
        "exhaustive cases={cases} epochs={batches} multi_rule_epochs={multi_batches} max_width={max_width}"
    );
}

#[test]
fn weighted_random_multi_piece_cases() {
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let mut multi_batches = 0usize;
    let mut max_width = 0usize;
    for case in 0..2000 {
        let mut pieces = Vec::new();
        let count = (next() % 5 + 1) as usize;
        for _ in 0..count {
            let len = (next() % 13 + 1) as usize;
            let mut ids = Vec::with_capacity(len);
            for _ in 0..len {
                ids.push((next() % 4) as u32);
            }
            pieces.push(Piece {
                ids,
                weight: next() % 9 + 1,
            });
        }
        let widths = check_case(pieces, 2);
        multi_batches += widths.iter().filter(|w| **w > 1).count();
        max_width = max_width.max(widths.iter().copied().max().unwrap_or(0));
        if case % 500 == 499 {
            eprintln!(
                "weighted cases={} multi_rule_epochs={multi_batches} max_width={max_width}",
                case + 1
            );
        }
    }
}

#[test]
fn targeted_overlap_adjacency_weights_and_piece_boundaries() {
    for n in 1..=40 {
        check_case(unweighted(&[0, 1].repeat(n)), 2); // ABAB and adjacent same-rule matches
        check_case(unweighted(&vec![0; n]), 2); // AA self-overlap; first rule stays a singleton batch
        check_case(unweighted(&[0, 1, 2, 0, 1, 2, 0, 1, 2]), 2); // newly created context pairs
    }
    let pieces = vec![
        Piece {
            ids: vec![0, 1, 2, 0, 1],
            weight: 7,
        },
        Piece {
            ids: vec![2, 0, 1, 2, 0],
            weight: 3,
        },
        Piece {
            ids: vec![1],
            weight: 999,
        },
    ];
    check_case(pieces, 2);
}

#[test]
fn parents_of_fresh_keys_are_not_preempted_inside_batch() {
    // Search every ternary corpus through length 9 for the tight case where a
    // generated key ties the parent winner frequency and could otherwise jump
    // ahead of a later old candidate. Exact trace equality is the oracle.
    let mut checked = 0usize;
    for len in 1..=9usize {
        for code in 0..3usize.pow(len as u32) {
            let mut x = code;
            let mut ids = vec![0; len];
            for id in &mut ids {
                *id = (x % 3) as u32;
                x /= 3;
            }
            check_case(unweighted(&ids), 2);
            checked += 1;
        }
    }
    eprintln!("preemption-counterexample search cases={checked}, mismatches=0");
}

#[test]
fn tied_fresh_key_has_later_ancestor_and_cannot_preempt_batch_prefix() {
    // Pair (0,1) is first by lex among three frequency-3 pairs. Replacing it
    // creates (2,new) and (new,3), each still at frequency 3. Their old
    // ancestors (2,0) and (1,3) are both frequency 3 and sort after (0,1);
    // (1,3) conflicts with the winner, so the certified batch must stop there.
    let mut ids = Vec::new();
    for _ in 0..3 {
        ids.extend([2, 0, 1, 3]);
    }
    let pieces = unweighted(&ids);
    let initial = recount(&pieces);
    assert_eq!(initial[&(0, 1)], 3);
    assert_eq!(initial[&(1, 3)], 3);
    assert_eq!(initial[&(2, 0)], 3);
    let batch = choose_batch(&initial, 2);
    assert_eq!(batch, vec![((0, 1), 3)]);
    let mut once = pieces.clone();
    let first_id = next_id(&once);
    replace_batch(&mut once, &batch, first_id);
    let after_one = recount(&once);
    let fresh = first_id;
    assert_eq!(after_one[&(2, fresh)], 3);
    assert_eq!(after_one[&(fresh, 3)], 3);
    let (rules, final_batch, widths) = certified(pieces.clone(), 2, 1000);
    let (serial_rules, final_serial) = serial(pieces, 2, 1000);
    assert_eq!(rules, serial_rules);
    assert_eq!(final_batch, final_serial);
    assert_eq!(widths.first(), Some(&1));
}

//! Batch-start token windows at physical region cuts. During the exclusive
//! phase, a region owns only its own mutable corpus slice; it never borrows
//! the complete corpus or another region's slice.

use super::{HEAD, ID_MASK, RegionCuts, Result};
use efficient_bpe_rust::TrainError;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Token {
    head: usize,
    id: u32,
    len: usize,
}

impl Token {
    fn head_raw(self) -> u32 {
        if self.id == 0 { 0 } else { self.id | HEAD }
    }

    fn tail_raw(self) -> u32 {
        if self.id == 0 {
            0
        } else if self.len == 1 {
            self.id | HEAD
        } else {
            self.id
        }
    }
}

#[derive(Clone, Debug)]
struct CutWindow {
    cut: usize,
    // K,L,A,B,C,D relative to the token covering the cut. A sentinel
    // occupies one slot and terminates traversal in its direction.
    tokens: [Option<Token>; 6],
}

impl CutWindow {
    fn anchor(&self) -> Token {
        self.tokens[2].expect("every cut has an anchor")
    }

    fn preceding(&self) -> Option<Token> {
        self.tokens[1].filter(|token| token.id != 0)
    }

    fn old_raw(&self, pos: usize, head_query: bool) -> Option<u32> {
        self.tokens.iter().flatten().find_map(|token| {
            if head_query && pos == token.head {
                Some(token.head_raw())
            } else if !head_query && pos == token.head + token.len - 1 {
                Some(token.tail_raw())
            } else {
                None
            }
        })
    }
}

fn token_at(corpus: &[AtomicU32], lengths: &[u32], head: usize) -> Result<Token> {
    let raw = corpus
        .get(head)
        .ok_or(TrainError::InternalInvariant(
            "snapshot head outside corpus",
        ))?
        .load(Ordering::Relaxed);
    if raw == 0 {
        return Ok(Token {
            head,
            id: 0,
            len: 1,
        });
    }
    if raw & HEAD == 0 {
        return Err(TrainError::InternalInvariant(
            "snapshot anchor is a bare tail or stale interior",
        ));
    }
    let id = raw & ID_MASK;
    let len = *lengths
        .get(id as usize)
        .ok_or(TrainError::InternalInvariant("snapshot ID outside lengths"))?
        as usize;
    if len == 0 || head.checked_add(len).is_none_or(|end| end >= corpus.len()) {
        return Err(TrainError::InternalInvariant(
            "snapshot token length crosses corpus end",
        ));
    }
    Ok(Token { head, id, len })
}

fn predecessor(corpus: &[AtomicU32], lengths: &[u32], token: Token) -> Result<Option<Token>> {
    if token.head == 0 {
        return Ok(None);
    }
    let end = token.head - 1;
    let raw = corpus[end].load(Ordering::Relaxed);
    if raw == 0 {
        return Ok(Some(Token {
            head: end,
            id: 0,
            len: 1,
        }));
    }
    let id = raw & ID_MASK;
    let len = *lengths
        .get(id as usize)
        .ok_or(TrainError::InternalInvariant(
            "snapshot predecessor ID outside lengths",
        ))? as usize;
    let head = token
        .head
        .checked_sub(len)
        .ok_or(TrainError::InternalInvariant(
            "snapshot predecessor starts before corpus",
        ))?;
    let prior = token_at(corpus, lengths, head)?;
    if prior.id != id || prior.len != len {
        return Err(TrainError::InternalInvariant(
            "snapshot predecessor endpoint disagrees with head",
        ));
    }
    Ok(Some(prior))
}

fn successor(corpus: &[AtomicU32], lengths: &[u32], token: Token) -> Result<Option<Token>> {
    let head = token
        .head
        .checked_add(token.len)
        .ok_or(TrainError::InternalInvariant(
            "snapshot successor overflows",
        ))?;
    if head >= corpus.len() {
        return Ok(None);
    }
    Ok(Some(token_at(corpus, lengths, head)?))
}

fn build_window(
    corpus: &[AtomicU32],
    lengths: &[u32],
    cut: usize,
    anchor_head: usize,
) -> Result<CutWindow> {
    let anchor = token_at(corpus, lengths, anchor_head)?;
    if anchor.id == 0 {
        if anchor.head != cut {
            return Err(TrainError::InternalInvariant(
                "snapshot sentinel anchor is not at cut",
            ));
        }
    } else if !(anchor.head <= cut && cut < anchor.head + anchor.len) {
        return Err(TrainError::InternalInvariant(
            "snapshot anchor does not cover cut",
        ));
    }
    let mut tokens = [None; 6];
    tokens[2] = Some(anchor);
    let mut cursor = anchor;
    for slot in (0..2).rev() {
        let Some(prior) = predecessor(corpus, lengths, cursor)? else {
            break;
        };
        tokens[slot] = Some(prior);
        if prior.id == 0 {
            break;
        }
        cursor = prior;
    }
    cursor = anchor;
    for place in tokens.iter_mut().skip(3) {
        let Some(next) = successor(corpus, lengths, cursor)? else {
            break;
        };
        *place = Some(next);
        if next.id == 0 {
            break;
        }
        cursor = next;
    }
    Ok(CutWindow { cut, tokens })
}

pub(super) struct BoundaryState {
    windows: Vec<CutWindow>,
}

impl BoundaryState {
    pub(super) fn new(corpus: &[AtomicU32], lengths: &[u32], cuts: &RegionCuts) -> Result<Self> {
        let mut windows = Vec::with_capacity(cuts.count().saturating_sub(1));
        for &cut in &cuts.cuts[1..cuts.count()] {
            // The prepared corpus starts with unit-length tokens, so every
            // physical cut is an initial token head or a sentinel.
            windows.push(build_window(corpus, lengths, cut, cut)?);
        }
        Ok(Self { windows })
    }

    pub(super) fn refresh(&mut self, corpus: &[AtomicU32], lengths: &[u32]) -> Result<()> {
        for window in &mut self.windows {
            let old = window.anchor();
            let anchor_head = if old.id == 0 {
                old.head
            } else {
                let raw = corpus[old.head].load(Ordering::Relaxed);
                if raw & HEAD != 0 {
                    old.head
                } else {
                    window
                        .preceding()
                        .ok_or(TrainError::InternalInvariant(
                            "swallowed snapshot anchor lacks old predecessor",
                        ))?
                        .head
                }
            };
            *window = build_window(corpus, lengths, window.cut, anchor_head)?;
        }
        Ok(())
    }

    pub(super) fn windows_len(&self) -> usize {
        self.windows.len()
    }

    pub(super) fn capacity_bytes(&self) -> usize {
        self.windows.capacity() * std::mem::size_of::<CutWindow>()
    }

    pub(super) fn accessor<'a>(
        &'a self,
        region: usize,
        cuts: &RegionCuts,
        local: &'a mut [AtomicU32],
    ) -> RegionAccess<'a> {
        let (lower, upper) = cuts.bounds(region);
        debug_assert_eq!(local.len(), upper - lower);
        RegionAccess {
            local,
            lower,
            upper,
            lower_window: region.checked_sub(1).map(|i| &self.windows[i]),
            upper_window: self.windows.get(region),
            deferred: Vec::new(),
            local_reads: 0,
            local_writes: 0,
            boundary_queries: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DeferredStore {
    pub pos: usize,
    pub value: u32,
}

pub(super) struct RegionAccess<'a> {
    local: &'a mut [AtomicU32],
    lower: usize,
    upper: usize,
    lower_window: Option<&'a CutWindow>,
    upper_window: Option<&'a CutWindow>,
    pub deferred: Vec<DeferredStore>,
    pub local_reads: usize,
    pub local_writes: usize,
    pub boundary_queries: usize,
}

impl RegionAccess<'_> {
    fn read_raw(&mut self, pos: usize, head_query: bool) -> Result<u32> {
        if self.lower <= pos && pos < self.upper {
            self.local_reads += 1;
            return Ok(*self.local[pos - self.lower].get_mut());
        }
        self.boundary_queries += 1;
        let window = if pos < self.lower {
            self.lower_window
        } else {
            self.upper_window
        }
        .ok_or(TrainError::InternalInvariant(
            "remote snapshot query has no exit cut",
        ))?;
        window
            .old_raw(pos, head_query)
            .ok_or(TrainError::InternalInvariant(
                "remote snapshot query falls outside allowed token endpoints",
            ))
    }

    pub(super) fn head_raw(&mut self, pos: usize) -> Result<u32> {
        self.read_raw(pos, true)
    }

    pub(super) fn tail_raw(&mut self, pos: usize) -> Result<u32> {
        self.read_raw(pos, false)
    }

    pub(super) fn put(&mut self, pos: usize, value: u32) {
        if self.lower <= pos && pos < self.upper {
            self.local_writes += 1;
            *self.local[pos - self.lower].get_mut() = value;
        } else {
            self.deferred.push(DeferredStore { pos, value });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(tokens: &[(u32, usize)]) -> (Vec<AtomicU32>, Vec<u32>) {
        let max_id = tokens.iter().map(|(id, _)| *id as usize).max().unwrap_or(0);
        let mut lengths = vec![1; max_id + 1];
        let mut raw = vec![0];
        for &(id, len) in tokens {
            if id == 0 {
                raw.push(0);
                continue;
            }
            lengths[id as usize] = len as u32;
            raw.push(id | HEAD);
            if len > 1 {
                raw.extend(std::iter::repeat_n(0, len - 2));
                raw.push(id);
            }
        }
        raw.push(0);
        (raw.into_iter().map(AtomicU32::new).collect(), lengths)
    }

    fn with_anchor(
        raw: &[AtomicU32],
        lengths: &[u32],
        cuts: &RegionCuts,
        heads: &[usize],
    ) -> BoundaryState {
        BoundaryState {
            windows: cuts.cuts[1..cuts.count()]
                .iter()
                .zip(heads)
                .map(|(&cut, &head)| build_window(raw, lengths, cut, head).unwrap())
                .collect(),
        }
    }

    #[test]
    fn distant_heads_tails_and_sentinel_are_in_constant_windows() {
        // Long A and B cross multiple cuts. K,L,A,B,C,D are all needed.
        let (mut raw, lengths) = corpus(&[(7, 1), (6, 5), (1, 8), (2, 6), (3, 4), (4, 3), (5, 1)]);
        let cuts = RegionCuts::new(raw.len(), 7).unwrap();
        let heads = [2, 7, 7, 15, 21, 25];
        let state = with_anchor(&raw, &lengths, &cuts, &heads);
        let region = cuts.of(7);
        let (lower, upper) = cuts.bounds(region);
        let mut access = state.accessor(region, &cuts, &mut raw[lower..upper]);
        assert_eq!(access.head_raw(15).unwrap(), 2 | HEAD);
        assert_eq!(access.head_raw(21).unwrap(), 3 | HEAD);
        assert_eq!(access.tail_raw(1).unwrap(), 7 | HEAD);
        assert!(access.boundary_queries >= 3);
    }

    #[test]
    fn delayed_remote_store_keeps_old_snapshot_until_join_then_moves_anchor() {
        let (mut raw, mut lengths) = corpus(&[(1, 2), (2, 4), (3, 1)]);
        lengths.push(6);
        let cuts = RegionCuts::new(raw.len(), 3).unwrap();
        let state = with_anchor(&raw, &lengths, &cuts, &[3, 3]);
        let region = cuts.of(1);
        let (lower, upper) = cuts.bounds(region);
        let deferred = {
            let mut access = state.accessor(region, &cuts, &mut raw[lower..upper]);
            access.put(1, 4 | HEAD);
            access.put(3, 0);
            access.put(6, 4);
            assert_eq!(access.head_raw(3).unwrap(), 2 | HEAD);
            std::mem::take(&mut access.deferred)
        };
        assert_eq!(raw[3].load(Ordering::Relaxed), 2 | HEAD);
        assert_eq!(deferred.len(), 2);
        for store in deferred {
            raw[store.pos].store(store.value, Ordering::Relaxed);
        }
        let mut state = state;
        state.refresh(&raw, &lengths).unwrap();
        assert_eq!(state.windows[0].anchor().head, 1);
    }

    #[test]
    fn six_token_endpoints_cover_all_short_length_and_cut_placements() {
        for shape in 0_u8..64 {
            let lengths_for = (0..6)
                .map(|i| 1 + usize::from((shape >> i) & 1))
                .collect::<Vec<_>>();
            let specification = (1..=6).zip(lengths_for.iter().copied()).collect::<Vec<_>>();
            let (mut raw, lengths) = corpus(&specification);
            let mut heads = [0_usize; 6];
            let mut next = 1;
            for (i, &len) in lengths_for.iter().enumerate() {
                heads[i] = next;
                next += len;
            }
            let [k, l, a, b, c, d] = heads;
            for cut in 1..raw.len() - 1 {
                let anchor = heads
                    .iter()
                    .copied()
                    .filter(|&head| head <= cut)
                    .max()
                    .unwrap_or(0);
                let cuts = RegionCuts {
                    cuts: vec![0, cut, raw.len()],
                };
                let state = with_anchor(&raw, &lengths, &cuts, &[anchor]);
                if a < cut {
                    let mut access = state.accessor(0, &cuts, &mut raw[..cut]);
                    for (pos, id) in [(b, 4_u32), (c, 5), (d, 6)] {
                        if pos >= cut {
                            assert_eq!(
                                access.head_raw(pos).unwrap(),
                                id | HEAD,
                                "right shape={shape}, cut={cut}, pos={pos}"
                            );
                        }
                    }
                } else {
                    let mut access = state.accessor(1, &cuts, &mut raw[cut..]);
                    for (pos, id, len) in [
                        (a - 1, 2_u32, lengths_for[1]),
                        (l - 1, 1_u32, lengths_for[0]),
                    ] {
                        if pos < cut {
                            let expected = if len == 1 { id | HEAD } else { id };
                            assert_eq!(
                                access.tail_raw(pos).unwrap(),
                                expected,
                                "left shape={shape}, cut={cut}, pos={pos}, K={k}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sentinel_and_refreshed_snapshot_expose_only_live_batch_start_ids() {
        let (mut raw, mut lengths) = corpus(&[(1, 1), (2, 1), (0, 1), (3, 1)]);
        let cut = 2;
        let cuts = RegionCuts {
            cuts: vec![0, cut, raw.len()],
        };
        let mut state = with_anchor(&raw, &lengths, &cuts, &[2]);
        {
            let mut access = state.accessor(0, &cuts, &mut raw[..cut]);
            assert_eq!(access.head_raw(3).unwrap(), 0);
            assert!(access.head_raw(4).is_err());
        }
        // A prior batch merges (1,2). The old B head becomes a bare tail;
        // after refresh a stale (3,2) query cannot see the retired old B.
        lengths.push(2);
        raw[1].store(4 | HEAD, Ordering::Relaxed);
        raw[2].store(4, Ordering::Relaxed);
        state.refresh(&raw, &lengths).unwrap();
        assert_eq!(state.windows[0].anchor().head, 1);
        assert_eq!(state.windows[0].anchor().id, 4);
    }

    #[test]
    fn long_257_and_513_tokens_cross_multiple_cuts_with_two_deferred_stores() {
        let (mut raw, mut lengths) = corpus(&[(1, 1), (2, 257), (3, 513), (4, 1)]);
        let cuts = RegionCuts::new(raw.len(), 4).unwrap();
        let token_heads = [1_usize, 2, 259, 772];
        let anchors = cuts.cuts[1..cuts.count()]
            .iter()
            .map(|&cut| {
                token_heads
                    .into_iter()
                    .filter(|&head| head <= cut)
                    .max()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut state = with_anchor(&raw, &lengths, &cuts, &anchors);
        let region = cuts.of(2);
        let (lower, upper) = cuts.bounds(region);
        let deferred = {
            let mut access = state.accessor(region, &cuts, &mut raw[lower..upper]);
            assert_eq!(access.head_raw(259).unwrap(), 3 | HEAD);
            assert_eq!(access.head_raw(772).unwrap(), 4 | HEAD);
            access.put(2, 5 | HEAD);
            access.put(259, 0);
            access.put(771, 5);
            assert_eq!(access.deferred.len(), 2);
            std::mem::take(&mut access.deferred)
        };
        lengths.push(770);
        for store in deferred {
            raw[store.pos].store(store.value, Ordering::Relaxed);
        }
        state.refresh(&raw, &lengths).unwrap();
        assert_eq!(state.windows[0].anchor().head, 2);
        assert_eq!(state.windows[1].anchor().head, 2);
        assert_eq!(state.windows[2].anchor().head, 2);
    }
}

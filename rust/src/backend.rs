//! Compact endpoint corpus used by both checked and trusted-access trainers.

#[derive(Debug, Clone, Copy)]
pub(crate) struct PairContext {
    pub before: Option<usize>,
    pub left_id: u32,
    pub right: usize,
    pub after: usize,
    pub right_id: u32,
}

pub(crate) struct Endpoints<const UNCHECKED: bool> {
    corpus: Vec<u32>,
    last: usize,
}

impl<const UNCHECKED: bool> Endpoints<UNCHECKED> {
    pub fn new(corpus: Vec<u32>) -> Self {
        // The public trainer validates the complete corpus before construction.
        let last = corpus.len() - 1;
        Self { corpus, last }
    }

    pub fn len(&self) -> usize {
        self.corpus.len()
    }

    /// Only called during the initial pass, after public input validation.
    pub fn initial_token(&self, pos: usize) -> u32 {
        self.read(pos)
    }

    #[inline(always)]
    fn read(&self, pos: usize) -> u32 {
        if UNCHECKED {
            // SAFETY: This private helper is called only at a validated input
            // position or a token boundary proven <= last by the surrounding
            // method. The trainer never exposes the backend to callers.
            unsafe { *self.corpus.get_unchecked(pos) }
        } else {
            self.corpus[pos]
        }
    }

    #[inline(always)]
    fn write(&mut self, pos: usize, value: u32) {
        if UNCHECKED {
            // SAFETY: merge_known receives positions produced by inspect_pair.
            // Both tokens are adjacent and after <= last; therefore pos,
            // right, and after-1 are within the validated corpus allocation.
            unsafe { *self.corpus.get_unchecked_mut(pos) = value }
        } else {
            self.corpus[pos] = value;
        }
    }

    /// A historical occurrence is accepted only while it remains a live pair.
    pub fn inspect_pair(&self, pos: usize, a: u32, b: u32, lengths: &[u32]) -> Option<PairContext> {
        if a == 0 || b == 0 || pos == 0 || pos >= self.last || self.read(pos) != a {
            return None;
        }
        let right = pos.checked_add(lengths[a as usize] as usize)?;
        if right >= self.last || self.read(right) != b {
            return None;
        }
        let after = right.checked_add(lengths[b as usize] as usize)?;
        if after > self.last {
            return None;
        }
        // A live token start has its predecessor token's ID at pos-1. The
        // endpoint representation keeps that ID even after interior clearing.
        let prior_length = lengths[self.read(pos - 1) as usize] as usize;
        let before = Some(pos.checked_sub(prior_length)?);
        Some(PairContext {
            before,
            left_id: before.map(|p| self.read(p)).unwrap_or(0),
            right,
            after,
            right_id: self.read(after),
        })
    }

    /// Lean endpoint update: two writes for a singleton right token, three
    /// for a longer one. Caller has just accepted inspect_pair's context.
    pub fn merge_known(&mut self, pos: usize, context: PairContext, new_id: u32) {
        self.write(pos, new_id);
        if context.after - context.right == 1 {
            self.write(context.right, new_id);
        } else {
            self.write(context.right, 0);
            self.write(context.after - 1, new_id);
        }
    }

    pub fn final_tokens(&self, lengths: &[u32]) -> Vec<u32> {
        let mut result = Vec::new();
        let mut pos = 0;
        loop {
            let token = self.read(pos);
            result.push(token);
            if pos == self.last {
                break;
            }
            let next = pos + lengths[token as usize] as usize;
            // The endpoint invariant places every next boundary within last.
            // Keep this assertion in both modes so a broken internal invariant
            // cannot turn the next unchecked read into out-of-bounds access.
            assert!(next > pos && next <= self.last);
            pos = next;
        }
        result
    }
}

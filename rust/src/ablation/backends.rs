//! Fixed-position corpus layouts under the same historical-occurrence contract.
//! All constructors receive the fully validated initial corpus from the trainer;
//! they also check the bounds needed for their own representation.
use crate::TrainError;

#[derive(Debug, Clone, Copy)]
pub struct Context {
    pub before: Option<usize>,
    pub left_id: u32,
    pub right: usize,
    pub after: usize,
    pub right_id: u32,
}

pub trait Corpus: Send + Sync {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError>
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn initial_token(&self, p: usize) -> u32;
    fn inspect_pair(&self, p: usize, a: u32, b: u32, lengths: &[u32]) -> Option<Context>;
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, new_length: u32);
    fn merge_with_lengths(
        &mut self,
        p: usize,
        ctx: Context,
        new_id: u32,
        new_length: u32,
        _lengths: &[u32],
    ) {
        self.merge_known(p, ctx, new_id, new_length);
    }
    fn final_tokens(&self, lengths: &[u32]) -> Vec<u32>;
    fn logical_bytes(&self) -> usize;
    fn capacity_bytes(&self) -> usize;
}

fn check_initial(corpus: &[u32], alphabet: usize) -> Result<(), TrainError> {
    if corpus.is_empty()
        || (corpus.len() as u128) >= (1_u128 << 32)
        || corpus[0] != 0
        || *corpus.last().unwrap() != 0
    {
        return Err(TrainError::InvalidInput(
            "corpus needs nonempty u32 positions and zero sentinels",
        ));
    }
    if corpus.iter().any(|&id| id as usize > alphabet) {
        return Err(TrainError::InvalidInput(
            "corpus ID exceeds initial alphabet",
        ));
    }
    Ok(())
}

#[inline(always)]
fn read<const UNCHECKED: bool, T: Copy>(items: &[T], pos: usize) -> T {
    if UNCHECKED {
        // SAFETY: this module is crate-private. Its trainer supplies positions
        // from validated initial passes or live boundaries; inspect_pair checks
        // historical candidates before deriving another index. Individual
        // call sites below document any tighter neighbor/endpoint bounds.
        unsafe { *items.get_unchecked(pos) }
    } else {
        items[pos]
    }
}

#[inline(always)]
fn write<const UNCHECKED: bool, T>(items: &mut [T], pos: usize, value: T) {
    if UNCHECKED {
        // SAFETY: merge_known is invoked only with a Context just returned by
        // inspect_pair; the retained positions lie in the original allocation.
        unsafe {
            *items.get_unchecked_mut(pos) = value;
        }
    } else {
        items[pos] = value;
    }
}

#[inline(always)]
fn token_length<const UNCHECKED: bool>(lengths: &[u32], id: u32) -> Option<usize> {
    if UNCHECKED {
        // SAFETY: in the crate-private trainer, every pair key and live corpus
        // ID was allocated before the current round, and its length has been
        // appended before inspect_pair. Historical keys never invent IDs.
        Some(read::<true, _>(lengths, id as usize) as usize)
    } else {
        lengths.get(id as usize).map(|&length| length as usize)
    }
}

#[inline]
fn next_tokens<F: Fn(usize) -> (u32, usize)>(last: usize, at: F) -> Vec<u32> {
    let mut out = Vec::new();
    let mut pos = 0;
    loop {
        let (id, next) = at(pos);
        out.push(id);
        if pos == last {
            break;
        }
        assert!(next > pos && next <= last, "backend boundary invariant");
        pos = next;
    }
    out
}

/// MODE=0 clears the whole interior, MODE=1 writes four endpoints, MODE=2
/// uses two or three writes. UNCHECKED specializes the same layout but keeps
/// public-facing indexing checked; the fused hot path has proven bounds.
pub struct Endpoint<const MODE: u8, const UNCHECKED: bool> {
    ids: Vec<u32>,
    last: usize,
}
impl<const MODE: u8, const UNCHECKED: bool> Endpoint<MODE, UNCHECKED> {
    #[inline(always)]
    fn known(&self, pos: usize) -> u32 {
        read::<UNCHECKED, _>(&self.ids, pos)
    }
    #[inline(always)]
    fn put(&mut self, pos: usize, id: u32) {
        write::<UNCHECKED, _>(&mut self.ids, pos, id);
    }
}
impl<const MODE: u8, const UNCHECKED: bool> Corpus for Endpoint<MODE, UNCHECKED> {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError> {
        check_initial(&corpus, alphabet)?;
        if MODE > 2 {
            return Err(TrainError::InvalidInput("unknown endpoint mode"));
        }
        let last = corpus.len() - 1;
        Ok(Self { ids: corpus, last })
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
    fn initial_token(&self, p: usize) -> u32 {
        self.known(p)
    }
    #[inline]
    fn inspect_pair(&self, p: usize, a: u32, b: u32, lengths: &[u32]) -> Option<Context> {
        if a == 0 || b == 0 || p == 0 || p >= self.last || self.known(p) != a {
            return None;
        }
        let right = p.checked_add(token_length::<UNCHECKED>(lengths, a)?)?;
        if right >= self.last || self.known(right) != b {
            return None;
        }
        let after = right.checked_add(token_length::<UNCHECKED>(lengths, b)?)?;
        if after > self.last {
            return None;
        }
        let prior = self.known(p - 1);
        let before = p.checked_sub(token_length::<UNCHECKED>(lengths, prior)?)?;
        Some(Context {
            before: Some(before),
            left_id: self.known(before),
            right,
            after,
            right_id: self.known(after),
        })
    }
    #[inline]
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, _new_length: u32) {
        if MODE == 0 {
            self.ids[p + 1..ctx.after].fill(0);
            self.put(p, new_id);
            self.put(ctx.after - 1, new_id);
        } else if MODE == 1 {
            self.put(ctx.right - 1, 0);
            self.put(ctx.right, 0);
            self.put(p, new_id);
            self.put(ctx.after - 1, new_id);
        } else {
            self.put(p, new_id);
            if ctx.after - ctx.right == 1 {
                self.put(ctx.right, new_id);
            } else {
                self.put(ctx.right, 0);
                self.put(ctx.after - 1, new_id);
            }
        }
    }
    fn final_tokens(&self, lengths: &[u32]) -> Vec<u32> {
        next_tokens(self.last, |p| {
            let id = self.known(p);
            (id, p + read::<UNCHECKED, _>(lengths, id as usize) as usize)
        })
    }
    fn logical_bytes(&self) -> usize {
        self.ids.len() * 4
    }
    fn capacity_bytes(&self) -> usize {
        self.ids.capacity() * 4
    }
}

/// A node per original position. RUN adds a fourth 4-byte column, but does
/// not perform word-level RLE; it is only the 16-byte layout ablation.
pub struct Linked<const RUN: bool, const UNCHECKED: bool = false> {
    val: Vec<u32>,
    prev: Vec<i32>,
    next: Vec<i32>,
    run: Vec<u32>,
    last: usize,
}
impl<const RUN: bool, const UNCHECKED: bool> Corpus for Linked<RUN, UNCHECKED> {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError> {
        check_initial(&corpus, alphabet)?;
        if corpus.len() > i32::MAX as usize {
            return Err(TrainError::InvalidInput(
                "i32 links require <=2^31-1 positions",
            ));
        }
        let n = corpus.len();
        let prev = (0..n).map(|p| p as i32 - 1).collect();
        let next = (0..n)
            .map(|p| if p + 1 == n { -1 } else { p as i32 + 1 })
            .collect();
        let run = if RUN {
            corpus.iter().map(|&id| u32::from(id != 0)).collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            val: corpus,
            prev,
            next,
            run,
            last: n - 1,
        })
    }
    fn len(&self) -> usize {
        self.val.len()
    }
    fn initial_token(&self, p: usize) -> u32 {
        read::<UNCHECKED, _>(&self.val, p)
    }
    #[inline]
    fn inspect_pair(&self, p: usize, a: u32, b: u32, _lengths: &[u32]) -> Option<Context> {
        if a == 0 || b == 0 || p >= self.last {
            return None;
        }
        let right = read::<UNCHECKED, _>(&self.next, p);
        if right < 0 || read::<UNCHECKED, _>(&self.val, p) != a {
            return None;
        }
        let right = right as usize;
        if right >= self.last || read::<UNCHECKED, _>(&self.val, right) != b {
            return None;
        }
        let prior = read::<UNCHECKED, _>(&self.prev, p);
        let before = (prior >= 0).then_some(prior as usize);
        let after = read::<UNCHECKED, _>(&self.next, right);
        if after < 0 {
            return None;
        }
        let after = after as usize;
        Some(Context {
            before,
            left_id: before
                .map(|p| read::<UNCHECKED, _>(&self.val, p))
                .unwrap_or(0),
            right,
            after,
            right_id: read::<UNCHECKED, _>(&self.val, after),
        })
    }
    #[inline]
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, _new_length: u32) {
        write::<UNCHECKED, _>(&mut self.val, p, new_id);
        write::<UNCHECKED, _>(&mut self.next, p, ctx.after as i32);
        write::<UNCHECKED, _>(&mut self.prev, ctx.after, p as i32);
        write::<UNCHECKED, _>(&mut self.val, ctx.right, 0);
        write::<UNCHECKED, _>(&mut self.next, ctx.right, -2);
        if RUN {
            write::<UNCHECKED, _>(&mut self.run, ctx.right, 0);
        }
    }
    fn final_tokens(&self, _lengths: &[u32]) -> Vec<u32> {
        next_tokens(self.last, |p| {
            (
                read::<UNCHECKED, _>(&self.val, p),
                read::<UNCHECKED, _>(&self.next, p).max(0) as usize,
            )
        })
    }
    fn logical_bytes(&self) -> usize {
        self.val.len() * if RUN { 16 } else { 12 }
    }
    fn capacity_bytes(&self) -> usize {
        (self.val.capacity() + self.prev.capacity() + self.next.capacity() + self.run.capacity())
            * 4
    }
}

struct BitLinks<const UNCHECKED: bool> {
    bits: Vec<u64>,
    skips: Vec<u32>,
    last: usize,
}
impl<const UNCHECKED: bool> BitLinks<UNCHECKED> {
    fn new(n: usize) -> Self {
        let words = n.div_ceil(64);
        let mut bits = vec![u64::MAX; words];
        bits[words - 1] = u64::MAX >> (63 - ((n - 1) % 64));
        Self {
            bits,
            skips: vec![0; words],
            last: n - 1,
        }
    }
    #[inline(always)]
    fn alive(&self, p: usize) -> bool {
        p <= self.last && read::<UNCHECKED, _>(&self.bits, p / 64) & (1_u64 << (p % 64)) != 0
    }
    #[inline(always)]
    fn next(&self, p: usize) -> Option<usize> {
        if p == self.last {
            return None;
        }
        let block = p / 64;
        let offset = p % 64;
        let here = if offset == 63 {
            0
        } else {
            read::<UNCHECKED, _>(&self.bits, block) >> (offset + 1)
        };
        if here != 0 {
            return Some(p + here.trailing_zeros() as usize + 1);
        }
        // A live p < last always has a later live sentinel; if its block has
        // no later bit, block+1 exists. A completely empty adjacent block has
        // its gap stored in skips[block+1].
        let adjacent = read::<UNCHECKED, _>(&self.bits, block + 1);
        if adjacent != 0 {
            return Some((block + 1) * 64 + adjacent.trailing_zeros() as usize);
        }
        Some(p + read::<UNCHECKED, _>(&self.skips, block + 1) as usize + 1)
    }
    #[inline(always)]
    fn prev(&self, p: usize) -> Option<usize> {
        if p == 0 {
            return None;
        }
        let block = p / 64;
        let offset = p % 64;
        let here = read::<UNCHECKED, _>(&self.bits, block) & ((1_u64 << offset) - 1);
        if here != 0 {
            return Some(block * 64 + 63 - here.leading_zeros() as usize);
        }
        // The leading zero sentinel remains live, so a known-live p > 0
        // either has a preceding bit here or has a preceding block.
        let adjacent = read::<UNCHECKED, _>(&self.bits, block - 1);
        if adjacent != 0 {
            return Some((block - 1) * 64 + 63 - adjacent.leading_zeros() as usize);
        }
        Some(p - read::<UNCHECKED, _>(&self.skips, block - 1) as usize - 1)
    }
    #[inline(always)]
    fn erase_and_update(&mut self, p: usize, right: usize, after: usize) {
        let right_block = right / 64;
        let old = read::<UNCHECKED, _>(&self.bits, right_block);
        write::<UNCHECKED, _>(&mut self.bits, right_block, old & !(1_u64 << (right % 64)));
        let first = p / 64;
        let last = after / 64;
        if last > first + 1 {
            let gap = (after - p - 1) as u32;
            write::<UNCHECKED, _>(&mut self.skips, first + 1, gap);
            write::<UNCHECKED, _>(&mut self.skips, last - 1, gap);
        }
    }
    fn logical_bytes(&self) -> usize {
        self.bits.len() * 8 + self.skips.len() * 4
    }
    fn capacity_bytes(&self) -> usize {
        self.bits.capacity() * 8 + self.skips.capacity() * 4
    }
}

pub struct BitmapU32<const UNCHECKED: bool = false> {
    ids: Vec<u32>,
    links: BitLinks<UNCHECKED>,
}
impl<const UNCHECKED: bool> Corpus for BitmapU32<UNCHECKED> {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError> {
        check_initial(&corpus, alphabet)?;
        let links = BitLinks::new(corpus.len());
        Ok(Self { ids: corpus, links })
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
    fn initial_token(&self, p: usize) -> u32 {
        read::<UNCHECKED, _>(&self.ids, p)
    }
    #[inline]
    fn inspect_pair(&self, p: usize, a: u32, b: u32, _lengths: &[u32]) -> Option<Context> {
        if a == 0
            || b == 0
            || p >= self.links.last
            || !self.links.alive(p)
            || read::<UNCHECKED, _>(&self.ids, p) != a
        {
            return None;
        }
        let right = self.links.next(p)?;
        if right >= self.links.last || read::<UNCHECKED, _>(&self.ids, right) != b {
            return None;
        }
        let before = self.links.prev(p);
        let after = self.links.next(right)?;
        Some(Context {
            before,
            left_id: before
                .map(|p| read::<UNCHECKED, _>(&self.ids, p))
                .unwrap_or(0),
            right,
            after,
            right_id: read::<UNCHECKED, _>(&self.ids, after),
        })
    }
    #[inline]
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, _new_length: u32) {
        self.links.erase_and_update(p, ctx.right, ctx.after);
        write::<UNCHECKED, _>(&mut self.ids, p, new_id);
    }
    fn final_tokens(&self, _lengths: &[u32]) -> Vec<u32> {
        next_tokens(self.links.last, |p| {
            (
                read::<UNCHECKED, _>(&self.ids, p),
                self.links.next(p).unwrap_or(0),
            )
        })
    }
    fn logical_bytes(&self) -> usize {
        self.ids.len() * 4 + self.links.logical_bytes()
    }
    fn capacity_bytes(&self) -> usize {
        self.ids.capacity() * 4 + self.links.capacity_bytes()
    }
}

/// Prezza-style halfword ID text: a live singleton stores one u16; a merged
/// token stores high and low halves in its first two physical positions.
pub struct Halfword<const UNCHECKED: bool = false> {
    text: Vec<u16>,
    links: BitLinks<UNCHECKED>,
}
impl<const UNCHECKED: bool> Halfword<UNCHECKED> {
    #[inline(always)]
    fn token(&self, p: usize) -> u32 {
        let hi = u32::from(read::<UNCHECKED, _>(&self.text, p));
        if p == self.links.last || self.links.alive(p + 1) {
            hi
        } else {
            (hi << 16) | u32::from(read::<UNCHECKED, _>(&self.text, p + 1))
        }
    }
}
impl<const UNCHECKED: bool> Corpus for Halfword<UNCHECKED> {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError> {
        check_initial(&corpus, alphabet)?;
        if alphabet >= 1 << 16 {
            return Err(TrainError::InvalidInput(
                "halfword initial alphabet must fit u16",
            ));
        }
        let links = BitLinks::new(corpus.len());
        Ok(Self {
            text: corpus.into_iter().map(|x| x as u16).collect(),
            links,
        })
    }
    fn len(&self) -> usize {
        self.text.len()
    }
    fn initial_token(&self, p: usize) -> u32 {
        u32::from(read::<UNCHECKED, _>(&self.text, p))
    }
    #[inline]
    fn inspect_pair(&self, p: usize, a: u32, b: u32, _lengths: &[u32]) -> Option<Context> {
        if a == 0 || b == 0 || p >= self.links.last || !self.links.alive(p) || self.token(p) != a {
            return None;
        }
        let right = self.links.next(p)?;
        if right >= self.links.last || self.token(right) != b {
            return None;
        }
        let before = self.links.prev(p);
        let after = self.links.next(right)?;
        Some(Context {
            before,
            left_id: before.map(|p| self.token(p)).unwrap_or(0),
            right,
            after,
            right_id: self.token(after),
        })
    }
    #[inline]
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, _new_length: u32) {
        self.links.erase_and_update(p, ctx.right, ctx.after);
        write::<UNCHECKED, _>(&mut self.text, p, (new_id >> 16) as u16);
        write::<UNCHECKED, _>(&mut self.text, p + 1, new_id as u16);
    }
    fn final_tokens(&self, _lengths: &[u32]) -> Vec<u32> {
        next_tokens(self.links.last, |p| {
            (self.token(p), self.links.next(p).unwrap_or(0))
        })
    }
    fn logical_bytes(&self) -> usize {
        self.text.len() * 2 + self.links.logical_bytes()
    }
    fn capacity_bytes(&self) -> usize {
        self.text.capacity() * 2 + self.links.capacity_bytes()
    }
}

/// Directed start/end tags with u16 text. NIBBLE=false: H3 (3N bytes),
/// NIBBLE=true: H2.5 (2N+ceil(N/2) bytes). Dead starts keep tag zero;
/// stale end tags may remain in the interior but are never accepted as starts.
pub struct Hybrid<const NIBBLE: bool, const UNCHECKED: bool = false> {
    text: Vec<u16>,
    tags: Vec<u8>,
    last: usize,
}
impl<const NIBBLE: bool, const UNCHECKED: bool> Hybrid<NIBBLE, UNCHECKED> {
    const SHORT_MAX: u8 = if NIBBLE { 6 } else { 127 };
    const LONG_START: u8 = if NIBBLE { 7 } else { 254 };
    const LONG_END: u8 = if NIBBLE { 13 } else { 255 };
    const END_BIAS: u8 = if NIBBLE { 6 } else { 126 };
    #[inline(always)]
    fn tag(&self, pos: usize) -> u8 {
        if NIBBLE {
            (read::<UNCHECKED, _>(&self.tags, pos >> 1) >> ((pos & 1) << 2)) & 15
        } else {
            read::<UNCHECKED, _>(&self.tags, pos)
        }
    }
    #[inline(always)]
    fn set_tag(&mut self, pos: usize, value: u8) {
        if NIBBLE {
            let shift = (pos & 1) << 2;
            let index = pos >> 1;
            let old = read::<UNCHECKED, _>(&self.tags, index);
            write::<UNCHECKED, _>(
                &mut self.tags,
                index,
                (old & !(15 << shift)) | (value << shift),
            );
        } else {
            write::<UNCHECKED, _>(&mut self.tags, pos, value);
        }
    }
    #[inline(always)]
    fn start_tag(tag: u8) -> bool {
        tag == 1 || (2..=Self::SHORT_MAX).contains(&tag) || tag == Self::LONG_START
    }
    #[inline(always)]
    fn token(&self, pos: usize) -> u32 {
        if self.tag(pos) == 1 {
            u32::from(read::<UNCHECKED, _>(&self.text, pos))
        } else {
            (u32::from(read::<UNCHECKED, _>(&self.text, pos)) << 16)
                | u32::from(read::<UNCHECKED, _>(&self.text, pos + 1))
        }
    }
    #[inline(always)]
    fn start_length(&self, pos: usize, lengths: &[u32]) -> Option<usize> {
        let tag = self.tag(pos);
        if tag == 1 {
            Some(1)
        } else if (2..=Self::SHORT_MAX).contains(&tag) {
            Some(tag as usize)
        } else if tag == Self::LONG_START {
            token_length::<UNCHECKED>(lengths, self.token(pos))
        } else {
            None
        }
    }
    #[inline(always)]
    fn predecessor(&self, pos: usize) -> Option<usize> {
        if pos == 0 {
            return None;
        }
        let end = pos - 1;
        let tag = self.tag(end);
        let length = if tag == 1 {
            1
        } else if tag == Self::LONG_END {
            if end == 0 {
                return None;
            }
            ((read::<UNCHECKED, _>(&self.text, end - 1) as usize) << 16)
                | read::<UNCHECKED, _>(&self.text, end) as usize
        } else if (Self::END_BIAS + 2..Self::END_BIAS + Self::SHORT_MAX + 1).contains(&tag) {
            (tag - Self::END_BIAS) as usize
        } else {
            return None;
        };
        pos.checked_sub(length)
    }
}
impl<const NIBBLE: bool, const UNCHECKED: bool> Corpus for Hybrid<NIBBLE, UNCHECKED> {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError> {
        check_initial(&corpus, alphabet)?;
        if alphabet >= 1 << 16 {
            return Err(TrainError::InvalidInput(
                "hybrid initial alphabet must fit u16",
            ));
        }
        let n = corpus.len();
        let tags = if NIBBLE {
            vec![0x11; n.div_ceil(2)]
        } else {
            vec![1; n]
        };
        Ok(Self {
            text: corpus.into_iter().map(|x| x as u16).collect(),
            tags,
            last: n - 1,
        })
    }
    fn len(&self) -> usize {
        self.text.len()
    }
    fn initial_token(&self, p: usize) -> u32 {
        u32::from(read::<UNCHECKED, _>(&self.text, p))
    }
    #[inline]
    fn inspect_pair(&self, p: usize, a: u32, b: u32, lengths: &[u32]) -> Option<Context> {
        if a == 0
            || b == 0
            || p == 0
            || p >= self.last
            || !Self::start_tag(self.tag(p))
            || self.token(p) != a
        {
            return None;
        }
        let right = p.checked_add(self.start_length(p, lengths)?)?;
        if right >= self.last || !Self::start_tag(self.tag(right)) || self.token(right) != b {
            return None;
        }
        let after = right.checked_add(self.start_length(right, lengths)?)?;
        if after > self.last || !Self::start_tag(self.tag(after)) {
            return None;
        }
        let before = self.predecessor(p)?;
        Some(Context {
            before: Some(before),
            left_id: self.token(before),
            right,
            after,
            right_id: self.token(after),
        })
    }
    #[inline]
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, new_length: u32) {
        self.set_tag(ctx.right, 0);
        let end = ctx.after - 1;
        if new_length <= u32::from(Self::SHORT_MAX) {
            self.set_tag(p, new_length as u8);
            self.set_tag(end, new_length as u8 + Self::END_BIAS);
        } else {
            self.set_tag(p, Self::LONG_START);
            self.set_tag(end, Self::LONG_END);
            write::<UNCHECKED, _>(&mut self.text, end - 1, (new_length >> 16) as u16);
            write::<UNCHECKED, _>(&mut self.text, end, new_length as u16);
        }
        write::<UNCHECKED, _>(&mut self.text, p, (new_id >> 16) as u16);
        write::<UNCHECKED, _>(&mut self.text, p + 1, new_id as u16);
    }
    fn final_tokens(&self, lengths: &[u32]) -> Vec<u32> {
        next_tokens(self.last, |p| {
            (self.token(p), p + self.start_length(p, lengths).unwrap())
        })
    }
    fn logical_bytes(&self) -> usize {
        self.text.len() * 2 + self.tags.len()
    }
    fn capacity_bytes(&self) -> usize {
        self.text.capacity() * 2 + self.tags.capacity()
    }
}

/// The exploratory Python HybridByteTags call sequence: validate both starts,
/// then query prev(pos), next(right), token(before), and token(after) separately.
/// Its writes are already context-based, so no merge-time length copy is needed.
pub struct UnfusedHybrid<const NIBBLE: bool = false, const UNCHECKED: bool = false> {
    inner: Hybrid<NIBBLE, UNCHECKED>,
}
impl<const NIBBLE: bool, const UNCHECKED: bool> Corpus for UnfusedHybrid<NIBBLE, UNCHECKED> {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError> {
        Ok(Self {
            inner: Hybrid::<NIBBLE, UNCHECKED>::new(corpus, alphabet)?,
        })
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn initial_token(&self, p: usize) -> u32 {
        self.inner.initial_token(p)
    }
    #[inline]
    fn inspect_pair(&self, p: usize, a: u32, b: u32, lengths: &[u32]) -> Option<Context> {
        let h = &self.inner;
        if a == 0 || b == 0 || p == 0 || p >= h.last {
            return None;
        }
        let tag = h.tag(p);
        let right = if tag == 1 {
            if u32::from(read::<UNCHECKED, _>(&h.text, p)) != a {
                return None;
            }
            p + 1
        } else if (2..=Hybrid::<NIBBLE, UNCHECKED>::SHORT_MAX).contains(&tag)
            || tag == Hybrid::<NIBBLE, UNCHECKED>::LONG_START
        {
            let id = (u32::from(read::<UNCHECKED, _>(&h.text, p)) << 16)
                | u32::from(read::<UNCHECKED, _>(&h.text, p + 1));
            if id != a {
                return None;
            }
            let length = if tag == Hybrid::<NIBBLE, UNCHECKED>::LONG_START {
                token_length::<UNCHECKED>(lengths, a)?
            } else {
                tag as usize
            };
            p.checked_add(length)?
        } else {
            return None;
        };
        if right >= h.last {
            return None;
        }
        let right_tag = h.tag(right);
        let right_value = if right_tag == 1 {
            u32::from(read::<UNCHECKED, _>(&h.text, right))
        } else if (2..=Hybrid::<NIBBLE, UNCHECKED>::SHORT_MAX).contains(&right_tag)
            || right_tag == Hybrid::<NIBBLE, UNCHECKED>::LONG_START
        {
            (u32::from(read::<UNCHECKED, _>(&h.text, right)) << 16)
                | u32::from(read::<UNCHECKED, _>(&h.text, right + 1))
        } else {
            return None;
        };
        if right_value != b {
            return None;
        }
        // Separate historical API calls. start_length re-reads the right tag
        // and (for long starts) its ID, as Python next(right) does.
        let before = h.predecessor(p)?;
        let after = right.checked_add(h.start_length(right, lengths)?)?;
        if after > h.last {
            return None;
        }
        Some(Context {
            before: Some(before),
            left_id: h.token(before),
            right,
            after,
            right_id: h.token(after),
        })
    }
    #[inline]
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, new_length: u32) {
        self.inner.merge_known(p, ctx, new_id, new_length);
    }
    fn final_tokens(&self, lengths: &[u32]) -> Vec<u32> {
        self.inner.final_tokens(lengths)
    }
    fn logical_bytes(&self) -> usize {
        self.inner.logical_bytes()
    }
    fn capacity_bytes(&self) -> usize {
        self.inner.capacity_bytes()
    }
}

/// Pre-fusion endpoint calls: pair_matches, next/prev/next, two token reads,
/// then merge recomputes next/next through the driver's shared length table.
/// This is an interface ablation, not a new corpus layout.
pub struct UnfusedEndpoint<const UNCHECKED: bool = false> {
    inner: Endpoint<1, UNCHECKED>,
}
impl<const UNCHECKED: bool> Corpus for UnfusedEndpoint<UNCHECKED> {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError> {
        Ok(Self {
            inner: Endpoint::<1, UNCHECKED>::new(corpus, alphabet)?,
        })
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn initial_token(&self, p: usize) -> u32 {
        self.inner.initial_token(p)
    }
    #[inline]
    fn inspect_pair(&self, p: usize, a: u32, b: u32, lengths: &[u32]) -> Option<Context> {
        let last = self.inner.last;
        if a == 0 || b == 0 || p >= last || self.inner.known(p) != a {
            return None;
        }
        let matched_right = p.checked_add(token_length::<UNCHECKED>(lengths, a)?)?;
        if matched_right >= last || self.inner.known(matched_right) != b {
            return None;
        }
        // These are separate old next(pos), prev(pos), next(right) queries.
        let right = p.checked_add(token_length::<UNCHECKED>(lengths, self.inner.known(p))?)?;
        let before = if p == 0 {
            None
        } else {
            let end_id = self.inner.known(p - 1);
            Some(p.checked_sub(token_length::<UNCHECKED>(lengths, end_id)?)?)
        };
        let after =
            right.checked_add(token_length::<UNCHECKED>(lengths, self.inner.known(right))?)?;
        if after > last {
            return None;
        }
        Some(Context {
            before,
            left_id: before.map(|p| self.inner.known(p)).unwrap_or(0),
            right,
            after,
            right_id: self.inner.known(after),
        })
    }
    #[inline]
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, new_length: u32) {
        self.inner.merge_known(p, ctx, new_id, new_length);
    }
    #[inline]
    fn merge_with_lengths(
        &mut self,
        p: usize,
        ctx: Context,
        new_id: u32,
        new_length: u32,
        lengths: &[u32],
    ) {
        let right = p + read::<UNCHECKED, _>(lengths, self.inner.known(p) as usize) as usize;
        let after =
            right + read::<UNCHECKED, _>(lengths, self.inner.known(right) as usize) as usize;
        debug_assert_eq!((right, after), (ctx.right, ctx.after));
        self.inner.merge_known(
            p,
            Context {
                right,
                after,
                ..ctx
            },
            new_id,
            new_length,
        );
    }
    fn final_tokens(&self, lengths: &[u32]) -> Vec<u32> {
        self.inner.final_tokens(lengths)
    }
    fn logical_bytes(&self) -> usize {
        self.inner.logical_bytes()
    }
    fn capacity_bytes(&self) -> usize {
        self.inner.capacity_bytes()
    }
}

/// Pre-fusion halfword calls: pair_matches already calls next; context
/// retrieval calls next/prev/next again, and merge calls next/next once more.
pub struct UnfusedHalfword<const UNCHECKED: bool = false> {
    inner: Halfword<UNCHECKED>,
}
impl<const UNCHECKED: bool> Corpus for UnfusedHalfword<UNCHECKED> {
    fn new(corpus: Vec<u32>, alphabet: usize) -> Result<Self, TrainError> {
        Ok(Self {
            inner: Halfword::<UNCHECKED>::new(corpus, alphabet)?,
        })
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn initial_token(&self, p: usize) -> u32 {
        self.inner.initial_token(p)
    }
    #[inline]
    fn inspect_pair(&self, p: usize, a: u32, b: u32, _lengths: &[u32]) -> Option<Context> {
        if a == 0
            || b == 0
            || p >= self.inner.links.last
            || !self.inner.links.alive(p)
            || self.inner.token(p) != a
        {
            return None;
        }
        let matched_right = self.inner.links.next(p)?;
        if matched_right >= self.inner.links.last || self.inner.token(matched_right) != b {
            return None;
        }
        let right = self.inner.links.next(p)?;
        let before = self.inner.links.prev(p);
        let after = self.inner.links.next(right)?;
        Some(Context {
            before,
            left_id: before.map(|p| self.inner.token(p)).unwrap_or(0),
            right,
            after,
            right_id: self.inner.token(after),
        })
    }
    #[inline]
    fn merge_known(&mut self, p: usize, ctx: Context, new_id: u32, new_length: u32) {
        let right = self.inner.links.next(p).expect("unfused left is live");
        let after = self.inner.links.next(right).expect("unfused right is live");
        debug_assert_eq!((right, after), (ctx.right, ctx.after));
        self.inner.merge_known(
            p,
            Context {
                right,
                after,
                ..ctx
            },
            new_id,
            new_length,
        );
    }
    fn final_tokens(&self, lengths: &[u32]) -> Vec<u32> {
        self.inner.final_tokens(lengths)
    }
    fn logical_bytes(&self) -> usize {
        self.inner.logical_bytes()
    }
    fn capacity_bytes(&self) -> usize {
        self.inner.capacity_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_trace<C: Corpus>(initial: Vec<u32>, alphabet: usize, rounds: usize) {
        let mut backend = C::new(initial.clone(), alphabet).unwrap();
        let mut lengths = vec![1_u32; alphabet + 1];
        let mut live: Vec<(usize, u32, usize)> = initial
            .iter()
            .enumerate()
            .map(|(p, &id)| (p, id, 1))
            .collect();
        let mut historical = Vec::<(usize, u32, u32)>::new();
        let mut seed = 0x8ab2_u64;
        for _ in 0..rounds {
            let candidates: Vec<usize> = (0..live.len() - 1)
                .filter(|&i| live[i].1 != 0 && live[i + 1].1 != 0)
                .collect();
            if candidates.is_empty() {
                break;
            }
            for &i in &candidates {
                historical.push((live[i].0, live[i].1, live[i + 1].1));
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let i = candidates[(seed as usize) % candidates.len()];
            let (p, a, la) = live[i];
            let (right, b, lb) = live[i + 1];
            let ctx = backend
                .inspect_pair(p, a, b, &lengths)
                .expect("valid live pair");
            assert_eq!(ctx.right, right);
            assert_eq!(ctx.after, right + lb);
            let newid = lengths.len() as u32;
            lengths.push((la + lb) as u32);
            backend.merge_with_lengths(p, ctx, newid, (la + lb) as u32, &lengths);
            live.splice(i..=i + 1, [(p, newid, la + lb)]);
            let expected: Vec<u32> = live.iter().map(|x| x.1).collect();
            assert_eq!(backend.final_tokens(&lengths), expected);
            for &(p, a, b) in &historical {
                let valid = live
                    .iter()
                    .position(|&(q, _, _)| q == p)
                    .and_then(|j| live.get(j + 1).map(|right| live[j].1 == a && right.1 == b))
                    .unwrap_or(false);
                assert_eq!(
                    backend.inspect_pair(p, a, b, &lengths).is_some(),
                    valid,
                    "historical position {p} pair {a},{b}"
                );
            }
        }
        assert!(backend.capacity_bytes() >= backend.logical_bytes());
    }

    #[test]
    fn all_layouts_match_naive_historical_trace() {
        for initial in [
            vec![0, 1, 2, 1, 2, 0, 1, 2, 1, 0],
            vec![0, 1, 1, 1, 1, 1, 0],
            vec![0, 1, 0, 2, 0],
        ] {
            run_trace::<Endpoint<0, false>>(initial.clone(), 2, 8);
            run_trace::<Endpoint<0, true>>(initial.clone(), 2, 8);
            run_trace::<Endpoint<1, false>>(initial.clone(), 2, 8);
            run_trace::<Endpoint<1, true>>(initial.clone(), 2, 8);
            run_trace::<Endpoint<2, false>>(initial.clone(), 2, 8);
            run_trace::<Endpoint<2, true>>(initial.clone(), 2, 8);
            run_trace::<Linked<false>>(initial.clone(), 2, 8);
            run_trace::<Linked<false, true>>(initial.clone(), 2, 8);
            run_trace::<Linked<true>>(initial.clone(), 2, 8);
            run_trace::<Linked<true, true>>(initial.clone(), 2, 8);
            run_trace::<BitmapU32>(initial.clone(), 2, 8);
            run_trace::<BitmapU32<true>>(initial.clone(), 2, 8);
            run_trace::<Halfword>(initial.clone(), 2, 8);
            run_trace::<Halfword<true>>(initial.clone(), 2, 8);
            run_trace::<Hybrid<false>>(initial.clone(), 2, 8);
            run_trace::<Hybrid<false, true>>(initial.clone(), 2, 8);
            run_trace::<Hybrid<true>>(initial.clone(), 2, 8);
            run_trace::<Hybrid<true, true>>(initial.clone(), 2, 8);
            run_trace::<UnfusedHybrid>(initial.clone(), 2, 8);
            run_trace::<UnfusedHybrid<false, true>>(initial.clone(), 2, 8);
            run_trace::<UnfusedHybrid<true>>(initial.clone(), 2, 8);
            run_trace::<UnfusedHybrid<true, true>>(initial.clone(), 2, 8);
            run_trace::<UnfusedEndpoint>(initial.clone(), 2, 8);
            run_trace::<UnfusedEndpoint<true>>(initial.clone(), 2, 8);
            run_trace::<UnfusedHalfword>(initial.clone(), 2, 8);
            run_trace::<UnfusedHalfword<true>>(initial, 2, 8);
        }
    }

    #[test]
    fn random_merges_cross_bitmap_words_and_tag_boundaries() {
        let mut initial = vec![0];
        initial.extend((0..136).map(|i| if i % 3 == 0 { 2 } else { 1 }));
        initial.push(0);
        run_trace::<BitmapU32>(initial.clone(), 2, 125);
        run_trace::<BitmapU32<true>>(initial.clone(), 2, 125);
        run_trace::<Halfword>(initial.clone(), 2, 125);
        run_trace::<Halfword<true>>(initial.clone(), 2, 125);
        run_trace::<Hybrid<false>>(initial.clone(), 2, 125);
        run_trace::<Hybrid<false, true>>(initial.clone(), 2, 125);
        run_trace::<Hybrid<true>>(initial.clone(), 2, 125);
        run_trace::<Hybrid<true, true>>(initial.clone(), 2, 125);
        run_trace::<UnfusedHybrid>(initial.clone(), 2, 125);
        run_trace::<UnfusedHybrid<false, true>>(initial.clone(), 2, 125);
        run_trace::<UnfusedHybrid<true>>(initial.clone(), 2, 125);
        run_trace::<UnfusedHybrid<true, true>>(initial.clone(), 2, 125);
        run_trace::<UnfusedEndpoint>(initial.clone(), 2, 125);
        run_trace::<UnfusedEndpoint<true>>(initial.clone(), 2, 125);
        run_trace::<UnfusedHalfword>(initial.clone(), 2, 125);
        run_trace::<UnfusedHalfword<true>>(initial, 2, 125);
    }

    fn grow_long<C: Corpus>() {
        let n = 65_536;
        let mut corpus = vec![1_u32; n + 2];
        corpus[0] = 0;
        corpus[n + 1] = 0;
        let mut backend = C::new(corpus, 1).unwrap();
        let mut lengths = vec![1; 65_536]; // First fresh ID=65536; low half is zero.
        for right in 2..=n {
            let a = (lengths.len() - 1) as u32;
            // First merge starts from initial ID 1, then from each fresh ID.
            let a = if right == 2 { 1 } else { a };
            let ctx = backend.inspect_pair(1, a, 1, &lengths).unwrap();
            assert_eq!(ctx.right, right);
            let new_id = lengths.len() as u32;
            let new_length = right as u32;
            lengths.push(new_length);
            backend.merge_known(1, ctx, new_id, new_length);
            if [2, 6, 7, 63, 64, 127, 128, 255, 256, 65535, 65536].contains(&right) {
                if right < n {
                    assert!(backend.inspect_pair(1, new_id, 1, &lengths).is_some());
                } else {
                    assert_eq!(backend.final_tokens(&lengths), vec![0, new_id, 0]);
                }
            }
        }
        assert_eq!(
            backend.final_tokens(&lengths),
            vec![0, (lengths.len() - 1) as u32, 0]
        );
    }

    #[test]
    fn long_cross_block_and_u32_ids() {
        grow_long::<BitmapU32>();
        grow_long::<BitmapU32<true>>();
        grow_long::<Halfword>();
        grow_long::<Halfword<true>>();
        grow_long::<Hybrid<false>>();
        grow_long::<Hybrid<false, true>>();
        grow_long::<Hybrid<true>>();
        grow_long::<Hybrid<true, true>>();
    }
}

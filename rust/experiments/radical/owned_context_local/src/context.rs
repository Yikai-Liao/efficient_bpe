//! Producer-local direct-address aggregation for one non-AA rule at a time.

use super::{
    BatchRule, BirthNode, Delta, Result, ScratchPlacement, TrainError, WorkerOutput,
    accumulate_delta, key, owner_for,
};

const LEFT: u32 = 0;
const RIGHT_OLD: u32 = 1;
const RIGHT_BIRTH: u32 = 2;

#[derive(Clone, Copy, Default)]
pub(super) struct Report {
    pub lookups: usize,
    pub hits: usize,
    pub groups: usize,
    pub flush_calls: usize,
    pub original_hash_updates: usize,
    pub flushed_hash_updates: usize,
}

struct ContextEntry {
    weight: u64,
    owner: usize,
    slot_key: u32,
    count: u32,
    head: u32,
    tail: u32,
}

pub(super) struct Scratch {
    directory: Box<[u32]>,
    entries: Vec<ContextEntry>,
    limit: usize,
    current: Option<(u32, u32, u32)>,
    pub report: Report,
}

impl Scratch {
    /// Move only the header, not its buffers, into this producer invocation.
    /// The empty placeholder is never used for planning. On a normal Result::Err
    /// the real scratch is restored before returning; on unwind both values drop.
    #[inline(never)]
    pub fn with_placement<T>(
        &mut self,
        placement: ScratchPlacement,
        run: impl FnOnce(&mut Self) -> T,
    ) -> T {
        match placement {
            ScratchPlacement::Borrowed => run(self),
            ScratchPlacement::Local => {
                let empty = Self {
                    directory: Box::new([]),
                    entries: Vec::new(),
                    limit: 0,
                    current: None,
                    report: Report::default(),
                };
                let mut local = std::mem::replace(self, empty);
                let result = run(&mut local);
                *self = local;
                result
            }
        }
    }

    pub fn new(vocabulary_bound: usize, entry_limit: usize) -> Self {
        let cells = vocabulary_bound
            .checked_mul(3)
            .expect("checked context directory size");
        assert!(cells > 0 && cells <= u32::MAX as usize);
        assert!((1..=65_536).contains(&entry_limit));
        let limit = entry_limit.min(cells);
        Self {
            directory: vec![u32::MAX; cells].into_boxed_slice(),
            entries: Vec::with_capacity(limit),
            limit,
            current: None,
            report: Report::default(),
        }
    }

    pub fn directory_bytes(&self) -> usize {
        self.directory.len() * std::mem::size_of::<u32>()
    }

    pub fn entry_capacity_bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<ContextEntry>()
    }

    pub fn switch_rule(&mut self, rule: &BatchRule, output: &mut WorkerOutput) -> Result<()> {
        let ids = (rule.a, rule.b, rule.new_id);
        if self.current != Some(ids) {
            self.flush(output)?;
            self.current = Some(ids);
        }
        Ok(())
    }

    fn entry(&mut self, family: u32, context: u32, output: &mut WorkerOutput) -> Result<usize> {
        debug_assert!(family < 3);
        debug_assert!((context as usize) < self.directory.len() / 3);
        let slot_key = context * 3 + family;
        // SAFETY: train has validated every initial ID, and reserved the whole
        // checked (initial vocabulary + max rules) ID domain in this directory.
        // Born contexts are selected rules' IDs in that same domain. Family<3.
        let prior = unsafe { *self.directory.get_unchecked(slot_key as usize) } as usize;
        self.report.lookups += 1;
        self.report.original_hash_updates += if family == LEFT { 2 } else { 1 };
        if self
            .entries
            .get(prior)
            .is_some_and(|entry| entry.slot_key == slot_key)
        {
            self.report.hits += 1;
            return Ok(prior);
        }
        if self.entries.len() == self.limit {
            self.flush(output)?;
        }
        let (_, b, fresh) = self
            .current
            .ok_or(TrainError::InternalInvariant("context rule not set"))?;
        let pair = match family {
            LEFT => key(context, fresh),
            RIGHT_OLD => key(b, context),
            RIGHT_BIRTH => key(fresh, context),
            _ => unreachable!(),
        };
        let index = self.entries.len();
        self.entries.push(ContextEntry {
            weight: 0,
            owner: owner_for(pair, output.routes.len()),
            slot_key,
            count: 0,
            head: u32::MAX,
            tail: u32::MAX,
        });
        // SAFETY: the same domain proof as above; index <= entry limit <=65536.
        unsafe {
            *self.directory.get_unchecked_mut(slot_key as usize) = index as u32;
        }
        Ok(index)
    }

    fn update(
        &mut self,
        family: u32,
        context: u32,
        pos: Option<u32>,
        weight: u64,
        output: &mut WorkerOutput,
    ) -> Result<()> {
        let index = self.entry(family, context, output)?;
        let entry = &mut self.entries[index];
        let next_weight = entry
            .weight
            .checked_add(weight)
            .ok_or(TrainError::Overflow("context weight exceeds u64"))?;
        let next_count = entry
            .count
            .checked_add(1)
            .ok_or(TrainError::Overflow("context occurrence count exceeds u32"))?;
        if let Some(pos) = pos {
            let born = &mut output.routes[entry.owner].born;
            let node = u32::try_from(born.len())
                .map_err(|_| TrainError::Overflow("context birth index exceeds u32"))?;
            if node == u32::MAX {
                return Err(TrainError::Overflow(
                    "context birth index collides with sentinel",
                ));
            }
            born.push(BirthNode {
                pos,
                next: entry.head,
            });
            if entry.head == u32::MAX {
                entry.tail = node;
            }
            entry.head = node;
        }
        entry.weight = next_weight;
        entry.count = next_count;
        Ok(())
    }

    /// # Safety
    /// `context` must be below the vocabulary bound passed to `new`.
    /// Only the checked trainer's old IDs and its bounded fresh IDs are allowed.
    pub unsafe fn left(
        &mut self,
        context: u32,
        pos: u32,
        weight: u64,
        output: &mut WorkerOutput,
    ) -> Result<()> {
        self.update(LEFT, context, Some(pos), weight, output)
    }

    /// # Safety
    /// `context` must be below the vocabulary bound passed to `new`.
    /// Only the checked trainer's old IDs and its bounded fresh IDs are allowed.
    pub unsafe fn right_old(
        &mut self,
        context: u32,
        weight: u64,
        output: &mut WorkerOutput,
    ) -> Result<()> {
        self.update(RIGHT_OLD, context, None, weight, output)
    }

    /// # Safety
    /// `context` must be below the vocabulary bound passed to `new`.
    /// Only the checked trainer's old IDs and its bounded fresh IDs are allowed.
    pub unsafe fn right_birth(
        &mut self,
        context: u32,
        pos: u32,
        weight: u64,
        output: &mut WorkerOutput,
    ) -> Result<()> {
        self.update(RIGHT_BIRTH, context, Some(pos), weight, output)
    }

    pub fn flush(&mut self, output: &mut WorkerOutput) -> Result<()> {
        if self.entries.is_empty() {
            return Ok(());
        }
        let (a, b, fresh) = self
            .current
            .ok_or(TrainError::InternalInvariant("context rule not set"))?;
        self.report.flush_calls += 1;
        for entry in self.entries.drain(..) {
            let context = entry.slot_key / 3;
            let family = entry.slot_key % 3;
            let delta = Delta {
                weight: entry.weight,
                occurrences: entry.count,
                head: u32::MAX,
            };
            self.report.groups += 1;
            self.report.flushed_hash_updates += if family == LEFT { 2 } else { 1 };
            if family == LEFT {
                let old_pair = key(context, a);
                let owner = owner_for(old_pair, output.routes.len());
                accumulate_delta(&mut output.routes[owner].delta, old_pair, delta)?;
            }
            let pair = match family {
                LEFT => key(context, fresh),
                RIGHT_OLD => key(b, context),
                RIGHT_BIRTH => key(fresh, context),
                _ => unreachable!(),
            };
            let route = &mut output.routes[entry.owner];
            let combined = route.delta.entry(pair).or_default();
            let weight = combined
                .weight
                .checked_add(entry.weight)
                .ok_or(TrainError::Overflow("routed context weight exceeds u64"))?;
            let count = combined
                .occurrences
                .checked_add(entry.count)
                .ok_or(TrainError::Overflow("routed context count exceeds u32"))?;
            if entry.head != u32::MAX {
                let tail = route.born.get_mut(entry.tail as usize).ok_or(
                    TrainError::InternalInvariant("context tail outside birth route"),
                )?;
                if tail.next != u32::MAX {
                    return Err(TrainError::InternalInvariant(
                        "context segment tail already linked",
                    ));
                }
                debug_assert!(combined.head == u32::MAX || combined.head < entry.tail);
                tail.next = combined.head;
                combined.head = entry.head;
            } else if family != RIGHT_OLD {
                return Err(TrainError::InternalInvariant(
                    "context birth group has no positions",
                ));
            }
            combined.weight = weight;
            combined.occurrences = count;
        }
        Ok(())
    }

    pub fn finish(&mut self, output: &mut WorkerOutput) -> Result<()> {
        self.flush(output)?;
        self.current = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{empty_worker, small_posting::SmallPosting};

    fn rule(a: u32, b: u32, new_id: u32) -> BatchRule {
        BatchRule {
            a,
            b,
            new_id,
            frequency: 0,
            a_length: 1,
            b_length: 1,
            posting: SmallPosting::default(),
        }
    }

    fn positions(output: &WorkerOutput, pair: u64) -> Vec<u32> {
        let route = &output.routes[owner_for(pair, output.routes.len())];
        let mut cursor = route.delta[&pair].head;
        let mut result = Vec::new();
        while cursor != u32::MAX {
            let node = &route.born[cursor as usize];
            result.push(node.pos);
            cursor = node.next;
        }
        result
    }

    #[test]
    fn local_header_restores_owned_buffers_and_error_state() {
        for placement in [ScratchPlacement::Borrowed, ScratchPlacement::Local] {
            let mut scratch = Scratch::new(32, 2);
            let address = (&scratch as *const Scratch) as usize;
            let directory = scratch.directory.as_ptr();
            let entries = scratch.entries.as_ptr();
            let result: std::result::Result<(), u32> =
                scratch.with_placement(placement, |active| {
                    let active_address = (active as *const Scratch) as usize;
                    assert_eq!(
                        active_address == address,
                        placement == ScratchPlacement::Borrowed
                    );
                    assert_eq!(active.directory.as_ptr(), directory);
                    assert_eq!(active.entries.as_ptr(), entries);
                    active.report.lookups = 37;
                    Err(9)
                });
            assert_eq!(result, Err(9));
            assert_eq!(scratch.report.lookups, 37);
            assert_eq!(scratch.directory.as_ptr(), directory);
            assert_eq!(scratch.entries.as_ptr(), entries);
        }
    }

    #[test]
    fn capacity_flush_rank_change_and_reused_stale_directory() {
        assert_eq!(std::mem::size_of::<ContextEntry>(), 32);
        let mut scratch = Scratch::new(32, 1);
        let mut first = empty_worker(4);
        scratch.switch_rule(&rule(1, 2, 10), &mut first).unwrap();
        // SAFETY: the literal context is below this test scratch's bound of 32.
        unsafe { scratch.left(3, 17, 2, &mut first) }.unwrap();
        // SAFETY: the literal context is below this test scratch's bound of 32.
        unsafe { scratch.right_old(4, 3, &mut first) }.unwrap(); // flush left, reuse slot 0
        // SAFETY: the literal context is below this test scratch's bound of 32.
        unsafe { scratch.left(3, 18, 5, &mut first) }.unwrap(); // stale directory must miss
        // SAFETY: the literal context is below this test scratch's bound of 32.
        unsafe { scratch.right_birth(4, 19, 3, &mut first) }.unwrap();
        scratch.switch_rule(&rule(5, 6, 11), &mut first).unwrap();
        // SAFETY: the literal context is below this test scratch's bound of 32.
        unsafe { scratch.left(3, 20, 7, &mut first) }.unwrap();
        scratch.finish(&mut first).unwrap();
        assert_eq!(positions(&first, key(3, 10)), vec![18, 17]);
        assert_eq!(positions(&first, key(3, 11)), vec![20]);
        let old = &first.routes[owner_for(key(3, 1), 4)].delta[&key(3, 1)];
        assert_eq!((old.weight, old.occurrences, old.head), (7, 2, u32::MAX));
        drop(first);
        let mut second = empty_worker(4);
        scratch.switch_rule(&rule(7, 8, 12), &mut second).unwrap();
        // SAFETY: the literal context is below this test scratch's bound of 32.
        unsafe { scratch.left(3, 25, 9, &mut second) }.unwrap();
        scratch.finish(&mut second).unwrap();
        assert_eq!(positions(&second, key(3, 12)), vec![25]);
    }
}

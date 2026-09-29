//! Persistent exact maxima of pair owners. Updating one owner does not require
//! polling every other owner. A head is published only after its owner has
//! applied all deltas for the completed epoch and validated its local heap.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Head {
    pub key: u64,
    pub frequency: u64,
    pub holders: u64,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Entry {
    key: u64,
    frequency: u64,
    owner: usize,
    generation: u64,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.key.cmp(&self.key))
            .then_with(|| other.owner.cmp(&self.owner))
            .then_with(|| self.generation.cmp(&other.generation))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) struct Frontier {
    current: Vec<(u64, Option<Head>)>,
    heap: BinaryHeap<Entry>,
}

impl Frontier {
    pub fn new(workers: usize) -> Self {
        assert!(workers > 0);
        Self {
            current: vec![(0, None); workers],
            heap: BinaryHeap::with_capacity(workers),
        }
    }

    pub fn replace(&mut self, owner: usize, head: Option<Head>) {
        let (generation, previous) = &mut self.current[owner];
        if *previous == head {
            return;
        }
        // At most one publication per owner per epoch; fresh token IDs bound
        // the number of epochs below 2^32 in the validated training contract.
        *generation += 1;
        *previous = head;
        if let Some(head) = head {
            self.heap.push(Entry {
                key: head.key,
                frequency: head.frequency,
                owner,
                generation: *generation,
            });
        }
        // Otherwise buried stale entries could accumulate for every epoch.
        // Rebuilding after > 4W entries charges O(W) work to at least O(W)
        // preceding publications, while bounding retained queue storage.
        if self.heap.len() > self.current.len().saturating_mul(4) {
            self.heap = self
                .current
                .iter()
                .enumerate()
                .filter_map(|(owner, &(generation, head))| {
                    head.map(|head| Entry {
                        key: head.key,
                        frequency: head.frequency,
                        owner,
                        generation,
                    })
                })
                .collect();
        }
    }

    pub fn best(&mut self) -> Option<(usize, Head)> {
        loop {
            let entry = *self.heap.peek()?;
            let (generation, head) = self.current[entry.owner];
            if generation == entry.generation {
                return Some((entry.owner, head.expect("current frontier entry")));
            }
            self.heap.pop();
        }
    }

    pub fn capacity_bytes(&self) -> usize {
        self.current.capacity() * std::mem::size_of::<(u64, Option<Head>)>()
            + self.heap.capacity() * std::mem::size_of::<Entry>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changed_owners_and_ties_match_a_complete_scan() {
        let mut frontier = Frontier::new(7);
        let mut expected = [None; 7];
        let mut state = 0x7d95_10fb_u64;
        for _ in 0..5000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let owner = (state >> 32) as usize % expected.len();
            let head = (state & 7 != 0).then_some(Head {
                // Disjoint key domains emulate unique pair ownership.
                key: ((state >> 16) % 13) * 7 + owner as u64,
                frequency: 1 + (state >> 8) % 9,
                holders: state | 1,
            });
            expected[owner] = head;
            frontier.replace(owner, head);
            let winner = expected
                .iter()
                .enumerate()
                .filter_map(|(i, &h)| h.map(|h| (i, h)))
                .max_by(|(_, a), (_, b)| {
                    a.frequency
                        .cmp(&b.frequency)
                        .then_with(|| b.key.cmp(&a.key))
                });
            assert_eq!(frontier.best(), winner);
            assert!(frontier.heap.len() <= 4 * expected.len());
        }
        for owner in 0..expected.len() {
            frontier.replace(owner, None);
        }
        assert_eq!(frontier.best(), None);
    }

    #[test]
    fn unchanged_head_does_not_accumulate_queue_entries() {
        let mut frontier = Frontier::new(1);
        let head = Head {
            key: 17,
            frequency: 5,
            holders: 1,
        };
        for _ in 0..1000 {
            frontier.replace(0, Some(head));
        }
        assert_eq!(frontier.heap.len(), 1);
        // A directory-only update also invalidates the old cached head.
        let updated = Head { holders: 3, ..head };
        frontier.replace(0, Some(updated));
        assert_eq!(frontier.best(), Some((0, updated)));
    }
}

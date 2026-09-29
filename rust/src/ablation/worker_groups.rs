//! One fixed-size bitmap for sparse scheduling at any worker count.
//! Up to 64 workers, bits identify workers exactly. Above that, each bit
//! identifies a contiguous group; expanding a mask may add idle workers but
//! never omits the worker that set a bit. Pair directories remain eight bytes.

#[derive(Clone, Copy)]
pub(super) struct WorkerGroups {
    workers: usize,
    group_size: usize,
}

impl WorkerGroups {
    pub fn new(workers: usize) -> Self {
        assert!(workers > 0);
        Self {
            workers,
            group_size: workers.div_ceil(64),
        }
    }

    pub fn bit(self, worker: usize) -> u64 {
        assert!(worker < self.workers);
        1_u64 << (worker / self.group_size)
    }

    pub fn all(self) -> u64 {
        let groups = self.workers.div_ceil(self.group_size);
        u64::MAX >> (64 - groups)
    }

    pub fn members(self, mask: u64) -> Members {
        Members {
            layout: self,
            mask: mask & self.all(),
            next: 0,
            end: 0,
        }
    }
}

pub(super) struct Members {
    layout: WorkerGroups,
    mask: u64,
    next: usize,
    end: usize,
}

impl Iterator for Members {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.next == self.end {
            if self.mask == 0 {
                return None;
            }
            let group = self.mask.trailing_zeros() as usize;
            self.mask &= self.mask - 1;
            self.next = group * self.layout.group_size;
            self.end = self
                .next
                .saturating_add(self.layout.group_size)
                .min(self.layout.workers);
        }
        let result = self.next;
        self.next += 1;
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_masks_are_conservative_without_duplicate_workers() {
        for workers in 1..=257 {
            let groups = WorkerGroups::new(workers);
            assert_eq!(
                groups.members(groups.all()).collect::<Vec<_>>(),
                (0..workers).collect::<Vec<_>>()
            );
            assert!(groups.members(0).next().is_none());
            for worker in 0..workers {
                let expanded: Vec<_> = groups.members(groups.bit(worker)).collect();
                assert!(expanded.contains(&worker));
                assert!(expanded.len() <= workers.div_ceil(64));
                if workers <= 64 {
                    assert_eq!(expanded, vec![worker]);
                }
            }
        }
    }
}

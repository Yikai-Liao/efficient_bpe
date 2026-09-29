//! One or two positions inline; counted longer postings reserve one Vec allocation.
//! Raw-part reconstruction is private to this module.

use efficient_bpe_rust::TrainError;
use std::mem::{self, ManuallyDrop};
use std::slice;

#[repr(C)]
union Payload {
    inline: [u32; 2],
    heap: *mut u32,
}

#[repr(C)]
pub(super) struct SmallPosting {
    len: u32,
    // Zero tags inline mode. Heap mode records Vec's actual allocation capacity.
    capacity: u32,
    payload: Payload,
}

// SAFETY: every heap allocation has one SmallPosting owner; mutation requires
// &mut self, and shared slices contain only initialized u32 values.
unsafe impl Send for SmallPosting {}
// SAFETY: immutable access never mutates the allocation or the union tag.
unsafe impl Sync for SmallPosting {}

impl Default for SmallPosting {
    fn default() -> Self {
        Self {
            len: 0,
            capacity: 0,
            payload: Payload { inline: [0, 0] },
        }
    }
}

impl SmallPosting {
    /// Reserve for a freshly counted key. A heap posting may initially be empty:
    /// its allocation is owned even while no position has been initialized.
    pub(super) fn with_capacity(count: u32) -> Result<Self, TrainError> {
        if count <= 2 {
            return Ok(Self::default());
        }
        let mut posting = Self::default();
        posting.install_heap(Vec::with_capacity((count as usize).max(4)))?;
        Ok(posting)
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub(super) fn is_inline(&self) -> bool {
        self.capacity == 0
    }

    /// Physical heap slots. The two inline slots are part of the map entry.
    #[inline]
    pub(super) fn allocated_capacity(&self) -> usize {
        if self.is_inline() {
            0
        } else {
            self.capacity as usize
        }
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[u32] {
        if self.is_inline() {
            debug_assert!(self.len <= 2);
            // SAFETY: inline is the active union field and both elements were
            // initialized by Default. Only the first len are exposed.
            let inline = unsafe { &self.payload.inline };
            &inline[..self.len()]
        } else {
            debug_assert!(self.len <= self.capacity && self.capacity >= 4);
            // SAFETY: heap came from a Vec<u32> allocation of capacity at
            // least len; its first len elements were initialized by push.
            unsafe { slice::from_raw_parts(self.payload.heap, self.len()) }
        }
    }

    #[inline]
    pub(super) fn as_mut_slice(&mut self) -> &mut [u32] {
        if self.is_inline() {
            debug_assert!(self.len <= 2);
            let len = self.len();
            // SAFETY: inline is active and initialized; &mut self is unique.
            let inline = unsafe { &mut self.payload.inline };
            &mut inline[..len]
        } else {
            debug_assert!(self.len <= self.capacity && self.capacity >= 4);
            // SAFETY: the allocation is uniquely owned and first len items
            // initialized; &mut self excludes other aliases.
            unsafe { slice::from_raw_parts_mut(self.payload.heap, self.len()) }
        }
    }

    /// Move the posting out and leave an empty inline posting behind.
    pub(super) fn take(&mut self) -> Self {
        mem::take(self)
    }

    /// Borrow the entire reserved payload of a fresh, empty heap posting.
    /// The returned MaybeUninit slice may be split among disjoint writers.
    pub(super) fn spare_capacity_mut(
        &mut self,
    ) -> Result<&mut [mem::MaybeUninit<u32>], TrainError> {
        if self.is_inline() || self.len != 0 {
            return Err(TrainError::InternalInvariant(
                "scatter requires a fresh empty heap posting",
            ));
        }
        // SAFETY: this is the unique Vec allocation recorded in heap mode.
        // MaybeUninit<u32> has u32's layout; len=0 means every cap slot is
        // spare, and the mutable borrow excludes every other access.
        Ok(unsafe {
            slice::from_raw_parts_mut(
                self.payload.heap.cast::<mem::MaybeUninit<u32>>(),
                self.capacity as usize,
            )
        })
    }

    /// # Safety
    /// The caller must have initialized every slot in `[0, count)` and joined
    /// every writer holding a split spare-capacity slice before this call.
    pub(super) unsafe fn finish_initialize(&mut self, count: u32) -> Result<(), TrainError> {
        if self.is_inline() || self.len != 0 || count > self.capacity {
            return Err(TrainError::InternalInvariant(
                "scatter initialized length exceeds fresh heap reservation",
            ));
        }
        self.len = count;
        Ok(())
    }

    fn install_heap(&mut self, vec: Vec<u32>) -> Result<(), TrainError> {
        // Every caller either holds inline data or used mem::take first.
        debug_assert!(self.is_inline());
        let len = u32::try_from(vec.len())
            .map_err(|_| TrainError::Overflow("posting length exceeds u32"))?;
        let capacity = u32::try_from(vec.capacity())
            .map_err(|_| TrainError::Overflow("posting capacity exceeds u32"))?;
        if capacity < 4 || len > capacity {
            return Err(TrainError::InternalInvariant("invalid heap posting Vec"));
        }
        // Conversion cannot fail beyond this point. Suppress Vec's destructor
        // only after checking both fields, then transfer its allocation.
        let mut vec = ManuallyDrop::new(vec);
        self.payload = Payload {
            heap: vec.as_mut_ptr(),
        };
        self.len = len;
        self.capacity = capacity;
        Ok(())
    }

    pub(super) fn push(&mut self, pos: u32) -> Result<(), TrainError> {
        let next_len = self
            .len
            .checked_add(1)
            .ok_or(TrainError::Overflow("posting length exceeds u32"))?;
        if self.is_inline() {
            if self.len < 2 {
                // SAFETY: active inline array is initialized; index is 0 or 1.
                unsafe {
                    self.payload.inline[self.len as usize] = pos;
                }
                self.len = next_len;
                return Ok(());
            }
            let mut vec = Vec::with_capacity(4);
            vec.extend_from_slice(self.as_slice());
            vec.push(pos);
            return self.install_heap(vec);
        }
        if self.len < self.capacity {
            // SAFETY: ptr is Vec allocation; len < capacity leaves one spare
            // uninitialized slot and &mut self is its unique owner.
            unsafe {
                self.payload.heap.add(self.len as usize).write(pos);
            }
            self.len = next_len;
            return Ok(());
        }
        let requested = (self.capacity as usize)
            .saturating_mul(2)
            .min(u32::MAX as usize)
            .max(next_len as usize);
        // The destination is empty before a Vec can reallocate or unwind.
        // The old raw pointer is owned only by the reconstructed Vec below.
        let old = ManuallyDrop::new(mem::take(self));
        // SAFETY: old is heap mode, and ptr/len/cap are the exact raw parts
        // recorded when its unique Vec allocation was last installed.
        let mut vec = unsafe {
            Vec::from_raw_parts(old.payload.heap, old.len as usize, old.capacity as usize)
        };
        vec.reserve_exact(requested - vec.len());
        vec.push(pos);
        self.install_heap(vec)
    }
}

impl Drop for SmallPosting {
    fn drop(&mut self) {
        if self.capacity != 0 {
            // SAFETY: heap mode uniquely owns a Vec allocation whose exact
            // raw pointer, initialized length, and capacity are stored here.
            unsafe {
                drop(Vec::from_raw_parts(
                    self.payload.heap,
                    self.len as usize,
                    self.capacity as usize,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_spare_slices_commit_only_after_complete_fill() {
        let mut posting = SmallPosting::with_capacity(17).unwrap();
        {
            let spare = posting.spare_capacity_mut().unwrap();
            let (left, rest) = spare.split_at_mut(7);
            let (right, _) = rest.split_at_mut(10);
            for (i, slot) in left.iter_mut().enumerate() {
                slot.write(i as u32);
            }
            for (i, slot) in right.iter_mut().enumerate() {
                slot.write((i + 7) as u32);
            }
        }
        // SAFETY: all 17 slots were written, and the two disjoint borrows ended.
        unsafe { posting.finish_initialize(17) }.unwrap();
        assert_eq!(posting.as_slice(), (0..17).collect::<Vec<_>>().as_slice());
        assert!(posting.spare_capacity_mut().is_err());

        let mut abandoned = SmallPosting::with_capacity(17).unwrap();
        abandoned.spare_capacity_mut().unwrap()[0].write(42);
        // A failed fill leaves len=0; Drop must never read the partial payload.
        drop(abandoned);
    }

    #[test]
    fn counted_reservation_empty_heap_fill_growth_and_take() {
        for count in [0_u32, 1, 2, 3, 17] {
            let mut posting = SmallPosting::with_capacity(count).unwrap();
            assert!(posting.is_empty());
            assert!(posting.as_slice().is_empty());
            assert!(posting.as_mut_slice().is_empty());
            assert_eq!(posting.is_inline(), count <= 2);
            if count > 2 {
                assert!(posting.allocated_capacity() >= count as usize);
            }
            for pos in 0..count {
                posting.push(pos).unwrap();
            }
            assert_eq!(
                posting.as_slice(),
                (0..count).collect::<Vec<_>>().as_slice()
            );
            if count > 2 {
                let reserved = posting.allocated_capacity();
                for pos in count..reserved as u32 {
                    posting.push(pos).unwrap();
                }
                assert_eq!(posting.allocated_capacity(), reserved);
                posting.push(u32::MAX).unwrap();
                assert_eq!(posting.len(), reserved + 1);
                assert_eq!(posting.as_slice()[reserved], u32::MAX);
            }
            let moved = posting.take();
            assert!(posting.is_inline());
            assert!(posting.is_empty());
            assert!(!moved.as_slice().is_empty() || count == 0);
            drop(moved);
        }
        // Empty heap mode must still free the reserved Vec on drop.
        drop(SmallPosting::with_capacity(17).unwrap());
    }

    #[test]
    fn position_payload_uses_all_u32_bits_across_inline_heap_and_take() {
        let values = [0, u32::MAX, 1_u32 << 31];
        let mut posting = SmallPosting::default();
        posting.push(values[0]).unwrap();
        posting.push(values[1]).unwrap();
        assert_eq!(posting.as_slice(), &values[..2]);
        posting.push(values[2]).unwrap();
        let moved = posting.take();
        assert_eq!(moved.as_slice(), values);
        assert!(posting.is_inline() && posting.is_empty());
    }

    #[test]
    fn layout_inline_growth_move_and_drop() {
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(mem::size_of::<SmallPosting>(), 16);
            assert_eq!(mem::align_of::<SmallPosting>(), 8);
        }
        let mut posting = SmallPosting::default();
        assert_eq!(posting.len(), 0);
        assert_eq!(posting.as_slice(), []);
        assert_eq!(posting.allocated_capacity(), 0);
        for n in 0..2 {
            posting.push(n).unwrap();
            assert!(posting.is_inline());
            assert_eq!(posting.as_slice(), (0..=n).collect::<Vec<_>>().as_slice());
        }
        posting.push(2).unwrap();
        assert!(!posting.is_inline());
        assert_eq!(posting.as_slice(), [0, 1, 2]);
        let mut moved = posting.take();
        assert_eq!(posting.len(), 0);
        for n in 3..1000 {
            moved.push(n).unwrap();
        }
        assert_eq!(moved.as_slice(), (0..1000).collect::<Vec<_>>().as_slice());
        moved.as_mut_slice().reverse();
        assert_eq!(moved.as_slice()[0], 999);
        drop(moved);
        posting.push(77).unwrap();
        assert_eq!(posting.as_slice(), [77]);
        let one = posting.take();
        posting.push(88).unwrap();
        assert_eq!(one.as_slice(), [77]);
        assert_eq!(posting.as_slice(), [88]);
        drop(one);
        posting.push(99).unwrap();
        let two = posting.take();
        assert_eq!(two.as_slice(), [88, 99]);
        assert_eq!(posting.len(), 0);
        drop(two);
    }

    #[test]
    fn grown_posting_can_move_between_threads() {
        let mut posting = SmallPosting::default();
        for n in 0..128 {
            posting.push(n).unwrap();
        }
        let posting = std::thread::spawn(move || {
            let mut posting = posting;
            for n in 128..512 {
                posting.push(n).unwrap();
            }
            assert_eq!(posting.as_slice()[511], 511);
            posting
        })
        .join()
        .unwrap();
        std::thread::scope(|scope| {
            scope
                .spawn(|| assert_eq!(posting.as_slice()[0], 0))
                .join()
                .unwrap();
        });
        drop(posting);
    }
}

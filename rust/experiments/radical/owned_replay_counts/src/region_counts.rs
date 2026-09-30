//! Ordered region/count metadata. Inline storage saves tiny heap allocations.

use super::{Result, TrainError};

pub(super) trait CountList: Default + Send {
    type Iter: Iterator<Item = (usize, u32)>;
    fn push(&mut self, region: usize, count: u32) -> Result<()>;
    fn len(&self) -> usize;
    fn heap_bytes(&self) -> usize;
    fn into_counts(self) -> Self::Iter;
}

#[derive(Default)]
pub(super) struct VecCounts(Vec<(usize, u32)>);

impl CountList for VecCounts {
    type Iter = std::vec::IntoIter<(usize, u32)>;
    fn push(&mut self, region: usize, count: u32) -> Result<()> {
        self.0.push((region, count));
        Ok(())
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn heap_bytes(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<(usize, u32)>()
    }
    fn into_counts(self) -> Self::Iter {
        self.0.into_iter()
    }
}

#[derive(Default)]
pub(super) enum InlineCounts {
    #[default]
    Empty,
    Inline {
        len: u8,
        slots: [[u32; 2]; 2],
    },
    Heap(Vec<[u32; 2]>),
}

pub(super) enum CountIter {
    Inline(std::iter::Take<std::array::IntoIter<[u32; 2], 2>>),
    Heap(std::vec::IntoIter<[u32; 2]>),
}

impl Iterator for CountIter {
    type Item = (usize, u32);
    fn next(&mut self) -> Option<Self::Item> {
        let next = match self {
            Self::Inline(iter) => iter.next(),
            Self::Heap(iter) => iter.next(),
        };
        next.map(|[region, count]| (region as usize, count))
    }
}

impl CountList for InlineCounts {
    type Iter = CountIter;
    fn push(&mut self, region: usize, count: u32) -> Result<()> {
        // T <= N <= u32::MAX is established by the corpus validation. Keep the
        // conversion checked so this container remains correct independently.
        let region =
            u32::try_from(region).map_err(|_| TrainError::Overflow("replay region exceeds u32"))?;
        let item = [region, count];
        match self {
            Self::Empty => {
                *self = Self::Inline {
                    len: 1,
                    slots: [item, [0, 0]],
                }
            }
            Self::Inline { len, slots } if *len < 2 => {
                slots[*len as usize] = item;
                *len += 1;
            }
            Self::Inline { slots, .. } => {
                let mut heap = Vec::with_capacity(4);
                heap.extend_from_slice(slots);
                heap.push(item);
                *self = Self::Heap(heap);
            }
            Self::Heap(heap) => heap.push(item),
        }
        Ok(())
    }
    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Inline { len, .. } => *len as usize,
            Self::Heap(heap) => heap.len(),
        }
    }
    fn heap_bytes(&self) -> usize {
        match self {
            Self::Heap(heap) => heap.capacity() * std::mem::size_of::<[u32; 2]>(),
            _ => 0,
        }
    }
    fn into_counts(self) -> Self::Iter {
        match self {
            Self::Empty => CountIter::Inline([[0; 2]; 2].into_iter().take(0)),
            Self::Inline { len, slots } => CountIter::Inline(slots.into_iter().take(len as usize)),
            Self::Heap(heap) => CountIter::Heap(heap.into_iter()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_counts_survive_inline_to_heap_transition() {
        for length in [0, 1, 2, 3, 4, 17] {
            let mut inline = InlineCounts::default();
            let mut control = VecCounts::default();
            for index in 0..length {
                inline.push(index * 7, (index + 1) as u32).unwrap();
                control.push(index * 7, (index + 1) as u32).unwrap();
            }
            assert_eq!(inline.len(), length);
            assert_eq!(inline.heap_bytes() == 0, length <= 2);
            assert_eq!(
                inline.into_counts().collect::<Vec<_>>(),
                control.into_counts().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn packed_regions_do_not_truncate_and_counts_keep_full_u32_domain() {
        let mut inline = InlineCounts::default();
        inline.push(u32::MAX as usize, u32::MAX).unwrap();
        assert_eq!(
            inline.into_counts().collect::<Vec<_>>(),
            vec![(u32::MAX as usize, u32::MAX)]
        );
        if usize::BITS > 32 {
            assert!(
                InlineCounts::default()
                    .push(u32::MAX as usize + 1, 1)
                    .is_err()
            );
        }
    }
}

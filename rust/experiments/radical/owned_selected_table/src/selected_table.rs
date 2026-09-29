//! One immutable selected-pair lookup table shared by all planning workers.

use super::SelectedLookupMode;
use efficient_bpe_rust::TrainError;
use std::collections::HashMap;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Slot {
    key: u64,
    new_id: u32,
    _padding: u32,
}

pub(super) struct FlatSelected {
    slots: Vec<Slot>,
    len: usize,
}

pub(super) enum SelectedLookup {
    Hash(HashMap<u64, u32>),
    Flat(FlatSelected),
}

pub(super) trait SelectedRead: Sync {
    fn lookup(&self, key: u64) -> Option<u32>;
}

impl SelectedRead for HashMap<u64, u32> {
    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        self.get(&key).copied()
    }
}

impl SelectedRead for FlatSelected {
    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        self.get(key)
    }
}

#[inline]
fn mix(mut key: u64) -> u64 {
    key ^= key >> 30;
    key = key.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    key ^= key >> 27;
    key = key.wrapping_mul(0x94d0_49bb_1331_11eb);
    key ^ (key >> 31)
}

impl FlatSelected {
    fn new(width: usize) -> Self {
        assert!((1..=256).contains(&width));
        let capacity = width.next_power_of_two() * 2;
        Self {
            slots: vec![Slot::default(); capacity],
            len: 0,
        }
    }

    fn insert(&mut self, key: u64, new_id: u32) -> Result<(), TrainError> {
        if key == 0 || new_id == 0 || (self.len + 1) * 2 > self.slots.len() {
            return Err(TrainError::InternalInvariant(
                "invalid selected flat table insertion",
            ));
        }
        let mask = self.slots.len() - 1;
        let mut slot_i = (mix(key) as usize) & mask;
        for _ in 0..self.slots.len() {
            let slot = &mut self.slots[slot_i];
            if slot.key == 0 {
                *slot = Slot {
                    key,
                    new_id,
                    _padding: 0,
                };
                self.len += 1;
                return Ok(());
            }
            if slot.key == key {
                return Err(TrainError::InternalInvariant("duplicate selected pair"));
            }
            slot_i = (slot_i + 1) & mask;
        }
        Err(TrainError::InternalInvariant(
            "selected flat table unexpectedly full",
        ))
    }

    #[inline]
    fn get(&self, key: u64) -> Option<u32> {
        if key == 0 {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut slot_i = (mix(key) as usize) & mask;
        for _ in 0..self.slots.len() {
            let slot = &self.slots[slot_i];
            if slot.key == key {
                return Some(slot.new_id);
            }
            if slot.key == 0 {
                return None;
            }
            slot_i = (slot_i + 1) & mask;
        }
        None
    }
}

impl SelectedLookup {
    pub(super) fn new(mode: SelectedLookupMode, width: usize) -> Self {
        match mode {
            SelectedLookupMode::Hash => Self::Hash(HashMap::with_capacity(width)),
            SelectedLookupMode::Flat => Self::Flat(FlatSelected::new(width)),
        }
    }

    pub(super) fn insert(&mut self, key: u64, new_id: u32) -> Result<(), TrainError> {
        match self {
            Self::Hash(map) => {
                if map.insert(key, new_id).is_some() {
                    return Err(TrainError::InternalInvariant("duplicate selected pair"));
                }
                Ok(())
            }
            Self::Flat(table) => table.insert(key, new_id),
        }
    }

    pub(super) fn flat_slots(&self) -> usize {
        match self {
            Self::Hash(_) => 0,
            Self::Flat(table) => table.slots.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collisions_compare_complete_keys_and_missing_terminates() {
        assert_eq!(std::mem::size_of::<Slot>(), 16);
        let mut table = FlatSelected::new(3);
        assert_eq!(table.slots.len(), 8);
        let first = 1_u64;
        let second = (2_u64..)
            .find(|key| (mix(*key) & 7) == (mix(first) & 7))
            .unwrap();
        assert_ne!(first, second);
        table.insert(first, 77).unwrap();
        table.insert(second, 88).unwrap();
        table.insert(u64::MAX, 99).unwrap();
        assert_eq!(table.get(first), Some(77));
        assert_eq!(table.get(second), Some(88));
        assert_eq!(table.get(u64::MAX), Some(99));
        assert_eq!(table.get(0), None);
        assert_eq!(table.get(second + 1), None);
        assert!(table.insert(first, 100).is_err());
    }

    #[test]
    fn maximum_batch_load_has_bounded_success_and_miss() {
        let mut table = FlatSelected::new(256);
        assert_eq!(table.slots.len(), 512);
        for id in 1..=256_u64 {
            table.insert(id, id as u32 + 1000).unwrap();
        }
        for id in 1..=256_u64 {
            assert_eq!(table.get(id), Some(id as u32 + 1000));
        }
        for id in 257..=512_u64 {
            assert_eq!(table.get(id), None);
        }
        assert_eq!(table.len, 256);
        assert!(table.insert(513, 1513).is_err());
    }
}

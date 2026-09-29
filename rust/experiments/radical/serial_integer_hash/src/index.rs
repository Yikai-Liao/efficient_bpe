//! The original ablation's u64 Combined index, parameterized only by its builder.

use efficient_bpe_rust::TrainError;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash};

pub trait Key: Copy + Eq + Ord + Hash + Debug {
    fn pair(a: u32, b: u32) -> Self;
    fn tokens(self) -> (u32, u32);
}

impl Key for u64 {
    #[inline]
    fn pair(a: u32, b: u32) -> Self {
        (u64::from(a) << 32) | u64::from(b)
    }

    #[inline]
    fn tokens(self) -> (u32, u32) {
        ((self >> 32) as u32, self as u32)
    }
}

#[derive(Default)]
struct Record {
    frequency: u64,
    positions: Vec<u32>,
}

pub struct Combined<H: BuildHasher + Default> {
    records: HashMap<u64, Record, H>,
}

impl<H: BuildHasher + Default> Default for Combined<H> {
    fn default() -> Self {
        Self {
            records: HashMap::with_hasher(H::default()),
        }
    }
}

impl<H: BuildHasher + Default> Combined<H> {
    pub const RECORD_BYTES: usize = 4;

    pub fn frequency(&self, key: u64) -> u64 {
        self.records.get(&key).map_or(0, |record| record.frequency)
    }

    pub fn add(&mut self, key: u64, weight: u64) -> Result<(), TrainError> {
        let frequency = &mut self.records.entry(key).or_default().frequency;
        *frequency = frequency
            .checked_add(weight)
            .ok_or(TrainError::Overflow("pair frequency exceeds u64"))?;
        Ok(())
    }

    pub fn subtract(&mut self, key: u64, weight: u64) -> Result<(), TrainError> {
        if let Some(record) = self.records.get_mut(&key) {
            record.frequency = record
                .frequency
                .checked_sub(weight)
                .ok_or(TrainError::InternalInvariant("negative pair frequency"))?;
        }
        Ok(())
    }

    pub fn record(&mut self, key: u64, weight: u64, pos: u32) -> Result<(), TrainError> {
        let record = self.records.entry(key).or_default();
        record.frequency = record
            .frequency
            .checked_add(weight)
            .ok_or(TrainError::Overflow("pair frequency exceeds u64"))?;
        record.positions.push(pos);
        Ok(())
    }

    pub fn append_if_tracked(&mut self, key: u64, pos: u32) -> bool {
        if let Some(record) = self.records.get_mut(&key) {
            record.positions.push(pos);
            true
        } else {
            false
        }
    }

    pub fn detach(&mut self, key: u64) -> std::vec::IntoIter<u32> {
        self.records
            .get_mut(&key)
            .map(|record| std::mem::take(&mut record.positions))
            .unwrap_or_default()
            .into_iter()
    }

    pub fn discard(&mut self, key: u64) {
        self.records.remove(&key);
    }

    pub fn entries(&self) -> Vec<(u64, u64)> {
        self.records
            .iter()
            .map(|(&key, record)| (key, record.frequency))
            .collect()
    }

    pub fn occurrence_bytes(&self) -> usize {
        self.records
            .values()
            .map(|record| record.positions.len() * 4)
            .sum()
    }

    pub fn metrics(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            (
                "position_capacity_bytes".into(),
                self.records
                    .values()
                    .map(|record| record.positions.capacity() * 4)
                    .sum::<usize>() as f64,
            ),
            (
                "combined_map_capacity".into(),
                self.records.capacity() as f64,
            ),
            (
                "combined_record_bytes".into(),
                std::mem::size_of::<Record>() as f64,
            ),
        ])
    }
}

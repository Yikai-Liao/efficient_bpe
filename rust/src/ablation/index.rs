use crate::TrainError;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

pub trait Key: Copy + Eq + Ord + Hash + Debug {
    fn pair(a: u32, b: u32) -> Self;
    fn tokens(self) -> (u32, u32);
}
impl Key for u64 {
    fn pair(a: u32, b: u32) -> Self {
        (u64::from(a) << 32) | u64::from(b)
    }
    fn tokens(self) -> (u32, u32) {
        ((self >> 32) as u32, self as u32)
    }
}
impl Key for (u32, u32) {
    fn pair(a: u32, b: u32) -> Self {
        (a, b)
    }
    fn tokens(self) -> (u32, u32) {
        self
    }
}

pub trait Index<K: Key>: Default {
    const RECORD_BYTES: usize;
    type Batch;
    type Selected: Copy;
    fn selected(&self, key: K) -> Self::Selected;
    fn subtract_selected(
        &mut self,
        selected: Self::Selected,
        weight: u64,
    ) -> Result<(), TrainError>;
    fn frequency(&self, key: K) -> u64;
    fn add(&mut self, key: K, weight: u64) -> Result<(), TrainError>;
    fn subtract(&mut self, key: K, weight: u64) -> Result<(), TrainError>;
    fn append(&mut self, key: K, pos: u32) -> Result<(), TrainError>;
    fn record(&mut self, key: K, weight: u64, pos: u32) -> Result<(), TrainError> {
        self.add(key, weight)?;
        self.append(key, pos)
    }
    fn append_if_tracked(&mut self, key: K, pos: u32) -> Result<bool, TrainError> {
        if self.frequency(key) == 0 {
            return Ok(false);
        }
        self.append(key, pos)?;
        Ok(true)
    }
    fn detach(&mut self, key: K) -> Self::Batch;
    fn next(&mut self, batch: &mut Self::Batch) -> Option<u32>;
    fn discard(&mut self, key: K, keep_frequency: bool);
    fn entries(&self) -> Vec<(K, u64)>;
    fn occurrence_bytes(&self) -> usize;
    fn metrics(&self) -> BTreeMap<String, f64>;
}

fn increase(value: &mut u64, weight: u64) -> Result<(), TrainError> {
    *value = value
        .checked_add(weight)
        .ok_or(TrainError::Overflow("pair frequency exceeds u64"))?;
    Ok(())
}
fn decrease(value: &mut u64, weight: u64) -> Result<(), TrainError> {
    *value = value
        .checked_sub(weight)
        .ok_or(TrainError::InternalInvariant("negative pair frequency"))?;
    Ok(())
}

pub struct Separate<K: Key> {
    frequencies: HashMap<K, u64>,
    positions: HashMap<K, Vec<u32>>,
}
impl<K: Key> Default for Separate<K> {
    fn default() -> Self {
        Self {
            frequencies: HashMap::new(),
            positions: HashMap::new(),
        }
    }
}
impl<K: Key> Index<K> for Separate<K> {
    const RECORD_BYTES: usize = 4;
    type Batch = std::vec::IntoIter<u32>;
    type Selected = K;
    fn selected(&self, key: K) -> K {
        key
    }
    fn subtract_selected(&mut self, key: K, weight: u64) -> Result<(), TrainError> {
        self.subtract(key, weight)
    }
    fn frequency(&self, key: K) -> u64 {
        self.frequencies.get(&key).copied().unwrap_or(0)
    }
    fn add(&mut self, key: K, weight: u64) -> Result<(), TrainError> {
        increase(self.frequencies.entry(key).or_default(), weight)
    }
    fn subtract(&mut self, key: K, weight: u64) -> Result<(), TrainError> {
        if let Some(f) = self.frequencies.get_mut(&key) {
            decrease(f, weight)?;
        }
        Ok(())
    }
    fn append(&mut self, key: K, pos: u32) -> Result<(), TrainError> {
        self.positions.entry(key).or_default().push(pos);
        Ok(())
    }
    fn detach(&mut self, key: K) -> Self::Batch {
        self.positions.remove(&key).unwrap_or_default().into_iter()
    }
    fn next(&mut self, batch: &mut Self::Batch) -> Option<u32> {
        batch.next()
    }
    fn discard(&mut self, key: K, keep_frequency: bool) {
        self.positions.remove(&key);
        if !keep_frequency {
            self.frequencies.remove(&key);
        }
    }
    fn entries(&self) -> Vec<(K, u64)> {
        self.frequencies.iter().map(|(&k, &f)| (k, f)).collect()
    }
    fn occurrence_bytes(&self) -> usize {
        self.positions.values().map(|v| v.len() * 4).sum()
    }
    fn metrics(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            (
                "position_capacity_bytes".into(),
                self.positions
                    .values()
                    .map(|v| v.capacity() * 4)
                    .sum::<usize>() as f64,
            ),
            (
                "position_map_capacity".into(),
                self.positions.capacity() as f64,
            ),
            (
                "frequency_map_capacity".into(),
                self.frequencies.capacity() as f64,
            ),
        ])
    }
}

#[derive(Default)]
struct Record {
    frequency: u64,
    positions: Vec<u32>,
}

/// Native-specific ablation: co-locate frequency and offsets in one hash table.
pub struct Combined<K: Key> {
    records: HashMap<K, Record>,
}
impl<K: Key> Default for Combined<K> {
    fn default() -> Self {
        Self {
            records: HashMap::new(),
        }
    }
}
impl<K: Key> Index<K> for Combined<K> {
    const RECORD_BYTES: usize = 4;
    type Batch = std::vec::IntoIter<u32>;
    type Selected = K;
    fn selected(&self, key: K) -> K {
        key
    }
    fn subtract_selected(&mut self, key: K, weight: u64) -> Result<(), TrainError> {
        self.subtract(key, weight)
    }
    fn frequency(&self, key: K) -> u64 {
        self.records.get(&key).map_or(0, |r| r.frequency)
    }
    fn add(&mut self, key: K, weight: u64) -> Result<(), TrainError> {
        increase(&mut self.records.entry(key).or_default().frequency, weight)
    }
    fn subtract(&mut self, key: K, weight: u64) -> Result<(), TrainError> {
        if let Some(r) = self.records.get_mut(&key) {
            decrease(&mut r.frequency, weight)?;
        }
        Ok(())
    }
    fn append(&mut self, key: K, pos: u32) -> Result<(), TrainError> {
        self.records.entry(key).or_default().positions.push(pos);
        Ok(())
    }
    fn record(&mut self, key: K, weight: u64, pos: u32) -> Result<(), TrainError> {
        let record = self.records.entry(key).or_default();
        increase(&mut record.frequency, weight)?;
        record.positions.push(pos);
        Ok(())
    }
    fn append_if_tracked(&mut self, key: K, pos: u32) -> Result<bool, TrainError> {
        if let Some(record) = self.records.get_mut(&key) {
            record.positions.push(pos);
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn detach(&mut self, key: K) -> Self::Batch {
        self.records
            .get_mut(&key)
            .map(|r| std::mem::take(&mut r.positions))
            .unwrap_or_default()
            .into_iter()
    }
    fn next(&mut self, batch: &mut Self::Batch) -> Option<u32> {
        batch.next()
    }
    fn discard(&mut self, key: K, keep_frequency: bool) {
        if keep_frequency {
            if let Some(r) = self.records.get_mut(&key) {
                r.positions = Vec::new();
            }
        } else {
            self.records.remove(&key);
        }
    }
    fn entries(&self) -> Vec<(K, u64)> {
        self.records
            .iter()
            .map(|(&k, r)| (k, r.frequency))
            .collect()
    }
    fn occurrence_bytes(&self) -> usize {
        self.records.values().map(|r| r.positions.len() * 4).sum()
    }
    fn metrics(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            (
                "position_capacity_bytes".into(),
                self.records
                    .values()
                    .map(|r| r.positions.capacity() * 4)
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

const NIL: u32 = u32::MAX;

/// Recyclable global position/next arrays, matching the Python arena ablation.
pub struct Arena<K: Key> {
    states: HashMap<K, u32>,
    frequency: Vec<u64>,
    head: Vec<u32>,
    tail: Vec<u32>,
    free_states: Vec<u32>,
    position: Vec<u32>,
    next_node: Vec<u32>,
    free_node: u32,
    active: usize,
    node_reuses: usize,
    state_reuses: usize,
}
impl<K: Key> Default for Arena<K> {
    fn default() -> Self {
        Self {
            states: HashMap::new(),
            frequency: vec![],
            head: vec![],
            tail: vec![],
            free_states: vec![],
            position: vec![],
            next_node: vec![],
            free_node: NIL,
            active: 0,
            node_reuses: 0,
            state_reuses: 0,
        }
    }
}
impl<K: Key> Arena<K> {
    fn state(&mut self, key: K) -> Result<u32, TrainError> {
        if let Some(&slot) = self.states.get(&key) {
            return Ok(slot);
        }
        let slot = if let Some(slot) = self.free_states.pop() {
            self.frequency[slot as usize] = 0;
            self.head[slot as usize] = NIL;
            self.tail[slot as usize] = NIL;
            self.state_reuses += 1;
            slot
        } else {
            if self.frequency.len() >= NIL as usize {
                return Err(TrainError::Overflow("arena state index exceeds u32"));
            }
            let slot = self.frequency.len() as u32;
            self.frequency.push(0);
            self.head.push(NIL);
            self.tail.push(NIL);
            slot
        };
        self.states.insert(key, slot);
        Ok(slot)
    }
    fn recycle(&mut self, node: u32) {
        self.next_node[node as usize] = self.free_node;
        self.free_node = node;
        self.active -= 1;
    }
    fn append_slot(&mut self, slot: usize, pos: u32) -> Result<(), TrainError> {
        let node = if self.free_node != NIL {
            let n = self.free_node;
            self.free_node = self.next_node[n as usize];
            self.position[n as usize] = pos;
            self.next_node[n as usize] = NIL;
            self.node_reuses += 1;
            n
        } else {
            if self.position.len() >= NIL as usize {
                return Err(TrainError::Overflow("arena occurrence index exceeds u32"));
            }
            let n = self.position.len() as u32;
            self.position.push(pos);
            self.next_node.push(NIL);
            n
        };
        if self.tail[slot] == NIL {
            self.head[slot] = node;
        } else {
            self.next_node[self.tail[slot] as usize] = node;
        }
        self.tail[slot] = node;
        self.active += 1;
        Ok(())
    }
}
impl<K: Key> Index<K> for Arena<K> {
    const RECORD_BYTES: usize = 8;
    type Batch = u32;
    type Selected = u32;
    fn selected(&self, key: K) -> u32 {
        self.states[&key]
    }
    fn subtract_selected(&mut self, slot: u32, weight: u64) -> Result<(), TrainError> {
        decrease(&mut self.frequency[slot as usize], weight)
    }
    fn frequency(&self, key: K) -> u64 {
        self.states
            .get(&key)
            .map_or(0, |&s| self.frequency[s as usize])
    }
    fn add(&mut self, key: K, weight: u64) -> Result<(), TrainError> {
        let slot = self.state(key)? as usize;
        increase(&mut self.frequency[slot], weight)
    }
    fn subtract(&mut self, key: K, weight: u64) -> Result<(), TrainError> {
        if let Some(&slot) = self.states.get(&key) {
            decrease(&mut self.frequency[slot as usize], weight)?;
        }
        Ok(())
    }
    fn append(&mut self, key: K, pos: u32) -> Result<(), TrainError> {
        let slot = self.state(key)? as usize;
        self.append_slot(slot, pos)
    }
    fn record(&mut self, key: K, weight: u64, pos: u32) -> Result<(), TrainError> {
        let slot = self.state(key)? as usize;
        increase(&mut self.frequency[slot], weight)?;
        self.append_slot(slot, pos)
    }
    fn append_if_tracked(&mut self, key: K, pos: u32) -> Result<bool, TrainError> {
        if let Some(&slot) = self.states.get(&key) {
            self.append_slot(slot as usize, pos)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn detach(&mut self, key: K) -> u32 {
        let Some(&slot) = self.states.get(&key) else {
            return NIL;
        };
        let slot = slot as usize;
        let node = self.head[slot];
        self.head[slot] = NIL;
        self.tail[slot] = NIL;
        node
    }
    fn next(&mut self, batch: &mut u32) -> Option<u32> {
        if *batch == NIL {
            return None;
        }
        let node = *batch;
        *batch = self.next_node[node as usize];
        let pos = self.position[node as usize];
        self.recycle(node);
        Some(pos)
    }
    fn discard(&mut self, key: K, keep_frequency: bool) {
        let Some(&slot) = self.states.get(&key) else {
            return;
        };
        let mut node = self.detach(key);
        while self.next(&mut node).is_some() {}
        if !keep_frequency {
            self.states.remove(&key);
            self.frequency[slot as usize] = 0;
            self.free_states.push(slot);
        }
    }
    fn entries(&self) -> Vec<(K, u64)> {
        self.states
            .iter()
            .map(|(&k, &s)| (k, self.frequency[s as usize]))
            .collect()
    }
    fn occurrence_bytes(&self) -> usize {
        self.position.len() * 8
    }
    fn metrics(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            (
                "arena_occurrence_logical_bytes".into(),
                (self.position.len() * 8) as f64,
            ),
            (
                "position_capacity_bytes".into(),
                ((self.position.capacity() + self.next_node.capacity()) * 4) as f64,
            ),
            (
                "arena_state_capacity_bytes".into(),
                (self.frequency.capacity() * 8
                    + (self.head.capacity() + self.tail.capacity() + self.free_states.capacity())
                        * 4) as f64,
            ),
            (
                "arena_occurrence_high_water".into(),
                self.position.len() as f64,
            ),
            ("arena_occurrence_active".into(), self.active as f64),
            ("arena_occurrence_reuses".into(), self.node_reuses as f64),
            ("arena_state_high_water".into(), self.frequency.len() as f64),
            ("arena_state_reuses".into(), self.state_reuses as f64),
            (
                "arena_state_map_capacity".into(),
                self.states.capacity() as f64,
            ),
        ])
    }
}

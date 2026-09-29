use super::index::Key;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Entry<K> {
    key: K,
    frequency: u64,
}
impl<K: Ord> Ord for Entry<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.key.cmp(&self.key))
    }
}
impl<K: Ord> PartialOrd for Entry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Queue<K: Key> {
    heap: BinaryHeap<Entry<K>>,
    high: Vec<K>,
    low: Vec<Vec<K>>,
    dirty: Vec<bool>,
    max_low: Option<usize>,
    minimum: u64,
    threshold: u64,
    unit: u64,
    bucket: bool,
    fallback: bool,
    pub pops: usize,
    pushes: usize,
    high_visits: usize,
    low_pops: usize,
    bucket_steps: usize,
    low_sorts: usize,
}

impl<K: Key> Queue<K> {
    /// mode 0: heap; 1: literal high/low; 2: divide scores by weight GCD.
    pub fn new(
        entries: Vec<(K, u64)>,
        minimum: u64,
        mass: u64,
        mode: u8,
        weight_gcd: u64,
        positions: usize,
    ) -> Self {
        let unit = if mode == 2 { weight_gcd.max(1) } else { 1 };
        let threshold = (mass / unit).isqrt().max(1);
        // An adversarial large weight must not cause an allocation proportional
        // to sqrt(weighted mass). Preserve exact semantics with a heap fallback.
        // The cap depends on physical input size and a documented upper bound,
        // not on the benchmark language or measured performance.
        let cap = (positions as u64).saturating_mul(2).clamp(1024, 1_048_576);
        let fallback = mode != 0 && threshold > cap;
        let bucket = mode != 0 && !fallback;
        let size = if bucket { threshold as usize } else { 0 };
        let mut result = Self {
            heap: BinaryHeap::new(),
            high: Vec::new(),
            low: vec![Vec::new(); size],
            dirty: vec![false; size],
            max_low: None,
            minimum,
            threshold,
            unit,
            bucket,
            fallback,
            pops: 0,
            pushes: 0,
            high_visits: 0,
            low_pops: 0,
            bucket_steps: 0,
            low_sorts: 0,
        };
        if bucket {
            for (k, f) in entries {
                result.place(k, f);
            }
        } else {
            result.heap = BinaryHeap::from(
                entries
                    .into_iter()
                    .filter_map(|(key, frequency)| {
                        (frequency >= minimum).then_some(Entry { key, frequency })
                    })
                    .collect::<Vec<_>>(),
            );
        }
        result
    }
    fn place(&mut self, key: K, frequency: u64) {
        if frequency < self.minimum {
            return;
        }
        let score = frequency / self.unit;
        if score >= self.threshold {
            self.high.push(key);
        } else {
            let level = score as usize;
            self.low[level].push(key);
            self.dirty[level] = true;
            self.max_low = Some(self.max_low.map_or(level, |old| old.max(level)));
        }
    }
    pub fn add(&mut self, key: K, frequency: u64) {
        if frequency < self.minimum {
            return;
        }
        if self.bucket {
            self.place(key, frequency);
        } else {
            self.heap.push(Entry { key, frequency });
            self.pushes += 1;
        }
    }
    pub fn pop(&mut self, mut get: impl FnMut(K) -> u64) -> Option<(K, u64)> {
        if !self.bucket {
            while let Some(entry) = self.heap.pop() {
                self.pops += 1;
                let current = get(entry.key);
                if current < self.minimum {
                    continue;
                }
                if current != entry.frequency {
                    self.add(entry.key, current);
                    continue;
                }
                return Some((entry.key, current));
            }
            return None;
        }
        let mut index = 0;
        let mut best: Option<(usize, K, u64)> = None;
        while index < self.high.len() {
            let key = self.high[index];
            self.high_visits += 1;
            let current = get(key);
            if current < self.minimum || current / self.unit < self.threshold {
                self.place(key, current);
                self.high.swap_remove(index);
                continue;
            }
            if best.is_none_or(|(_, k, f)| current > f || (current == f && key < k)) {
                best = Some((index, key, current));
            }
            index += 1;
        }
        if let Some((index, key, current)) = best {
            self.high.swap_remove(index);
            return Some((key, current));
        }
        while let Some(level) = self.max_low {
            if (level as u64) < self.minimum.div_ceil(self.unit) {
                break;
            }
            if self.low[level].is_empty() {
                self.max_low = level.checked_sub(1);
                self.bucket_steps += 1;
                continue;
            }
            if self.dirty[level] {
                self.low[level].sort_unstable_by(|a, b| b.cmp(a));
                self.dirty[level] = false;
                self.low_sorts += 1;
            }
            let key = self.low[level].pop().unwrap();
            self.low_pops += 1;
            let current = get(key);
            if current >= self.minimum && current / self.unit == level as u64 {
                return Some((key, current));
            }
            self.place(key, current);
        }
        None
    }
    pub fn metrics(&self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("heap_pushes".into(), self.pushes as f64),
            ("high_scan_visits".into(), self.high_visits as f64),
            ("low_candidate_pops".into(), self.low_pops as f64),
            ("low_bucket_steps".into(), self.bucket_steps as f64),
            ("low_sort_calls".into(), self.low_sorts as f64),
            ("bucket_threshold".into(), self.threshold as f64),
            ("bucket_weight_unit".into(), self.unit as f64),
            (
                "bucket_heap_fallback".into(),
                u8::from(self.fallback) as f64,
            ),
            (
                "queue_capacity_bytes".into(),
                (self.heap.capacity() * std::mem::size_of::<Entry<K>>()
                    + self.high.capacity() * std::mem::size_of::<K>()
                    + self.low.capacity() * std::mem::size_of::<Vec<K>>()
                    + self
                        .low
                        .iter()
                        .map(|v| v.capacity() * std::mem::size_of::<K>())
                        .sum::<usize>()
                    + self.dirty.capacity() * std::mem::size_of::<bool>()) as f64,
            ),
        ])
    }
}

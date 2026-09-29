//! One byte per original position for topology only; token IDs live elsewhere.
use crate::TrainError;

pub struct ByteSpans {
    tags: Vec<u8>,
}

impl ByteSpans {
    pub fn new(n: usize) -> Result<Self, TrainError> {
        if n == 0 || (n as u128) >= (1_u128 << 32) {
            return Err(TrainError::InvalidInput(
                "ByteSpans needs 1..2^32-1 positions",
            ));
        }
        Ok(Self { tags: vec![1; n] })
    }

    pub fn len(&self) -> usize {
        self.tags.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn logical_bytes(&self) -> usize {
        self.tags.len()
    }
    pub fn capacity_bytes(&self) -> usize {
        self.tags.capacity()
    }

    fn decode_forward(&self, pos: usize) -> Option<u32> {
        let tag = *self.tags.get(pos)?;
        if (1..=63).contains(&tag) {
            return Some(u32::from(tag));
        }
        if tag != 126 || pos.checked_add(5)? >= self.tags.len() {
            return None;
        }
        let mut value = 0_u64;
        for j in 0..5 {
            let digit = self.tags[pos + 1 + j];
            if digit < 128 {
                return None;
            }
            value |= u64::from(digit & 127) << (7 * j);
        }
        u32::try_from(value).ok().filter(|&length| length >= 64)
    }

    fn decode_backward(&self, end: usize) -> Option<u32> {
        let tag = *self.tags.get(end)?;
        if tag == 1 {
            return Some(1);
        }
        if (64..=125).contains(&tag) {
            return Some(u32::from(tag - 62));
        }
        if tag != 127 || end < 5 {
            return None;
        }
        let mut value = 0_u64;
        for j in 0..5 {
            let digit = self.tags[end - 1 - j];
            if digit < 128 {
                return None;
            }
            value |= u64::from(digit & 127) << (7 * j);
        }
        u32::try_from(value).ok().filter(|&length| length >= 64)
    }

    pub fn length(&self, pos: usize) -> Option<u32> {
        self.decode_forward(pos)
    }

    pub fn next(&self, pos: usize) -> Option<usize> {
        let next = pos.checked_add(self.length(pos)? as usize)?;
        (next < self.len()).then_some(next)
    }

    pub fn prev(&self, pos: usize) -> Option<usize> {
        let end = pos.checked_sub(1)?;
        let length = self.decode_backward(end)? as usize;
        let start = pos.checked_sub(length)?;
        (self.length(start)? as usize == length).then_some(start)
    }

    pub fn merge_known(
        &mut self,
        pos: usize,
        right: usize,
        after: usize,
    ) -> Result<(), TrainError> {
        let left = self
            .length(pos)
            .ok_or(TrainError::InvalidInput("left is not a start"))? as usize;
        let right_len =
            self.length(right)
                .ok_or(TrainError::InvalidInput("right is not a start"))? as usize;
        if pos.checked_add(left) != Some(right)
            || right.checked_add(right_len) != Some(after)
            || after > self.len()
        {
            return Err(TrainError::InvalidInput(
                "ByteSpans merge requires adjacent spans",
            ));
        }
        let length = after - pos;
        let length_u32 = u32::try_from(length)
            .map_err(|_| TrainError::Overflow("ByteSpans length exceeds u32"))?;
        self.tags[right - 1] = 0;
        self.tags[right] = 0;
        let end = after - 1;
        if length <= 63 {
            self.tags[pos] = length as u8;
            self.tags[end] = length as u8 + 62;
        } else {
            self.tags[pos] = 126;
            self.tags[end] = 127;
            for j in 0..5 {
                let digit = 128 | (((length_u32 >> (7 * j)) & 127) as u8);
                self.tags[pos + 1 + j] = digit;
                self.tags[end - 1 - j] = digit;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directional_and_random_topology_matches_naive() {
        let mut seed = 0x829_u64;
        for &n in &[2, 3, 63, 64, 65, 127, 128, 255, 256, 513, 1024] {
            for mode in 0..3 {
                let mut spans = ByteSpans::new(n).unwrap();
                let mut lengths = vec![1_usize; n];
                while lengths.len() > 1 {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let k = match mode {
                        0 => 0,
                        1 => lengths.len() - 2,
                        _ => (seed as usize) % (lengths.len() - 1),
                    };
                    let pos: usize = lengths[..k].iter().sum();
                    let right = pos + lengths[k];
                    let after = right + lengths[k + 1];
                    spans.merge_known(pos, right, after).unwrap();
                    let joined = lengths[k] + lengths[k + 1];
                    lengths.splice(k..=k + 1, [joined]);
                    let mut current = 0;
                    let mut expected = vec![None; n];
                    for &length in &lengths {
                        expected[current] = Some(length as u32);
                        current += length;
                    }
                    for (p, &want) in expected.iter().enumerate() {
                        assert_eq!(spans.length(p), want, "n={n} mode={mode} p={p}");
                    }
                    let mut pos = 0;
                    let mut previous = None;
                    for &length in &lengths {
                        assert_eq!(spans.prev(pos), previous);
                        let expected_next = (pos + length < n).then_some(pos + length);
                        assert_eq!(spans.next(pos), expected_next);
                        previous = Some(pos);
                        pos += length;
                    }
                    assert_eq!(spans.logical_bytes(), n);
                }
            }
        }
    }

    #[test]
    fn rejects_nonadjacent_merge() {
        let mut spans = ByteSpans::new(4).unwrap();
        assert!(spans.merge_known(0, 2, 3).is_err());
        assert_eq!(spans.length(0), Some(1));
    }

    #[test]
    fn long_lengths_cross_u8_and_u16_boundaries() {
        let n = 65_536;
        let mut spans = ByteSpans::new(n).unwrap();
        for length in 2..=n {
            spans.merge_known(0, length - 1, length).unwrap();
            if [63, 64, 255, 256, 65535, 65536].contains(&length) {
                assert_eq!(spans.length(0), Some(length as u32));
                assert_eq!(spans.next(0), (length < n).then_some(length));
                if length < n {
                    assert_eq!(spans.prev(length), Some(0));
                }
            }
        }
        assert_eq!(spans.logical_bytes(), n);
        assert_eq!(spans.length(n - 1), None);
    }
}

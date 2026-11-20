const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const MIX_FIRST: u64 = 0xbf58_476d_1ce4_e5b9;
const MIX_SECOND: u64 = 0x94d0_49bb_1331_11eb;
const UNIT_SCALE: f64 = 1.0 / 4_294_967_296.0;

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_FIRST);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_SECOND);
        z ^ (z >> 31)
    }

    pub(crate) fn unit(&mut self) -> f64 {
        let high = u32::try_from(self.next_u64() >> 32).unwrap_or(u32::MAX);
        f64::from(high) * UNIT_SCALE
    }

    pub(crate) fn below(&mut self, bound: usize) -> usize {
        let Ok(bound) = u64::try_from(bound) else {
            return 0;
        };
        let value = self.next_u64().checked_rem(bound).unwrap_or(0);
        usize::try_from(value).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_outputs_for_seed_zero_match_the_reference() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    }

    #[test]
    fn values_stay_within_their_bounds() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..1000 {
            assert!((0.0..1.0).contains(&rng.unit()));
            assert!(rng.below(5) < 5);
        }
        assert_eq!(rng.below(0), 0);
    }
}

use cosmosearch_core::Signal;

use crate::table::{SIGNALS, slot};
use crate::{Error, Result};

/// What one count of each signal is worth, as starting values to tune for your traffic.
/// A search only passed over the shard, and a hit or a warm hint says little more; a read
/// means someone used a passage from it; unmet demand is a search that ended inconclusive
/// without reaching the shard.
const DEFAULT_WEIGHTS: [(Signal, f64); 5] = [
    (Signal::Search, 1.0),
    (Signal::Hit, 1.0),
    (Signal::Warm, 1.0),
    (Signal::Read, 3.0),
    (Signal::Unmet, 5.0),
];

/// The share of each neighbouring shard's heat a shard is credited with: enough that a hot
/// run of keys lifts the shards beside it, and too little to promote a shard nobody uses.
const DEFAULT_NEIGHBOUR_SHARE: f64 = 0.1;

/// Heat for each mebibyte of cost at which a shard is promoted. A starting value: a shard of
/// 1 GiB needs about 51, such as 17 reads within a half-life.
const DEFAULT_PROMOTE_AT: f64 = 0.05;

/// Heat for each mebibyte of cost below which a promoted shard is demoted: a fifth of the
/// promotion threshold, so heat must decay for more than two half-lives in between.
const DEFAULT_DEMOTE_BELOW: f64 = 0.01;

/// How a [`ThresholdPolicy`] values heat and where it promotes and demotes.
///
/// [`ThresholdPolicy`]: crate::ThresholdPolicy
#[derive(Debug, Clone, PartialEq)]
pub struct PolicySettings {
    pub(crate) weights: [f64; SIGNALS],
    pub(crate) neighbour_share: f64,
    pub(crate) promote_at: f64,
    pub(crate) demote_below: f64,
}

impl Default for PolicySettings {
    fn default() -> Self {
        let mut weights = [0.0; SIGNALS];
        for (signal, weight) in DEFAULT_WEIGHTS {
            if let Some(held) = slot(signal).and_then(|slot| weights.get_mut(slot)) {
                *held = weight;
            }
        }
        PolicySettings {
            weights,
            neighbour_share: DEFAULT_NEIGHBOUR_SHARE,
            promote_at: DEFAULT_PROMOTE_AT,
            demote_below: DEFAULT_DEMOTE_BELOW,
        }
    }
}

impl PolicySettings {
    /// Weights of 1 for a search, a hit and a warm hint, 3 for a read and 5 for unmet demand,
    /// a neighbour share of 0.1, promotion at 0.05 and demotion below 0.01 heat per mebibyte.
    pub fn new() -> Self {
        PolicySettings::default()
    }

    /// Sets what one count of `signal` is worth.
    #[must_use]
    pub fn weight(mut self, signal: Signal, weight: f64) -> Self {
        if let Some(held) = slot(signal).and_then(|slot| self.weights.get_mut(slot)) {
            *held = weight;
        }
        self
    }

    /// Sets the share of each neighbouring shard's heat a shard is credited with; zero turns
    /// neighbour heat off.
    #[must_use]
    pub fn neighbour_share(mut self, share: f64) -> Self {
        self.neighbour_share = share;
        self
    }

    /// Sets the heat for each mebibyte of cost at which a shard is promoted, and the lower
    /// one below which a promoted shard is demoted. The gap between them keeps a shard from
    /// moving up and down as its heat hovers.
    #[must_use]
    pub fn thresholds(mut self, promote_at: f64, demote_below: f64) -> Self {
        self.promote_at = promote_at;
        self.demote_below = demote_below;
        self
    }

    pub(crate) fn check(&self) -> Result<()> {
        let usable = |value: f64| value.is_finite() && value >= 0.0;
        if !self.weights.iter().copied().all(usable) {
            return Err(invalid("a signal's weight must be finite and not negative"));
        }
        if !usable(self.neighbour_share) {
            return Err(invalid(
                "the neighbour share must be finite and not negative",
            ));
        }
        if !usable(self.demote_below) || !self.promote_at.is_finite() {
            return Err(invalid("the thresholds must be finite and not negative"));
        }
        if self.promote_at <= self.demote_below {
            return Err(invalid(
                "the promotion threshold must be above the demotion threshold",
            ));
        }
        Ok(())
    }
}

fn invalid(reason: &'static str) -> Error {
    Error::InvalidSettings { reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_settings_that_could_flap_or_are_not_numbers() {
        PolicySettings::new().check().unwrap();
        for settings in [
            PolicySettings::new().thresholds(1.0, 1.0),
            PolicySettings::new().thresholds(1.0, 2.0),
            PolicySettings::new().thresholds(f64::INFINITY, 1.0),
            PolicySettings::new().thresholds(1.0, -1.0),
            PolicySettings::new().weight(Signal::Read, f64::NAN),
            PolicySettings::new().weight(Signal::Read, -1.0),
            PolicySettings::new().neighbour_share(-0.1),
        ] {
            assert!(
                matches!(settings.check(), Err(Error::InvalidSettings { .. })),
                "{settings:?}"
            );
        }
    }
}

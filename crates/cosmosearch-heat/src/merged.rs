use std::collections::BTreeMap;

use cosmosearch_store::ObjectKey;

use crate::codec::MergedParts;
use crate::{HalfLife, Heat, Result, WriterId};

pub(crate) const MERGED: &str = "merged";

/// A merged version of an index's heat: every writer's snapshots up to the newest each one
/// had when it was merged.
#[derive(Debug, Clone)]
pub struct MergedHeat {
    version: u64,
    clock: u64,
    writers: BTreeMap<WriterId, u64>,
    heat: Heat,
}

impl MergedHeat {
    pub(crate) fn new(parts: MergedParts, half_life: HalfLife) -> Self {
        MergedHeat {
            version: parts.version,
            clock: parts.clock,
            writers: parts.writers,
            heat: Heat::from_table(parts.table, half_life),
        }
    }

    pub(crate) fn empty(half_life: HalfLife) -> Self {
        MergedHeat::new(MergedParts::default(), half_life)
    }

    /// The version, 0 before the first merge.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The merged heat.
    pub fn heat(&self) -> &Heat {
        &self.heat
    }

    /// Takes the merged heat.
    pub fn into_heat(self) -> Heat {
        self.heat
    }

    pub(crate) fn clock_ms(&self) -> u64 {
        self.clock
    }

    /// The highest snapshot of `writer` this version holds, if it holds any and still
    /// tracks the writer.
    pub(crate) fn merged_through(&self, writer: &WriterId) -> Option<u64> {
        self.writers.get(writer).copied()
    }

    pub(crate) fn into_parts(self) -> MergedParts {
        MergedParts {
            version: self.version,
            clock: self.clock,
            writers: self.writers,
            table: self.heat.into_table(),
        }
    }
}

/// The key of merged version `version` within the heat scope: `merged/v<version>`.
pub(crate) fn version_key(version: u64) -> Result<ObjectKey> {
    Ok(ObjectKey::new(format!("{MERGED}/v{version:012}"))?)
}

/// The merged version a key within the heat scope names, if it names one.
pub(crate) fn parse_version_key(key: &str) -> Option<u64> {
    let digits = key.strip_prefix(MERGED)?.strip_prefix("/v")?;
    let version: u64 = digits.parse().ok()?;
    (format!("{version:012}") == digits && version > 0).then_some(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_keys_parse_back() {
        let key = version_key(12).unwrap();
        assert_eq!(key.as_str(), "merged/v000000000012");
        assert_eq!(parse_version_key(key.as_str()), Some(12));
        for key in [
            "merged/v12",
            "merged/v000000000000",
            "merged/000000000012",
            "deltas/v000000000012",
        ] {
            assert_eq!(parse_version_key(key), None, "{key}");
        }
    }
}

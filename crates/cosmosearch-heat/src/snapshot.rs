use cosmosearch_store::ObjectKey;

use crate::{Result, WriterId};

pub(crate) const DELTAS: &str = "deltas";

/// One heat snapshot of a writer: the heat it recorded since its previous one. A writer's
/// snapshots are numbered from 1 with no gaps.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Snapshot {
    writer: WriterId,
    seq: u64,
}

impl Snapshot {
    pub(crate) fn new(writer: WriterId, seq: u64) -> Self {
        Snapshot { writer, seq }
    }

    /// The writer that wrote it.
    pub fn writer(&self) -> &WriterId {
        &self.writer
    }

    /// Its number among the writer's snapshots.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Its key within the heat scope: `deltas/<writer id>/<seq>`.
    pub(crate) fn key(&self) -> Result<ObjectKey> {
        Ok(ObjectKey::new(format!(
            "{DELTAS}/{}/{:012}",
            self.writer, self.seq
        ))?)
    }

    /// The snapshot a key within the heat scope names, if it names one.
    pub(crate) fn parse(key: &str) -> Option<Self> {
        let (writer, seq) = key
            .strip_prefix(DELTAS)?
            .strip_prefix('/')?
            .split_once('/')?;
        let seq_value: u64 = seq.parse().ok()?;
        if format!("{seq_value:012}") != seq || seq_value == 0 {
            return None;
        }
        Some(Snapshot::new(WriterId::parse(writer)?, seq_value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse_back() {
        let writer = WriterId::new(1_800_000_000_000).unwrap();
        let snapshot = Snapshot::new(writer.clone(), 42);
        let key = snapshot.key().unwrap();
        assert_eq!(key.as_str(), format!("deltas/{writer}/000000000042"));
        assert_eq!(Snapshot::parse(key.as_str()), Some(snapshot));
        for key in [
            format!("deltas/{writer}/42"),
            format!("deltas/{writer}/000000000000"),
            format!("deltas/{writer}"),
            format!("merged/{writer}/000000000001"),
            "deltas/not-a-writer/000000000001".to_owned(),
        ] {
            assert_eq!(Snapshot::parse(&key), None, "{key}");
        }
    }
}

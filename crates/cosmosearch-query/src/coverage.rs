use std::collections::BTreeSet;

use cosmosearch_core::{Filter, Scope, ShardId};
use cosmosearch_format::{FileEntry, FileReason, FileStatus, FileTable};

/// How many files of a file table in a search's scope that pass its filter are in each state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    searched: u64,
    partly: u64,
    pending: u64,
    stale: u64,
    failed: u64,
    unsupported: u64,
}

impl Counts {
    /// The files of `table` in `scope` that pass `filter`, by state.
    pub(crate) fn of(table: &FileTable, scope: &Scope, filter: &Filter) -> Self {
        let mut counts = Counts::default();
        for file in table.iter() {
            if scope.contains(file.key()) && filter.matches(file.attributes()) {
                counts.count(file);
            }
        }
        counts
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "a file state added later counts as not searched until coverage decides on it"
    )]
    fn count(&mut self, file: &FileEntry) {
        let stale = matches!(file.reason(), Some(FileReason::SourceMoved));
        let slot = match file.status() {
            FileStatus::Ready => &mut self.searched,
            FileStatus::Pending if stale => &mut self.stale,
            FileStatus::Failed => &mut self.failed,
            FileStatus::Unsupported => &mut self.unsupported,
            _ => &mut self.pending,
        };
        *slot = slot.saturating_add(1);
        if file.status() == FileStatus::Ready && file.unread_parts() > 0 {
            self.partly = self.partly.saturating_add(1);
        }
    }

    /// Whether any file was counted.
    pub(crate) fn any(&self) -> bool {
        *self != Counts::default()
    }

    /// These counts and `other`'s together.
    pub(crate) fn plus(self, other: Counts) -> Counts {
        Counts {
            searched: self.searched.saturating_add(other.searched),
            partly: self.partly.saturating_add(other.partly),
            pending: self.pending.saturating_add(other.pending),
            stale: self.stale.saturating_add(other.stale),
            failed: self.failed.saturating_add(other.failed),
            unsupported: self.unsupported.saturating_add(other.unsupported),
        }
    }
}

/// What a search looked at: the files in its scope that pass its filter, by state, in the
/// shards it opened, and the shards of its scope it did not search.
///
/// The files of a shard it did not open are not counted, since counting them would mean
/// opening the shard.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Coverage {
    counts: Counts,
    unread: BTreeSet<ShardId>,
    unexplored: BTreeSet<ShardId>,
}

impl Coverage {
    pub(crate) fn new(
        counts: Counts,
        unread: BTreeSet<ShardId>,
        unexplored: BTreeSet<ShardId>,
    ) -> Self {
        Coverage {
            counts,
            unread,
            unexplored,
        }
    }

    /// Files whose current content was searched.
    pub fn searched(&self) -> u64 {
        self.counts.searched
    }

    /// Of the searched files, those searched only in part, because their extraction could
    /// not read every part, such as PDF pages that hold only an image.
    pub fn partly_searched(&self) -> u64 {
        self.counts.partly
    }

    /// Files waiting for an indexing job, so not searched.
    pub fn pending(&self) -> u64 {
        self.counts.pending
    }

    /// Files whose recorded version is gone from the data location, so not searched until a
    /// newer change is indexed.
    pub fn stale(&self) -> u64 {
        self.counts.stale
    }

    /// Files whose extraction failed, so not searched.
    pub fn failed(&self) -> u64 {
        self.counts.failed
    }

    /// Files no extractor reads or with no text. They cannot be searched, so they do not make
    /// coverage incomplete.
    pub fn unsupported(&self) -> u64 {
        self.counts.unsupported
    }

    /// Shards left unread because a segment of theirs stayed missing. Each needs a rebuild.
    pub fn unread(&self) -> &BTreeSet<ShardId> {
        &self.unread
    }

    /// Shards in scope the search did not search: those beyond its shard limit and, when its
    /// time ran out before any hit was found, those it was searching.
    pub fn unexplored(&self) -> &BTreeSet<ShardId> {
        &self.unexplored
    }

    /// Whether every file the search could search was searched whole: none pending, stale,
    /// failed or searched only in part, and no shard unread or unexplored.
    pub fn is_complete(&self) -> bool {
        self.counts.partly == 0
            && self.counts.pending == 0
            && self.counts.stale == 0
            && self.counts.failed == 0
            && self.unread.is_empty()
            && self.unexplored.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use cosmosearch_core::{AttributeValue, Attributes, ETag, ExtractFailure, ManifestVersion};

    use super::*;

    fn table() -> FileTable {
        let mut table = FileTable::new();
        let etag = ETag::new("9f3a").unwrap();
        for key in [
            "docs/a.txt",
            "docs/b.txt",
            "docs/c.txt",
            "docs/d.txt",
            "docs/e.txt",
            "logs/f.txt",
        ] {
            table.insert(FileEntry::pending(
                key,
                5,
                etag.clone(),
                ManifestVersion::FIRST,
            ));
        }
        table.mark_ready("docs/a.txt", 0);
        table.mark_ready("logs/f.txt", 2);
        table.record("docs/b.txt", FileReason::SourceMoved);
        table.record(
            "docs/c.txt",
            FileReason::Extraction(ExtractFailure::Malformed),
        );
        table.record("docs/d.txt", FileReason::Extraction(ExtractFailure::NoText));
        let legal =
            Attributes::from([("owner".to_owned(), AttributeValue::One("legal".to_owned()))]);
        table.describe("docs/a.txt", None, legal);
        table
    }

    #[test]
    fn counts_files_in_scope_that_pass_the_filter_by_state() {
        let docs = Counts::of(&table(), &Scope::prefix("docs/"), &Filter::new());
        let expected = Counts {
            searched: 1,
            pending: 1,
            stale: 1,
            failed: 1,
            unsupported: 1,
            ..Counts::default()
        };
        assert_eq!(docs, expected);
        let all = Counts::of(&table(), &Scope::all(), &Filter::new());
        assert_eq!(all.searched, 2);
        assert_eq!(all.partly, 1, "the log file was read only in part");
        let legal = Counts::of(
            &table(),
            &Scope::all(),
            &Filter::new().equals("owner", "legal"),
        );
        assert_eq!(
            legal,
            Counts {
                searched: 1,
                ..Counts::default()
            }
        );
        assert!(!Counts::of(&table(), &Scope::prefix("none/"), &Filter::new()).any());
        assert_eq!(docs.plus(legal).searched, 2);
    }

    #[test]
    fn coverage_is_complete_only_with_nothing_left_unseen() {
        let shard = || BTreeSet::from([ShardId::new("a").unwrap()]);
        let done = Counts {
            searched: 3,
            unsupported: 2,
            ..Counts::default()
        };
        assert!(Coverage::new(done, BTreeSet::new(), BTreeSet::new()).is_complete());
        assert!(!Coverage::new(done, shard(), BTreeSet::new()).is_complete());
        assert!(!Coverage::new(done, BTreeSet::new(), shard()).is_complete());
        for unseen in [
            Counts { pending: 1, ..done },
            Counts { stale: 1, ..done },
            Counts { failed: 1, ..done },
            Counts { partly: 1, ..done },
        ] {
            assert!(!Coverage::new(unseen, BTreeSet::new(), BTreeSet::new()).is_complete());
        }
    }
}

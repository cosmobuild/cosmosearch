use cosmosearch_core::{Budget, Filter, Query, Scope};

use crate::{Error, Result};

/// How many hits a search returns unless set.
const DEFAULT_LIMIT: usize = 10;

/// One search: what to look for, where, among which files, how many hits to return and what
/// it may spend.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub(crate) query: Query,
    pub(crate) scope: Scope,
    pub(crate) filter: Filter,
    pub(crate) limit: usize,
    pub(crate) budget: Budget,
}

impl SearchRequest {
    /// The most hits a search may ask for. Each index plug-in sets aside room for its
    /// candidates in every segment it searches, so a larger limit would let one request
    /// claim memory without bound.
    pub const MAX_LIMIT: usize = 1_000;

    /// A search for `query` over every file, with no filter and no limit on what it spends,
    /// for 10 hits.
    pub fn new(query: Query) -> Self {
        SearchRequest {
            query,
            scope: Scope::all(),
            filter: Filter::new(),
            limit: DEFAULT_LIMIT,
            budget: Budget::new(),
        }
    }

    /// Searches only the files in `scope`. Shards holding none of its keys are not opened.
    #[must_use]
    pub fn scope(mut self, scope: Scope) -> Self {
        self.scope = scope;
        self
    }

    /// Searches only the files that pass `filter`.
    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = filter;
        self
    }

    /// Returns at most `limit` hits. A search asking for more than [`Self::MAX_LIMIT`] is
    /// refused.
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Spends at most `budget`. The shards of the scope beyond its shard limit, taken in key
    /// order, are not opened and are reported unexplored. When its time runs out the search
    /// returns what it has and says so.
    #[must_use]
    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    pub(crate) fn check(&self) -> Result<()> {
        if self.limit > Self::MAX_LIMIT {
            return Err(Error::LimitTooLarge { limit: self.limit });
        }
        Ok(())
    }
}

/// How a search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The search returned passages. Whether they answer the question is for the caller to
    /// judge; the search's coverage says what it looked at.
    Found,
    /// The search returned no passage. That is no evidence the answer is absent: its
    /// coverage says which files and shards it did not see.
    Inconclusive,
}

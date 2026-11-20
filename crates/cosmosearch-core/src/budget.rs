use std::time::Duration;

/// What one search may spend: how many shards it may open and how long it may run. Nothing
/// is limited unless set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Budget {
    shards: Option<usize>,
    time: Option<Duration>,
}

impl Budget {
    /// A budget that limits nothing.
    pub fn new() -> Self {
        Budget::default()
    }

    /// Opens at most `shards` shards; zero opens none.
    #[must_use]
    pub fn max_shards(mut self, shards: usize) -> Self {
        self.shards = Some(shards);
        self
    }

    /// Runs for at most `time` from the start of the search, including any refresh of the
    /// manifest it waits on.
    #[must_use]
    pub fn max_time(mut self, time: Duration) -> Self {
        self.time = Some(time);
        self
    }

    /// How many shards a search may open, if limited.
    pub fn shards(&self) -> Option<usize> {
        self.shards
    }

    /// How long a search may run, if limited.
    pub fn time(&self) -> Option<Duration> {
        self.time
    }
}

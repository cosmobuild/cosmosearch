//! Records file changes for cosmosearch and turns them into index work.

#![warn(clippy::arithmetic_side_effects)]

mod change;
mod error;
mod indexer;
mod job;
mod live_vectors;
mod promotion;
mod state;

pub use change::{Applied, AttributeChange, ChangeOutcome, FileChange, FileUpsert, StoreFields};
pub use error::{Error, Result};
pub use indexer::Indexer;
pub use job::{
    Batch, BuiltSegment, ExtractOutcome, ExtractedFile, FileJob, Plan, PublishOutcome, Rebuild,
};
pub use promotion::Promotion;
pub use state::FileState;

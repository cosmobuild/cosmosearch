use serde::{Deserialize, Serialize};

/// What one call spent on storage and models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Usage {
    object_reads: u64,
    bytes: u64,
    tokens: u64,
}

impl Usage {
    /// Nothing spent.
    pub fn new() -> Self {
        Usage::default()
    }

    /// Usage of `object_reads` requests that returned `bytes` in all.
    pub fn of_reads(object_reads: u64, bytes: u64) -> Self {
        Usage {
            object_reads,
            bytes,
            tokens: 0,
        }
    }

    /// Usage of a model reading `tokens` tokens.
    pub fn of_tokens(tokens: u64) -> Self {
        Usage {
            tokens,
            ..Usage::default()
        }
    }

    /// Requests that read an object or part of one.
    pub fn object_reads(&self) -> u64 {
        self.object_reads
    }

    /// Bytes those requests returned.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Tokens a model read.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// This usage and `other` together, each count stopping at its largest value.
    #[must_use]
    pub fn saturating_add(self, other: Usage) -> Usage {
        Usage {
            object_reads: self.object_reads.saturating_add(other.object_reads),
            bytes: self.bytes.saturating_add(other.bytes),
            tokens: self.tokens.saturating_add(other.tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_every_count_and_saturates() {
        let reads = Usage::of_reads(2, 100);
        let tokens = Usage::of_tokens(7);
        let both = reads.saturating_add(tokens).saturating_add(reads);
        assert_eq!(
            (both.object_reads(), both.bytes(), both.tokens()),
            (4, 200, 7)
        );
        let full = Usage::of_tokens(u64::MAX).saturating_add(tokens);
        assert_eq!(full.tokens(), u64::MAX);
    }
}

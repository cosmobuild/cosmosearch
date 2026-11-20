use std::fmt;

use serde::{Deserialize, Serialize};

use crate::SourceRef;

/// The BLAKE3 hash of what an index reads for a chunk, normalized: its headings and its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkHash([u8; 32]);

impl ChunkHash {
    /// Hashes `text` with runs of whitespace collapsed and the ends trimmed.
    pub fn of(text: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        let mut words = text.split_whitespace();
        if let Some(first) = words.next() {
            hasher.update(first.as_bytes());
        }
        for word in words {
            hasher.update(b" ");
            hasher.update(word.as_bytes());
        }
        ChunkHash(*hasher.finalize().as_bytes())
    }

    /// A hash read back from storage.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        ChunkHash(bytes)
    }

    /// The raw digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ChunkHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

/// What an index reads for a passage: the headings above it, outermost first, one to a line,
/// then its text, so a passage deep in a long section still carries its section's title.
pub fn indexed_text(heading_path: &[String], text: &str) -> String {
    let mut indexed = heading_path.join("\n");
    if !indexed.is_empty() {
        indexed.push('\n');
    }
    indexed.push_str(text);
    indexed
}

#[derive(Serialize, Deserialize)]
struct ChunkFields {
    source: SourceRef,
    heading_path: Vec<String>,
    text: String,
}

impl From<ChunkFields> for Chunk {
    fn from(fields: ChunkFields) -> Self {
        Chunk::new(fields.source, fields.heading_path, fields.text)
    }
}

impl From<Chunk> for ChunkFields {
    fn from(chunk: Chunk) -> Self {
        ChunkFields {
            source: chunk.source,
            heading_path: chunk.heading_path,
            text: chunk.text,
        }
    }
}

/// A passage of a file with its ref and hash. No text exists without its ref.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "ChunkFields", into = "ChunkFields")]
pub struct Chunk {
    source: SourceRef,
    heading_path: Vec<String>,
    text: String,
    hash: ChunkHash,
}

impl Chunk {
    pub(crate) fn new(source: SourceRef, heading_path: Vec<String>, text: String) -> Self {
        let hash = ChunkHash::of(&indexed_text(&heading_path, &text));
        Chunk {
            source,
            heading_path,
            text,
            hash,
        }
    }

    /// Where this text came from.
    pub fn source(&self) -> &SourceRef {
        &self.source
    }

    /// The headings above this text, outermost first.
    pub fn heading_path(&self) -> &[String] {
        &self.heading_path
    }

    /// The text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// What an index reads for this chunk: its headings, one to a line, then its text.
    pub fn indexed_text(&self) -> String {
        indexed_text(&self.heading_path, &self.text)
    }

    /// The hash of what an index reads, so equal hashes mean equal headings and text.
    pub fn hash(&self) -> ChunkHash {
        self.hash
    }
}

impl fmt::Debug for Chunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chunk")
            .field("source", &self.source)
            .field("hash", &self.hash)
            .field("text_bytes", &self.text.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_published_blake3_value() {
        assert_eq!(
            ChunkHash::of("hello world").to_string(),
            "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24"
        );
    }

    #[test]
    fn ignores_differences_in_whitespace_only() {
        let base = ChunkHash::of("refund policy applies");
        for same in [
            "  refund policy applies\n",
            "refund\tpolicy\r\n  applies",
            "refund\u{a0}policy\u{2003}applies",
        ] {
            assert_eq!(ChunkHash::of(same), base, "{same:?}");
        }
        for different in [
            "Refund policy applies",
            "refundpolicy applies",
            "refund policy",
        ] {
            assert_ne!(ChunkHash::of(different), base, "{different:?}");
        }
    }

    #[test]
    fn a_stored_chunk_recomputes_its_hash() {
        let json = r#"{"source":"s3://example-bucket/a.txt#bytes=0-5@etag=e","heading_path":[],"text":"hello"}"#;
        let chunk: Chunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.hash(), ChunkHash::of("hello"));
        let again: Chunk = serde_json::from_str(&serde_json::to_string(&chunk).unwrap()).unwrap();
        assert_eq!(again, chunk);
    }

    #[test]
    fn headings_lead_the_indexed_text_and_change_the_hash() {
        let under = |headings: &[&str]| -> Chunk {
            let json = serde_json::json!({
                "source": "s3://example-bucket/a.txt#bytes=0-5@etag=e",
                "heading_path": headings,
                "text": "Total assets 352,583",
            });
            serde_json::from_value(json).unwrap()
        };
        let sheet = under(&["Annual Report", "Balance Sheets"]);
        assert_eq!(
            sheet.indexed_text(),
            "Annual Report\nBalance Sheets\nTotal assets 352,583"
        );
        assert_eq!(under(&[]).indexed_text(), "Total assets 352,583");
        assert_eq!(under(&[]).hash(), ChunkHash::of("Total assets 352,583"));
        assert_ne!(sheet.hash(), under(&[]).hash());
        assert_ne!(sheet.hash(), under(&["Risk Factors"]).hash());
    }

    #[test]
    fn blank_text_has_one_hash() {
        assert_eq!(ChunkHash::of(" \n\t"), ChunkHash::of(""));
    }
}

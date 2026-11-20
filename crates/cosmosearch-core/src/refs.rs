use std::fmt;
use std::num::{NonZeroU32, NonZeroU64};
use std::ops::Range;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{ETag, ElementPath, Error, Location, Result};

/// A half-open byte range `start..end` within an object, never empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ByteRange {
    start: u64,
    end: u64,
}

impl ByteRange {
    /// Creates the range `start..end`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRange`] unless `start < end`.
    pub fn new(start: u64, end: u64) -> Result<Self> {
        if start < end {
            Ok(ByteRange { start, end })
        } else {
            Err(Error::InvalidRange { start, end })
        }
    }

    /// First byte, inclusive.
    pub fn start(self) -> u64 {
        self.start
    }

    /// End of the range, exclusive.
    pub fn end(self) -> u64 {
        self.end
    }

    /// Number of bytes in the range, at least 1.
    pub fn len(self) -> u64 {
        self.end.abs_diff(self.start)
    }

    /// Always `false`: a `ByteRange` holds at least one byte.
    pub fn is_empty(self) -> bool {
        false
    }
}

impl From<ByteRange> for Range<u64> {
    fn from(range: ByteRange) -> Self {
        range.start..range.end
    }
}

/// An inclusive run of records in a table, counting from 1, never empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RowRange {
    first: NonZeroU64,
    last: NonZeroU64,
}

impl RowRange {
    /// Creates the run from record `first` to record `last`, both included.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRows`] if `last` comes before `first`.
    pub fn new(first: NonZeroU64, last: NonZeroU64) -> Result<Self> {
        if first <= last {
            Ok(RowRange { first, last })
        } else {
            Err(Error::InvalidRows {
                first: first.get(),
                last: last.get(),
            })
        }
    }

    /// The first record.
    pub fn first(self) -> NonZeroU64 {
        self.first
    }

    /// The last record, included.
    pub fn last(self) -> NonZeroU64 {
        self.last
    }
}

/// Where a passage sits within its source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Locator {
    /// A byte range, for files read as raw bytes.
    Bytes(ByteRange),
    /// A page, counting from 1. It identifies a page, not an exact passage.
    Page(NonZeroU32),
    /// A paragraph of a word-processing document, where its passage starts.
    Element(ElementPath),
    /// The records of a table a passage holds, counting from 1 after the header.
    Rows(RowRange),
    /// A value in a structured document, counting every value from 1 in document order, where
    /// its passage starts.
    Value(NonZeroU64),
}

impl fmt::Display for Locator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Locator::Bytes(range) => write!(f, "bytes={}-{}", range.start, range.end),
            Locator::Page(page) => write!(f, "page={page}"),
            Locator::Element(path) => write!(f, "element={path}"),
            Locator::Rows(rows) => write!(f, "rows={}-{}", rows.first, rows.last),
            Locator::Value(value) => write!(f, "value={value}"),
        }
    }
}

/// A passage of one version of one file, such as `s3://bucket/key#bytes=START-END@etag=ETAG`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SourceRef {
    location: Location,
    locator: Locator,
    etag: ETag,
}

impl SourceRef {
    /// Creates a ref.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRef`] if the location has no key.
    pub fn new(location: Location, locator: Locator, etag: ETag) -> Result<Self> {
        if location.key().is_empty() {
            return Err(Error::InvalidRef {
                input: location.to_string(),
                reason: "the key is empty",
            });
        }
        Ok(SourceRef {
            location,
            locator,
            etag,
        })
    }

    /// The source file.
    pub fn location(&self) -> &Location {
        &self.location
    }

    /// Where the passage sits within the file.
    pub fn locator(&self) -> Locator {
        self.locator
    }

    /// The ETag of the file version the locator was taken from.
    pub fn etag(&self) -> &ETag {
        &self.etag
    }

    /// Whether this ref points into the file version with the given ETag.
    pub fn is_current(&self, etag: &ETag) -> bool {
        &self.etag == etag
    }
}

impl fmt::Display for SourceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}@etag={}", self.location, self.locator, self.etag)
    }
}

fn parse_locator(text: &str) -> std::result::Result<Locator, &'static str> {
    if let Some(range) = text.strip_prefix("bytes=") {
        let (start, end) = range
            .split_once('-')
            .ok_or("byte range must be `START-END`")?;
        let start = parse_canonical_u64(start)
            .ok_or("range start must be a number without leading zeros")?;
        let end =
            parse_canonical_u64(end).ok_or("range end must be a number without leading zeros")?;
        let range =
            ByteRange::new(start, end).map_err(|_cause| "range start must be below its end")?;
        Ok(Locator::Bytes(range))
    } else if let Some(page) = text.strip_prefix("page=") {
        parse_canonical_u64(page)
            .and_then(|page| u32::try_from(page).ok())
            .and_then(NonZeroU32::new)
            .map(Locator::Page)
            .ok_or("page must be a number from 1 without leading zeros")
    } else if let Some(path) = text.strip_prefix("element=") {
        ElementPath::parse(path)
            .map(Locator::Element)
            .ok_or("element must be `bN/pN` or `bN/rN/cN/pN`, numbers from 1")
    } else if let Some(rows) = text.strip_prefix("rows=") {
        let reason = "rows must be `FIRST-LAST`, numbers from 1 without leading zeros";
        let (first, last) = rows.split_once('-').ok_or(reason)?;
        let number = |part| parse_canonical_u64(part).and_then(NonZeroU64::new);
        let range = RowRange::new(number(first).ok_or(reason)?, number(last).ok_or(reason)?)
            .map_err(|_cause| "the first row must not come after the last")?;
        Ok(Locator::Rows(range))
    } else if let Some(value) = text.strip_prefix("value=") {
        parse_canonical_u64(value)
            .and_then(NonZeroU64::new)
            .map(Locator::Value)
            .ok_or("value must be a number from 1 without leading zeros")
    } else {
        Err("locator must be `bytes=`, `page=`, `element=`, `rows=` or `value=`")
    }
}

impl FromStr for Locator {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self> {
        parse_locator(input).map_err(|reason| Error::InvalidRef {
            input: input.to_owned(),
            reason,
        })
    }
}

impl FromStr for SourceRef {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self> {
        let invalid = |reason| Error::InvalidRef {
            input: input.to_owned(),
            reason,
        };
        let (location, fragment) = input
            .split_once('#')
            .ok_or_else(|| invalid("missing `#` and its locator"))?;
        let location: Location = location.parse()?;
        let (locator, etag) = fragment
            .split_once("@etag=")
            .ok_or_else(|| invalid("fragment must be `LOCATOR@etag=ETAG`"))?;
        let locator = parse_locator(locator).map_err(invalid)?;
        SourceRef::new(location, locator, ETag::new(etag)?)
    }
}

impl TryFrom<String> for SourceRef {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

impl From<SourceRef> for String {
    fn from(source_ref: SourceRef) -> Self {
        source_ref.to_string()
    }
}

pub(crate) fn parse_canonical_u64(s: &str) -> Option<u64> {
    let canonical =
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && (s == "0" || !s.starts_with('0'));
    if canonical { s.parse().ok() } else { None }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::Scheme;

    fn r(bucket: &str, key: &str, start: u64, end: u64, etag: &str) -> SourceRef {
        SourceRef::new(
            Location::new(Scheme::S3, bucket, key).unwrap(),
            Locator::Bytes(ByteRange::new(start, end).unwrap()),
            ETag::new(etag).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn parses_the_documented_form() {
        let parsed: SourceRef =
            "s3://example-bucket/contracts/acme.pdf#bytes=18200-19750@etag=9f3a"
                .parse()
                .unwrap();
        assert_eq!(parsed.location().authority(), "example-bucket");
        assert_eq!(parsed.location().key(), "contracts/acme.pdf");
        let Locator::Bytes(range) = parsed.locator() else {
            panic!("expected a byte locator");
        };
        assert_eq!(range.start(), 18200);
        assert_eq!(range.end(), 19750);
        assert_eq!(range.len(), 1550);
        assert_eq!(Range::from(range), 18200..19750);
        assert_eq!(parsed.etag().as_str(), "9f3a");
    }

    #[test]
    fn round_trips() {
        let cases = [
            r("example-bucket", "docs/readme.md", 0, 1, "abc"),
            r(
                "example-bucket",
                "a b/c d.txt",
                5,
                10,
                "d41d8cd98f00b204e9800998ecf8427e-12",
            ),
            r("example-bucket", "100%/x#y@etag=z", 1, 2, "e"),
            r(
                "example-bucket",
                "\u{fc}ber/na\u{ef}ve.txt",
                0,
                u64::MAX,
                "e",
            ),
            r("example-bucket", "tab\there\nnewline", 7, 8, "e"),
            r("a.b-c", "/leading/slash/", 0, 1, "e"),
        ];
        for original in cases {
            let printed = original.to_string();
            let parsed: SourceRef = printed.parse().unwrap();
            assert_eq!(parsed, original, "{printed}");
            assert_eq!(parsed.to_string(), printed);
        }
    }

    #[test]
    fn a_file_ref_round_trips() {
        let printed = "file:///srv/docs/a.txt#bytes=0-4@etag=e";
        assert_eq!(printed.parse::<SourceRef>().unwrap().to_string(), printed);
    }

    #[test]
    fn escapes_only_what_it_must() {
        let printed = r("example-bucket", "100%/x#y z", 1, 2, "e").to_string();
        assert_eq!(
            printed,
            "s3://example-bucket/100%25/x%23y z#bytes=1-2@etag=e"
        );
    }

    #[test]
    fn decodes_any_escape_in_the_key() {
        let parsed: SourceRef = "s3://example-bucket/a%20b#bytes=0-1@etag=e"
            .parse()
            .unwrap();
        assert_eq!(parsed.location().key(), "a b");
    }

    #[test]
    fn rejects_malformed_refs() {
        let cases = [
            "",
            "s3:/example-bucket/k#bytes=0-1@etag=e",
            "http://example-bucket/k#bytes=0-1@etag=e",
            "s3://example-bucket/k",
            "s3://example-bucket#bytes=0-1@etag=e",
            "s3://example-bucket/#bytes=0-1@etag=e",
            "s3://example-bucket/k#bytes=1-1@etag=e",
            "s3://example-bucket/k#bytes=2-1@etag=e",
            "s3://example-bucket/k#bytes=01-2@etag=e",
            "s3://example-bucket/k#bytes=+1-2@etag=e",
            "s3://example-bucket/k#bytes=-2@etag=e",
            "s3://example-bucket/k#bytes=0-18446744073709551616@etag=e",
            "s3://example-bucket/k#bytes=0-1",
            "s3://example-bucket/k#bytes=0-1@etag=",
            "s3://example-bucket/k#bytes=0-1@etag=a b",
            "s3://example-bucket/k#range=0-1@etag=e",
            "s3://example-bucket/k%2#bytes=0-1@etag=e",
            "s3://example-bucket/k#page=0@etag=e",
            "s3://example-bucket/k#page=01@etag=e",
            "s3://example-bucket/k#page=x@etag=e",
            "s3://example-bucket/k#page=@etag=e",
            "s3://example-bucket/k#page=4294967296@etag=e",
            "s3://example-bucket/k#page=1",
        ];
        for case in cases {
            case.parse::<SourceRef>().unwrap_err();
        }
    }

    #[test]
    fn a_page_ref_round_trips() {
        let text = "s3://example-bucket/contracts/acme.pdf#page=12@etag=9f3a";
        let parsed: SourceRef = text.parse().unwrap();
        assert_eq!(
            parsed.locator(),
            Locator::Page(NonZeroU32::new(12).unwrap())
        );
        assert_eq!(parsed.to_string(), text);
    }

    #[test]
    fn element_rows_and_value_refs_round_trip() {
        for locator in [
            "element=b4/p2",
            "element=b9/r3/c2/p1",
            "rows=1-1",
            "rows=7-12",
            "value=42",
        ] {
            let text = format!("s3://example-bucket/k#{locator}@etag=e");
            let parsed: SourceRef = text.parse().unwrap();
            assert_eq!(parsed.to_string(), text);
            assert_eq!(
                parsed.locator().to_string().parse::<Locator>().unwrap(),
                parsed.locator()
            );
        }
    }

    #[test]
    fn rejects_malformed_element_rows_and_value_locators() {
        for locator in [
            "element=",
            "element=b0/p1",
            "element=b1",
            "rows=0-1",
            "rows=2-1",
            "rows=1",
            "rows=01-2",
            "value=0",
            "value=",
            "value=01",
        ] {
            locator.parse::<Locator>().unwrap_err();
        }
    }

    #[test]
    fn keeps_the_cause_of_a_bad_part() {
        let err = "s3://example-bucket/k#bytes=0-1@etag=a b"
            .parse::<SourceRef>()
            .unwrap_err();
        assert!(matches!(err, Error::InvalidETag(_)), "{err}");
    }

    #[test]
    fn staleness_compares_etags() {
        let source_ref = r("example-bucket", "k", 0, 1, "v1");
        assert!(source_ref.is_current(&ETag::new("\"v1\"").unwrap()));
        assert!(!source_ref.is_current(&ETag::new("v2").unwrap()));
    }

    #[test]
    fn serializes_as_a_string() {
        let source_ref = r("example-bucket", "k", 0, 1, "e");
        let json = serde_json::to_string(&source_ref).unwrap();
        assert_eq!(json, "\"s3://example-bucket/k#bytes=0-1@etag=e\"");
        assert_eq!(
            serde_json::from_str::<SourceRef>(&json).unwrap(),
            source_ref
        );
    }

    #[test]
    fn new_needs_a_key_and_a_non_empty_range() {
        let etag = ETag::new("e").unwrap();
        let locator = Locator::Bytes(ByteRange::new(0, 1).unwrap());
        let root = Location::new(Scheme::S3, "example-bucket", "").unwrap();
        SourceRef::new(root, locator, etag).unwrap_err();
        ByteRange::new(3, 3).unwrap_err();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn any_ref_round_trips(
            key in any::<String>().prop_filter("needs a key", |k| !k.is_empty()),
            start in 0..1_000_000_000_000u64,
            len in 1..1000u64,
        ) {
            let source_ref = r("example-bucket", &key, start, start.saturating_add(len), "e");
            let printed = source_ref.to_string();
            prop_assert_eq!(printed.parse::<SourceRef>().unwrap(), source_ref);
        }
    }
}

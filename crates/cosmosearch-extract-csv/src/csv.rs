use std::num::NonZeroU64;

use cosmosearch_core::{
    ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile, LimitKind, Locator,
    MAX_CHUNK_BYTES, Passage, RowRange, text_ranges,
};
use csv::{ReaderBuilder, StringRecord};

const CSV_EXTENSIONS: [&str; 1] = ["csv"];
const BYTE_ORDER_MARK: &[u8] = b"\xEF\xBB\xBF";
const ROW_SEPARATOR: &str = "\n\n";

/// Extracts comma-separated files in UTF-8. The first record is the header, and each value
/// is labelled with its column's name, or its column number from 1 when the header has none.
///
/// Rows are grouped into passages, each located by the [`RowRange`] of the records it holds,
/// counting from 1 after the header.
#[derive(Debug, Clone)]
pub struct Csv(ExtractorProfile);

impl Csv {
    /// Creates the extractor.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed profile id is not valid, which is a bug.
    pub fn new() -> Result<Self, cosmosearch_core::Error> {
        Ok(Csv(ExtractorProfile::new("csv-2")?))
    }
}

impl Extractor for Csv {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &CSV_EXTENSIONS
    }

    fn extract(&self, file: &[u8], limits: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        let file = file.strip_prefix(BYTE_ORDER_MARK).unwrap_or(file);
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(file);
        let mut record = StringRecord::new();
        if !read(&mut reader, &mut record)? {
            return Ok(Extraction::new(Vec::new()));
        }
        let header: Vec<String> = record.iter().map(|name| name.trim().to_owned()).collect();
        let mut builder = Builder::default();
        let mut number = NonZeroU64::MIN;
        let mut chars = 0usize;
        while read(&mut reader, &mut record)? {
            let text = row_text(&header, &record);
            chars = chars.saturating_add(text.chars().count());
            if chars > limits.chars() {
                return Err(ExtractFailure::LimitExceeded(LimitKind::Chars));
            }
            builder.add(number, text)?;
            number = number.checked_add(1).ok_or(ExtractFailure::Malformed)?;
        }
        builder.flush()?;
        Ok(Extraction::new(builder.found))
    }
}

fn read(
    reader: &mut csv::Reader<&[u8]>,
    record: &mut StringRecord,
) -> Result<bool, ExtractFailure> {
    reader
        .read_record(record)
        .map_err(|_cause| ExtractFailure::Malformed)
}

fn row_text(header: &[String], record: &StringRecord) -> String {
    let mut lines = Vec::new();
    for (column, value) in record.iter().enumerate() {
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        match header.get(column).filter(|name| !name.is_empty()) {
            Some(name) => lines.push(format!("{name}: {value}")),
            None => lines.push(format!("{}: {value}", column.saturating_add(1))),
        }
    }
    lines.join("\n")
}

#[derive(Default)]
struct Builder {
    open: Option<(NonZeroU64, NonZeroU64, String)>,
    found: Vec<Passage>,
}

impl Builder {
    fn add(&mut self, number: NonZeroU64, text: String) -> Result<(), ExtractFailure> {
        if text.is_empty() {
            return Ok(());
        }
        if text.len() > MAX_CHUNK_BYTES {
            self.flush()?;
            let rows = rows(number, number)?;
            for range in text_ranges(&text, 0..text.len()) {
                if let Some(piece) = text.get(range) {
                    self.found.push(Passage::new(rows, Vec::new(), piece));
                }
            }
            return Ok(());
        }
        match self.open.as_mut() {
            Some((_, last, joined))
                if joined
                    .len()
                    .saturating_add(ROW_SEPARATOR.len())
                    .saturating_add(text.len())
                    <= MAX_CHUNK_BYTES =>
            {
                joined.push_str(ROW_SEPARATOR);
                joined.push_str(&text);
                *last = number;
            }
            Some(_) | None => {
                self.flush()?;
                self.open = Some((number, number, text));
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ExtractFailure> {
        if let Some((first, last, text)) = self.open.take() {
            self.found
                .push(Passage::new(rows(first, last)?, Vec::new(), text));
        }
        Ok(())
    }
}

fn rows(first: NonZeroU64, last: NonZeroU64) -> Result<Locator, ExtractFailure> {
    RowRange::new(first, last)
        .map(Locator::Rows)
        .map_err(|_cause| ExtractFailure::Malformed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extractor() -> Csv {
        Csv::new().unwrap()
    }

    fn extract(file: &str) -> Result<Vec<Passage>, ExtractFailure> {
        extractor()
            .extract(file.as_bytes(), &ExtractLimits::default())
            .map(Extraction::into_passages)
    }

    fn shape(found: &[Passage]) -> Vec<(String, &str)> {
        found
            .iter()
            .map(|p| (p.locator().to_string(), p.text()))
            .collect()
    }

    #[test]
    fn damaged_files_give_passages_or_an_error_and_never_a_panic() {
        let good = b"name,amount,note\nrefund,30,\"within, days\"\nfee,\"5\",\"two\nlines\"\n,,\n";
        cosmosearch_core::extract_damaged(&extractor(), good);
    }

    #[test]
    fn labels_values_with_the_header_and_groups_rows() {
        let file = "\u{feff}name,team,\nAda,core,x\n\nBo,,\n,,\n\"Cy\nCo\",\"a, b\",y,extra\n";
        let found = extract(file).unwrap();
        assert_eq!(
            shape(&found),
            [(
                "rows=1-4".to_owned(),
                "name: Ada\nteam: core\n3: x\n\nname: Bo\n\nname: Cy\nCo\nteam: a, b\n3: y\n4: extra"
            )]
        );
    }

    #[test]
    fn a_header_alone_has_no_text_and_bad_utf8_is_malformed() {
        assert_eq!(extract("a,b\n").unwrap().len(), 0);
        assert_eq!(extract("").unwrap().len(), 0);
        let bad = Csv::new()
            .unwrap()
            .extract(b"a\n\xff\xfe\n", &ExtractLimits::default());
        assert_eq!(bad.unwrap_err(), ExtractFailure::Malformed);
    }

    #[test]
    fn chunks_stay_within_the_limit_and_resolve_by_their_rows() {
        let mut lines = vec!["id,note".to_owned()];
        for id in 0..300 {
            lines.push(format!("{id},a short note about record {id}"));
        }
        lines.push(format!("300,{}", "long words ".repeat(400)));
        let file = lines.join("\n");
        let found = extract(&file).unwrap();
        assert!(found.len() > 3);
        assert!(found.iter().all(|p| p.text().len() <= MAX_CHUNK_BYTES));
        let again = extract(&file).unwrap();
        for passage in &found {
            let resolved: Vec<&str> = again
                .iter()
                .filter(|p| p.locator() == passage.locator())
                .map(Passage::text)
                .collect();
            let same: Vec<&str> = found
                .iter()
                .filter(|p| p.locator() == passage.locator())
                .map(Passage::text)
                .collect();
            assert_eq!(resolved, same);
        }
        let Some(Locator::Rows(first)) = found.first().map(Passage::locator) else {
            panic!("expected rows");
        };
        assert_eq!(first.first().get(), 1);
        let long: Vec<&Passage> = found
            .iter()
            .filter(|p| p.locator().to_string() == "rows=301-301")
            .collect();
        assert!(long.len() > 1);
        assert!(long[0].text().starts_with("id: 300\nnote: long words"));
    }

    #[test]
    fn stops_at_the_character_limit() {
        let limits = ExtractLimits::builder().chars(10).build().unwrap();
        let found = Csv::new().unwrap().extract(b"a\n0123456789\n", &limits);
        assert_eq!(
            found.unwrap_err(),
            ExtractFailure::LimitExceeded(LimitKind::Chars)
        );
    }
}

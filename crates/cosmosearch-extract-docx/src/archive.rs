use std::io::{Cursor, Read};

use cosmosearch_core::{ExtractFailure, ExtractLimits, LimitKind};
use zip::ZipArchive;
use zip::result::ZipError;

/// A zip archive whose parts are read within the caller's limits.
pub(crate) struct Archive<'file> {
    zip: ZipArchive<Cursor<&'file [u8]>>,
    limits: ExtractLimits,
}

impl<'file> Archive<'file> {
    pub(crate) fn open(file: &'file [u8], limits: &ExtractLimits) -> Result<Self, ExtractFailure> {
        let zip = ZipArchive::new(Cursor::new(file)).map_err(|_cause| ExtractFailure::Malformed)?;
        if zip.len() > limits.archive_entries() {
            return Err(ExtractFailure::LimitExceeded(LimitKind::ArchiveEntries));
        }
        Ok(Archive {
            zip,
            limits: *limits,
        })
    }

    /// The bytes of the part `name`, or `None` if the archive has no such part.
    ///
    /// The declared size and ratio are checked before reading, and the bytes read after, so a
    /// part that lies about its size is still stopped at the limit.
    pub(crate) fn part(&mut self, name: &str) -> Result<Option<Vec<u8>>, ExtractFailure> {
        let entry = match self.zip.by_name(name) {
            Ok(entry) => entry,
            Err(ZipError::FileNotFound) => return Ok(None),
            Err(_cause) => return Err(ExtractFailure::Malformed),
        };
        let max_bytes = self.limits.archive_bytes();
        let max_by_ratio = entry
            .compressed_size()
            .saturating_mul(u64::from(self.limits.archive_ratio()));
        check_size(entry.size(), max_bytes, max_by_ratio)?;
        let mut bytes = Vec::new();
        entry
            .take(max_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|_cause| ExtractFailure::Malformed)?;
        let read = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        check_size(read, max_bytes, max_by_ratio)?;
        Ok(Some(bytes))
    }
}

fn check_size(size: u64, max_bytes: u64, max_by_ratio: u64) -> Result<(), ExtractFailure> {
    if size > max_bytes {
        Err(ExtractFailure::LimitExceeded(LimitKind::ArchiveBytes))
    } else if size > max_by_ratio {
        Err(ExtractFailure::LimitExceeded(LimitKind::ArchiveRatio))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_checked_before_ratio() {
        let bytes = ExtractFailure::LimitExceeded(LimitKind::ArchiveBytes);
        let ratio = ExtractFailure::LimitExceeded(LimitKind::ArchiveRatio);
        assert_eq!(check_size(10, 10, 10), Ok(()));
        assert_eq!(check_size(11, 10, 100), Err(bytes));
        assert_eq!(check_size(11, 100, 10), Err(ratio));
        assert_eq!(check_size(200, 100, 10), Err(bytes));
    }

    #[test]
    fn something_that_is_not_a_zip_is_malformed() {
        for file in [&b""[..], b"PK\x03\x04garbage", b"plain text"] {
            assert!(
                matches!(
                    Archive::open(file, &ExtractLimits::default()),
                    Err(ExtractFailure::Malformed)
                ),
                "{file:?}"
            );
        }
    }
}

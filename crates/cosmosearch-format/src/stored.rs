use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::Schema;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use crate::error::{EncodingError, malformed};
use crate::{Error, Result};

const FORMAT_KEY: &str = "cosmosearch.format";

fn encoding(source: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Error {
    Error::Encoding(EncodingError::new(source))
}

pub(crate) struct Stored {
    pub(crate) batches: Vec<RecordBatch>,
    pub(crate) metadata: BTreeMap<String, String>,
}

pub(crate) fn write(
    schema: Schema,
    columns: Vec<ArrayRef>,
    format: u32,
    metadata: Vec<(&str, String)>,
) -> Result<Vec<u8>> {
    let schema = Arc::new(schema);
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).map_err(encoding)?;
    let mut pairs = vec![KeyValue::new(FORMAT_KEY.to_owned(), format.to_string())];
    pairs.extend(
        metadata
            .into_iter()
            .map(|(key, value)| KeyValue::new(key.to_owned(), value)),
    );
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(pairs))
        .build();
    let mut bytes = Vec::new();
    let mut writer =
        ArrowWriter::try_new(&mut bytes, schema, Some(properties)).map_err(encoding)?;
    writer.write(&batch).map_err(encoding)?;
    writer.close().map_err(encoding)?;
    Ok(bytes)
}

#[expect(
    clippy::disallowed_methods,
    reason = "the Parquet reader can panic on malformed bytes, which are untrusted input"
)]
pub(crate) fn read(bytes: Bytes, format: u32) -> Result<Stored> {
    catch_unwind(AssertUnwindSafe(|| read_unguarded(bytes, format)))
        .map_err(|_payload| malformed("the Parquet reader could not process the file"))?
}

fn read_unguarded(bytes: Bytes, format: u32) -> Result<Stored> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(encoding)?;
    let metadata: BTreeMap<String, String> = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .into_iter()
        .flatten()
        .filter_map(|pair| Some((pair.key.clone(), pair.value.clone()?)))
        .collect();
    let found = metadata
        .get(FORMAT_KEY)
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| malformed("the format version is missing"))?;
    if found != format {
        return Err(Error::UnsupportedFormat { format: found });
    }
    let reader = builder.build().map_err(encoding)?;
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(encoding)?;
    Ok(Stored { batches, metadata })
}

/// A column of `batch` that must exist, have this type and hold no nulls.
pub(crate) fn required<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a T> {
    let found = optional::<T>(batch, name)?;
    if found.null_count() == 0 {
        Ok(found)
    } else {
        Err(malformed("a required column holds nulls"))
    }
}

/// A column of `batch` that must exist and have this type.
pub(crate) fn optional<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<T>())
        .ok_or_else(|| malformed("a column is missing or has the wrong type"))
}

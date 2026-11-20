use cosmosearch_core::{AttributeValue, Attributes, ETag, ManifestVersion};

/// A file's size and ETag, when the host already knows them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreFields {
    pub(crate) size: u64,
    pub(crate) etag: ETag,
}

impl StoreFields {
    /// The fields of one version of a file.
    pub fn new(size: u64, etag: ETag) -> Self {
        StoreFields { size, etag }
    }
}

/// A file added or changed by a host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileUpsert {
    pub(crate) key: String,
    pub(crate) fields: Option<StoreFields>,
    pub(crate) display_name: Option<String>,
    pub(crate) attributes: Attributes,
    pub(crate) expected: Option<ManifestVersion>,
}

impl FileUpsert {
    /// A file by key. Without [`FileUpsert::fields`], its size and ETag are read with one HEAD.
    pub fn new(key: impl Into<String>) -> Self {
        FileUpsert {
            key: key.into(),
            fields: None,
            display_name: None,
            attributes: Attributes::new(),
            expected: None,
        }
    }

    /// Supplies the size and ETag, so the file is not read.
    #[must_use]
    pub fn fields(mut self, fields: StoreFields) -> Self {
        self.fields = Some(fields);
        self
    }

    /// Sets the name to show for this file.
    #[must_use]
    pub fn display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Sets one attribute.
    #[must_use]
    pub fn attribute(mut self, name: impl Into<String>, value: AttributeValue) -> Self {
        self.attributes.insert(name.into(), value);
        self
    }

    /// Applies only if the file's content version is still `version`.
    #[must_use]
    pub fn expected(mut self, version: ManifestVersion) -> Self {
        self.expected = Some(version);
        self
    }
}

/// New attributes and display name for a file already in the index. The file is not read
/// again and keeps its content version and status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeChange {
    pub(crate) key: String,
    pub(crate) display_name: Option<String>,
    pub(crate) attributes: Attributes,
    pub(crate) expected: Option<ManifestVersion>,
}

impl AttributeChange {
    /// Replaces what the index holds for this file with no display name and no attributes,
    /// until some are set.
    pub fn new(key: impl Into<String>) -> Self {
        AttributeChange {
            key: key.into(),
            display_name: None,
            attributes: Attributes::new(),
            expected: None,
        }
    }

    /// Sets the name to show for this file.
    #[must_use]
    pub fn display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Sets one attribute.
    #[must_use]
    pub fn attribute(mut self, name: impl Into<String>, value: AttributeValue) -> Self {
        self.attributes.insert(name.into(), value);
        self
    }

    /// Applies only if the file's content version is still `version`.
    #[must_use]
    pub fn expected(mut self, version: ManifestVersion) -> Self {
        self.expected = Some(version);
        self
    }
}

/// One change a host records.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FileChange {
    /// A file is new or has changed.
    Upsert(FileUpsert),
    /// A file is gone. With `expected`, applies only at that content version.
    Remove {
        /// The file's key.
        key: String,
        /// The content version the file must still have.
        expected: Option<ManifestVersion>,
    },
    /// A file is to be tried again, for example after a parser fix or a raised limit. It gets
    /// a new content version and waits for a job. With `expected`, applies only at that
    /// content version.
    Retry {
        /// The file's key.
        key: String,
        /// The content version the file must still have.
        expected: Option<ManifestVersion>,
    },
    /// A file's attributes and display name are replaced; its content stays as indexed.
    Attributes(AttributeChange),
}

/// What became of one change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChangeOutcome {
    /// The change is recorded.
    Applied,
    /// The index already reflects this.
    Unchanged,
    /// The file is no longer at the expected version, so nothing was done.
    Stale,
    /// There is no such file: in the data location for an addition, in the index for a retry
    /// or an attribute change.
    NotFound,
}

/// What a call recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applied {
    pub(crate) version: Option<ManifestVersion>,
    pub(crate) outcomes: Vec<ChangeOutcome>,
}

impl Applied {
    /// The manifest version that recorded the changes, or `None` if nothing changed.
    pub fn version(&self) -> Option<ManifestVersion> {
        self.version
    }

    /// What became of each change, in the order given.
    pub fn outcomes(&self) -> &[ChangeOutcome] {
        &self.outcomes
    }
}

#[cfg(feature = "s3")]
use std::fmt;
use std::sync::Arc;
#[cfg(feature = "s3")]
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use cosmosearch_core::{ByteRange, ETag};
use futures_util::StreamExt;
use futures_util::stream::{self, BoxStream};
#[cfg(feature = "s3")]
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
#[cfg(feature = "s3")]
use object_store::{ClientOptions, RetryConfig};
use object_store::{
    Error as OsError, GetOptions, ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, PutOptions,
};

use crate::{BackendError, Error, ObjectInfo, ObjectKey, ReadStore, Result, WriteStore};

/// A backend over the `object_store` crate.
#[derive(Debug, Clone)]
pub struct ObjectStoreBackend {
    store: Arc<dyn ObjectStore>,
    reserves_upload_names: bool,
}

impl ObjectStoreBackend {
    /// Uses a ready `object_store` store.
    pub fn from_object_store(store: impl ObjectStore + 'static) -> Self {
        ObjectStoreBackend {
            store: Arc::new(store),
            reserves_upload_names: false,
        }
    }

    /// The local filesystem. Keys are absolute paths without their leading `/`.
    ///
    /// Symbolic links are followed, so a link inside a data location brings the files it
    /// points to into the location. Index only folders where the host controls the links.
    ///
    /// A file's ETag comes from its inode, size and modification time, so a file rewritten in
    /// place may keep its ETag. Replace files by renaming a new file over them.
    #[cfg(feature = "fs")]
    pub fn local() -> Self {
        ObjectStoreBackend {
            store: Arc::new(object_store::local::LocalFileSystem::new()),
            reserves_upload_names: true,
        }
    }

    /// An S3 or S3-compatible bucket.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Refused`] if the options cannot make a client.
    #[cfg(feature = "s3")]
    pub fn s3(options: &S3Options) -> Result<Self> {
        let refused = |source: OsError| Error::Refused {
            key: options.bucket.clone(),
            source: BackendError::new(source),
        };
        let mut retry = RetryConfig::default();
        if let Some(max_retries) = options.max_retries {
            retry.max_retries = max_retries;
        }
        // The client options carry `allow_http` too, so it is set here and not on the
        // builder, where `with_client_options` would replace it.
        let mut client = ClientOptions::new().with_allow_http(options.allow_http);
        if let Some(timeout) = options.timeout {
            client = client.with_timeout(timeout);
        }
        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(&options.bucket)
            .with_region(&options.region)
            .with_client_options(client)
            .with_retry(retry);
        if let Some(endpoint) = &options.endpoint {
            builder = builder.with_endpoint(endpoint);
        }
        if let Some(credentials) = &options.credentials {
            builder = builder
                .with_access_key_id(&credentials.access_key_id)
                .with_secret_access_key(&credentials.secret_access_key);
        }
        Ok(Self::from_object_store(builder.build().map_err(refused)?))
    }
}

#[cfg(feature = "s3")]
#[derive(Clone)]
struct Credentials {
    access_key_id: String,
    secret_access_key: String,
}

#[cfg(feature = "s3")]
impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Credentials(redacted)")
    }
}

/// How to reach an S3-compatible bucket. Made with [`S3Options::builder`].
#[cfg(feature = "s3")]
#[derive(Debug, Clone)]
pub struct S3Options {
    bucket: String,
    region: String,
    endpoint: Option<String>,
    credentials: Option<Credentials>,
    allow_http: bool,
    max_retries: Option<usize>,
    timeout: Option<Duration>,
}

#[cfg(feature = "s3")]
impl S3Options {
    /// Starts building options for `bucket`.
    pub fn builder(bucket: impl Into<String>) -> S3OptionsBuilder {
        S3OptionsBuilder(S3Options {
            bucket: bucket.into(),
            region: "us-east-1".to_owned(),
            endpoint: None,
            credentials: None,
            allow_http: false,
            max_retries: None,
            timeout: None,
        })
    }
}

/// Builds [`S3Options`].
#[cfg(feature = "s3")]
#[derive(Debug, Clone)]
pub struct S3OptionsBuilder(S3Options);

#[cfg(feature = "s3")]
impl S3OptionsBuilder {
    /// The region, `us-east-1` if not set.
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.0.region = region.into();
        self
    }

    /// The endpoint URL of an S3-compatible store.
    #[must_use]
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.0.endpoint = Some(endpoint.into());
        self
    }

    /// Static credentials. Without them, the client looks for instance credentials.
    #[must_use]
    pub fn credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.0.credentials = Some(Credentials {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
        });
        self
    }

    /// Allows plain HTTP, for a local store.
    #[must_use]
    pub fn allow_http(mut self, allow: bool) -> Self {
        self.0.allow_http = allow;
        self
    }

    /// How many times a retryable request is retried.
    #[must_use]
    pub fn max_retries(mut self, max_retries: usize) -> Self {
        self.0.max_retries = Some(max_retries);
        self
    }

    /// How long one request may take.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.0.timeout = Some(timeout);
        self
    }

    /// Finishes the options.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKey`] for an empty bucket name.
    pub fn build(self) -> Result<S3Options> {
        if self.0.bucket.is_empty() {
            return Err(Error::InvalidKey {
                key: String::new(),
                reason: "the bucket name must not be empty",
            });
        }
        Ok(self.0)
    }
}

fn to_path(key: &ObjectKey) -> Result<Path> {
    let invalid = || Error::InvalidKey {
        key: key.to_string(),
        reason: "the storage client cannot represent this key",
    };
    match Path::parse(key.as_str()) {
        Ok(path) if path.as_ref() == key.as_str() => Ok(path),
        Ok(_) | Err(_) => Err(invalid()),
    }
}

fn has_upload_suffix(key: &ObjectKey) -> bool {
    key.file_name()
        .split_once('#')
        .is_some_and(|(_, suffix)| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()))
}

async fn is_folder_marker(store: &dyn ObjectStore, meta: &ObjectMeta) -> bool {
    meta.size == 0
        && matches!(
            store.head(&meta.location).await,
            Err(OsError::NotFound { .. })
        )
}

fn list_error(scope: String, err: OsError) -> Error {
    if is_refusal(&err) {
        Error::Refused {
            key: scope,
            source: BackendError::new(err),
        }
    } else {
        Error::Backend {
            key: scope,
            source: BackendError::new(err),
        }
    }
}

fn is_refusal(err: &OsError) -> bool {
    matches!(
        err,
        OsError::PermissionDenied { .. }
            | OsError::Unauthenticated { .. }
            | OsError::NotImplemented { .. }
            | OsError::NotSupported { .. }
    )
}

fn read_error(key: &ObjectKey, err: OsError, expected: Option<&ETag>) -> Error {
    match (&err, expected) {
        (OsError::NotFound { .. }, _) => Error::NotFound { key: key.clone() },
        (OsError::Precondition { .. } | OsError::NotModified { .. }, Some(expected)) => {
            Error::Stale {
                key: key.clone(),
                expected: expected.clone(),
            }
        }
        _ if is_refusal(&err) => Error::Refused {
            key: key.to_string(),
            source: BackendError::new(err),
        },
        _ => Error::Backend {
            key: key.to_string(),
            source: BackendError::new(err),
        },
    }
}

fn write_error(key: &ObjectKey, err: OsError) -> Error {
    if matches!(
        err,
        OsError::AlreadyExists { .. } | OsError::Precondition { .. }
    ) {
        Error::AlreadyExists { key: key.clone() }
    } else if is_refusal(&err) {
        Error::Refused {
            key: key.to_string(),
            source: BackendError::new(err),
        }
    } else {
        Error::OutcomeUnknown {
            key: key.clone(),
            source: BackendError::new(err),
        }
    }
}

fn to_info(meta: &ObjectMeta) -> Result<ObjectInfo> {
    let key = ObjectKey::new(meta.location.as_ref())?;
    let etag = meta.e_tag.as_deref().ok_or_else(|| Error::Backend {
        key: key.to_string(),
        source: BackendError::new("the store reported no ETag"),
    })?;
    let etag = ETag::new(etag)?;
    Ok(ObjectInfo::new(key, meta.size, etag).with_last_modified(meta.last_modified.into()))
}

impl ObjectStoreBackend {
    fn path(&self, key: &ObjectKey) -> Result<Path> {
        if self.reserves_upload_names && has_upload_suffix(key) {
            return Err(Error::InvalidKey {
                key: key.to_string(),
                reason: "the local filesystem backend reserves names ending in `#<digits>`",
            });
        }
        to_path(key)
    }

    async fn range_failure(
        &self,
        key: &ObjectKey,
        path: &Path,
        range: ByteRange,
        failure: Error,
    ) -> Error {
        let past_the_end = matches!(failure, Error::Backend { .. })
            && matches!(self.store.head(path).await, Ok(meta) if meta.size <= range.start());
        if past_the_end {
            Error::ShortRead { key: key.clone() }
        } else {
            failure
        }
    }
}

#[async_trait]
impl ReadStore for ObjectStoreBackend {
    async fn read_range(
        &self,
        key: &ObjectKey,
        range: ByteRange,
        expected: &ETag,
    ) -> Result<Bytes> {
        let path = self.path(key)?;
        let options = GetOptions {
            if_match: Some(expected.quoted()),
            range: Some((range.start()..range.end()).into()),
            ..GetOptions::default()
        };
        let result = match self.store.get_opts(&path, options).await {
            Ok(result) => result,
            Err(err) => {
                let failure = read_error(key, err, Some(expected));
                return Err(self.range_failure(key, &path, range, failure).await);
            }
        };
        let bytes = result
            .bytes()
            .await
            .map_err(|err| read_error(key, err, Some(expected)))?;
        if u64::try_from(bytes.len()).ok() == Some(range.len()) {
            Ok(bytes)
        } else {
            Err(Error::ShortRead { key: key.clone() })
        }
    }

    async fn head(&self, key: &ObjectKey) -> Result<ObjectInfo> {
        let path = self.path(key)?;
        let meta = self
            .store
            .head(&path)
            .await
            .map_err(|err| read_error(key, err, None))?;
        to_info(&meta)
    }

    fn list<'a>(
        &'a self,
        prefix: Option<&'a ObjectKey>,
        after: Option<&'a ObjectKey>,
    ) -> BoxStream<'a, Result<ObjectInfo>> {
        let paths = prefix
            .map(to_path)
            .transpose()
            .and_then(|prefix| Ok((prefix, after.map(to_path).transpose()?)));
        let (prefix_path, after_path) = match paths {
            Ok(paths) => paths,
            Err(err) => return stream::once(async move { Err(err) }).boxed(),
        };
        let listing = match &after_path {
            Some(after) => self.store.list_with_offset(prefix_path.as_ref(), after),
            None => self.store.list(prefix_path.as_ref()),
        };
        let scope = prefix.map(ToString::to_string).unwrap_or_default();
        let store = Arc::clone(&self.store);
        listing
            .filter_map(move |item| {
                let (store, scope) = (Arc::clone(&store), scope.clone());
                async move {
                    match item {
                        Ok(meta) if is_folder_marker(store.as_ref(), &meta).await => None,
                        Ok(meta) => Some(to_info(&meta)),
                        Err(err) => Some(Err(list_error(scope, err))),
                    }
                }
            })
            .boxed()
    }
}

#[async_trait]
impl WriteStore for ObjectStoreBackend {
    async fn get_range(&self, key: &ObjectKey, range: ByteRange) -> Result<Bytes> {
        let path = self.path(key)?;
        let options = GetOptions {
            range: Some((range.start()..range.end()).into()),
            ..GetOptions::default()
        };
        let result = self
            .store
            .get_opts(&path, options)
            .await
            .map_err(|err| read_error(key, err, None))?;
        let bytes = result
            .bytes()
            .await
            .map_err(|err| read_error(key, err, None))?;
        if u64::try_from(bytes.len()).ok() == Some(range.len()) {
            Ok(bytes)
        } else {
            Err(Error::ShortRead { key: key.clone() })
        }
    }

    async fn get(&self, key: &ObjectKey) -> Result<Bytes> {
        let path = self.path(key)?;
        let result = self
            .store
            .get(&path)
            .await
            .map_err(|err| read_error(key, err, None))?;
        result
            .bytes()
            .await
            .map_err(|err| read_error(key, err, None))
    }

    async fn put_create(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        let path = self.path(key)?;
        let options = PutOptions {
            mode: PutMode::Create,
            ..PutOptions::default()
        };
        self.store
            .put_opts(&path, body.into(), options)
            .await
            .map(|_put| ())
            .map_err(|err| write_error(key, err))
    }

    async fn put_overwrite(&self, key: &ObjectKey, body: Bytes) -> Result<()> {
        let path = self.path(key)?;
        self.store
            .put_opts(&path, body.into(), PutOptions::default())
            .await
            .map(|_put| ())
            .map_err(|err| write_error(key, err))
    }

    async fn delete(&self, key: &ObjectKey) -> Result<()> {
        let path = self.path(key)?;
        match self.store.delete(&path).await {
            Ok(()) | Err(OsError::NotFound { .. }) => Ok(()),
            Err(err) => Err(read_error(key, err, None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(value: &str) -> ObjectKey {
        ObjectKey::new(value).unwrap()
    }

    #[cfg(feature = "s3")]
    #[test]
    fn credentials_never_appear_in_debug_output() {
        let options = S3Options::builder("example-bucket")
            .credentials("visible-id", "visible-secret")
            .build()
            .unwrap();
        let printed = format!("{options:?}");
        assert!(!printed.contains("visible"), "{printed}");
    }

    #[cfg(feature = "fs")]
    #[tokio::test]
    async fn the_local_backend_refuses_names_it_reserves() {
        let backend = ObjectStoreBackend::local();
        let err = backend.head(&key("tmp/report#2")).await.unwrap_err();
        assert!(matches!(err, Error::InvalidKey { .. }), "{err}");
    }

    #[test]
    fn plain_keys_convert_unchanged() {
        for value in ["a", "docs/a b.txt", "\u{fc}ber/x%20y", "a/.hidden/b"] {
            assert_eq!(to_path(&key(value)).unwrap().as_ref(), value);
        }
    }

    #[test]
    fn refuses_keys_it_would_reject_or_rewrite() {
        for value in ["a//b", "/x/", "/x", "x/", "a/./b", "a/../b", "a\tb"] {
            let err = to_path(&key(value)).unwrap_err();
            assert!(matches!(err, Error::InvalidKey { .. }), "{value:?}: {err}");
        }
    }
}

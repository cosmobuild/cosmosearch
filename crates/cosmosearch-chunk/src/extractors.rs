use std::collections::BTreeMap;

use cosmosearch_core::{ExtractorProfile, MediaType, SharedExtractor};

use crate::{Markdown, PlainText, Result};

/// The extractors a host registered: each found by its profile, and by the media types and
/// file name extensions it reads.
#[derive(Debug, Clone, Default)]
pub struct Extractors {
    profiles: BTreeMap<ExtractorProfile, SharedExtractor>,
    extensions: BTreeMap<String, ExtractorProfile>,
    media_types: BTreeMap<String, ExtractorProfile>,
}

impl Extractors {
    /// No extractors.
    pub fn new() -> Self {
        Extractors::default()
    }

    /// The extractors built into the kernel: plain text and Markdown.
    ///
    /// # Errors
    ///
    /// Returns an error if a fixed profile id is not valid, which is a bug.
    pub fn builtin() -> Result<Self> {
        let mut extractors = Extractors::new();
        extractors.insert(SharedExtractor::new(PlainText::new()?));
        extractors.insert(SharedExtractor::new(Markdown::new()?));
        Ok(extractors)
    }

    /// Adds `extractor`, which then reads the media types and extensions it names in place of
    /// any added before. Returns `false`, and changes nothing, if an extractor with the same
    /// profile is there.
    pub fn insert(&mut self, extractor: SharedExtractor) -> bool {
        let profile = extractor.get().profile().clone();
        if self.profiles.contains_key(&profile) {
            return false;
        }
        for extension in extractor.get().extensions() {
            self.extensions
                .insert(extension.to_ascii_lowercase(), profile.clone());
        }
        for media_type in extractor.get().media_types() {
            self.media_types
                .insert(media_type.to_ascii_lowercase(), profile.clone());
        }
        self.profiles.insert(profile, extractor);
        true
    }

    /// The extractor with this profile.
    pub fn get(&self, profile: &ExtractorProfile) -> Option<&SharedExtractor> {
        self.profiles.get(profile)
    }

    /// The extractor that reads the file with this key and media type: the one that reads the
    /// media type, or else the one that reads the key's extension.
    pub fn for_file(&self, key: &str, media_type: Option<&MediaType>) -> Option<&SharedExtractor> {
        media_type
            .and_then(|media_type| self.media_types.get(media_type.as_str()))
            .and_then(|profile| self.profiles.get(profile))
            .or_else(|| self.for_key(key))
    }

    fn for_key(&self, key: &str) -> Option<&SharedExtractor> {
        let name = key.rsplit('/').next().unwrap_or(key);
        let (stem, extension) = name.rsplit_once('.')?;
        if stem.is_empty() {
            return None;
        }
        let profile = self.extensions.get(&extension.to_ascii_lowercase())?;
        self.profiles.get(profile)
    }

    /// Every registered profile, in order.
    pub fn profiles(&self) -> impl Iterator<Item = &ExtractorProfile> {
        self.profiles.keys()
    }

    /// Whether an extractor with this profile is registered.
    pub fn contains(&self, profile: &ExtractorProfile) -> bool {
        self.profiles.contains_key(profile)
    }
}

#[cfg(test)]
mod tests {
    use cosmosearch_core::{ExtractFailure, ExtractLimits, Extraction, Extractor};

    use super::*;

    #[derive(Debug)]
    struct Named(
        ExtractorProfile,
        &'static [&'static str],
        &'static [&'static str],
    );

    impl Extractor for Named {
        fn profile(&self) -> &ExtractorProfile {
            &self.0
        }

        fn extensions(&self) -> &[&str] {
            self.1
        }

        fn media_types(&self) -> &[&str] {
            self.2
        }

        fn extract(&self, _: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
            Ok(Extraction::new(vec![]))
        }
    }

    fn named(profile: &str, extensions: &'static [&'static str]) -> SharedExtractor {
        SharedExtractor::new(Named(
            ExtractorProfile::new(profile).unwrap(),
            extensions,
            &[],
        ))
    }

    fn chosen(extractors: &Extractors, key: &str) -> Option<String> {
        chosen_as(extractors, key, None)
    }

    fn chosen_as(extractors: &Extractors, key: &str, media_type: Option<&str>) -> Option<String> {
        let media_type = media_type.map(|value| MediaType::new(value).unwrap());
        extractors
            .for_file(key, media_type.as_ref())
            .map(|found| found.get().profile().to_string())
    }

    #[test]
    fn a_media_type_an_extractor_reads_wins_over_the_extension() {
        let mut extractors = Extractors::builtin().unwrap();
        assert!(extractors.insert(SharedExtractor::new(Named(
            ExtractorProfile::new("pdf-1").unwrap(),
            &["pdf"],
            &["Application/PDF"],
        ))));
        let cases = [
            ("docs/report", Some("application/pdf"), Some("pdf-1")),
            (
                "docs/report.txt",
                Some("application/pdf; v=2"),
                Some("pdf-1"),
            ),
            ("docs/notes.txt", Some("text/markdown"), Some("markdown-2")),
            ("docs/notes.md", Some("image/png"), Some("markdown-2")),
            ("docs/notes", Some("image/png"), None),
            ("docs/notes.txt", None, Some("text-2")),
        ];
        for (key, media_type, expected) in cases {
            assert_eq!(
                chosen_as(&extractors, key, media_type).as_deref(),
                expected,
                "{key} {media_type:?}"
            );
        }
    }

    #[test]
    fn chooses_extensions_ignoring_case() {
        let extractors = Extractors::builtin().unwrap();
        let cases = [
            ("docs/a.txt", Some("text-2")),
            ("docs/B.TXT", Some("text-2")),
            ("docs/notes.md", Some("markdown-2")),
            ("docs/notes.markdown", Some("markdown-2")),
            ("docs/v1.2/readme", None),
            ("docs/archive.tar.gz", None),
            ("docs/.txt", None),
            ("docs/no-extension", None),
            ("docs/report.pdf", None),
        ];
        for (key, expected) in cases {
            assert_eq!(chosen(&extractors, key).as_deref(), expected, "{key}");
        }
    }

    #[test]
    fn a_later_extractor_takes_over_an_extension_and_both_stay_registered() {
        let mut extractors = Extractors::builtin().unwrap();
        assert!(extractors.insert(named("my-text-2", &["TXT"])));
        assert_eq!(
            chosen(&extractors, "docs/a.txt").as_deref(),
            Some("my-text-2")
        );
        let text = ExtractorProfile::new("text-2").unwrap();
        assert!(extractors.get(&text).is_some());
        let profiles: Vec<&str> = extractors
            .profiles()
            .map(ExtractorProfile::as_str)
            .collect();
        assert_eq!(profiles, ["markdown-2", "my-text-2", "text-2"]);
    }

    #[test]
    fn refuses_a_second_extractor_with_the_same_profile() {
        let mut extractors = Extractors::new();
        assert!(extractors.insert(named("pdf-1", &["pdf"])));
        assert!(!extractors.insert(named("pdf-1", &["docx"])));
        assert_eq!(chosen(&extractors, "a.docx"), None);
    }
}

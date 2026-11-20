use std::collections::BTreeMap;

use cosmosearch_core::{ExtractorProfile, SharedExtractor};

use crate::{Markdown, PlainText, Result};

/// The extractors a host registered: each found by its profile, and by the file name
/// extensions it reads.
#[derive(Debug, Clone, Default)]
pub struct Extractors {
    by_profile: BTreeMap<ExtractorProfile, SharedExtractor>,
    by_extension: BTreeMap<String, ExtractorProfile>,
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

    /// Adds `extractor`, which then reads the extensions it names in place of any added before.
    /// Returns `false`, and changes nothing, if an extractor with the same profile is there.
    pub fn insert(&mut self, extractor: SharedExtractor) -> bool {
        let profile = extractor.get().profile().clone();
        if self.by_profile.contains_key(&profile) {
            return false;
        }
        for extension in extractor.get().extensions() {
            self.by_extension
                .insert(extension.to_ascii_lowercase(), profile.clone());
        }
        self.by_profile.insert(profile, extractor);
        true
    }

    /// The extractor with this profile.
    pub fn get(&self, profile: &ExtractorProfile) -> Option<&SharedExtractor> {
        self.by_profile.get(profile)
    }

    /// The extractor that reads the file with this key, chosen by its extension.
    pub fn for_key(&self, key: &str) -> Option<&SharedExtractor> {
        let name = key.rsplit('/').next().unwrap_or(key);
        let (stem, extension) = name.rsplit_once('.')?;
        if stem.is_empty() {
            return None;
        }
        let profile = self.by_extension.get(&extension.to_ascii_lowercase())?;
        self.by_profile.get(profile)
    }

    /// Every registered profile, in order.
    pub fn profiles(&self) -> impl Iterator<Item = &ExtractorProfile> {
        self.by_profile.keys()
    }

    /// Whether an extractor with this profile is registered.
    pub fn contains(&self, profile: &ExtractorProfile) -> bool {
        self.by_profile.contains_key(profile)
    }
}

#[cfg(test)]
mod tests {
    use cosmosearch_core::{ExtractFailure, ExtractLimits, Extraction, Extractor};

    use super::*;

    #[derive(Debug)]
    struct Named(ExtractorProfile, &'static [&'static str]);

    impl Extractor for Named {
        fn profile(&self) -> &ExtractorProfile {
            &self.0
        }

        fn extensions(&self) -> &[&str] {
            self.1
        }

        fn extract(&self, _: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
            Ok(Extraction::new(vec![]))
        }
    }

    fn named(profile: &str, extensions: &'static [&'static str]) -> SharedExtractor {
        SharedExtractor::new(Named(ExtractorProfile::new(profile).unwrap(), extensions))
    }

    fn chosen(extractors: &Extractors, key: &str) -> Option<String> {
        extractors
            .for_key(key)
            .map(|found| found.get().profile().to_string())
    }

    #[test]
    fn chooses_by_extension_ignoring_case() {
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

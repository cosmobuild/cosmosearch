use tantivy::tokenizer::Language as Snowball;

/// A language whose words BM25 can reduce to their stems, so that `refunds` matches
/// `refund`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Language {
    /// Arabic.
    Arabic,
    /// Danish.
    Danish,
    /// Dutch.
    Dutch,
    /// English.
    English,
    /// Finnish.
    Finnish,
    /// French.
    French,
    /// German.
    German,
    /// Greek.
    Greek,
    /// Hungarian.
    Hungarian,
    /// Italian.
    Italian,
    /// Norwegian.
    Norwegian,
    /// Portuguese.
    Portuguese,
    /// Romanian.
    Romanian,
    /// Russian.
    Russian,
    /// Spanish.
    Spanish,
    /// Swedish.
    Swedish,
    /// Tamil.
    Tamil,
    /// Turkish.
    Turkish,
}

impl Language {
    /// The language's ISO 639-1 code, which names it in the plug-in's kind.
    pub fn code(self) -> &'static str {
        match self {
            Language::Arabic => "ar",
            Language::Danish => "da",
            Language::Dutch => "nl",
            Language::English => "en",
            Language::Finnish => "fi",
            Language::French => "fr",
            Language::German => "de",
            Language::Greek => "el",
            Language::Hungarian => "hu",
            Language::Italian => "it",
            Language::Norwegian => "no",
            Language::Portuguese => "pt",
            Language::Romanian => "ro",
            Language::Russian => "ru",
            Language::Spanish => "es",
            Language::Swedish => "sv",
            Language::Tamil => "ta",
            Language::Turkish => "tr",
        }
    }

    pub(crate) fn snowball(self) -> Snowball {
        match self {
            Language::Arabic => Snowball::Arabic,
            Language::Danish => Snowball::Danish,
            Language::Dutch => Snowball::Dutch,
            Language::English => Snowball::English,
            Language::Finnish => Snowball::Finnish,
            Language::French => Snowball::French,
            Language::German => Snowball::German,
            Language::Greek => Snowball::Greek,
            Language::Hungarian => Snowball::Hungarian,
            Language::Italian => Snowball::Italian,
            Language::Norwegian => Snowball::Norwegian,
            Language::Portuguese => Snowball::Portuguese,
            Language::Romanian => Snowball::Romanian,
            Language::Russian => Snowball::Russian,
            Language::Spanish => Snowball::Spanish,
            Language::Swedish => Snowball::Swedish,
            Language::Tamil => Snowball::Tamil,
            Language::Turkish => Snowball::Turkish,
        }
    }
}

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A value a host attaches to a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum AttributeValue {
    /// A single string.
    One(String),
    /// A list of strings.
    Many(Vec<String>),
}

/// A file's attributes, by name.
pub type Attributes = BTreeMap<String, AttributeValue>;

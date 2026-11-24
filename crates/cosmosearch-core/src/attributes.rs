use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Timestamp;

/// A finite number a host attaches to a file, kept as a 64-bit float.
#[derive(Debug, Clone, Copy)]
pub struct Number(f64);

impl Number {
    /// The number, or `None` for NaN or an infinity.
    pub fn new(value: f64) -> Option<Self> {
        value.is_finite().then_some(Number(value + 0.0))
    }

    /// The number as a float.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Number {}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Number {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Serialize for Number {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse::<f64>()
            .ok()
            .and_then(Number::new)
            .ok_or_else(|| serde::de::Error::custom("the number is not a finite decimal"))
    }
}

/// A value a host attaches to a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "Stored", into = "Stored")]
#[non_exhaustive]
pub enum AttributeValue {
    /// A single string.
    One(String),
    /// A list of strings.
    Many(Vec<String>),
    /// A number.
    Number(Number),
    /// A point in time, such as a contract's signing date.
    Time(Timestamp),
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Stored {
    One(String),
    Many(Vec<String>),
    Number { number: Number },
    Time { time: Timestamp },
}

impl From<Stored> for AttributeValue {
    fn from(stored: Stored) -> Self {
        match stored {
            Stored::One(value) => AttributeValue::One(value),
            Stored::Many(values) => AttributeValue::Many(values),
            Stored::Number { number } => AttributeValue::Number(number),
            Stored::Time { time } => AttributeValue::Time(time),
        }
    }
}

impl From<AttributeValue> for Stored {
    fn from(value: AttributeValue) -> Self {
        match value {
            AttributeValue::One(value) => Stored::One(value),
            AttributeValue::Many(values) => Stored::Many(values),
            AttributeValue::Number(number) => Stored::Number { number },
            AttributeValue::Time(time) => Stored::Time { time },
        }
    }
}

/// A file's attributes, by name.
pub type Attributes = BTreeMap<String, AttributeValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_of_value_round_trips_through_json() {
        let attributes = Attributes::from([
            ("owner".to_owned(), AttributeValue::One("legal".to_owned())),
            (
                "labels".to_owned(),
                AttributeValue::Many(vec!["a".to_owned(), "b".to_owned()]),
            ),
            (
                "amount".to_owned(),
                AttributeValue::Number(Number::new(12.5).unwrap()),
            ),
            (
                "signed".to_owned(),
                AttributeValue::Time(Timestamp::from_millis(-5)),
            ),
        ]);
        let json = serde_json::to_string(&attributes).unwrap();
        assert_eq!(
            json,
            r#"{"amount":{"number":"12.5"},"labels":["a","b"],"owner":"legal","signed":{"time":-5}}"#
        );
        let back: Attributes = serde_json::from_str(&json).unwrap();
        assert_eq!(back, attributes);
    }

    #[test]
    fn a_number_round_trips_exactly_and_a_bad_one_is_refused() {
        for value in [
            4.764_216_654_515_026_3e86,
            -874_941_090_656_061.5,
            0.1,
            f64::MIN_POSITIVE,
        ] {
            let value = AttributeValue::Number(Number::new(value).unwrap());
            let json = serde_json::to_string(&value).unwrap();
            assert_eq!(
                serde_json::from_str::<AttributeValue>(&json).unwrap(),
                value
            );
        }
        for json in [
            r#"{"number":"NaN"}"#,
            r#"{"number":"inf"}"#,
            r#"{"number":"x"}"#,
        ] {
            assert!(
                serde_json::from_str::<AttributeValue>(json).is_err(),
                "{json}"
            );
        }
    }

    #[test]
    fn numbers_are_finite_and_zero_has_one_sign() {
        assert!(Number::new(f64::NAN).is_none());
        assert!(Number::new(f64::INFINITY).is_none());
        assert_eq!(Number::new(-0.0), Number::new(0.0));
        assert!(Number::new(-1.0) < Number::new(0.5));
    }
}

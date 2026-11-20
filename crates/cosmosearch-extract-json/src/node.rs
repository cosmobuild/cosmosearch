use std::fmt;

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

/// A JSON value that keeps object members in document order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Node {
    /// A string, number, boolean or null, as text.
    Scalar(String),
    /// An array.
    Array(Vec<Node>),
    /// An object, its members in document order.
    Object(Vec<(String, Node)>),
}

impl Node {
    /// The members of an object or array with their reference tokens, in document order.
    pub(crate) fn members(&self) -> Vec<(String, &Node)> {
        match self {
            Node::Object(members) => members.iter().map(|(key, v)| (key.clone(), v)).collect(),
            Node::Array(items) => items
                .iter()
                .enumerate()
                .map(|(index, v)| (index.to_string(), v))
                .collect(),
            Node::Scalar(_) => Vec::new(),
        }
    }

    /// The text of a value with no members.
    pub(crate) fn scalar(&self) -> &str {
        match self {
            Node::Scalar(text) => text,
            Node::Array(_) => "[]",
            Node::Object(_) => "{}",
        }
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(NodeVisitor)
    }
}

struct NodeVisitor;

impl<'de> Visitor<'de> for NodeVisitor {
    type Value = Node;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Node, E> {
        Ok(Node::Scalar(v.to_string()))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Node, E> {
        Ok(Node::Scalar(v.to_string()))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Node, E> {
        Ok(Node::Scalar(v.to_string()))
    }

    fn visit_f64<E>(self, v: f64) -> Result<Node, E> {
        Ok(Node::Scalar(v.to_string()))
    }

    fn visit_str<E>(self, v: &str) -> Result<Node, E> {
        Ok(Node::Scalar(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> Result<Node, E> {
        Ok(Node::Scalar(v))
    }

    fn visit_unit<E>(self) -> Result<Node, E> {
        Ok(Node::Scalar("null".to_owned()))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Node, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(Node::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Node, A::Error> {
        let mut members = Vec::new();
        while let Some(member) = map.next_entry()? {
            members.push(member);
        }
        Ok(Node::Object(members))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_document_order_and_scalar_text() {
        let node: Node =
            serde_json::from_str(r#"{"b": 1, "a": [true, null, -2, 1.5, "s"], "c": {}}"#).unwrap();
        let scalar = |text: &str| Node::Scalar(text.to_owned());
        assert_eq!(
            node,
            Node::Object(vec![
                ("b".to_owned(), scalar("1")),
                (
                    "a".to_owned(),
                    Node::Array(vec![
                        scalar("true"),
                        scalar("null"),
                        scalar("-2"),
                        scalar("1.5"),
                        scalar("s")
                    ])
                ),
                ("c".to_owned(), Node::Object(Vec::new())),
            ])
        );
    }
}

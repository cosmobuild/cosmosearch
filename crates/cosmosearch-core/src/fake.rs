use crate::{
    EmbedError, Embedder, EmbeddingIdentity, Embeddings, RerankError, RerankScores, Reranker,
    RerankerIdentity, Usage,
};

/// An embedder for tests that derives each vector from a hash of its prefixed text.
///
/// Equal texts get equal vectors, different texts almost always differ, and every word
/// separated by whitespace counts as one token.
#[derive(Debug, Clone)]
pub struct FakeEmbedder(EmbeddingIdentity);

impl FakeEmbedder {
    /// An embedder giving vectors of `identity`.
    pub fn new(identity: EmbeddingIdentity) -> Self {
        FakeEmbedder(identity)
    }

    fn embed(&self, prefix: &str, texts: &[&str]) -> Embeddings {
        let dimensions = usize::try_from(self.0.dimensions().get()).unwrap_or(usize::MAX);
        let max_tokens = u64::from(self.0.max_tokens().get());
        let mut tokens = 0u64;
        let vectors = texts
            .iter()
            .map(|text| {
                let input = format!("{prefix}{text}");
                let words = u64::try_from(input.split_whitespace().count()).unwrap_or(u64::MAX);
                tokens = tokens.saturating_add(words.min(max_tokens));
                vector(&input, dimensions, self.0.normalized())
            })
            .collect();
        Embeddings::new(vectors, Usage::of_tokens(tokens))
    }
}

fn vector(input: &str, dimensions: usize, normalized: bool) -> Vec<f32> {
    let mut bytes = vec![0u8; dimensions];
    blake3::Hasher::new()
        .update(input.as_bytes())
        .finalize_xof()
        .fill(&mut bytes);
    let raw: Vec<f32> = bytes
        .iter()
        .map(|byte| f32::from(*byte) / 127.5 - 1.0)
        .collect();
    let length = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if normalized && length > 0.0 {
        raw.iter().map(|x| x / length).collect()
    } else {
        raw
    }
}

impl Embedder for FakeEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.0
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        Ok(self.embed(self.0.document_prefix(), texts))
    }

    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError> {
        Ok(self.embed(self.0.query_prefix(), &[text]))
    }
}

/// A reranker for tests that derives each score from a hash of the query and passage.
///
/// Equal pairs get equal scores between 0 and 1, different pairs almost always differ, and
/// every word separated by whitespace counts as one token, up to the identity's limit.
#[derive(Debug, Clone)]
pub struct FakeReranker(RerankerIdentity);

impl FakeReranker {
    /// A reranker giving scores of `identity`.
    pub fn new(identity: RerankerIdentity) -> Self {
        FakeReranker(identity)
    }
}

impl Reranker for FakeReranker {
    fn identity(&self) -> &RerankerIdentity {
        &self.0
    }

    fn rerank(&self, query: &str, passages: &[&str]) -> Result<RerankScores, RerankError> {
        let max_tokens = u64::from(self.0.max_tokens().get());
        let query_words = query.split_whitespace().count();
        let mut tokens = 0u64;
        let mut scores = Vec::with_capacity(passages.len());
        for passage in passages {
            let words = query_words.saturating_add(passage.split_whitespace().count());
            let words = u64::try_from(words).unwrap_or(u64::MAX);
            tokens = tokens.saturating_add(words.min(max_tokens));
            let mut bytes = [0u8; 2];
            blake3::Hasher::new()
                .update(&u64::try_from(query.len()).unwrap_or(u64::MAX).to_le_bytes())
                .update(query.as_bytes())
                .update(passage.as_bytes())
                .finalize_xof()
                .fill(&mut bytes);
            scores.push(f32::from(u16::from_le_bytes(bytes)) / 65_536.0);
        }
        Ok(RerankScores::new(scores, Usage::of_tokens(tokens)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SharedEmbedder, SharedReranker};

    fn shared() -> SharedEmbedder {
        let identity = EmbeddingIdentity::builder("fake", "1", 8, 4)
            .query_prefix("query: ")
            .build()
            .unwrap();
        SharedEmbedder::new(FakeEmbedder::new(identity))
    }

    #[test]
    fn gives_one_stable_unit_vector_per_text() {
        let embedder = shared();
        let first = embedder.get().embed_documents(&["a b", "c"]).unwrap();
        let again = embedder.get().embed_documents(&["a b", "c"]).unwrap();
        assert_eq!(first, again);
        assert_eq!(first.vectors().len(), 2);
        assert_ne!(first.vectors()[0], first.vectors()[1]);
        for vector in first.vectors() {
            assert_eq!(vector.len(), 8);
            let length = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((length - 1.0).abs() < 1e-5, "{length}");
        }
        assert_eq!(first.usage().tokens(), 3);
    }

    #[test]
    fn a_query_is_prefixed_and_its_tokens_are_capped() {
        let embedder = shared();
        let query = embedder.get().embed_query("c").unwrap();
        let document = embedder.get().embed_documents(&["c"]).unwrap();
        assert_ne!(query.vectors(), document.vectors());
        assert_eq!(query.usage().tokens(), 2);
        let long = embedder
            .get()
            .embed_query("one two three four five")
            .unwrap();
        assert_eq!(long.usage().tokens(), 4);
    }

    fn reranker() -> SharedReranker {
        let identity = RerankerIdentity::new("fake-reranker", "1", 4).unwrap();
        SharedReranker::new(FakeReranker::new(identity))
    }

    #[test]
    fn gives_one_stable_score_per_passage_in_order() {
        let reranker = reranker();
        let passages = ["refunds take thirty days", "parcels ship weekly", "x"];
        let first = reranker.get().rerank("refund", &passages).unwrap();
        let again = reranker.get().rerank("refund", &passages).unwrap();
        assert_eq!(first, again);
        assert_eq!(first.scores().len(), 3);
        assert!(first.scores().iter().all(|s| (0.0..1.0).contains(s)));
        assert_ne!(first.scores()[0], first.scores()[1]);
        let reversed = reranker
            .get()
            .rerank("refund", &[passages[2], passages[1], passages[0]])
            .unwrap();
        assert_eq!(reversed.scores()[2], first.scores()[0]);
        let other = reranker.get().rerank("parcel", &passages).unwrap();
        assert_ne!(other.scores()[0], first.scores()[0]);
    }

    #[test]
    fn counts_the_words_of_each_pair_up_to_the_limit() {
        let scores = reranker()
            .get()
            .rerank("refund", &["refunds take thirty days", "x"])
            .unwrap();
        assert_eq!(scores.usage().tokens(), 4 + 2);
        let none = reranker().get().rerank("refund", &[]).unwrap();
        assert_eq!(none.scores(), &[] as &[f32]);
        assert_eq!(none.usage().tokens(), 0);
    }
}

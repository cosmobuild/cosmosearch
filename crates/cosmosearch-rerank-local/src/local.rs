use std::fmt;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::ModernBert;
use cosmosearch_core::{RerankError, RerankScores, Reranker, RerankerIdentity, Usage};
use tokenizers::{Tokenizer, TruncationParams};

use crate::encoder;
use crate::failure::failed;
use crate::head::Head;

const ETTIN_RERANKER_17M: &str = "cross-encoder/ettin-reranker-17m-v1";
const ETTIN_RERANKER_17M_REVISION: &str = "9e4aa35321a6dd1a43ca313f500c4b4f7cfb5cc6";
const ETTIN_RERANKER_17M_MAX_TOKENS: u32 = 512;
const ENCODER_PREFIX: &str = "model.";
const PREFIXED_EMBEDDINGS: &str = "model.embeddings.tok_embeddings.weight";

/// The identity of the default model, `ettin-reranker-17m-v1` at the revision this crate was
/// tested with, reading 512 tokens of each query and passage.
///
/// # Errors
///
/// Returns an error only if the fixed identity is invalid, which is a bug.
pub fn ettin_reranker_17m_v1() -> Result<RerankerIdentity, cosmosearch_core::Error> {
    RerankerIdentity::new(
        ETTIN_RERANKER_17M,
        ETTIN_RERANKER_17M_REVISION,
        ETTIN_RERANKER_17M_MAX_TOKENS,
    )
}

/// The files of a ModernBERT cross-encoder as Sentence Transformers saves them, read by the
/// host from wherever it keeps them.
#[derive(Clone)]
pub struct RerankerFiles {
    config: Vec<u8>,
    tokenizer: Vec<u8>,
    weights: Vec<u8>,
    dense: Vec<u8>,
    norm: Vec<u8>,
    score: Vec<u8>,
}

impl RerankerFiles {
    /// The encoder's `config.json`, `tokenizer.json` and `model.safetensors`, then the
    /// scoring head's `2_Dense`, `3_LayerNorm` and `4_Dense` `model.safetensors`.
    pub fn new(
        config: Vec<u8>,
        tokenizer: Vec<u8>,
        weights: Vec<u8>,
        dense: Vec<u8>,
        norm: Vec<u8>,
        score: Vec<u8>,
    ) -> Self {
        RerankerFiles {
            config,
            tokenizer,
            weights,
            dense,
            norm,
            score,
        }
    }
}

impl fmt::Debug for RerankerFiles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankerFiles")
            .field("config_bytes", &self.config.len())
            .field("tokenizer_bytes", &self.tokenizer.len())
            .field("weights_bytes", &self.weights.len())
            .field("dense_bytes", &self.dense.len())
            .field("norm_bytes", &self.norm.len())
            .field("score_bytes", &self.score.len())
            .finish()
    }
}

/// Scores passages on the CPU with a ModernBERT cross-encoder, one query and passage pair at
/// a time, each cut to the identity's token limit.
pub struct LocalReranker {
    identity: RerankerIdentity,
    tokenizer: Tokenizer,
    encoder: ModernBert,
    head: Head,
    device: Device,
}

impl LocalReranker {
    /// Loads the model in `files` to give scores of `identity`. `identity` must match the
    /// files: this cannot be checked beyond the token limit.
    ///
    /// # Errors
    ///
    /// Returns [`RerankError::Model`] if the files cannot be read as a ModernBERT
    /// cross-encoder, or the model reads fewer tokens than `identity`.
    pub fn new(identity: RerankerIdentity, files: RerankerFiles) -> Result<Self, RerankError> {
        let device = Device::Cpu;
        let config = encoder::parse(&files.config)?;
        let max_tokens = usize::try_from(identity.max_tokens().get()).map_err(failed)?;
        if max_tokens > config.max_position_embeddings {
            return Err(failed(
                "the model reads fewer tokens than the identity says",
            ));
        }
        let mut tokenizer = Tokenizer::from_bytes(&files.tokenizer).map_err(failed)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_tokens,
                ..TruncationParams::default()
            }))
            .map_err(failed)?;
        tokenizer.with_padding(None);
        let weights = VarBuilder::from_buffered_safetensors(files.weights, DType::F32, &device)
            .map_err(failed)?;
        let encoder = ModernBert::load(unprefixed(weights), &config).map_err(failed)?;
        let head = Head::load(
            &files.dense,
            &files.norm,
            &files.score,
            config.hidden_size,
            &device,
        )?;
        Ok(LocalReranker {
            identity,
            tokenizer,
            encoder,
            head,
            device,
        })
    }

    fn score(&self, query: &str, passage: &str) -> Result<(f32, u64), RerankError> {
        let encoding = self
            .tokenizer
            .encode((query, passage), true)
            .map_err(failed)?;
        let tokens = u64::try_from(encoding.get_ids().len()).unwrap_or(u64::MAX);
        let score = self
            .forward(encoding.get_ids(), encoding.get_attention_mask())
            .map_err(failed)?;
        Ok((score, tokens))
    }

    fn forward(&self, ids: &[u32], mask: &[u32]) -> candle_core::Result<f32> {
        let ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mask = Tensor::new(mask, &self.device)?.unsqueeze(0)?;
        let hidden = self.encoder.forward(&ids, &mask)?;
        let scores = self.head.forward(&hidden)?;
        scores
            .first()
            .copied()
            .ok_or_else(|| candle_core::Error::Msg("the head gave no score".to_owned()))
    }
}

/// Sentence Transformers saves the encoder's weights without the `model.` prefix candle's
/// loader asks for, so the prefix is dropped from each name asked for when the file lacks it.
fn unprefixed(weights: VarBuilder<'static>) -> VarBuilder<'static> {
    if weights.contains_tensor(PREFIXED_EMBEDDINGS) {
        return weights;
    }
    weights.rename_f(|name: &str| name.strip_prefix(ENCODER_PREFIX).unwrap_or(name).to_owned())
}

impl fmt::Debug for LocalReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalReranker")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl Reranker for LocalReranker {
    fn identity(&self) -> &RerankerIdentity {
        &self.identity
    }

    fn rerank(&self, query: &str, passages: &[&str]) -> Result<RerankScores, RerankError> {
        let mut scores = Vec::with_capacity(passages.len());
        let mut tokens = 0u64;
        for passage in passages {
            let (score, read) = self.score(query, passage)?;
            if !score.is_finite() {
                return Err(failed("the model gave a score that is not finite"));
            }
            scores.push(score);
            tokens = tokens.saturating_add(read);
        }
        Ok(RerankScores::new(scores, Usage::of_tokens(tokens)))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn model_file(name: &str) -> Vec<u8> {
        #[expect(
            clippy::disallowed_methods,
            reason = "a test, not the library, chooses where the model files are"
        )]
        let root = std::env::var("COSMOSEARCH_TEST_MODELS")
            .expect("COSMOSEARCH_TEST_MODELS names the folder that holds the model files");
        let path = Path::new(&root).join("ettin-reranker-17m-v1").join(name);
        #[expect(clippy::disallowed_methods, reason = "a test reads a local model file")]
        std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    fn files() -> RerankerFiles {
        RerankerFiles::new(
            model_file("config.json"),
            model_file("tokenizer.json"),
            model_file("model.safetensors"),
            model_file("2_Dense/model.safetensors"),
            model_file("3_LayerNorm/model.safetensors"),
            model_file("4_Dense/model.safetensors"),
        )
    }

    fn reranker() -> LocalReranker {
        LocalReranker::new(ettin_reranker_17m_v1().unwrap(), files()).unwrap()
    }

    const QUERY: &str = "Which planet is known as the Red Planet?";
    const PASSAGES: [&str; 4] = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ];

    const CARD_SCORES: [f32; 4] = [6.59375, 10.5, 8.5625, 10.0];
    const CARD_TOLERANCE: f32 = 0.03;

    #[test]
    #[ignore = "needs the model files; set COSMOSEARCH_TEST_MODELS"]
    fn reproduces_the_model_card_scores_and_order() {
        let reranker = reranker();
        let found = reranker.rerank(QUERY, &PASSAGES).unwrap();
        for (got, want) in found.scores().iter().zip(CARD_SCORES) {
            assert!(
                (got - want).abs() < CARD_TOLERANCE,
                "{:?} against {CARD_SCORES:?}",
                found.scores()
            );
        }
        let order = |scores: &[f32]| {
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
            order
        };
        assert_eq!(order(found.scores()), order(&CARD_SCORES));
        assert_eq!(order(found.scores()), [1, 3, 2, 0]);
    }

    #[test]
    #[ignore = "needs the model files; set COSMOSEARCH_TEST_MODELS"]
    fn a_passage_scores_the_same_alone_or_among_others_and_tokens_are_counted() {
        let reranker = reranker();
        let all = reranker.rerank(QUERY, &PASSAGES).unwrap();
        let alone = reranker.rerank(QUERY, &PASSAGES[1..2]).unwrap();
        assert_eq!(alone.scores(), &all.scores()[1..2]);
        assert!(all.usage().tokens() > 4 * 20, "{:?}", all.usage());
        assert!(alone.usage().tokens() < all.usage().tokens());
        let none = reranker.rerank(QUERY, &[]).unwrap();
        assert_eq!(none.scores(), &[] as &[f32]);
        assert_eq!(none.usage().tokens(), 0);
    }

    #[test]
    #[ignore = "needs the model files; set COSMOSEARCH_TEST_MODELS"]
    fn cuts_a_long_pair_to_the_identitys_token_limit() {
        let files = files();
        let short =
            RerankerIdentity::new(ETTIN_RERANKER_17M, ETTIN_RERANKER_17M_REVISION, 64).unwrap();
        let reranker = LocalReranker::new(short, files).unwrap();
        let long = format!("{} {}", PASSAGES[1], "filler words ".repeat(100));
        let scores = reranker.rerank(QUERY, &[&long, PASSAGES[1]]).unwrap();
        let alone = reranker.rerank(QUERY, &[PASSAGES[1]]).unwrap();
        assert_eq!(
            scores.usage().tokens(),
            64 + alone.usage().tokens(),
            "{scores:?}"
        );
        assert!(scores.scores().iter().all(|score| score.is_finite()));
    }

    #[test]
    #[ignore = "needs the model files; set COSMOSEARCH_TEST_MODELS"]
    fn refuses_more_tokens_than_the_model_reads() {
        let files = files();
        let long = RerankerIdentity::new("m", "r", 8192).unwrap();
        assert!(matches!(
            LocalReranker::new(long, files),
            Err(RerankError::Model(_))
        ));
    }

    #[test]
    fn rejects_files_that_are_not_a_model() {
        let files = RerankerFiles::new(
            b"{}".to_vec(),
            b"{}".to_vec(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        assert!(matches!(
            LocalReranker::new(ettin_reranker_17m_v1().unwrap(), files),
            Err(RerankError::Model(_))
        ));
    }
}

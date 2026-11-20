use std::fmt;
use std::num::NonZeroUsize;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use cosmosearch_core::{EmbedCause, EmbedError, Embedder, EmbeddingIdentity, Embeddings, Usage};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const MDBR_LEAF_IR: &str = "MongoDB/mdbr-leaf-ir";
const MDBR_LEAF_IR_REVISION: &str = "4262131b32c3182bd06e67e92ae69d7bd66e0c5c";
const MDBR_LEAF_IR_DIMENSIONS: u32 = 768;
const MDBR_LEAF_IR_MAX_TOKENS: u32 = 512;
const MDBR_LEAF_IR_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";
const DEFAULT_BATCH: usize = 4;
const DENSE_WEIGHT: &str = "linear.weight";
const DENSE_BIAS: &str = "linear.bias";

/// The identity of the default model, `mdbr-leaf-ir` at the revision this crate was tested
/// with, giving 768 dimensions. Use [`EmbeddingIdentity::truncated`] for shorter vectors.
///
/// # Errors
///
/// Returns an error only if the fixed identity is invalid, which is a bug.
pub fn mdbr_leaf_ir() -> Result<EmbeddingIdentity, cosmosearch_core::Error> {
    EmbeddingIdentity::builder(
        MDBR_LEAF_IR,
        MDBR_LEAF_IR_REVISION,
        MDBR_LEAF_IR_DIMENSIONS,
        MDBR_LEAF_IR_MAX_TOKENS,
    )
    .query_prefix(MDBR_LEAF_IR_QUERY_PREFIX)
    .build()
}

/// The files of a BERT sentence-embedding model as Sentence Transformers saves them, read by
/// the host from wherever it keeps them.
#[derive(Clone)]
pub struct ModelFiles {
    config: Vec<u8>,
    tokenizer: Vec<u8>,
    weights: Vec<u8>,
    dense: Option<Vec<u8>>,
}

impl ModelFiles {
    /// The model's `config.json`, `tokenizer.json` and `model.safetensors`.
    pub fn new(config: Vec<u8>, tokenizer: Vec<u8>, weights: Vec<u8>) -> Self {
        ModelFiles {
            config,
            tokenizer,
            weights,
            dense: None,
        }
    }

    /// Adds the projection after pooling, `2_Dense/model.safetensors`.
    #[must_use]
    pub fn dense(mut self, weights: Vec<u8>) -> Self {
        self.dense = Some(weights);
        self
    }
}

impl fmt::Debug for ModelFiles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelFiles")
            .field("config_bytes", &self.config.len())
            .field("tokenizer_bytes", &self.tokenizer.len())
            .field("weights_bytes", &self.weights.len())
            .field("dense_bytes", &self.dense.as_ref().map(Vec::len))
            .finish()
    }
}

/// Embeds text on the CPU with a BERT model, mean pooling, an optional projection and, when
/// the identity says so, normalization.
pub struct LocalEmbedder {
    identity: EmbeddingIdentity,
    tokenizer: Tokenizer,
    bert: BertModel,
    dense: Option<Linear>,
    batch: NonZeroUsize,
    threads: NonZeroUsize,
    device: Device,
}

fn failed(cause: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> EmbedError {
    EmbedError::Model(EmbedCause::new(cause))
}

fn load_dense(weights: &[u8], device: &Device) -> Result<(Linear, usize), EmbedError> {
    let mut tensors = candle_core::safetensors::load_buffer(weights, device).map_err(failed)?;
    let weight = tensors
        .remove(DENSE_WEIGHT)
        .ok_or_else(|| failed("the projection has no weight"))?;
    let bias = tensors.remove(DENSE_BIAS);
    let (out, _) = weight.dims2().map_err(failed)?;
    Ok((Linear::new(weight, bias), out))
}

impl LocalEmbedder {
    /// Loads the model in `files` to give vectors of `identity`, embedding a few texts at a
    /// time. `identity` must match the files: this cannot be checked beyond its sizes.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError::Model`] if the files cannot be read as a model, if the model
    /// gives fewer dimensions than `identity`, or reads fewer tokens than it.
    pub fn new(identity: EmbeddingIdentity, files: ModelFiles) -> Result<Self, EmbedError> {
        let device = Device::Cpu;
        let config: Config = serde_json::from_slice(&files.config).map_err(failed)?;
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
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));
        let vb = VarBuilder::from_buffered_safetensors(files.weights, DType::F32, &device)
            .map_err(failed)?;
        let bert = BertModel::load(vb, &config).map_err(failed)?;
        let (dense, output) = match &files.dense {
            Some(weights) => {
                let (dense, output) = load_dense(weights, &device)?;
                (Some(dense), output)
            }
            None => (None, config.hidden_size),
        };
        let dimensions = usize::try_from(identity.dimensions().get()).map_err(failed)?;
        if dimensions > output {
            return Err(failed(
                "the model gives fewer dimensions than the identity says",
            ));
        }
        Ok(LocalEmbedder {
            identity,
            tokenizer,
            bert,
            dense,
            batch: NonZeroUsize::new(DEFAULT_BATCH).unwrap_or(NonZeroUsize::MIN),
            threads: NonZeroUsize::MIN,
            device,
        })
    }

    /// Sets how many texts run through the model together; 4 unless set. Larger batches
    /// pad shorter texts and need far more memory.
    #[must_use]
    pub fn batch_size(mut self, size: NonZeroUsize) -> Self {
        self.batch = size;
        self
    }

    /// Sets how many threads embed batches side by side; 1 unless set.
    #[must_use]
    pub fn threads(mut self, threads: NonZeroUsize) -> Self {
        self.threads = threads;
        self
    }

    fn embed(&self, prefix: &str, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        let inputs: Vec<String> = texts.iter().map(|text| format!("{prefix}{text}")).collect();
        let batches: Vec<&[String]> = inputs.chunks(self.batch.get()).collect();
        let per_thread = batches.len().div_ceil(self.threads.get()).max(1);
        if per_thread >= batches.len() {
            return self.embed_batches(&batches);
        }
        let groups = std::thread::scope(|scope| {
            let mut running = Vec::with_capacity(self.threads.get());
            for group in batches.chunks(per_thread) {
                running.push(scope.spawn(move || self.embed_batches(group)));
            }
            let mut groups = Vec::with_capacity(running.len());
            for thread in running {
                groups.push(
                    thread
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
                );
            }
            groups
        });
        let mut vectors = Vec::with_capacity(inputs.len());
        let mut usage = Usage::default();
        for group in groups {
            let group = group?;
            usage = usage.saturating_add(group.usage());
            vectors.extend(group.into_vectors());
        }
        Ok(Embeddings::new(vectors, usage))
    }

    fn embed_batches(&self, batches: &[&[String]]) -> Result<Embeddings, EmbedError> {
        let mut vectors = Vec::new();
        let mut tokens = 0u64;
        for batch in batches {
            let (batch_vectors, batch_tokens) = self.embed_batch(batch)?;
            vectors.extend(batch_vectors);
            tokens = tokens.saturating_add(batch_tokens);
        }
        Ok(Embeddings::new(vectors, Usage::of_tokens(tokens)))
    }

    fn embed_batch(&self, inputs: &[String]) -> Result<(Vec<Vec<f32>>, u64), EmbedError> {
        let encodings = self
            .tokenizer
            .encode_batch(inputs.to_vec(), true)
            .map_err(failed)?;
        let tokens: usize = encodings
            .iter()
            .map(|encoding| {
                encoding
                    .get_attention_mask()
                    .iter()
                    .filter(|m| **m == 1)
                    .count()
            })
            .sum();
        let ids = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_ids(), &self.device))
            .collect::<Result<Vec<_>, _>>()
            .map_err(failed)?;
        let masks = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_attention_mask(), &self.device))
            .collect::<Result<Vec<_>, _>>()
            .map_err(failed)?;
        let vectors = self.forward(&ids, &masks).map_err(failed)?;
        Ok((vectors, u64::try_from(tokens).unwrap_or(u64::MAX)))
    }

    fn forward(&self, ids: &[Tensor], masks: &[Tensor]) -> candle_core::Result<Vec<Vec<f32>>> {
        let ids = Tensor::stack(ids, 0)?;
        let mask = Tensor::stack(masks, 0)?;
        let types = ids.zeros_like()?;
        let hidden = self.bert.forward(&ids, &types, Some(&mask))?;
        let weights = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
        let pooled = hidden
            .broadcast_mul(&weights)?
            .sum(1)?
            .broadcast_div(&weights.sum(1)?)?;
        let projected = match &self.dense {
            Some(dense) => dense.forward(&pooled)?,
            None => pooled,
        };
        let dimensions = usize::try_from(self.identity.dimensions().get()).unwrap_or(usize::MAX);
        let kept = projected.narrow(D::Minus1, 0, dimensions)?;
        let out = if self.identity.normalized() {
            let length = kept.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
            kept.broadcast_div(&length)?
        } else {
            kept
        };
        out.to_vec2::<f32>()
    }
}

impl fmt::Debug for LocalEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalEmbedder")
            .field("identity", &self.identity)
            .field("batch", &self.batch)
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

impl Embedder for LocalEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Embeddings, EmbedError> {
        self.embed(self.identity.document_prefix(), texts)
    }

    fn embed_query(&self, text: &str) -> Result<Embeddings, EmbedError> {
        self.embed(self.identity.query_prefix(), &[text])
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
        let path = Path::new(&root).join("mdbr-leaf-ir").join(name);
        #[expect(clippy::disallowed_methods, reason = "a test reads a local model file")]
        std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    fn files() -> ModelFiles {
        ModelFiles::new(
            model_file("config.json"),
            model_file("tokenizer.json"),
            model_file("model.safetensors"),
        )
        .dense(model_file("2_Dense/model.safetensors"))
    }

    fn embedder(identity: EmbeddingIdentity) -> LocalEmbedder {
        LocalEmbedder::new(identity, files()).unwrap()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let length = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (length(a) * length(b))
    }

    const QUERIES: [&str; 2] = [
        "What is machine learning?",
        "How does neural network training work?",
    ];
    const DOCUMENTS: [&str; 2] = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms \
         that can learn from data.",
        "Neural networks are trained through backpropagation, adjusting weights to minimize \
         prediction errors.",
    ];

    fn similarities(embedder: &LocalEmbedder) -> Vec<[f32; 2]> {
        let documents = embedder.embed_documents(&DOCUMENTS).unwrap();
        QUERIES
            .iter()
            .map(|query| {
                let found = embedder.embed_query(query).unwrap();
                let query = found.vectors().first().unwrap();
                [
                    cosine(query, &documents.vectors()[0]),
                    cosine(query, &documents.vectors()[1]),
                ]
            })
            .collect()
    }

    fn assert_near(found: &[[f32; 2]], expected: [[f32; 2]; 2]) {
        for (row, want) in found.iter().zip(expected) {
            for (got, want) in row.iter().zip(want) {
                assert!((got - want).abs() < 5e-5, "{found:?} against {expected:?}");
            }
        }
    }

    #[test]
    #[ignore = "needs the model files; set COSMOSEARCH_TEST_MODELS"]
    fn reproduces_the_model_card_at_full_and_truncated_size() {
        let identity = mdbr_leaf_ir().unwrap();
        let full = embedder(identity.clone());
        assert_near(&similarities(&full), [[0.6857, 0.4598], [0.4238, 0.5723]]);
        let short = embedder(identity.truncated(256).unwrap());
        let found = similarities(&short);
        assert_near(&found, [[0.7136, 0.4989], [0.4567, 0.6022]]);
        let vector = &short.embed_query(QUERIES[0]).unwrap().into_vectors()[0];
        assert_eq!(vector.len(), 256);
    }

    #[test]
    #[ignore = "needs the model files; set COSMOSEARCH_TEST_MODELS"]
    fn repeats_give_identical_vectors_and_count_tokens() {
        let embedder = embedder(mdbr_leaf_ir().unwrap());
        let first = embedder.embed_documents(&DOCUMENTS).unwrap();
        let again = embedder.embed_documents(&DOCUMENTS).unwrap();
        assert_eq!(first, again);
        assert_eq!(first.vectors().len(), 2);
        assert!(first.vectors().iter().all(|vector| vector.len() == 768));
        assert!(first.usage().tokens() > 20);
        let one = embedder
            .batch_size(NonZeroUsize::MIN)
            .embed_documents(&DOCUMENTS)
            .unwrap();
        for (alone, batched) in one.vectors().iter().zip(first.vectors()) {
            let drift = alone
                .iter()
                .zip(batched)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(drift < 1e-5, "{drift}");
        }
    }

    #[test]
    #[ignore = "needs the model files; set COSMOSEARCH_TEST_MODELS"]
    fn threads_give_the_same_vectors_in_the_same_order() {
        let embedder = embedder(mdbr_leaf_ir().unwrap());
        let texts: Vec<String> = (0..4).map(|i| format!("Passage number {i}.")).collect();
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embedder = embedder.batch_size(NonZeroUsize::MIN);
        let alone = embedder.embed_documents(&texts).unwrap();
        let threaded = embedder
            .threads(NonZeroUsize::new(2).unwrap())
            .embed_documents(&texts)
            .unwrap();
        assert_eq!(alone, threaded);
        assert_eq!(threaded.vectors().len(), texts.len());
    }

    #[test]
    #[ignore = "needs the model files; set COSMOSEARCH_TEST_MODELS"]
    fn refuses_more_dimensions_or_tokens_than_the_model_has() {
        let sample = files();
        let wide = EmbeddingIdentity::builder("m", "r", 1024, 512)
            .build()
            .unwrap();
        assert!(matches!(
            LocalEmbedder::new(wide, sample.clone()),
            Err(EmbedError::Model(_))
        ));
        let long = EmbeddingIdentity::builder("m", "r", 768, 4096)
            .build()
            .unwrap();
        assert!(matches!(
            LocalEmbedder::new(long, sample),
            Err(EmbedError::Model(_))
        ));
    }

    #[test]
    fn rejects_files_that_are_not_a_model() {
        let files = ModelFiles::new(b"{}".to_vec(), b"{}".to_vec(), b"".to_vec());
        assert!(matches!(
            LocalEmbedder::new(mdbr_leaf_ir().unwrap(), files),
            Err(EmbedError::Model(_))
        ));
    }
}

use candle_core::{Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear};
use cosmosearch_core::RerankError;

use crate::failure::failed;

const DENSE_WEIGHT: &str = "linear.weight";
const DENSE_BIAS: &str = "linear.bias";
const NORM_WEIGHT: &str = "norm.weight";
const NORM_BIAS: &str = "norm.bias";
/// PyTorch's default, which the Sentence Transformers layer norm module keeps.
const NORM_EPS: f64 = 1e-5;

/// The scoring head Sentence Transformers saves after a cross-encoder: the first token's
/// vector, a dense layer with GELU, a layer norm and a dense layer to one score.
pub(crate) struct Head {
    dense: Linear,
    norm: LayerNorm,
    score: Linear,
}

impl Head {
    /// Loads the head from the weights of `2_Dense`, `3_LayerNorm` and `4_Dense`, checking
    /// that it reads vectors of `width` and gives one score.
    pub(crate) fn load(
        dense: &[u8],
        norm: &[u8],
        score: &[u8],
        width: usize,
        device: &Device,
    ) -> Result<Self, RerankError> {
        let (dense, dense_shape) = linear(dense, device)?;
        let (score, score_shape) = linear(score, device)?;
        if dense_shape != (width, width) || score_shape != (1, width) {
            return Err(failed("the head's layers do not fit the encoder"));
        }
        let mut tensors = candle_core::safetensors::load_buffer(norm, device).map_err(failed)?;
        let weight = tensors
            .remove(NORM_WEIGHT)
            .ok_or_else(|| failed("the head's layer norm has no weight"))?;
        let bias = tensors
            .remove(NORM_BIAS)
            .ok_or_else(|| failed("the head's layer norm has no bias"))?;
        if weight.dims() != [width] || bias.dims() != [width] {
            return Err(failed("the head's layer norm does not fit the encoder"));
        }
        Ok(Head {
            dense,
            norm: LayerNorm::new(weight, bias, NORM_EPS),
            score,
        })
    }

    /// Scores each sequence of `hidden`, the encoder's output of shape (sequences, tokens,
    /// width), from its first token.
    pub(crate) fn forward(&self, hidden: &Tensor) -> candle_core::Result<Vec<f32>> {
        let first = hidden.get_on_dim(1, 0)?;
        let dense = self.dense.forward(&first)?.gelu_erf()?;
        let normed = self.norm.forward(&dense)?;
        self.score.forward(&normed)?.flatten_all()?.to_vec1::<f32>()
    }
}

fn linear(weights: &[u8], device: &Device) -> Result<(Linear, (usize, usize)), RerankError> {
    let mut tensors = candle_core::safetensors::load_buffer(weights, device).map_err(failed)?;
    let weight = tensors
        .remove(DENSE_WEIGHT)
        .ok_or_else(|| failed("a dense layer of the head has no weight"))?;
    let bias = tensors.remove(DENSE_BIAS);
    let shape = weight.dims2().map_err(failed)?;
    if bias.as_ref().is_some_and(|bias| bias.dims() != [shape.0]) {
        return Err(failed(
            "a dense layer of the head has a bias of the wrong size",
        ));
    }
    Ok((Linear::new(weight, bias), shape))
}

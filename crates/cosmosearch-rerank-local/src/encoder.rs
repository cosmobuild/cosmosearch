use candle_transformers::models::modernbert::Config;
use cosmosearch_core::RerankError;
use serde::Deserialize;

use crate::failure::failed;

/// The fields of a ModernBERT `config.json` the encoder needs. Newer files give the rotary
/// bases in `rope_parameters`, older ones as `global_rope_theta` and `local_rope_theta`.
#[derive(Debug, Deserialize)]
struct EncoderConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    norm_eps: f64,
    pad_token_id: u32,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    #[serde(default)]
    local_rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    full_attention: Rope,
    sliding_attention: Rope,
}

#[derive(Debug, Deserialize)]
struct Rope {
    rope_theta: f64,
}

/// Reads a ModernBERT `config.json` into candle's encoder config.
pub(crate) fn parse(bytes: &[u8]) -> Result<Config, RerankError> {
    let config: EncoderConfig = serde_json::from_slice(bytes).map_err(failed)?;
    let ropes = config.rope_parameters.as_ref();
    let global_rope_theta = config
        .global_rope_theta
        .or(ropes.map(|r| r.full_attention.rope_theta))
        .ok_or_else(|| failed("the config gives no rotary base for full attention"))?;
    let local_rope_theta = config
        .local_rope_theta
        .or(ropes.map(|r| r.sliding_attention.rope_theta))
        .ok_or_else(|| failed("the config gives no rotary base for sliding attention"))?;
    if config.global_attn_every_n_layers == 0
        || config.hidden_size.checked_rem(config.num_attention_heads) != Some(0)
    {
        return Err(failed("the config's layer sizes do not fit together"));
    }
    Ok(Config {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        intermediate_size: config.intermediate_size,
        max_position_embeddings: config.max_position_embeddings,
        layer_norm_eps: config.norm_eps,
        pad_token_id: config.pad_token_id,
        global_attn_every_n_layers: config.global_attn_every_n_layers,
        global_rope_theta,
        local_attention: config.local_attention,
        local_rope_theta,
        classifier_config: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZES: &str = r#""vocab_size": 50368, "hidden_size": 256, "num_hidden_layers": 7,
        "num_attention_heads": 4, "intermediate_size": 384, "max_position_embeddings": 7999,
        "norm_eps": 1e-5, "pad_token_id": 50283, "global_attn_every_n_layers": 3,
        "local_attention": 128"#;

    #[test]
    fn reads_rotary_bases_in_either_form() {
        let newer = format!(
            r#"{{{SIZES}, "rope_parameters": {{"full_attention": {{"rope_theta": 160000.0}},
            "sliding_attention": {{"rope_theta": 10000.0}}}}, "id2label": {{"0": "LABEL_0"}},
            "label2id": {{"LABEL_0": 0}}}}"#
        );
        let older =
            format!(r#"{{{SIZES}, "global_rope_theta": 160000.0, "local_rope_theta": 10000.0}}"#);
        for json in [newer, older] {
            let config = parse(json.as_bytes()).unwrap();
            assert_eq!(config.hidden_size, 256);
            assert_eq!(config.global_attn_every_n_layers, 3);
            assert!((config.global_rope_theta - 160_000.0).abs() < 1e-9);
            assert!((config.local_rope_theta - 10_000.0).abs() < 1e-9);
            assert!((config.layer_norm_eps - 1e-5).abs() < 1e-12);
        }
    }

    #[test]
    fn refuses_a_config_it_cannot_run() {
        let no_rope = format!("{{{SIZES}}}");
        let odd_heads = format!(
            r#"{{{}, "global_rope_theta": 1.0, "local_rope_theta": 1.0}}"#,
            SIZES.replace(r#""num_attention_heads": 4"#, r#""num_attention_heads": 3"#)
        );
        for json in [no_rope, odd_heads, "{}".to_owned()] {
            assert!(
                matches!(parse(json.as_bytes()), Err(RerankError::Model(_))),
                "{json}"
            );
        }
    }
}

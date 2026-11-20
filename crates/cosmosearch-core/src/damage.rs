use crate::{ExtractLimits, Extractor};

const BREAKING: [u8; 12] = [
    0x00, 0xff, b'{', b'}', b'[', b']', b'"', b'<', b'>', b',', b'\n', b'#',
];

/// Gives `extractor` damaged copies of `good`, each byte in turn replaced by bytes that break
/// structure, and every prefix of it. A panic fails the calling test; passages or an error
/// are both fine.
pub fn extract_damaged(extractor: &dyn Extractor, good: &[u8]) {
    let limits = ExtractLimits::default();
    for at in 0..good.len() {
        for byte in BREAKING {
            let mut bad = good.to_vec();
            if let Some(slot) = bad.get_mut(at) {
                *slot = byte;
            }
            let _outcome = extractor.extract(&bad, &limits);
        }
    }
    for len in 0..good.len() {
        let _outcome = extractor.extract(good.get(..len).unwrap_or_default(), &limits);
    }
}

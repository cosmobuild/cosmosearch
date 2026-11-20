#![expect(
    clippy::cast_possible_truncation,
    reason = "a fixture builder writing a few small test archives"
)]

const LOCAL_HEADER: u32 = 0x0403_4b50;
const CENTRAL_HEADER: u32 = 0x0201_4b50;
const END_OF_DIRECTORY: u32 = 0x0605_4b50;
const STORED: u16 = 0;
const VERSION: u16 = 20;

pub(crate) struct Part<'a> {
    pub(crate) name: &'a str,
    pub(crate) body: &'a [u8],
    pub(crate) declared: Option<u32>,
}

impl<'a> Part<'a> {
    pub(crate) fn new(name: &'a str, body: &'a [u8]) -> Self {
        Part {
            name,
            body,
            declared: None,
        }
    }
}

pub(crate) fn zip(parts: &[Part<'_>]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for part in parts {
        let offset = out.len() as u32;
        let crc = crc32(part.body);
        let size = part.body.len() as u32;
        let declared = part.declared.unwrap_or(size);
        header(
            &mut out,
            LOCAL_HEADER,
            false,
            crc,
            size,
            declared,
            part.name,
            0,
        );
        out.extend_from_slice(part.body);
        header(
            &mut central,
            CENTRAL_HEADER,
            true,
            crc,
            size,
            declared,
            part.name,
            offset,
        );
    }
    let start = out.len() as u32;
    let count = parts.len() as u16;
    let length = central.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&END_OF_DIRECTORY.to_le_bytes());
    for value in [0u16, 0, count, count] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&start.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the fields of a zip header"
)]
fn header(
    out: &mut Vec<u8>,
    signature: u32,
    central: bool,
    crc: u32,
    size: u32,
    declared: u32,
    name: &str,
    offset: u32,
) {
    out.extend_from_slice(&signature.to_le_bytes());
    if central {
        out.extend_from_slice(&VERSION.to_le_bytes());
    }
    for value in [VERSION, 0, STORED, 0, 0] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in [crc, size, declared] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    if central {
        for value in [0u16, 0, 0] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.extend_from_slice(name.as_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

pub(crate) fn docx(body: &str, styles: Option<&str>) -> Vec<u8> {
    let document = format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>"#
    );
    let mut parts = vec![Part::new("word/document.xml", document.as_bytes())];
    if let Some(styles) = styles {
        parts.push(Part::new("word/styles.xml", styles.as_bytes()));
    }
    zip(&parts)
}

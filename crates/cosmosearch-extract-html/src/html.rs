use std::convert::Infallible;

use cosmosearch_core::{
    ByteRange, ExtractFailure, ExtractLimits, Extraction, Extractor, ExtractorProfile, Locator,
    MAX_CHUNK_BYTES, Passage,
};
use html5gum::{DefaultEmitter, Token, Tokenizer};

const HTML_EXTENSIONS: [&str; 2] = ["html", "htm"];
const HTML_MEDIA_TYPES: [&str; 2] = ["text/html", "application/xhtml+xml"];
const BYTE_ORDER_MARK: char = '\u{feff}';
const LONGEST_ENTITY_BYTES: usize = 40;
const HEADINGS: [&[u8]; 6] = [b"h1", b"h2", b"h3", b"h4", b"h5", b"h6"];
const HIDDEN: [&[u8]; 10] = [
    b"script",
    b"style",
    b"template",
    b"textarea",
    b"iframe",
    b"noembed",
    b"noframes",
    b"noscript",
    b"xmp",
    b"plaintext",
];
const BLOCKS: [&[u8]; 39] = [
    b"address",
    b"article",
    b"aside",
    b"blockquote",
    b"body",
    b"br",
    b"caption",
    b"center",
    b"dd",
    b"details",
    b"dialog",
    b"dir",
    b"div",
    b"dl",
    b"dt",
    b"fieldset",
    b"figcaption",
    b"figure",
    b"footer",
    b"form",
    b"head",
    b"header",
    b"hr",
    b"html",
    b"li",
    b"main",
    b"menu",
    b"nav",
    b"ol",
    b"p",
    b"pre",
    b"section",
    b"summary",
    b"table",
    b"td",
    b"th",
    b"title",
    b"tr",
    b"ul",
];
const TITLE: &[u8] = b"title";

/// Extracts HTML in UTF-8, one section per heading, with each section's heading path.
#[derive(Debug, Clone)]
pub struct Html(ExtractorProfile);

impl Html {
    /// Creates the extractor.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed profile id is not valid, which is a bug.
    pub fn new() -> Result<Self, cosmosearch_core::Error> {
        Ok(Html(ExtractorProfile::new("html-1")?))
    }
}

impl Extractor for Html {
    fn profile(&self) -> &ExtractorProfile {
        &self.0
    }

    fn extensions(&self) -> &[&str] {
        &HTML_EXTENSIONS
    }

    fn media_types(&self) -> &[&str] {
        &HTML_MEDIA_TYPES
    }

    fn extract(&self, file: &[u8], _: &ExtractLimits) -> Result<Extraction, ExtractFailure> {
        let Ok(text) = std::str::from_utf8(file) else {
            return Err(ExtractFailure::Malformed);
        };
        let base = if text.starts_with(BYTE_ORDER_MARK) {
            BYTE_ORDER_MARK.len_utf8()
        } else {
            0
        };
        let body = text.get(base..).unwrap_or_default();
        let mut walk = Walk::new(body, base);
        for token in tokens(body) {
            walk.token(token);
        }
        Ok(Extraction::new(walk.finish()))
    }
}

fn tokens(body: &str) -> impl Iterator<Item = Token<usize>> {
    let mut emitter = DefaultEmitter::<usize>::new_with_span();
    emitter.naively_switch_states(true);
    Tokenizer::new_with_emitter(body, emitter).map(|token| match token {
        Ok(token) => token,
        Err(never) => match never as Infallible {},
    })
}

fn visible_text(markup: &str) -> String {
    let mut text = String::new();
    for token in tokens(markup) {
        if let Token::String(found) = token {
            text.push_str(&String::from_utf8_lossy(&found.value.0));
        }
    }
    text
}

fn collapsed(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_one_of(name: &[u8], names: &[&[u8]]) -> bool {
    names.contains(&name)
}

#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct Atom {
    span: Span,
    text: String,
}

#[derive(Debug, Default)]
struct Block {
    atoms: Vec<Atom>,
}

impl Block {
    fn text(&self) -> String {
        let joined: String = self.atoms.iter().map(|a| a.text.as_str()).collect();
        collapsed(&joined)
    }

    fn extent(&self) -> Option<Span> {
        let mut visible = self.atoms.iter().filter(|a| !a.text.trim().is_empty());
        let first = visible.next()?;
        let last = visible.next_back().unwrap_or(first);
        Some(Span {
            start: first.span.start,
            end: last.span.end,
        })
    }

    fn into_piece(self) -> Option<Piece> {
        let span = self.extent()?;
        Some(Piece {
            span,
            text: self.text(),
        })
    }
}

#[derive(Debug)]
struct Piece {
    span: Span,
    text: String,
}

struct Hidden {
    name: Vec<u8>,
    depth: usize,
}

struct Walk<'a> {
    body: &'a str,
    base: usize,
    block: Block,
    pending: Option<Piece>,
    open_heading: Option<usize>,
    headings: Vec<(usize, String)>,
    hidden: Option<Hidden>,
    title_start: Option<usize>,
    found: Vec<Passage>,
}

impl<'a> Walk<'a> {
    fn new(body: &'a str, base: usize) -> Self {
        Walk {
            body,
            base,
            block: Block::default(),
            pending: None,
            open_heading: None,
            headings: Vec::new(),
            hidden: None,
            title_start: None,
            found: Vec::new(),
        }
    }

    fn token(&mut self, token: Token<usize>) {
        match token {
            Token::StartTag(tag) => self.start_tag(&tag.name.0, tag.span.start),
            Token::EndTag(tag) => self.end_tag(&tag.name.0),
            Token::String(text) => {
                let span = Span {
                    start: text.span.start,
                    end: text.span.end,
                };
                self.text(span, &text.value.0);
            }
            Token::Comment(_) | Token::Doctype(_) | Token::Error(_) => {}
        }
    }

    fn start_tag(&mut self, name: &[u8], start: usize) {
        if let Some(hidden) = &mut self.hidden {
            if hidden.name == name {
                hidden.depth = hidden.depth.saturating_add(1);
            }
            return;
        }
        if is_one_of(name, &HIDDEN) {
            self.hidden = Some(Hidden {
                name: name.to_vec(),
                depth: 1,
            });
            return;
        }
        if let Some(level) = HEADINGS.iter().position(|h| *h == name) {
            self.finish_block();
            self.flush();
            self.open_heading = Some(level);
        } else if is_one_of(name, &BLOCKS) {
            self.finish_block();
        }
        if name == TITLE {
            self.title_start = Some(start);
        }
    }

    fn end_tag(&mut self, name: &[u8]) {
        if let Some(hidden) = &mut self.hidden {
            if hidden.name == name {
                hidden.depth = hidden.depth.saturating_sub(1);
                if hidden.depth == 0 {
                    self.hidden = None;
                }
            }
            return;
        }
        if is_one_of(name, &BLOCKS) || is_one_of(name, &HEADINGS) {
            self.finish_block();
        }
        if name == TITLE {
            self.title_start = None;
        }
    }

    fn text(&mut self, span: Span, decoded: &[u8]) {
        if self.hidden.is_some() {
            return;
        }
        let start = self.title_start.unwrap_or(span.start);
        self.block.atoms.push(Atom {
            span: Span {
                start,
                end: span.end,
            },
            text: String::from_utf8_lossy(decoded).into_owned(),
        });
    }

    fn finish_block(&mut self) {
        let block = std::mem::take(&mut self.block);
        let text = block.text();
        let heading = self.open_heading.take();
        if text.is_empty() || (self.title_start.is_some() && text.len() > MAX_CHUNK_BYTES) {
            return;
        }
        if let Some(level) = heading {
            self.headings.retain(|(open, _)| *open < level);
            self.headings.push((level, text.clone()));
        }
        if text.len() > MAX_CHUNK_BYTES {
            self.flush();
            for piece in split_block(self.body, block) {
                self.emit(&piece);
            }
        } else if let Some(span) = block.extent() {
            self.add(span, &text);
        }
    }

    fn add(&mut self, span: Span, text: &str) {
        if let Some(open) = &mut self.pending {
            let joined = open.text.len().saturating_add(1).saturating_add(text.len());
            if joined <= MAX_CHUNK_BYTES {
                open.span.end = span.end;
                open.text.push('\n');
                open.text.push_str(text);
                return;
            }
        }
        self.flush();
        self.pending = Some(Piece {
            span,
            text: text.to_owned(),
        });
    }

    fn flush(&mut self) {
        if let Some(piece) = self.pending.take() {
            self.emit(&piece);
        }
    }

    fn emit(&mut self, piece: &Piece) {
        let start = u64::try_from(self.base.saturating_add(piece.span.start));
        let end = u64::try_from(self.base.saturating_add(piece.span.end));
        let (Ok(start), Ok(end)) = (start, end) else {
            return;
        };
        let Ok(range) = ByteRange::new(start, end) else {
            return;
        };
        let path = self.headings.iter().map(|(_, t)| t.clone()).collect();
        self.found.push(Passage::new(
            Locator::Bytes(range),
            path,
            piece.text.as_str(),
        ));
    }

    fn finish(mut self) -> Vec<Passage> {
        self.finish_block();
        self.flush();
        self.found
    }
}

fn split_block(body: &str, block: Block) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut open = Block::default();
    let mut open_bytes = 0usize;
    for atom in block.atoms {
        if atom.text.len() > MAX_CHUNK_BYTES {
            pieces.extend(std::mem::take(&mut open).into_piece());
            open_bytes = 0;
            pieces.extend(split_markup(body, atom.span));
            continue;
        }
        if open_bytes.saturating_add(atom.text.len()) > MAX_CHUNK_BYTES {
            pieces.extend(std::mem::take(&mut open).into_piece());
            open_bytes = 0;
        }
        open_bytes = open_bytes.saturating_add(atom.text.len());
        open.atoms.push(atom);
    }
    pieces.extend(open.into_piece());
    pieces
}

fn split_markup(body: &str, span: Span) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut start = span.start;
    while span.end.saturating_sub(start) > MAX_CHUNK_BYTES {
        let limit = body.floor_char_boundary(start.saturating_add(MAX_CHUNK_BYTES));
        let Some(window) = body.get(start..limit).filter(|w| !w.is_empty()) else {
            break;
        };
        let cut = start.saturating_add(safe_cut(window));
        pieces.extend(piece_of(body, start, cut));
        start = cut;
    }
    pieces.extend(piece_of(body, start, span.end));
    pieces
}

fn safe_cut(window: &str) -> usize {
    if let Some(at) = window.rfind(char::is_whitespace).filter(|at| *at > 0) {
        return at;
    }
    let tail_start = window.floor_char_boundary(window.len().saturating_sub(LONGEST_ENTITY_BYTES));
    let open_entity = window.get(tail_start..).and_then(|tail| {
        let at = tail.rfind('&')?;
        let unfinished = !tail.get(at..)?.contains(';');
        unfinished.then_some(tail_start.saturating_add(at))
    });
    open_entity.filter(|at| *at > 0).unwrap_or(window.len())
}

fn piece_of(body: &str, start: usize, end: usize) -> Option<Piece> {
    let text = collapsed(&visible_text(body.get(start..end)?));
    (!text.is_empty()).then_some(Piece {
        span: Span { start, end },
        text,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn extract(file: &str) -> Vec<Passage> {
        Html::new()
            .unwrap()
            .extract(file.as_bytes(), &ExtractLimits::default())
            .unwrap()
            .into_passages()
    }

    fn check(file: &str, expected: &[(&[&str], &str)]) {
        let found = extract(file);
        let actual: Vec<(Vec<&str>, &str)> = found
            .iter()
            .map(|p| {
                let path = p.heading_path().iter().map(String::as_str).collect();
                (path, p.text())
            })
            .collect();
        let expected: Vec<(Vec<&str>, &str)> =
            expected.iter().map(|(p, t)| (p.to_vec(), *t)).collect();
        assert_eq!(actual, expected, "{file:?}");
        assert_ranges_give_back_the_text(file, &found);
    }

    fn slice<'a>(file: &'a str, passage: &Passage) -> &'a str {
        let Locator::Bytes(range) = passage.locator() else {
            panic!("expected a byte locator");
        };
        let start = usize::try_from(range.start()).unwrap();
        let end = usize::try_from(range.end()).unwrap();
        file.get(start..end).expect("range inside the file")
    }

    fn assert_ranges_give_back_the_text(file: &str, found: &[Passage]) {
        let mut previous_end = 0;
        for passage in found {
            let Locator::Bytes(range) = passage.locator() else {
                panic!("expected a byte locator");
            };
            assert!(range.start() >= previous_end, "passages overlap");
            previous_end = range.end();
            assert!(passage.text().len() <= MAX_CHUNK_BYTES, "passage too large");
            let again = extract(slice(file, passage));
            let texts: Vec<&str> = again.iter().map(Passage::text).collect();
            assert_eq!(texts, [passage.text()], "{:?}", slice(file, passage));
        }
    }

    #[test]
    fn damaged_files_give_passages_or_an_error_and_never_a_panic() {
        let good = b"<!doctype html><html><head><title>Terms</title><script>let a = '<p>';</script></head><body><h1>Refunds</h1><ul><li>Within <b>30</b> days &amp; more</li></ul><table><tr><td>a</td></tr></table><!-- note --><p hidden>x</p></body></html>";
        cosmosearch_core::extract_damaged(&Html::new().unwrap(), good);
    }

    #[test]
    fn follows_the_heading_hierarchy() {
        check(
            "<p>intro</p><h1>A</h1><p>one</p><h2>B</h2><p>two</p><h2>C</h2><p>three</p>",
            &[
                (&[], "intro"),
                (&["A"], "A\none"),
                (&["A", "B"], "B\ntwo"),
                (&["A", "C"], "C\nthree"),
            ],
        );
    }

    #[test]
    fn removes_tags_and_decodes_entities() {
        check(
            "<p>Fish &amp; <b>chips</b>, caf&eacute;&#33;</p>",
            &[(&[], "Fish & chips, café!")],
        );
    }

    #[test]
    fn inline_tags_do_not_separate_words_and_blocks_do() {
        check(
            "<p><b>un</b>break<i>able</i></p><p>next</p>",
            &[(&[], "unbreakable\nnext")],
        );
        check(
            "<li>one</li><li>two</li><td>a</td><td>b</td>",
            &[(&[], "one\ntwo\na\nb")],
        );
        check("a<br>b", &[(&[], "a\nb")]);
    }

    #[test]
    fn collapses_whitespace_inside_a_block() {
        check("<p>  one \n\t two  </p>", &[(&[], "one two")]);
    }

    #[test]
    fn hides_what_a_reader_never_sees() {
        check(
            "<p>a</p><script>var hidden = '<p>x</p>';</script><style>p{}</style>\
             <!-- note --><template><p>t</p></template><textarea>typed</textarea><p>b</p>",
            &[(&[], "a\nb")],
        );
    }

    #[test]
    fn nested_hidden_elements_end_at_the_matching_tag() {
        check(
            "<template><template>x</template>y</template><p>seen</p>",
            &[(&[], "seen")],
        );
    }

    #[test]
    fn keeps_the_title_and_reads_markup_inside_it_as_text() {
        check("<title>a <b> c</title><p>d</p>", &[(&[], "a <b> c\nd")]);
    }

    #[test]
    fn leaves_out_a_title_too_long_for_one_chunk() {
        let title = "x".repeat(MAX_CHUNK_BYTES + 1);
        check(
            &format!("<title>{title}</title><p>body</p>"),
            &[(&[], "body")],
        );
        check(&format!("<title>{title}<p>swallowed"), &[]);
    }

    #[test]
    fn survives_broken_markup() {
        check("<p>open <b>never closed", &[(&[], "open never closed")]);
        check("a < b and c > d", &[(&[], "a < b and c > d")]);
        check(
            "<h2>unclosed heading<p>body</p>",
            &[(&["unclosed heading"], "unclosed heading\nbody")],
        );
    }

    #[test]
    fn finds_nothing_without_visible_text() {
        for file in [
            "",
            "<p> </p>",
            "<script>x</script>",
            "<!-- c -->",
            "\u{feff}",
        ] {
            assert_eq!(extract(file).len(), 0, "{file:?}");
        }
    }

    #[test]
    fn leaves_a_byte_order_mark_out_of_every_range() {
        let file = "\u{feff}<p>hello</p>";
        let found = extract(file);
        assert_eq!(found[0].text(), "hello");
        assert_eq!(slice(file, &found[0]), "hello");
    }

    #[test]
    fn refuses_invalid_utf8() {
        let failure = Html::new()
            .unwrap()
            .extract(b"<p>\xff</p>", &ExtractLimits::default())
            .unwrap_err();
        assert_eq!(failure, ExtractFailure::Malformed);
    }

    #[test]
    fn splits_a_long_block_without_breaking_an_entity() {
        let long = format!("<p>{}</p>", "&amp;".repeat(3_000));
        let found = extract(&long);
        assert!(found.len() > 1);
        let joined: String = found.iter().map(Passage::text).collect();
        assert_eq!(joined, "&".repeat(3_000));
        assert_ranges_give_back_the_text(&long, &found);
    }

    #[test]
    fn a_long_section_keeps_its_path_on_every_chunk() {
        let body = "<p>word word word</p>".repeat(400);
        let file = format!("<h1>A</h1>{body}");
        let found = extract(&file);
        assert!(found.len() > 1);
        assert!(found.iter().all(|p| p.heading_path() == ["A"]));
        assert_ranges_give_back_the_text(&file, &found);
    }

    fn markup() -> impl Strategy<Value = String> {
        let long = "x".repeat(2_500);
        let long_accents = "é".repeat(1_300);
        let long_words = "word ".repeat(600);
        let tokens: Vec<&'static str> = vec![
            "word ",
            "other ",
            "\n",
            "<p>",
            "</p>",
            "<div>",
            "</div>",
            "<b>",
            "</b>",
            "<br>",
            "<h1>",
            "</h1>",
            "<h2>T</h2>",
            "<li>",
            "<td>",
            "&amp;",
            "&#65;",
            "&",
            "<",
            ">",
            "a < b ",
            "<!-- c -->",
            "<script>s y</script>",
            "<style>",
            "</style>",
            "<title>",
            "</title>",
            "<template>",
            "</template>",
            "</x>",
            "<x y=\"z\">",
            "ü ",
            Box::leak(long.into_boxed_str()),
            Box::leak(long_accents.into_boxed_str()),
            Box::leak(long_words.into_boxed_str()),
        ];
        prop::collection::vec(prop::sample::select(tokens), 0..120).prop_map(|p| p.concat())
    }

    fn words() -> impl Strategy<Value = Vec<&'static str>> {
        let tokens: Vec<&'static str> = vec!["alpha", "beta", "gamma"];
        prop::collection::vec(prop::sample::select(tokens), 1..1_200)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn extracting_a_range_gives_back_its_text(file in markup()) {
            assert_ranges_give_back_the_text(&file, &extract(&file));
        }

        #[test]
        fn no_word_is_lost_or_added(words in words()) {
            let file = words
                .iter()
                .enumerate()
                .map(|(i, word)| if i % 7 == 0 { format!("<p>{word} ") } else { format!("<b>{word}</b> ") })
                .collect::<String>();
            let found = extract(&file);
            let kept: Vec<&str> = found.iter().flat_map(|p| p.text().split_whitespace()).collect();
            prop_assert_eq!(kept, words);
        }
    }
}

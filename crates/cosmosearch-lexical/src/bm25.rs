use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use cosmosearch_core::{
    AllowedRows, Built, BuiltFile, Chunk, Hit, Hits, Index, IndexCause, IndexError, IndexKind,
    Level, Query, RowSet, SegmentEntry, SegmentFile, SegmentFileName, SegmentId, SegmentSource,
    Usage,
};
use futures_util::{FutureExt, StreamExt, TryStreamExt, future, stream};
use tantivy::collector::TopDocs;
use tantivy::directory::{Directory, RamDirectory};
use tantivy::index::SegmentComponent;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::doc_predicate_query::{DocPredicateQuery, FunctionPredicate};
use tantivy::query::{
    Bm25StatisticsProvider, BooleanQuery, BoostQuery, Occur, PhraseQuery, Query as TantivyQuery,
    TermQuery,
};
use tantivy::schema::{FAST, Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer};
use tantivy::{
    DocId, Index as TantivyIndex, IndexSettings, IndexWriter, ReloadPolicy, Searcher,
    SegmentReader, TantivyDocument, Term,
};

use crate::Language;
use crate::bundle::{Bundle, write_bundle};

const TEXT: &str = "text";
const NAME: &str = "name";
const NAME_WEIGHT: f32 = 0.5;
const ROW: &str = "row";
const META_FILE: &str = "meta.json";
const FILE_PREFIX: &str = "lex-";
const BUILD_THREADS: usize = 1;
const BUILD_MEMORY_BYTES: usize = 64 * 1024 * 1024;
const OPEN_CONCURRENCY: usize = 16;
const WARM_CONCURRENCY: usize = 32;
const STATISTICS_CONCURRENCY: usize = 32;
const SEGMENT_COMPONENTS: [SegmentComponent; 6] = [
    SegmentComponent::Postings,
    SegmentComponent::Positions,
    SegmentComponent::FastFields,
    SegmentComponent::FieldNorms,
    SegmentComponent::Terms,
    SegmentComponent::Store,
];
const HOT_EXTENSIONS: [&str; 2] = ["fieldnorm", "fast"];
const FILTER_SCORE: f32 = 0.0;
const DEFAULT_TOKENIZER: &str = "default";
const LONGEST_WORD: usize = 40;

fn engine(source: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> IndexError {
    IndexError::Engine(IndexCause::new(source))
}

fn field(schema: &Schema, name: &str) -> Result<Field, IndexError> {
    schema
        .get_field(name)
        .map_err(|_cause| IndexError::Malformed {
            reason: "a segment is missing a field",
        })
}

/// How the text of chunks and queries becomes words: split on what is not a letter or digit,
/// words longer than 40 bytes dropped, lower-cased, and with a language, stemmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Analyzer(Option<Language>);

impl Analyzer {
    fn tokenizer(self) -> String {
        match self.0 {
            None => DEFAULT_TOKENIZER.to_owned(),
            Some(language) => format!("stem-{}", language.code()),
        }
    }

    fn register(self, index: &TantivyIndex) {
        if let Some(language) = self.0 {
            let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
                .filter(RemoveLongFilter::limit(LONGEST_WORD))
                .filter(LowerCaser)
                .filter(Stemmer::new(language.snowball()))
                .build();
            index.tokenizers().register(&self.tokenizer(), analyzer);
        }
    }
}

struct Fields {
    text: Field,
    name: Field,
    row: Field,
}

fn schema(analyzer: Analyzer) -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let indexing = |option| {
        let indexing = TextFieldIndexing::default()
            .set_tokenizer(&analyzer.tokenizer())
            .set_index_option(option);
        TextOptions::default().set_indexing_options(indexing)
    };
    let text = builder.add_text_field(TEXT, indexing(IndexRecordOption::WithFreqsAndPositions));
    let name = builder.add_text_field(NAME, indexing(IndexRecordOption::WithFreqs));
    let row = builder.add_u64_field(ROW, FAST);
    (builder.build(), Fields { text, name, row })
}

/// What the name field holds for each chunk: for the first chunk of each file, the file's
/// display name and the last part of its key without the extension; nothing for the others.
fn names(chunks: &[Chunk]) -> Vec<Option<String>> {
    let mut seen = BTreeSet::new();
    chunks
        .iter()
        .map(|chunk| {
            let key = chunk.source().location().key();
            if !seen.insert(key) {
                return None;
            }
            let last = key.rsplit('/').next().unwrap_or(key);
            let stem = match last.rsplit_once('.') {
                Some((stem, _)) if !stem.is_empty() => stem,
                _ => last,
            };
            Some(match chunk.name() {
                Some(name) => format!("{name}\n{stem}"),
                None => stem.to_owned(),
            })
        })
        .collect()
}

/// A BM25 keyword index, one bundled file per segment.
///
/// A chunk matches if it holds any word of the query. The first chunk of each file also
/// matches the words of the file's name, its display name and the last part of its key
/// without the extension, with half the weight of its text. For a query made with
/// [`Query::with_exact_phrases`], text in double quotes is a phrase: a chunk matches only if it
/// holds those words next to each other, in that order, whatever their case and the
/// punctuation between them. Every phrase must match, and the other words rank the chunks that
/// do. A quote without its pair is ignored.
///
/// Words are split on what is not a letter or digit and lower-cased. [`Bm25::stemmed`] also
/// reduces them to their stems in one language. How words are made is part of the plug-in's
/// kind, `bm25` or `bm25-stem-<code>`, so a plug-in searches only the segments built the
/// same way, and an index changing it builds its keyword segments again.
#[derive(Debug, Clone)]
pub struct Bm25 {
    kind: IndexKind,
    analyzer: Analyzer,
    scoring_tasks: NonZeroUsize,
}

impl Bm25 {
    /// Creates the plug-in, which does not stem.
    ///
    /// # Errors
    ///
    /// Returns an error if the fixed kind is not valid, which is a bug.
    pub fn new() -> Result<Self, cosmosearch_core::Error> {
        Ok(Bm25 {
            kind: IndexKind::new("bm25")?,
            analyzer: Analyzer(None),
            scoring_tasks: NonZeroUsize::MIN,
        })
    }

    /// Creates the plug-in, which reduces words to their stems in `language`, so that a search
    /// for `refund` also finds `refunds` and `refunded`. Its kind is `bm25-stem-` followed by
    /// the language's code, such as `bm25-stem-en`.
    ///
    /// # Errors
    ///
    /// Returns an error if the kind is not valid, which is a bug.
    pub fn stemmed(language: Language) -> Result<Self, cosmosearch_core::Error> {
        Ok(Bm25 {
            kind: IndexKind::new(format!("bm25-stem-{}", language.code()))?,
            analyzer: Analyzer(Some(language)),
            scoring_tasks: NonZeroUsize::MIN,
        })
    }

    /// Sets how many blocking tasks score a search's segments side by side; 1 unless set.
    #[must_use]
    pub fn scoring_tasks(mut self, tasks: NonZeroUsize) -> Self {
        self.scoring_tasks = tasks;
        self
    }

    fn file_of<'a>(&self, segment: &'a SegmentEntry) -> Option<&'a SegmentFile> {
        if segment.built_by().index() != Some(&self.kind) {
            return None;
        }
        segment
            .files()
            .iter()
            .find(|file| file.name().as_str().starts_with(FILE_PREFIX))
    }
}

fn collect_files(
    index: &TantivyIndex,
    directory: &RamDirectory,
) -> Result<Vec<(PathBuf, Vec<u8>)>, IndexError> {
    let mut files = vec![(
        PathBuf::from(META_FILE),
        directory
            .atomic_read(Path::new(META_FILE))
            .map_err(engine)?,
    )];
    for meta in index.searchable_segment_metas().map_err(engine)? {
        for component in SEGMENT_COMPONENTS {
            let path = meta.relative_path(component);
            if directory.exists(&path).map_err(engine)? {
                let bytes = directory
                    .open_read(&path)
                    .map_err(engine)?
                    .read_bytes()
                    .map_err(engine)?;
                files.push((path, bytes.as_slice().to_vec()));
            }
        }
    }
    Ok(files)
}

fn is_always_hot(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| HOT_EXTENSIONS.contains(&extension))
}

fn open_reader(bundle: Bundle) -> Result<(TantivyIndex, Searcher), IndexError> {
    open_reader_with(bundle, Analyzer(None))
}

fn open_reader_with(
    bundle: Bundle,
    analyzer: Analyzer,
) -> Result<(TantivyIndex, Searcher), IndexError> {
    let index = TantivyIndex::open(bundle).map_err(engine)?;
    analyzer.register(&index);
    let reader: tantivy::IndexReader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .map_err(engine)?;
    let searcher = reader.searcher();
    Ok((index, searcher))
}

fn build_bundle(
    texts: &[(String, Option<String>)],
    analyzer: Analyzer,
) -> Result<Vec<u8>, IndexError> {
    let (schema, fields) = schema(analyzer);
    let directory = RamDirectory::create();
    let index = TantivyIndex::create(directory.clone(), schema, IndexSettings::default())
        .map_err(engine)?;
    analyzer.register(&index);
    let mut writer: IndexWriter<TantivyDocument> = index
        .writer_with_num_threads(BUILD_THREADS, BUILD_MEMORY_BYTES)
        .map_err(engine)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for (position, (chunk_text, name)) in texts.iter().enumerate() {
        let mut document = TantivyDocument::default();
        document.add_text(fields.text, chunk_text);
        if let Some(name) = name {
            document.add_text(fields.name, name);
        }
        document.add_u64(fields.row, u64::try_from(position).map_err(engine)?);
        writer.add_document(document).map_err(engine)?;
    }
    writer.commit().map_err(engine)?;
    let segments = index.searchable_segment_ids().map_err(engine)?;
    if segments.len() > 1 {
        writer.merge(&segments).wait().map_err(engine)?;
    }
    writer.wait_merging_threads().map_err(engine)?;
    write_bundle(
        collect_files(&index, &directory)?,
        is_always_hot,
        touch_what_opening_needs,
    )
}

fn touch_what_opening_needs(bundle: Bundle) -> Result<(), IndexError> {
    let (index, searcher) = open_reader(bundle)?;
    let text = field(&index.schema(), TEXT)?;
    let name = field(&index.schema(), NAME)?;
    for reader in searcher.segment_readers() {
        reader.inverted_index(text).map_err(engine)?;
        reader.inverted_index(name).map_err(engine)?;
        reader.fast_fields().u64(ROW).map_err(engine)?;
    }
    Ok(())
}

#[derive(Clone)]
struct OpenSegment {
    id: SegmentId,
    rows: Arc<RowSet>,
    index: TantivyIndex,
    searcher: Searcher,
}

async fn open_segment(
    source: SegmentSource,
    id: SegmentId,
    file: SegmentFile,
    rows: Arc<RowSet>,
    analyzer: Analyzer,
) -> Result<OpenSegment, IndexError> {
    let bundle = Bundle::open(source, id.clone(), &file).await?;
    let (index, searcher) = open_reader_with(bundle, analyzer)?;
    Ok(OpenSegment {
        id,
        rows,
        index,
        searcher,
    })
}

/// Whether `rows` holds the row of every document of `segment`, so the search needs no
/// predicate to keep the others out.
fn allows_every_document(segment: &OpenSegment) -> Result<bool, IndexError> {
    for reader in segment.searcher.segment_readers() {
        let column = reader.fast_fields().u64(ROW).map_err(engine)?;
        let bounds = column.min_value()..column.max_value().saturating_add(1);
        if !column.get_cardinality().is_full() || !segment.rows.covers(bounds) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn only_rows(rows: Arc<RowSet>) -> Box<dyn TantivyQuery> {
    let predicate = FunctionPredicate::from(move |reader: &SegmentReader| {
        let column = reader.fast_fields().u64(ROW)?;
        let rows = Arc::clone(&rows);
        Ok(move |doc: DocId| column.first(doc).is_some_and(|row| rows.contains(row)))
    });
    scoring_nothing(DocPredicateQuery::from(predicate))
}

/// Keeps every document, yet scores through the same conjunction as [`only_rows`]: without
/// it the engine sums term scores in another order and a score can differ in its last bit.
fn every_document() -> Box<dyn TantivyQuery> {
    let predicate = FunctionPredicate::from(|_: &SegmentReader| Ok(|_: DocId| true));
    scoring_nothing(DocPredicateQuery::from(predicate))
}

fn scoring_nothing(filter: DocPredicateQuery) -> Box<dyn TantivyQuery> {
    Box::new(BoostQuery::new(Box::new(filter), FILTER_SCORE))
}

/// The words of a query: those outside quotes, any of which may match the text or, with
/// less weight, the name, and each quoted phrase, with every word at its position in the
/// phrase.
struct Wanted {
    loose: Vec<Term>,
    names: Vec<Term>,
    phrases: Vec<Vec<(usize, Term)>>,
}

impl Wanted {
    fn of(segment: &OpenSegment, query: &Query) -> Result<Self, IndexError> {
        let schema = segment.index.schema();
        let field = field(&schema, TEXT)?;
        let name = self::field(&schema, NAME)?;
        let mut analyzer = segment.index.tokenizer_for_field(field).map_err(engine)?;
        let mut tokens = |text: &str| {
            let mut found = Vec::new();
            analyzer.token_stream(text).process(&mut |token| {
                found.push((token.position, Term::from_field_text(field, &token.text)));
            });
            found
        };
        let (outside, quoted) = if query.exact_phrases() {
            split_phrases(query.text())
        } else {
            (query.text().to_owned(), Vec::new())
        };
        let mut loose: Vec<Term> = tokens(&outside).into_iter().map(|(_, term)| term).collect();
        loose.sort_unstable();
        loose.dedup();
        let names = loose
            .iter()
            .filter_map(|term| {
                let value = term.value();
                value.as_str().map(|word| Term::from_field_text(name, word))
            })
            .collect();
        let phrases = quoted
            .into_iter()
            .map(&mut tokens)
            .filter(|phrase| !phrase.is_empty())
            .collect();
        Ok(Wanted {
            loose,
            names,
            phrases,
        })
    }

    fn is_empty(&self) -> bool {
        self.loose.is_empty() && self.phrases.is_empty()
    }

    /// Every distinct word, and whether a phrase needs its positions.
    fn terms(&self) -> BTreeMap<Term, bool> {
        let mut terms: BTreeMap<Term, bool> = self
            .loose
            .iter()
            .chain(&self.names)
            .map(|term| (term.clone(), false))
            .collect();
        for (_, term) in self.phrases.iter().flatten() {
            terms.insert(term.clone(), true);
        }
        terms
    }

    fn query(&self, kept: Box<dyn TantivyQuery>) -> BooleanQuery {
        let text: Box<dyn TantivyQuery> =
            Box::new(BooleanQuery::new_multiterms_query(self.loose.clone()));
        let name: Box<dyn TantivyQuery> = Box::new(BoostQuery::new(
            Box::new(BooleanQuery::new_multiterms_query(self.names.clone())),
            NAME_WEIGHT,
        ));
        let loose: Box<dyn TantivyQuery> = Box::new(BooleanQuery::new(vec![
            (Occur::Should, text),
            (Occur::Should, name),
        ]));
        if self.phrases.is_empty() {
            return BooleanQuery::new(vec![(Occur::Must, loose), (Occur::Must, kept)]);
        }
        let mut clauses = Vec::with_capacity(self.phrases.len().saturating_add(2));
        if !self.loose.is_empty() {
            clauses.push((Occur::Should, loose));
        }
        for phrase in &self.phrases {
            let required: Box<dyn TantivyQuery> = match phrase.as_slice() {
                [(_, term)] => Box::new(TermQuery::new(term.clone(), IndexRecordOption::WithFreqs)),
                _ => Box::new(PhraseQuery::new_with_offset(phrase.clone())),
            };
            clauses.push((Occur::Must, required));
        }
        clauses.push((Occur::Must, kept));
        BooleanQuery::new(clauses)
    }
}

/// The text outside double quotes, and the text inside each pair of them. A quote without
/// its pair is left out, and the text after it counts as outside.
fn split_phrases(text: &str) -> (String, Vec<&str>) {
    let parts: Vec<&str> = text.split('"').collect();
    let mut outside = String::new();
    let mut quoted = Vec::new();
    for (at, part) in parts.iter().enumerate() {
        if at % 2 == 1 && at.saturating_add(1) < parts.len() {
            quoted.push(*part);
        } else {
            outside.push(' ');
            outside.push_str(part);
        }
    }
    (outside, quoted)
}

async fn warm(segment: &OpenSegment, term: &Term, with_positions: bool) -> Result<(), IndexError> {
    for reader in segment.searcher.segment_readers() {
        let inverted = reader.inverted_index(term.field()).map_err(engine)?;
        inverted
            .warm_postings(term, with_positions)
            .await
            .map_err(engine)?;
    }
    Ok(())
}

struct GlobalStatistics {
    docs: u64,
    tokens: BTreeMap<Field, u64>,
    doc_freq: BTreeMap<Term, u64>,
}

impl Bm25StatisticsProvider for GlobalStatistics {
    fn total_num_tokens(&self, field: Field) -> tantivy::Result<u64> {
        Ok(self.tokens.get(&field).copied().unwrap_or(0))
    }

    fn total_num_docs(&self) -> tantivy::Result<u64> {
        Ok(self.docs)
    }

    fn doc_freq(&self, term: &Term) -> tantivy::Result<u64> {
        Ok(self.doc_freq.get(term).copied().unwrap_or(0))
    }
}

/// One segment's share of the statistics: its documents, its tokens in each field the terms
/// are in, and each term's document frequency, in the order of the terms.
async fn segment_statistics(
    segment: &OpenSegment,
    terms: &[Term],
) -> Result<(u64, BTreeMap<Field, u64>, Vec<u64>), IndexError> {
    let fields: BTreeSet<Field> = terms.iter().map(Term::field).collect();
    let mut docs = 0u64;
    let mut tokens = BTreeMap::new();
    for reader in segment.searcher.segment_readers() {
        docs = docs.saturating_add(u64::from(reader.max_doc()));
        for field in &fields {
            let inverted = reader.inverted_index(*field).map_err(engine)?;
            let total: &mut u64 = tokens.entry(*field).or_insert(0);
            *total = total.saturating_add(inverted.total_num_tokens());
        }
    }
    let mut doc_freq = Vec::with_capacity(terms.len());
    for term in terms {
        doc_freq.push(
            segment
                .searcher
                .doc_freq_async(term)
                .await
                .map_err(engine)?,
        );
    }
    Ok((docs, tokens, doc_freq))
}

async fn gather_statistics(
    segments: &[OpenSegment],
    terms: &[Term],
) -> Result<GlobalStatistics, IndexError> {
    let gathering: Vec<_> = segments
        .iter()
        .map(|segment| segment_statistics(segment, terms))
        .collect();
    let shares: Vec<(u64, BTreeMap<Field, u64>, Vec<u64>)> = stream::iter(gathering)
        .buffer_unordered(STATISTICS_CONCURRENCY)
        .try_collect()
        .await?;
    let mut statistics = GlobalStatistics {
        docs: 0,
        tokens: BTreeMap::new(),
        doc_freq: BTreeMap::new(),
    };
    for (docs, tokens, doc_freq) in shares {
        statistics.docs = statistics.docs.saturating_add(docs);
        for (field, found) in tokens {
            let total = statistics.tokens.entry(field).or_insert(0);
            *total = total.saturating_add(found);
        }
        for (term, found) in terms.iter().zip(doc_freq) {
            let total = statistics.doc_freq.entry(term.clone()).or_insert(0);
            *total = total.saturating_add(found);
        }
    }
    Ok(statistics)
}

/// Splits `segments` into at most `tasks` groups of about equal documents.
fn share_out(mut segments: Vec<OpenSegment>, tasks: NonZeroUsize) -> Vec<Vec<OpenSegment>> {
    segments.sort_by_key(|segment| Reverse(segment.searcher.num_docs()));
    let mut groups: Vec<(u64, Vec<OpenSegment>)> = Vec::new();
    for segment in segments {
        let docs = segment.searcher.num_docs();
        if groups.len() < tasks.get() {
            groups.push((docs, vec![segment]));
        } else if let Some(lightest) = groups.iter_mut().min_by_key(|(load, _)| *load) {
            lightest.0 = lightest.0.saturating_add(docs);
            lightest.1.push(segment);
        }
    }
    groups.into_iter().map(|(_, group)| group).collect()
}

fn best(mut hits: Vec<Hit>, limit: usize) -> Vec<Hit> {
    hits.sort_by(|a, b| {
        b.score()
            .total_cmp(&a.score())
            .then_with(|| a.segment().cmp(b.segment()))
            .then_with(|| a.row().cmp(&b.row()))
    });
    hits.truncate(limit);
    hits
}

async fn score(
    segments: Vec<OpenSegment>,
    wanted: Wanted,
    statistics: GlobalStatistics,
    limit: usize,
    tasks: NonZeroUsize,
) -> Result<Vec<Hit>, IndexError> {
    let wanted = Arc::new(wanted);
    let statistics = Arc::new(statistics);
    let running: Vec<_> = share_out(segments, tasks)
        .into_iter()
        .map(|group| {
            let wanted = Arc::clone(&wanted);
            let statistics = Arc::clone(&statistics);
            tokio::task::spawn_blocking(move || collect_hits(&group, &wanted, &statistics, limit))
        })
        .collect();
    let mut hits = Vec::new();
    for found in future::try_join_all(running).await.map_err(engine)? {
        hits.extend(found?);
    }
    Ok(best(hits, limit))
}

fn collect_hits(
    segments: &[OpenSegment],
    wanted: &Wanted,
    statistics: &GlobalStatistics,
    limit: usize,
) -> Result<Vec<Hit>, IndexError> {
    let collector = TopDocs::with_limit(limit).order_by_score();
    let mut hits = Vec::new();
    for segment in segments {
        let kept = if allows_every_document(segment)? {
            every_document()
        } else {
            only_rows(Arc::clone(&segment.rows))
        };
        let query = wanted.query(kept);
        let found = segment
            .searcher
            .search_with_statistics_provider(&query, &collector, statistics)
            .map_err(engine)?;
        for (score, address) in found {
            let column = segment
                .searcher
                .segment_reader(address.segment_ord)
                .fast_fields()
                .u64(ROW)
                .map_err(engine)?;
            let row = column.first(address.doc_id).ok_or(IndexError::Malformed {
                reason: "a matched document has no row",
            })?;
            hits.push(Hit::new(segment.id.clone(), row, score));
        }
    }
    Ok(best(hits, limit))
}

#[async_trait]
impl Index for Bm25 {
    fn kind(&self) -> &IndexKind {
        &self.kind
    }

    fn level(&self) -> Level {
        Level::L1
    }

    async fn build(&self, segment: &SegmentId, chunks: &[Chunk]) -> Result<Built, IndexError> {
        let name = SegmentFileName::new(format!("{FILE_PREFIX}{segment}.bin")).map_err(engine)?;
        let texts: Vec<(String, Option<String>)> = chunks
            .iter()
            .map(Chunk::indexed_text)
            .zip(names(chunks))
            .collect();
        let analyzer = self.analyzer;
        let bundle = tokio::task::spawn_blocking(move || build_bundle(&texts, analyzer))
            .await
            .map_err(engine)??;
        Ok(Built::new(vec![BuiltFile::new(name, bundle)], Usage::new()))
    }

    async fn search(
        &self,
        source: &SegmentSource,
        segments: &[SegmentEntry],
        allowed: &AllowedRows,
        query: &Query,
        limit: usize,
    ) -> Result<Hits, IndexError> {
        AssertUnwindSafe(self.search_unguarded(source, segments, allowed, query, limit))
            .catch_unwind()
            .await
            .map_err(|_payload| IndexError::Malformed {
                reason: "the search engine could not read a segment",
            })?
    }
}

impl Bm25 {
    /// Tantivy can panic on damaged segment files, which come from the metadata store, so
    /// [`Index::search`] runs this with panics caught.
    async fn search_unguarded(
        &self,
        source: &SegmentSource,
        segments: &[SegmentEntry],
        allowed: &AllowedRows,
        query: &Query,
        limit: usize,
    ) -> Result<Hits, IndexError> {
        if limit == 0 {
            return Ok(Hits::default());
        }
        let targets: Vec<(SegmentId, SegmentFile, Arc<RowSet>)> = segments
            .iter()
            .filter_map(|segment| {
                let rows = allowed.rows(segment.id())?;
                let file = self.file_of(segment)?;
                Some((segment.id().clone(), file.clone(), Arc::clone(rows)))
            })
            .collect();
        let source = source.clone();
        let analyzer = self.analyzer;
        let opened: Vec<OpenSegment> = stream::iter(targets)
            .map(move |(id, file, rows)| open_segment(source.clone(), id, file, rows, analyzer))
            .buffer_unordered(OPEN_CONCURRENCY)
            .try_collect()
            .await?;
        let Some(first) = opened.first() else {
            return Ok(Hits::default());
        };
        let wanted = Wanted::of(first, query)?;
        if wanted.is_empty() {
            return Ok(Hits::default());
        }
        let terms = wanted.terms();
        let warming: Vec<(OpenSegment, Term, bool)> = opened
            .iter()
            .flat_map(|segment| {
                terms
                    .iter()
                    .map(|(term, positions)| (segment.clone(), term.clone(), *positions))
            })
            .collect();
        stream::iter(warming)
            .map(|(segment, term, positions)| async move { warm(&segment, &term, positions).await })
            .buffer_unordered(WARM_CONCURRENCY)
            .try_collect::<Vec<()>>()
            .await?;
        let terms: Vec<Term> = terms.into_keys().collect();
        let statistics = gather_statistics(&opened, &terms).await?;
        let hits = score(opened, wanted, statistics, limit, self.scoring_tasks).await?;
        Ok(Hits::new(hits, Usage::new()))
    }
}

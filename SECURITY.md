# Security

## Reporting a vulnerability

Please report vulnerabilities privately, through the **Security** tab of this repository
("Report a vulnerability"), not in a public issue. Say what you found, how to reproduce it,
and which version or commit you used.

## Versions and warranty

cosmosearch is provided as is, without warranty of any kind, under the terms of its
licences. Any fix goes into a new release; earlier releases are not updated.

## Known limits

These are part of the current design. They are documented rather than fixed, and reports
that sharpen them are welcome.

**Untrusted files.** Text extraction runs inside your process. A crafted PDF can make the
PDF parser use a lot of memory or run for a long time, which can crash or stall the
process, and that can't be caught. Every file is limited in size, characters and chunks,
and in pages, archive entries, expansion, decoded bytes and nesting depth where the format
has them. A PDF whose compressed streams decode to more than `ExtractLimits::decoded_bytes`
fails before the parser reads it, and so does a PDF that this check cannot load.

Extraction happens in two places. Indexing extracts every file once. The engine extracts a
file again whenever it reads text back: `read`, `expand` and `chunks`, snippets that aren't
in the passage cache, and reranking. If you index files you don't trust, run indexing's
extract step in a separate process with memory and time limits, and do the same for the
engine: serve searches from a process you can restart, or turn off snippets and reranking
and call `read`, `expand` and `chunks` only from such a process.

**Panic messages.** A parser that panics fails only its file: cosmosearch catches the panic
and reports the file as failed, without the panic's message. Rust's panic hook runs first,
though, and by default prints the message to stderr, where it can quote document text. If
your files are sensitive, install a panic hook that leaves the message out
(`std::panic::set_hook`), or keep the stderr of the processes that extract out of your logs.
cosmosearch does not set a hook, because the hook belongs to the whole process.

**The metadata location is trusted.** cosmosearch writes its index files and reads them back
as trusted data. Anyone who can write to the metadata location can make a reader decode
crafted files. Keep write access to it as narrow as access to the data itself.

**Stores.** Commits rely on create-only writes, and cosmosearch refuses a store that ignores
them when it creates an index. Reads rely on `If-Match` to pin a file's version. A store that
ignored `If-Match` would return changed bytes, and that is not checked yet.

**Access control.** cosmosearch has no users or permissions. Scopes and filters narrow a
search, but `read` and `expand` return any indexed passage whose ref they are given, and
`file` and `chunks` describe and read any indexed file whose key they are given. If refs or
keys come from your users, call `read_within`, `expand_within`, `file_within` and
`chunks_within` with the same scope and filter you give that user's searches, or check that
the user may access the file yourself. A search returns at most 1,000 hits; query length is
not limited, so bound it yourself.

**Local files.** The local filesystem backend follows symbolic links, so a link inside a
data folder brings its target's files into the index. Only index folders whose links you
control. Its ETags come from a file's inode, size and modification time, so a file
rewritten in place may not be seen as changed; replace files atomically by renaming.

**Disk cache.** The disk cache owns a `cosmosearch-blocks` folder inside the directory you
give it, refuses a folder holding files it didn't write, and writes with the process's
default permissions, so give it a private directory.

**Credentials.** cosmosearch takes credentials and API keys from you and never logs them;
`Debug` output and errors redact them. The HTTP plug-ins use https unless you explicitly
allow plain http, never follow redirects (so neither the key nor your text can be sent
elsewhere), and refuse responses over 64 MiB.

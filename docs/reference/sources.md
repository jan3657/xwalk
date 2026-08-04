# Loading records

The entire extension contract for input data is one type alias:

```python
RecordSource = Iterable[Record]
```

A `RecordSource` is anything iterable that yields `Record`s. A list works. A generator
reading 10M rows works. A function querying your warehouse works. Nothing in xwalk
needs to know your dataset exists — sources feed `MemoryStore.from_source`, index
builders, and `run_batch_sync` alike.

A `Record` is `Record(id: str, fields: Mapping[str, Any])`. The ID must be a non-empty
string (a non-`str` raises `TypeError`, a blank one `ValueError`); `fields` is whatever
the source produced.

The shipped loaders:

| Loader | Extra | Reads |
|---|---|---|
| `csv_source` | *(none)* | CSV or TSV files, one record per row |
| `jsonl_source` | *(none)* | JSON Lines, one record per non-blank line |
| `obo_source` | *(none)* | OBO ontologies, one record per `[Term]` stanza |
| `owl_source` | `ontology` | OWL/RDF files via rdflib, one record per `owl:Class` |
| `sql_source` | `sql` | any SQLAlchemy URL plus a query, streamed in chunks |

## `csv_source`

```python
def csv_source(
    path: str | Path,
    *,
    id_column: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    multivalue_columns: Sequence[str] = (),
    multivalue_sep: str = "|",
) -> Iterator[Record]
```

Yields one `Record` per row. The `id_column` cell, stripped, becomes `Record.id` and is
removed from `fields`; every other column lands in `fields` under its header name.
Plain cells are stripped strings, and a blank cell is `""`. Columns named in
`multivalue_columns` are split on `multivalue_sep` into lists with each part stripped
and blank parts dropped — a blank cell becomes `[]`, never `[""]`. Extra cells beyond
the header are silently discarded. Pass `delimiter="\t"` for TSV. The CSV field size
limit is raised at import, so long definition cells do not trip the stdlib default.

Raises `ValueError` when `id_column` is not in the header (the message lists the
columns found) and when a row's ID cell is blank (the message names the file and the
1-based line number; the header is line 1).

## `jsonl_source`

```python
def jsonl_source(
    path: str | Path,
    *,
    id_field: str,
    encoding: str = "utf-8",
) -> Iterator[Record]
```

Yields one `Record` per non-blank line; blank lines are skipped. The `id_field` value
is popped from the object, coerced with `str()`, and stripped to form `Record.id` — a
numeric ID `5` becomes `"5"`. Everything else in the object becomes `fields` verbatim:
no coercion, so lists and nested objects survive, which makes JSONL the natural format
for multi-valued fields without a separator convention.

Raises `ValueError` when a line is not valid JSON, when the object has no `id_field`,
or when the ID is blank after stripping — each message names the file and line number.

## `obo_source`

```python
def obo_source(
    path: str | Path,
    *,
    id_prefix: str | None = None,
    include_obsolete: bool = False,
    encoding: str = "utf-8",
) -> Iterator[Record]
```

Yields one `Record` per `[Term]` stanza. No dependencies: OBO is line-oriented stanzas,
and sixty lines of parsing beats pulling rdflib in for it. Stanzas of other kinds
(`[Typedef]`, ...) and stanzas without an `id:` are skipped.

`Record.id` is the `id:` value. `fields` always contains exactly five keys, shared with
`owl_source` so a single `doc` template works against either:

| Field | Type | From |
|---|---|---|
| `label` | `str` | `name:` (empty string if absent) |
| `synonyms` | `list[str]` | every `synonym:` line, quoted text extracted |
| `definition` | `str` | `def:`, quoted text extracted |
| `parents` | `list[str]` | every `is_a:` target |
| `obsolete` | `bool` | `is_obsolete: true` |

Trailing ` ! comment` text is stripped from values; `def:` and `synonym:` take the
leading quoted string with `\"` unescaped. `id_prefix` keeps only terms whose ID starts
with it (e.g. `"CHEBI:"`). Obsolete terms are skipped unless `include_obsolete=True`.

## `owl_source`

```python
def owl_source(
    path: str | Path,
    *,
    format: str | None = None,
    id_prefix: str | None = None,
    label_predicates: Sequence[str] = DEFAULT_LABEL_PREDICATES,
    synonym_predicates: Sequence[str] = DEFAULT_SYNONYM_PREDICATES,
    definition_predicates: Sequence[str] = DEFAULT_DEFINITION_PREDICATES,
    include_obsolete: bool = False,
) -> Iterator[Record]
```

Requires `xwalk[ontology]`. Parses the file with rdflib (`format=None` lets rdflib
guess) and yields one `Record` per `owl:Class` with a URI, sorted by URI so output
order is deterministic; blank-node class expressions are skipped.

`Record.id` is `curie(uri)`: an OBO PURL
(`http://purl.obolibrary.org/obo/CHEBI_17234`) compacts to `CHEBI:17234`, a fragment
URI compacts to the part after `#`, anything else passes through unchanged.
`id_prefix` filters on the compacted ID. A class is obsolete when any
`owl:deprecated` value is `"true"` (case-insensitive); obsolete classes are skipped
unless `include_obsolete=True`.

`fields` carries the same five names as `obo_source` — `label`, `synonyms`,
`definition`, `parents`, `obsolete`. `label` and `definition` take the first value
found walking the predicate lists in order; `synonyms` is the sorted, de-duplicated
union across all synonym predicates; `parents` is the sorted CURIEs of `subClassOf`
targets that are URIs. The defaults cover `rdfs:label`, the four oboInOwl synonym
predicates (exact, related, narrow, broad), and `IAO:0000115` plus
`skos:definition` — override the predicate tuples for vocabularies that use others.

## `sql_source`

```python
def sql_source(
    url: str,
    query: str,
    *,
    id_column: str,
    params: Mapping[str, Any] | None = None,
    multivalue_columns: Sequence[str] = (),
    multivalue_sep: str = "|",
    chunk_size: int = 1000,
) -> Iterator[Record]
```

Requires `xwalk[sql]`. Runs `query` (with bound `params`) against a SQLAlchemy `url`
and streams the result in chunks of `chunk_size` rather than materialising it —
"point at your warehouse" is exactly the case where loading the whole result set
defeats the purpose.

The `id_column` value is popped, coerced with `str()`, and stripped into `Record.id`.
Remaining columns land in `fields`: `multivalue_columns` follow the same
split-strip-drop rules as `csv_source` (a `NULL` becomes `[]`), while other columns
keep their database-typed values with `NULL` mapped to `""`. The engine is disposed in
a `finally` block, so abandoning the generator mid-stream never strands a warehouse
connection in the pool.

Raises `ValueError` when `id_column` is not among the result columns (the message
lists them) and when a row's ID is `NULL` or blank (the message shows the row's
remaining fields, since a database row has no line number).

## Writing your own source

Any generator of `Record`s is a source. There is no base class to inherit and no
registration step:

```python
from xwalk.records import Record

def issue_source(client, project):
    for issue in client.iter_issues(project):
        yield Record(
            id=str(issue.key),
            fields={"label": issue.title, "synonyms": issue.aliases},
        )

store = MemoryStore.from_source(issue_source(client, "XW"))
```

The only contract is the one `Record` enforces: a non-empty `str` ID and a `Mapping`
of fields. Coerce IDs with `str()` yourself — `Record` rejects an `int` rather than
guessing.

## Optional dependencies

The base install stays small on purpose: matching two CSVs with an API-hosted model
should not download torch. Loaders that need a heavy library call
`require(extra, module, purpose=...)`, which imports the module or raises
`MissingExtra` — a subclass of `ImportError` — that says which extra to install, never
a bare `ImportError` from a library you have never heard of. The message shape is:

```
owl_source requires the optional dependency 'rdflib', which is part of xwalk[ontology].

    pip install 'xwalk[ontology]'

underlying import error: No module named 'rdflib'
```

| Extra | Pulls | Needed for |
|---|---|---|
| `ontology` | rdflib | `owl_source`; `obo_source` needs nothing |
| `sql` | SQLAlchemy | `sql_source` |
| `dense` | sentence-transformers, torch, faiss-cpu | `SentenceTransformerEncoder` |
| `litellm` | litellm | `LiteLLMClient` |
| `all` | all of the above | — |

## Gotchas

- **Errors surface at iteration time, not call time.** Every loader is a generator, so
  `csv_source("missing.csv", id_column="id")` returns without touching the file; the
  `FileNotFoundError` — and even the id-column check — fires on the first record. The
  same applies to `MissingExtra`: `sql_source(...)` without SQLAlchemy raises only
  when iterated. Wrap the *iteration* in your error handling, not the call.
- **Sources are one-shot.** A generator is exhausted after a single pass; to feed both
  a `MemoryStore` and an index builder, materialise first: `targets = list(source)`.
- **Blank cells differ by shape.** A blank plain cell is `""`; a blank multivalue cell
  is `[]`, never `[""]`. Templates and `exact_fields` handle both, but custom code
  should not assume one shape.
- **IDs are always strings.** `jsonl_source` and `sql_source` coerce with `str()`, so
  a numeric key `5` becomes `"5"` — normalise your gold labels to match. A custom
  source must coerce too; `Record` raises `TypeError` on a non-`str` ID and
  `ValueError` on a blank one.
- **`sql_source` field values keep their database types.** Non-multivalue columns are
  not stringified; a `date` stays a `date`. Only `NULL` is rewritten, to `""`.
- **`obo_source` reads `[Term]` stanzas only.** Typedefs, instance stanzas, and
  header-level tags are ignored; a term without an `id:` is dropped silently.
- **Ontology loaders share field names.** Both emit exactly `label`, `synonyms`,
  `definition`, `parents`, `obsolete`, so one `doc` template — and one
  `exact_fields=("label", "synonyms")` — serves OBO and OWL alike.

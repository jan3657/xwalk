# Ref_zivila food crosswalk plan

## Objective

Populate the six destination columns in `data/Ref_zivila.csv` with defensible,
versioned links to:

- FoodOn
- SNOMED CT
- FoodEx2
- the Hansard/Historical Thesaurus semantic taxonomy used by FoodBase and
  CafeteriaFCD
- ChEMBL, where the source row actually denotes a bioactive molecule
- UMLS, where a meaningful Metathesaurus concept exists

The goal is to process every source row, not to force every row to have a code in
every resource. A blank adjudicated result means that the resource has no suitable
concept. Forced broad or category-only matches are worse than explicit no-matches.

## What is already established

### Source data

`data/Ref_zivila.csv` is a valid UTF-8, comma-delimited CSV with:

- 2,030 data rows;
- 2,030 unique, UUID-shaped `ID` values;
- no malformed-width rows;
- no duplicate bundles across `SHORT_NAME_SLO`, `NAME_SLO`, `TAG_NAME`, and
  `NAME_ENG`;
- a populated `FGNM` category on every row (106 distinct categories);
- at least one Slovenian name on every row;
- an English name on 1,933 rows, leaving 97 without English;
- six destination columns that are currently entirely blank.

The name fields are complementary rather than interchangeable. `NAME_ENG` is the
best retrieval query when present. Slovenian names and `FGNM` are valuable context.
`TAG_NAME` often carries processing state. Some source strings contain curation
notes such as `NE POTRDI` or uncertainty about a translation; these must be retained
as warnings and must not be treated as part of a food name.

### xwalk fit

xwalk is designed for this job. It combines local candidate retrieval with LLM
selection, independent scoring, optional verification, resumable SQLite ledgers,
review overlays, and evaluation against gold mappings. The existing
`examples/cafeteria_fcd` example already demonstrates FoodOn matching.

One xwalk run maps one source collection to one target collection and returns at
most one target ID for a source row. Therefore the baseline design is one independent
job and ledger per vocabulary. It assumes one most-specific canonical result per
destination column. If a column is intended to contain every valid parent, multiple
co-equal concepts, or multiple FoodEx2 facets, that requires an explicit multi-label
extension or a post-processing representation agreed in advance.

### OpenRouter smoke test

The configured OpenRouter endpoint successfully answered through xwalk using:

```yaml
llm:
  kind: openai_compat
  model: minimax/minimax-m3:free
  base_url: https://openrouter.ai/api/v1
  api_key_env: XWALK_TEST_API_KEY
  profile: unknown
```

The smoke response was valid JSON. `profile: unknown` is deliberate: it uses xwalk's
prompt-level JSON contract and local validation without claiming that every routed
free provider enforces JSON Schema.

The free endpoint is a pilot tool, not a production throughput plan. With
`max_attempts: 1` and verification disabled, a normal record uses one selector call
and one scorer call. A 20-row pilot can therefore need 40 calls. Default free-tier
limits can be lower than a single full pilot, and the full six-resource run can need
tens of thousands of calls if retries and verification are enabled.

## Output contract

Keep three distinct artifacts:

1. `Ref_zivila_mapped.csv`: the original columns in the original row order, with
   only adjudicated canonical IDs written to the six destination columns.
2. `mapping_long.csv`: one row per source and vocabulary, including source ID,
   vocabulary, matched ID, target label, confidence, status, reason, method,
   target release, and run fingerprint.
3. One untouched xwalk run directory per vocabulary containing `ledger.sqlite`,
   `mapping.csv`, `results.jsonl`, and `manifest.json`.

Use these canonical cell formats unless a downstream consumer requires something
else:

| Column | Cell value |
|---|---|
| `FOODON` | `FOODON:nnnnnnnn` |
| `SNOMEDCT` | `SNOMEDCT:<SCTID>` |
| `FOODEX` | the release's FoodEx2 code, including an agreed facet encoding if used |
| `HANSARD` | the exact thematic tag, for example `AG.01.h.02.i` |
| `Chembl (Maybe)` | `CHEMBL...` |
| `UMLS (maybe)` | UMLS CUI, for example `C0000005` |

Do not put confidence, labels, explanations, or sentinel strings such as `N/A` in
these cells. Those belong in `mapping_long.csv`. Blank means adjudicated no-match.
If multi-label output is later approved, use a documented `|` separator and never a
comma, since several source names contain commas.

## Directory layout

Keep reusable code and configuration separate from downloaded or licensed data:

```text
examples/ref_zivila/
  prepare_source.py
  prepare_targets.py
  merge_results.py
  jobs/<vocabulary>/job.yaml
  jobs/<vocabulary>/slots.yaml
data/ref_zivila/
  source.csv
  targets/<vocabulary>.*
  snapshots/<vocabulary>/...
  manifests/<vocabulary>.json
  gold/<vocabulary>.csv
runs/ref_zivila/<vocabulary>/...
```

Do not commit SNOMED CT or UMLS licensed content. Each target manifest must record
the release/version, source URL or distribution name, download date, checksum,
license/access note, extraction filters, record count, and extraction script commit.

## Phase 1: freeze semantics before mapping

Make these decisions explicit in a short mapping contract:

1. A source row maps to one most-specific target concept per vocabulary.
2. A broader concept is not accepted merely because no exact concept exists. It is
   either review-only or no-match, depending on the vocabulary policy.
3. Processing state, species, animal part, preparation, and concentration/fat level
   are identity-bearing when the target vocabulary represents them.
4. `FGNM` is supporting context, not the entity label.
5. Several source variants may legitimately map to the same target; xwalk's duplicate
   target report is diagnostic, not a uniqueness violation.
6. For FoodEx2, decide whether the requested cell is a base term only or a complete
   base-plus-facets code. The latter is preferable for detailed foods but cannot be
   generated by stock one-target xwalk unless valid composite codes are materialized
   as target records.
7. For Hansard, choose the deepest defensible thematic tag and decide whether animal
   (`AE`) and plant (`AF`) tags are allowed alongside food/drink (`AG`). The baseline
   one-cell contract chooses one primary tag.
8. UMLS means a CUI, not a source-specific code already present in the Metathesaurus.

## Phase 2: prepare a clean source view

Generate `data/ref_zivila/source.csv`; never use the output CSV itself directly as
the source of independent jobs. This prevents empty or partially populated target
columns from leaking into prompts and changing source fingerprints between runs.

Recommended staging fields:

```text
ID,mention_en,short_name_slo,name_slo,tag_name_slo,fgnm,aliases,curation_note
```

Rules:

- preserve every raw source value in the original CSV;
- normalize Unicode to NFC, trim whitespace, and normalize only comparison keys to
  case-folded forms;
- use `NAME_ENG` as `mention_en` when present;
- extract editorial phrases such as `NE POTRDI`, `ne vem`, and translation comments
  into `curation_note` without silently deleting the evidence;
- for the 97 rows without English, create a reviewed English retrieval alias or add
  a multilingual dense retriever; do not rely on English BM25 to retrieve from a
  Slovenian-only query;
- place all distinct raw names in `aliases`, using `|` as the multivalue separator;
- validate row count, UUID uniqueness, and a stable source checksum on every build.

Suggested source templates:

```yaml
templates:
  query: "{{ mention_en }} {{ aliases | join(' ') }}"
  context: |-
    English name: {{ mention_en }}
    Slovenian full name: {{ name_slo }}
    Slovenian tag name: {{ tag_name_slo }}
    Slovenian short name: {{ short_name_slo }}
    Food group: {{ fgnm }}
    Curation warning: {{ curation_note }}
```

## Phase 3: acquire and normalize target snapshots

### FoodOn

- Start from the stable `foodon.owl` release and pin its checksum/date.
- For the baseline, xwalk can load OWL directly with `id_prefix: FOODON:` and
  `include_obsolete: false`.
- For production, flatten it to a target CSV so parent labels, definitions, exact,
  related, narrow, and broad synonyms, and database cross-references can all be
  indexed and displayed. The stock OWL loader emits parent IDs but does not enrich
  them to labels or expose xrefs.
- Use FoodOn xrefs and any validated FoodOntoMap links as deterministic evidence
  before asking the LLM.

### SNOMED CT

- Obtain a licensed, pinned International or appropriate national edition. Record
  which edition and effective date is used.
- Extract active concepts and active English descriptions from the RF2 Snapshot,
  with the preferred term as `label`, other active terms as `synonyms`, the FSN and
  semantic tag as context, and active `is-a` parents.
- Build a food-relevant subset rather than indexing the entire clinical terminology.
  Include the applicable substance/product branches and only justified organism or
  body-structure concepts for ingredients such as fish species or animal organs.
- Prefix IDs in the normalized target file as `SNOMEDCT:<conceptId>`.
- Never redistribute the extracted target table or labels unless the license permits
  it.

### FoodEx2

- Export a pinned catalogue from EFSA's current Catalogue Browser/distribution and
  retain hierarchy, scope notes, reportable flags, and facet metadata.
- Do not treat FoodEx2 as a flat synonym list. It is a hierarchical, faceted coding
  system.
- Pilot base-term mapping first. For production, either materialize allowed complete
  descriptors as target records or implement a second constrained facet-assignment
  stage after base-term selection.
- Validate generated codes with EFSA's interpreting/checking tooling where available.

### Hansard taxonomy

- This column refers to the Historical Thesaurus thematic codes used in the Hansard,
  FoodBase, and CafeteriaFCD work (`AG.01...`, with possible `AE...`/`AF...`), not the
  separate, much smaller USAS `F1/F2` tagset.
- Build target records from the exact taxonomy version used for the project, including
  tag, category label, parent path, and licensed/available lexical examples.
- Category labels alone are too coarse for retrieval: enrich the records with food
  phrases already annotated in FoodBase/CafeteriaFCD, subject to their licenses and
  with provenance retained.
- Expect high coverage but low granularity. A tag such as fat/oil can correctly cover
  many distinct foods that FoodOn keeps separate.

### ChEMBL feasibility branch

- ChEMBL contains bioactive, drug-like molecules, not ordinary foods. Do not run all
  2,030 rows against it as though it were a food ontology.
- First identify chemical-like rows using `FGNM` and names: additives, acids, vitamins,
  isolated nutrients, sweeteners, and similar substances.
- Normalize ChEMBL molecule records from the pinned database release using molecule
  preferred names and synonyms.
- Require identity at the chemical-substance level. `lemon`, `milk`, or `olive oil`
  must not map to one constituent molecule.
- Compare this branch with ChEBI. ChEBI is likely the more appropriate ontology for
  food chemicals; if ChEMBL IDs are still required, use structure- or identifier-based
  UniChem cross-references after a ChEBI match where possible.

### UMLS feasibility branch

- UMLS is a metathesaurus of many source vocabularies, not a food ontology. Pin the
  release and comply with both the UMLS license and source-vocabulary restrictions.
- Prefer deterministic derivation: after a SNOMED CT match, join the SCTID through
  `MRCONSO.RRF` to its UMLS CUI. Record this as method `snomed_to_umls`, not as an
  independent LLM decision.
- For remaining direct UMLS matches, create a constrained subset using selected source
  vocabularies, English preferred terms/synonyms, suppression flags, and relevant
  semantic types. Do not index the complete Metathesaurus indiscriminately.
- Keep the CUI in the output cell and retain every supporting source atom in the audit
  table.

## Phase 4: deterministic evidence before LLM calls

For each vocabulary, resolve and record these tiers in order:

1. existing authoritative xref from a previously accepted target;
2. normalized exact preferred-label match;
3. normalized exact synonym match;
4. approved external crosswalk (for example SCTID to CUI);
5. xwalk retrieval plus LLM selection/scoring;
6. human review;
7. explicit no-match.

Do not silently auto-accept every lexical exact match. Exact strings can still differ
by food state or sense. Either validate high-risk exact matches with the scorer or
restrict auto-acceptance to a manually reviewed rule set. Every output must retain a
`method` and evidence provenance in `mapping_long.csv`.

## Phase 5: configure one xwalk job per vocabulary

Use a common job shape, with vocabulary-specific target templates and prompt slots.
A safe MiniMax pilot policy is:

```yaml
retrievers:
  - kind: bm25
    name: bm25
    limit: 50
    exact_fields: [label, synonyms]

llm:
  kind: openai_compat
  model: minimax/minimax-m3:free
  base_url: https://openrouter.ai/api/v1
  api_key_env: XWALK_TEST_API_KEY
  profile: unknown

policy:
  max_attempts: 1
  accept_at: 0.85
  review_floor: 0.50
  verify_band: null
  audit_rate: 0.0
  concurrency: 1

selector:
  max_candidates: 25
  max_candidate_tokens: 8000
```

The pilot thresholds are conservative placeholders, not calibrated truth. Production
settings should normally restore 2-3 attempts, conditional verification around the
accept threshold, a small audit rate, and higher concurrency only after provider rate
limits are known.

Target documents should index label, synonyms, definition/scope note, and parent labels.
Candidate blocks should show the canonical ID, label, synonyms, definition/scope,
and hierarchy path. Each vocabulary needs its own slots file because an acceptable
specificity relation in Hansard is not acceptable in FoodOn, and FoodEx2 facet rules do
not apply to SNOMED CT.

All slots should enforce:

- food identity before lexical similarity;
- processing state, source organism, animal part, and preparation distinctions;
- no ingredient-to-dish or dish-to-ingredient substitution;
- no constituent-molecule mapping for whole foods;
- abstention when no candidate denotes the same entity;
- broader/narrower matches below the automatic threshold unless the mapping contract
  explicitly allows them.

## Phase 6: staged pilot under the free limit

### Technical pilot

Already complete: one structured-response transport call succeeded.

### Retrieval-only check

Before spending more calls, manually select 20-30 rows spanning:

- exact English labels;
- processed foods;
- raw/cooked pairs;
- species and animal parts;
- composite dishes;
- additives/chemicals;
- rows without English;
- rows carrying curation warnings;
- common and rare `FGNM` groups.

Inspect top-25 and top-50 candidates for every target. Fix source aliases, target
documents, and subset filters until the correct target is usually present. An LLM
cannot recover a target that retrieval never presents.

### MiniMax quality pilot

- Run 2 shared records against each ready vocabulary to validate every job contract
  (normally no more than 20 calls for five vocabularies).
- Use the remaining daily allowance for a 10-12 row FoodOn pilot (normally no more
  than 20-24 calls).
- Keep `max_attempts: 1`, `verify_band: null`, and `concurrency: 1` for this stage.
- Manually adjudicate every pilot result, including no-matches.
- Repeat a small fixed sample in a separate run to measure instability, since the
  free route may not provide deterministic backend behavior.
- Do not scale MiniMax until it meets the precision gate on gold data and its JSON,
  abstention, latency, and rate-limit behavior are acceptable.

## Phase 7: gold set and gates

Create a shared, stratified gold set before a full run. Start with at least 100-150
source rows covering all high-volume `FGNM` groups plus every identified risk stratum.
For each vocabulary, label the correct ID or an explicit empty `gold_ids` value.
Review ambiguous records with a food-domain expert and record target release versions.

Use xwalk evaluation to inspect:

- accepted precision;
- automatic coverage and review rate;
- no-match precision and recall;
- recall at any status;
- never-retrieved, selector-truncated, and LLM-misjudged failures;
- calibration/threshold curves;
- behavior by `FGNM`, English availability, specificity, and curation-warning strata.

Recommended launch gates:

- candidate recall@50 at least 0.98 on non-empty gold mappings;
- accepted precision at least 0.95 overall, with no high-risk stratum below 0.90;
- no-match behavior explicitly measured rather than folded into accuracy;
- malformed/provider failures below 1%;
- a domain expert approves all prompt rules and the FoodEx2 output semantics.

If calibration is poor, do not tune `accept_at` as if confidence were meaningful.
Improve prompts/model or send all LLM matches to review.

## Phase 8: production and review

For each vocabulary:

```bash
set -a
source .env
set +a

.venv/bin/xwalk index \
  --job examples/ref_zivila/jobs/<vocabulary>/job.yaml \
  --out runs/ref_zivila/<vocabulary>/index

.venv/bin/xwalk match \
  --job examples/ref_zivila/jobs/<vocabulary>/job.yaml \
  --index runs/ref_zivila/<vocabulary>/index \
  --out runs/ref_zivila/<vocabulary>

.venv/bin/xwalk review export \
  --run runs/ref_zivila/<vocabulary> \
  --out runs/ref_zivila/<vocabulary>/review.csv
```

Run in resumable batches. Preserve every ledger and manifest. Review all
`needs_review` results, all curation-warning rows, all unsupported-resource proposals,
and a random audit sample of automatic matches. Apply review as xwalk's immutable
overlay; never hand-edit `mapping.csv`.

After adjudication, merge results by `ID` into a new CSV, validate that it contains
exactly 2,030 rows in original order, validate every nonblank ID against the pinned
target snapshot, and emit coverage/status summaries by vocabulary and `FGNM`.

## Phase 9: cross-resource consistency checks

These checks flag review candidates; they do not manufacture mappings:

- a SNOMED CT code and UMLS CUI must agree with `MRCONSO` when the UMLS result was
  derived from SNOMED CT;
- FoodOn xrefs should agree with accepted SNOMED/UMLS links where a current xref exists;
- ChEMBL must only occur on chemical-substance rows and should have structure or
  authoritative identifier evidence where possible;
- FoodEx2 facets must agree with explicit source properties such as raw/cooked,
  dried, canned, sweetened, animal species, and body part;
- Hansard should be at an appropriate depth but is expected to be coarser than FoodOn;
- broad target reuse across many source rows is reviewed by category, not rejected
  merely because xwalk reports duplicate targets.

## Immediate implementation order

1. Approve the one-result-per-column and FoodEx2 facet contracts.
2. Add source-preparation and invariant tests.
3. Acquire and pin FoodOn; build the enriched target and FoodOn job.
4. Run retrieval-only analysis and the 10-12 row MiniMax FoodOn pilot.
5. Create and adjudicate the stratified gold set; calibrate the FoodOn run.
6. Acquire licensed SNOMED CT and UMLS snapshots; build SNOMED extraction and
   deterministic SCTID-to-CUI mapping.
7. Acquire/export FoodEx2 and implement base-term first, facet assignment second.
8. Build the versioned Hansard target from taxonomy plus licensed annotated phrases.
9. Run a chemical-row feasibility study comparing ChEMBL with ChEBI; retain ChEMBL
   only if it adds meaningful identifiers.
10. Run, review, cross-check, and merge each production vocabulary independently.

## Main blockers and decisions

- SNOMED CT and UMLS require licensed access and careful redistribution controls.
- FoodEx2 catalogue access/export and the exact representation of facets must be
  settled before coding.
- The exact Hansard thematic taxonomy snapshot and usable lexical material must be
  located and licensed; USAS `F1/F2` is not a substitute.
- ChEMBL is probably inappropriate for most rows. ChEBI should be evaluated as the
  chemical ontology even if a ChEMBL cross-reference column is ultimately retained.
- xwalk currently returns one target per run. True multi-label output needs either an
  explicit extension or a constrained iterative/post-processing stage.
- The free MiniMax route has enough capacity for smoke tests and small pilots, not a
  reliable end-to-end production run at default free limits.

# The NER site on real-world texts, 4 November 2026

Visitors reported "not JSON" errors with the GPT and Anthropic models on paper abstracts.
This records the test that reproduced them, what was wrong, and what changed. Texts
are not committed (abstract copyright varies); they are rebuilt from the sources below.

## Corpus

| name | source | characters |
|---|---|---|
| diet_diabetes | PubMed 25940230 | 1,713 |
| drug_interaction | PubMed 16513449 | 2,745 |
| food_allergy | PubMed 25705822 | 2,364 |
| cancer_genes | PubMed 35026309 | 1,525 |
| microbiome_polyphenol | PubMed 32605083 | 1,320 |
| structured_long | PubMed 39861407 | 1,697 |
| fermented_food | PubMed 32749526 | 1,493 |
| toxicology | PubMed 17687698 | 1,516 |
| wiki_hummus | Wikipedia en:Hummus (summary) | 216 |
| wiki_mediterranean | Wikipedia en:Mediterranean_diet (summary) | 627 |
| wiki_cheese | Wikipedia en:Cheese (summary) | 569 |
| wiki_sl_potica | Wikipedia sl:Potica (summary, Slovenian) | 121 |
| long_combo | the first three abstracts joined, cut to 4,990 | 4,990 |

Each text was posted to `/api/ner` with each extraction model (Jev selecting), and five
of them with each LLM selector: first against production (the site's own key), then
against a local site with the production configuration (full Neon catalogs, same
models) and a temporary key, before and after the fixes.

## What was wrong, and the fixes

| # | Problem | Evidence | Fix |
|---|---|---|---|
| 1 | GPT-6 Luna and Claude Haiku 5.5 extraction failed: "not a JSON object (cut off at the token limit)" | production: Luna 3 of 13 texts, Haiku 2 of 13; the raw response had `reasoning_tokens: 2048`, empty content | OpenRouter `reasoning: {enabled: false}` on every site LLM request (`service.OpenRouterClient`); after: Luna 13 of 13, 21 to 25 entities on abstracts, 4 to 6 s |
| 2 | Claude Haiku 5.5 under a strict `json_schema` with reasoning off returned one entity with zero offsets | direct probe on PubMed 35026309 | Anthropic models get the `anthropic-compat` profile (JSON asked in the prompt) |
| 3 | Claude Haiku 5.5 still unreliable for extraction | same text, three runs: 19, 4, 17 entities; stops after the first sentence at random; once non-JSON; a "read the whole text" rule did not help; with reasoning on it never leaves reasoning (8,000 of 8,000 tokens) | removed from the production extraction models; kept as a selector, where it works (15 of 15 requests, no failed pick) |
| 4 | Genes extracted in full-catalog mode although the prompt no longer asks for them, using the 25-entity budget | Gemini typed 6 to 9 mentions `gene` on the cancer and drug abstracts | the system prompt and the JSON schema list only the routed types plus "other"; a stray `gene` becomes "other" |
| 5 | Jev "failed" picks that were not failures | "same 0.43; NONE 0.69": Jev leaned to "none of these" just under `none_at` 0.70, which the policy calls needs_review with no term | a review pick may have no term ("Unsure, needs review" on the page); a real Jev error now names the error instead of the partial signals |
| 6 | Mapping took 15 to 76 s on 25-entity texts; one request 102 s, at the 100 s deadline | one embeddings request per entity, 4 at a time: 25 singles 52 to 67 s, one batch of 25 3 to 10 s (measured) | `indexes.PrefetchEncoder`: mapping embeds every entity's query in one request first; the searches read their vectors from memory. After: map 0.5 to 8.5 s on the same texts |
| 7 | `/api/health` said `llm: ok` with an expired key while every POST failed with 502 | the morning's outage | health asks OpenRouter's free `GET /api/v1/key` (cached 5 min, 1 min when refused) and reports a refused or spent key with the reason |

## Results

Extraction, Jev selecting (13 texts each):

| model | before: answered | after: answered | after: mean entities |
|---|---|---|---|
| google/gemini-3.1-flash-lite | 13 | 13 | 17.8 (identical across three runs) |
| openai/gpt-6-luna | 3 | 13 | 17.0 (21 to 25 on long abstracts) |
| anthropic/claude-haiku-5.5 | 2 | 13 | 9.7 (erratic: see fix 3) |

LLM selectors after the fixes (Gemini extracting; 5 texts each): GPT-6 Luna, Claude
Haiku 5.5 and Gemini 3.1 Flash Lite all answered every entity, no failed pick; about
25,000 to 150,000 selector prompt tokens a request (about $0.003 to $0.015), where Jev
reported $0.001 to $0.003 for the same texts.

## Not changed (for the operator to decide)

- **The 25-entity limit** cuts long abstracts: Gemini and Luna stop at 25 with 10 to 40%
  of the text still unread, and "other" mentions (people, places, study names) count
  towards the 25. Options: raise `NER_SITE_MAX_ENTITIES`, or leave "other" out of the
  default prompt.
- **Slovenian text**: extraction works (Potica, pogača), but the ontologies are English,
  so nothing matches.

# Recall and Cost on the Decider Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the retrieval-recall gap on the decider path, cut its cost by about two thirds, and make per-domain threshold fitting routine, with a pass/fail test for every improvement.

**Architecture:** One offline recall harness turns retrieval work into tests. BM25 gains an analyzer and fuzzy terms; a dense multilingual retriever joins it; a graph hop and a conditional LLM rewrite extend the candidate set only on records the screen missed. The screen's repeated preamble moves into the shared state. `xwalk fit` gains a hold-out split and writes fitted policies back into job files.

**Tech Stack:** Python 3.10+, tantivy 0.26 (`TextAnalyzerBuilder`, `fuzzy_term_query`), `xwalk[dense]` (sentence-transformers, torch, faiss-cpu) installed with `uv` through the site proxy, pytest with `asyncio_mode = "auto"`.

**Spec:** `docs/superpowers/specs/2026-09-23-jev-recall-and-cost-design.md`

## Global Constraints

- Branch `jev-recall-cost`, off `jev-decider`. Python floor 3.10; line length 100; every task passes `ruff check src tests scripts`, `ruff format --check` on its files, `mypy`, and the full `pytest` suite apart from the three pre-existing failures from the user's untracked files (docs index orphan, ruff and mypy on `scripts/compare_runs.py`).
- The LLM path stays behaviourally unchanged; every new retriever and decider option defaults off.
- No decision moves out of Jev. The rewrite LLM (A3) only proposes query strings.
- Every improvement has a test whose threshold is written in the test. Live tests are marked `integration`, skip without `XWALK_TEST_API_KEY`, and cost cents; the plan says the expected cost per task.
- Shell commands that reach the network need the proxy: `source ./_env.sh` or export `https_proxy=http://www-proxy.ijs.si:8080`.
- Commit per task, message ending `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Do not push unless asked.

## Baselines the tests compare against

Recall@200 with the shipped BM25 (from the decider evaluation, first attempt): cafeteria_fcd 0.94, chebi 0.60, ncbi_disease 0.58, nlm_gene 0.70; mean 0.705. Decider accuracy at the shipped policy: 0.90 / 0.58 / 0.46 / 0.58. Ref_zivila: 62,740 prompt tokens per record.

---

### Task 1: The retrieval-recall harness (A0)

**Files:** Create `src/xwalk/evaluate/recall.py`, `scripts/retrieval_recall.py`, `tests/test_retrieval_recall.py`.

**Interfaces:**
- `retrieval_recall(job: JobSpec, gold: GoldSet, *, index_dir: Path, ks: Sequence[int] = (10, 50, 200)) -> dict[int, float]` builds the job's retrievers into `index_dir`, renders `templates.render_queries` per source, runs `xwalk.retrieve.retrieve`, and returns recall@k = fraction of labelled rows (excluding gold no-match rows) whose gold set intersects the top-k fused candidates.
- `scripts/retrieval_recall.py JOB GOLD [--k 200]` prints one line per k.
- `tests/test_retrieval_recall.py` defines `SAMPLES = {"cafeteria_fcd": 0.94, "chebi": 0.60, "ncbi_disease": 0.58, "nlm_gene": 0.70}` and a helper `recall_for(example, mutate=None)` that loads `examples/<ex>/job_jev.yaml`, applies `mutate(data)` to the YAML dict if given, builds into `tmp_path`, and returns recall@200.

- [ ] Test (pass/fail): `test_shipped_bm25_reproduces_the_baselines` asserts `abs(recall_for(ex) - SAMPLES[ex]) <= 0.02` for every sample. This pins the harness to the numbers the evaluation reported; if it fails, the harness is wrong, not the retriever.
- [ ] Implement, run, commit `feat(evaluate): offline retrieval-recall harness`.

### Task 2: Hold-out and write-back for `xwalk fit` (C1)

**Files:** Modify `src/xwalk/decide/fit.py`, `src/xwalk/cli/main.py`; Test `tests/test_decide_fit.py`, `tests/test_cli.py`.

**Interfaces:**
- `fit_thresholds(..., choose_grid: Sequence[float] = (0.3, 0.5, 0.7))` adds `choose_at` to `FitPoint` and the sweep.
- `fit_holdout(results, gold, *, base, seed_fingerprint: str, target_precision, ...) -> tuple[FitReport, FitPoint | None]`: splits labelled ids with `Partitioner` (existing `evaluate/partition.py`) into dev/holdout halves, fits on dev, and evaluates the recommended point on holdout, returning the dev report and the holdout point (accepted, correct, precision, coverage).
- CLI: `fit --holdout` prints the holdout line; `fit --write-job OUT.yaml` requires `--job` and writes a copy with `policy.accept_at`, `policy.property_floor`, `policy.choose_at` set to the recommendation.

- [ ] Tests (pass/fail): (a) `test_holdout_split_is_deterministic_and_disjoint`: two calls give identical halves, no id in both, sizes differ by at most one. (b) `test_holdout_reports_the_recommended_point_on_unseen_rows`: on the synthetic results from the existing fit tests duplicated to 16 rows, the holdout point's `accepted + correct` are computed only over holdout ids (assert against a hand count). (c) `test_write_job_round_trips`: after `fit --job fixtures/job_tiny_jev.yaml --write-job out.yaml`, `load_job(out.yaml).build_decision_policy().accept_at == report.recommended.accept_at` (use the `_build_decider` monkeypatch and a `FakeDecider` run as in the existing CLI fit tests).
- [ ] Implement, run, commit `feat(fit): hold-out evaluation and write-back of fitted thresholds`.

### Task 3: Ref_zivila gold from the adjudication (C2)

**Files:** Create `examples/ref_zivila/gold_adjudicated.csv`, `scripts/adjudication_to_gold.py`; Test `tests/test_ref_zivila_gold.py`.

- [ ] Script: read `docs/superpowers/specs/2026-09-22-jev-adjudicated-sample.csv`; emit `source_id,gold_ids` with `jev` → `jev_id`, `qwen` → `qwen_id`, `both_acceptable` → `qwen_id|jev_id`, skip `both_wrong` and `unsure`; write a leading comment line `# biased sample: only Qwen/Jev disagreements from the 2026-09-22 run` (check `load_gold_csv` tolerates a comment line; if not, put the note in the plan and the README instead).
- [ ] Test (pass/fail): `load_gold_csv` on the file yields at least 35 labelled rows, every id starts with `FOODON:`, and every `both_acceptable` row has two ids.
- [ ] Evaluation (not a unit test): `xwalk fit --run runs/ref_zivila/foodon_jev --gold examples/ref_zivila/gold_adjudicated.csv --job examples/ref_zivila/jobs/foodon/job_jev.yaml --holdout`. Record the recommended point in the spec's Results section for this plan.
- [ ] Commit `feat(ref_zivila): gold file from the adjudicated disagreements`.

### Task 4: Stemming and fuzzy terms for BM25 (A1)

**Files:** Modify `src/xwalk/retrieval/bm25.py`, `src/xwalk/config.py` (`RetrieverSpec.analyzer: Literal["default", "en_stem"] = "default"`, `fuzzy_distance: int = 0`); Test `tests/test_bm25.py`, `tests/test_retrieval_recall.py`.

**Implementation notes:** register the analyzer with `index.register_tokenizer("en_stem", tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple()).filter(tantivy.Filter.lowercase()).filter(tantivy.Filter.ascii_fold()).filter(tantivy.Filter.stemmer("english")).build())` and declare the `text` field with `tokenizer_name=analyzer` in `_build_schema(analyzer)`; store `analyzer` and `fuzzy_distance` in the index metadata and the fingerprint. In `_search_sync`, when `fuzzy_distance > 0`, add `Query.boost_query(Query.fuzzy_term_query(schema, "text", term, distance=fuzzy_distance, transposition_cost_one=True), 0.5)` as a `Should` clause for each whitespace-split query term of length 5 or more. Check the exact tantivy 0.26 Python signatures in the venv before coding (`help(tantivy.TextAnalyzerBuilder)`, `help(tantivy.Query.fuzzy_term_query)`).

- [ ] Unit tests (pass/fail): with a three-record index (`anesthetic`, `anaesthetic agents`, `glucose`): default analyzer, query `anesthetics` returns nothing; `analyzer: en_stem` returns the `anesthetic` record; `fuzzy_distance: 1` with the default analyzer makes query `anaesthetic` return both spellings; the fingerprint differs across the three configurations; an index built with one analyzer refuses to open with another (metadata check).
- [ ] Recall gate (pass/fail): `test_stem_and_fuzzy_lift_recall` uses `recall_for(ex, mutate=set analyzer en_stem and fuzzy_distance 1)`; asserts no sample below `SAMPLES[ex] - 0.0` and `mean >= 0.705 + 0.03`. If it fails, report the per-sample numbers and stop the task for a ruling rather than loosening the threshold.
- [ ] Flip the four sample `job_jev.yaml` files and the Ref_zivila job to the new settings only if the gate passed. Commit `feat(bm25): stemming analyzer and fuzzy terms, opt-in`.

### Task 5: Hoist the screen preamble (B1)

**Files:** Modify `src/xwalk/decide/questions.py` (`questions_version` 2; `rules_state() -> dict`; `screen_question(key)` short form), `src/xwalk/stages/screen.py` (state gains `"rules": questions.rules_state()`); Test `tests/test_decide_questions.py`, `tests/test_screen.py`, `tests/test_decide_integration.py`; docs `docs/reference/decide.md` (Screener section, which also needs `ScreenFailed` and the oversized-candidate note from the parked items).

- [ ] Size test (pass/fail): build a 50-candidate chunk request twice with the Ref_zivila slots (`examples/ref_zivila/jobs/foodon/slots.yaml`): once through a `QuestionSet` frozen at version 1 (keep a `legacy_screen_question` for the test only, or construct the old string in the test), once through the new path; `len(json.dumps(new)) <= 0.45 * len(json.dumps(old))`.
- [ ] Unit tests: the new instruction names `candidates.<key>`, `source`, `rules.entity`, `rules.hard_rules`, `rules.same`, `rules.different` and nothing from `domain_brief` verbatim; `rules_state()` carries every hard rule verbatim; the fingerprint changes with the version bump.
- [ ] Live gate (integration, about $0.04): `scripts/run_jev_eval.sh` on the four samples into `runs/jev_eval_v2/`; a test-like script `scripts/compare_gates.py` prints per-domain accuracy deltas against the recorded baselines (0.90 / 0.58 / 0.46 / 0.58) and tokens per record from the manifests; pass if every domain is within −0.02 and the mean is not lower, and tokens per record are at least 50% lower. If accuracy fails, implement the alternative: a once-per-chunk `rules` question-less instruction is not possible in the API, so instead put the rules in the state and keep only `rules.same` / `rules.different` as per-noul `criteria` (about 40 tokens), and re-gate.
- [ ] Commit `feat(decide): hoist the screen preamble into the shared state`.

### Task 6: Dense multilingual retrieval (A2)

**Files:** venv install; modify the four sample `job_jev.yaml` and `examples/ref_zivila/jobs/foodon/job_jev.yaml`; create `examples/ref_zivila/gold_slo_mini.csv`; Test `tests/test_retrieval_recall.py` (marked `dense`), `tests/test_ref_zivila_gold.py`.

- [ ] Install: `source ./_env.sh; uv pip install --python .venv/bin/python -e '.[dense]'`; pull the model once with `SentenceTransformerEncoder("intfloat/multilingual-e5-small")`. Record versions in the report. If the download fails through the proxy, fall back to `BAAI/bge-small-en-v1.5` (already cached) for the English gate and report the Slovenian gate as blocked.
- [ ] Slovenian mini gold: ten rows `source_id,gold_ids` for Ref_zivila source ids whose `mention_en` is empty, hand-mapped by looking the Slovenian name up against `data/ref_zivila/targets/foodon.csv` labels (the probe already established `Pšenična moka` → wheat flour, `Sveži grah` → green pea, `Kodrolistni ohrovt, zamrznjeno` → kale (frozen), `Svinjina, stegno` → pork leg). Note in the file header that the ids were chosen by a person from labels, not from a run.
- [ ] Gates (pass/fail, `dense` marker): English, `recall_for(ex, mutate=add dense retriever)` mean at least 0.08 above the Task 4 mean and no sample more than 0.02 below Task 4; Slovenian, dense recall@50 on the mini gold at least 0.7 and BM25-only at most 0.2.
- [ ] Commit `feat(examples): dense multilingual retriever beside BM25 on the decider jobs`.

### Task 7: Graph hop over the target (A4), with a go/no-go

**Files:** Create `scripts/hop_headroom.py`; then (only on go) modify `src/xwalk/decide/matcher.py`, `src/xwalk/config.py` (`DeciderSpec.hierarchy_field: str | None = None`, `hop_band`, `hop_top`), create `src/xwalk/decide/hop.py` (`Neighbourhood.from_store(store, field) -> parents/children maps`, `expand(seed_ids, k) -> list[str]`); Test `tests/test_decide_hop.py`, `tests/test_decide_matcher.py`.

- [ ] Go/no-go (pass/fail on the numbers, not the code): `scripts/hop_headroom.py RUN_DIR GOLD FIELD` reads a decider ledger and reports, over labelled misses, the share whose gold id is a parent or child of any retrieved candidate. Run it on `runs/jev_eval/cafeteria_fcd` with its gold and on `runs/ref_zivila/foodon_jev` with `examples/ref_zivila/gold_adjudicated.csv`. Go if either share is at least 0.10; otherwise record both numbers in the spec and skip the rest of this task.
- [ ] Unit tests: hop happens only when the first screen's best is inside `hop_band`; only the top `hop_top` seeds expand; already-present ids are not re-added; exactly one extra screen chunk; the expansion candidates carry a synthetic `RetrievalHit(retriever="hop", rank=i)` so traces show where they came from.
- [ ] Live gate (integration, about $0.01): cafeteria_fcd accuracy not lower than Task 5's and at least one previously missed gold row matched.
- [ ] Commit `feat(decide): one-hop expansion over the target hierarchy, opt-in`.

### Task 8: Conditional query rewrite (A3)

**Files:** Modify `src/xwalk/config.py` (`DeciderSpec.rewrite: LLMSpec | None = None`), `src/xwalk/decide/matcher.py` (an optional `QueryRewriter`; second retrieve+screen on miss; merge by record id keeping the max probability); Test `tests/test_decide_matcher.py`, `tests/test_config.py`.

- [ ] Unit tests (pass/fail) with `FakeLLM` scripted to return two queries: the rewriter is called only when the first screen's best is below `screen_floor` or no candidates were found; at most once per record; the second screen's probabilities are merged (a record seen in both keeps the higher); `Attempt.query` lists all queries tried, joined by ` | `; usage sums both screens plus the rewrite call; a rewrite LLM error is a note, not a failure (the first screen's result stands).
- [ ] Live gate (integration, about $0.06): ncbi_disease and nlm_gene accuracy at least 10 points above the Task 6 run, matched precision unchanged.
- [ ] Commit `feat(decide): rewrite queries with an LLM only on screen misses`.

### Task 9: Final measurement and results

- [ ] Re-run the four samples with everything on (Tasks 4, 5, 6, 7 if go, 8) into `runs/jev_eval_v3/`; run `fit --holdout --write-job` for each and commit the written policies into the sample jobs; re-run Ref_zivila once (about $2) with the fitted policy and compare against the 2026-09-22 run with `scripts/compare_runs.py`.
- [ ] Append a Results section to the spec: recall@200 per task per domain (from the harness), accuracy and precision per domain at each gate, tokens and cost per record before and after, Ref_zivila status counts before and after, and which of A3/A4 earned their keep.
- [ ] Commit `eval: recall and cost improvements measured on the gold samples and Ref_zivila`.

## Deferred

The two parked items from the decider branch's final review are folded into Task 5's docs step (`ScreenFailed`, oversized-candidate note) and Task 5's screen edit (propagate outer cancellation into chunk tasks: wrap the `asyncio.wait` in `try/except BaseException: await _cancel(tasks); raise`).

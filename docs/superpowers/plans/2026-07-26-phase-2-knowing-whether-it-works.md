# xwalk Phase 2 — Knowing Whether It Works Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a run into an answer to "is this good, and where should I spend effort?" — operational metrics, a three-way failure decomposition, and a prompt optimizer that improves slots against labelled data without leaking the test set.

**Architecture:** `evaluate/` reads a completed run's ledger plus a gold file and produces numbers; nothing in it re-runs the matcher. `prompts/author.py` asks a strong model for *slots only*, validated through Phase 1's `contract.py` before anything is written. `prompts/optimize.py` drives rounds over three partitions — failures from **prompt-train** are shown to the optimising model, **validation** chooses the retained round and the stopping point, **test** is evaluated exactly once at the end.

**Tech Stack:** Everything from Phase 1, plus nothing new. No numpy, no sklearn — the metrics are counts and ratios, and adding a numerical stack to a base install that deliberately excludes torch would be inconsistent.

**Prerequisite:** Phase 1 is complete and its "Definition of done" checklist passes.

**Spec:** `docs/superpowers/specs/2026-07-26-xwalk-generalized-matching-library-design.md`, sections "Prompt authoring and optimization" and "Evaluation and diagnostics".

## Global Constraints

Every task's requirements implicitly include this section, in addition to Phase 1's.

- **The test partition is evaluated once**, after the final prompt is selected. Never per round. Reporting it each round turns it into a second validation set, which is the same mistake one level up.
- **Nothing in `evaluate/` calls an LLM or a retriever.** It reads the ledger. This keeps evaluation free, repeatable, and safe to run on a colleague's machine.
- **Gold ID normalization and alias expansion are user-supplied callables**, never library-owned hooks. This is the whole point of the rewrite — `expand_ctd_eval_ids` lives in *your* code.
- **The optimizer never writes a prompt that fails `validate_contract`.** A bad round produces a poor rubric, never a broken prompt.
- **Any command that spends money prints an estimated call count and cost before starting**, and honours `max_calls`.
- **`mypy --strict`, `ruff check`, `ruff format --check` must pass** at the end of every task. Commit at the end of every task.
- **Metric definitions live in one place and are documented in the docstring** where they are computed. A metric whose definition is folded into a caller is a metric nobody can trust.

---

## Plan review — corrections applied 2026-07-28

Reviewed against the as-built Phase 1 tree (`src/xwalk/`, 406 tests green at `515f792`).
The code blocks below have been edited in place; this section records *why*, so a later
reader does not "restore" a defect.

### Already done — do not re-apply

**Task 2 Step 1** (`elapsed_seconds` on `Attempt`/`MatchResult`, populated in `matcher.py`,
carried in `serde.py`, exported by `batch.py`) shipped in commit `515f792`, together with
`finish_reason`, which this plan never anticipated. Re-applying it duplicates fields and
breaks `serde`. Step 1 is now a verification step only.

### Defects that would have failed at runtime

| # | Where | Defect | Fix |
|---|---|---|---|
| 1 | Task 2, `test_calibration_warning_fires_when_bands_overlap` | 1 correct / 2 incorrect confidences, but `_MIN_CALIBRATION_SAMPLES = 3` requires 3 of **each**. `calibration_warning` returns `None`, so the test fails. | Fixture widened to 3 correct + 3 incorrect with overlapping means. |
| 2 | Task 2, `test_no_calibration_warning_when_bands_separate` | 2 correct / 1 incorrect — passes, but because there was *not enough data*, not because the bands separate. A green test pinning nothing. | Widened to 3 + 3 with genuinely separated means. |
| 3 | Task 5, `test_scorer_ignores_never_retrieved_cases` | Asserts `[]`, but `NEVER` has `status=MATCHED` with a wrong `matched_id`, which is `wrong_accept` — the implementation keeps it, and **is right to**. A confident accept of a wrong candidate when nothing correct was retrieved is the canonical scorer failure: rejecting in that situation is the scorer's entire job. Excluding it would blind the optimiser to abstention behaviour. | Test inverted and renamed; the "model never saw the right answer" argument applies to the **selector**, which must choose among what it was shown, not to the scorer. |
| 4 | Task 7, `make_factory` | `sequence` is indexed per *factory call*, and the call order is baseline → train → candidate → test. `[False, True]` therefore makes the prompt-train run **correct**, which yields no failures, so the optimiser exits with "no failures" before ever calling the model. `test_an_improving_round_is_retained` and `test_a_round_producing_invalid_slots_is_skipped_not_fatal` both fail. | Factory rewritten to be driven by *accuracy per prompt*, keyed on `slots.domain_brief`, so correctness is a property of the prompt under test rather than of call ordering. |
| 5 | Task 7, `test_estimate_calls_counts_the_test_partition_once` | `estimate_calls(rounds=1, 0,0,100) == 201` and `(rounds=4, …) == 204` — the per-round optimiser call makes them differ, so the assertion is simply false. | Rewritten to compare the *test partition's contribution* (`with_test - without_test`) across round counts, which is what the property actually says. |
| 6 | Task 7, `test_only_prompt_train_failures_reach_the_optimising_model` | Source ids `s0…s29` — `"s2" in prompt` is satisfied by `"s20"`, so a train id can spuriously fail the leak check. | Ids zero-padded to `s00…s29`. |
| 7 | Task 8, `test_a_calibration_warning_appears_prominently_when_present` | Same defect as #1: 1 correct / 2 incorrect. | Widened to 3 + 3. |

### Lint and type gates (would fail every task's `ruff check`)

- Every module and test block imports ABCs from `typing` (`Iterable`, `Mapping`, `Sequence`,
  `Callable`, `AbstractSet`). `ruff` 0.16 raises **UP035** on all of them. All rewritten to
  `collections.abc` — exactly the sweep Phase 1 needed.
- `tests/test_metrics.py`, `tests/test_ceiling.py`, `tests/test_failures.py` import `pytest`
  without using it → **F401**.
- `partition.py` uses `.encode("utf-8")` → **UP012**.
- `optimize.py` imports `xwalk.llm.parsing` after `xwalk.prompts.contract` → **I001**.
- Over-length lines in `metrics.evaluate_results` (the `recalled` generator) and
  `Partitioner.__post_init__` → **E501**.
- `GoldSet.__iter__` is annotated `-> Iterable[str]` but returns an iterator; the iteration
  protocol wants `Iterator[str]`.
- `optimize.py`'s `type(case)(**{**case.__dict__, …})` defeats `mypy --strict`. The plan
  already flags this in Task 7 Step 3; it is now written correctly in Step 2 as
  `dataclasses.replace`, and `_render_optimiser_prompt` takes `Sequence[FailureCase]`
  rather than `Sequence[Any]`.

### Drift between "Interfaces" and the code blocks

- Task 2 advertises `evaluate_results(results, gold, *, duplicate_targets=None)`; no such
  parameter exists or is needed (`Ledger.duplicate_targets` already covers the ledger case).
  Removed from the interface line.
- Task 4 omits `partition_of` and `ids_in`, which Tasks 7 and 8 both import. Added.
- Task 5's `FailureCase` field list omits `source_fields`, `status`, and `reason`, all of
  which the code block defines and `render_failure` reads. Added.
- Task 7's `estimate_calls` charges the baseline for `n_prompt_train + n_validation`, but the
  baseline only runs validation. Corrected to `n_validation`.

### Test-count claims

Stated vs actual: Task 2 `24 → 23`, Task 7 `16 → 15`. Tasks 1, 3, 4, 5, 6 are accurate.

### Task 8's worked example

The example README invokes `python -m examples.chemistry.run` and
`python -m examples.chemistry.evaluate`, neither of which the plan creates. Rather than
shipping a README that documents commands that do not exist, Task 8 now writes real
`examples/chemistry/` artefacts (`slots.yaml`, `templates.yaml`, `gold.csv`, `targets.csv`,
`sources.csv`, `run.py`, `evaluate_run.py`) and the README describes what is actually there.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/xwalk/evaluate/__init__.py` | Public re-exports |
| `src/xwalk/evaluate/gold.py` | `GoldSet`, loading, user-supplied normalization/expansion |
| `src/xwalk/evaluate/metrics.py` | `EvalReport` and every operational number |
| `src/xwalk/evaluate/ceiling.py` | never-retrieved / truncated / misjudged decomposition |
| `src/xwalk/evaluate/partition.py` | deterministic three-way split |
| `src/xwalk/evaluate/failures.py` | role-specific failure selection |
| `src/xwalk/prompts/author.py` | LLM-assisted drafting of slots |
| `src/xwalk/prompts/optimize.py` | the three-partition optimization loop |
| `src/xwalk/records.py` (modify) | add `elapsed_seconds` to `Attempt` and `MatchResult` |
| `tests/fixtures/gold_tiny.csv` | gold labels for the Phase 1 fixtures |

---

## Task 1: Gold labels

**Files:**
- Create: `src/xwalk/evaluate/__init__.py`, `src/xwalk/evaluate/gold.py`
- Create: `tests/fixtures/gold_tiny.csv`
- Test: `tests/test_gold.py`

**Interfaces:**
- Consumes: nothing from Phase 1 beyond the standard library.
- Produces: `GoldSet` (frozen, wrapping `Mapping[str, frozenset[str]]`) with `.get(source_id) -> frozenset[str] | None`, `.__contains__`, `.__len__`, `.no_match_ids -> frozenset[str]`; `load_gold_csv(path, *, source_column="source_id", gold_column="gold_ids", separator="|", normalize=None, expand=None) -> GoldSet`; `load_gold_jsonl(path, *, source_field="source_id", gold_field="gold_ids", normalize=None, expand=None) -> GoldSet`. Tasks 2, 3, 6, 8 consume it.

**Design note:** an empty `gold_ids` cell means **"the correct answer is no match"** — a first-class label, not a missing one. A source record absent from the file entirely is unlabelled and is excluded from every metric. Conflating those two would silently inflate no-match recall.

- [ ] **Step 1: Create `tests/fixtures/gold_tiny.csv`**

```csv
source_id,gold_ids
s1,CHEBI:17234
s2,CHEBI:17234
s3,CHEBI:17992
s4,
```

`s4` (unobtainium) has an empty cell — its correct answer is no match.

- [ ] **Step 2: Write the failing test**

Create `tests/test_gold.py`:

```python
import json

import pytest

from xwalk.evaluate.gold import GoldSet, load_gold_csv, load_gold_jsonl


def test_loads_one_entry_per_row(targets_csv):  # noqa: ARG001
    from tests.conftest import FIXTURES

    gold = load_gold_csv(FIXTURES / "gold_tiny.csv")
    assert len(gold) == 4


def test_a_populated_cell_becomes_a_one_element_set():
    from tests.conftest import FIXTURES

    gold = load_gold_csv(FIXTURES / "gold_tiny.csv")
    assert gold.get("s1") == frozenset({"CHEBI:17234"})


def test_an_empty_cell_means_the_answer_is_no_match():
    from tests.conftest import FIXTURES

    gold = load_gold_csv(FIXTURES / "gold_tiny.csv")
    assert gold.get("s4") == frozenset()
    assert "s4" in gold


def test_an_absent_source_id_is_unlabelled_not_no_match():
    from tests.conftest import FIXTURES

    gold = load_gold_csv(FIXTURES / "gold_tiny.csv")
    assert gold.get("s99") is None
    assert "s99" not in gold


def test_no_match_ids_lists_exactly_the_empty_labels():
    from tests.conftest import FIXTURES

    gold = load_gold_csv(FIXTURES / "gold_tiny.csv")
    assert gold.no_match_ids == frozenset({"s4"})


def test_multi_gold_splits_on_the_separator(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,A|B|C\n", encoding="utf-8")
    assert load_gold_csv(path).get("s1") == frozenset({"A", "B", "C"})


def test_blank_parts_are_dropped(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,A||B|\n", encoding="utf-8")
    assert load_gold_csv(path).get("s1") == frozenset({"A", "B"})


def test_whitespace_around_ids_is_stripped(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1, A | B \n", encoding="utf-8")
    assert load_gold_csv(path).get("s1") == frozenset({"A", "B"})


def test_a_duplicate_source_id_is_rejected(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,A\ns1,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_gold_csv(path)


def test_a_missing_column_is_reported_clearly(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("id,gold\ns1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_id"):
        load_gold_csv(path)


def test_a_user_supplied_normalizer_runs_on_every_id(tmp_path):
    """This replaces the paper repo's normalize_ncbi_gene_prediction hook."""
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,3|7\n", encoding="utf-8")
    gold = load_gold_csv(path, normalize=lambda i: f"NCBIGene:{i}" if i.isdigit() else i)
    assert gold.get("s1") == frozenset({"NCBIGene:3", "NCBIGene:7"})


def test_a_normalizer_returning_none_drops_the_id(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,A|SKIP\n", encoding="utf-8")
    gold = load_gold_csv(path, normalize=lambda i: None if i == "SKIP" else i)
    assert gold.get("s1") == frozenset({"A"})


def test_a_user_supplied_expander_adds_aliases(tmp_path):
    """This replaces the paper repo's expand_ctd_eval_ids hook."""
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,MESH:D001\n", encoding="utf-8")
    aliases = {"MESH:D001": {"MESH:D001", "OMIM:100"}}
    gold = load_gold_csv(path, expand=lambda ids: frozenset().union(
        *(aliases.get(i, {i}) for i in ids)
    ) if ids else frozenset())
    assert gold.get("s1") == frozenset({"MESH:D001", "OMIM:100"})


def test_the_expander_is_not_called_for_a_no_match_label(tmp_path):
    calls = []
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,\n", encoding="utf-8")

    def expand(ids):
        calls.append(ids)
        return ids

    gold = load_gold_csv(path, expand=expand)
    assert gold.get("s1") == frozenset()
    assert calls == []


def test_jsonl_gold_accepts_a_list(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"source_id": "s1", "gold_ids": ["A", "B"]}) + "\n",
                    encoding="utf-8")
    assert load_gold_jsonl(path).get("s1") == frozenset({"A", "B"})


def test_jsonl_gold_accepts_a_bare_string(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"source_id": "s1", "gold_ids": "A"}) + "\n", encoding="utf-8")
    assert load_gold_jsonl(path).get("s1") == frozenset({"A"})


def test_jsonl_gold_accepts_an_empty_list_as_no_match(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"source_id": "s1", "gold_ids": []}) + "\n", encoding="utf-8")
    gold = load_gold_jsonl(path)
    assert gold.get("s1") == frozenset() and "s1" in gold


def test_gold_set_is_constructible_directly():
    gold = GoldSet({"s1": frozenset({"A"})})
    assert gold.get("s1") == frozenset({"A"})
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_gold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.evaluate'`.

- [ ] **Step 4: Write `src/xwalk/evaluate/gold.py`**

```python
"""Gold labels.

Two things that look alike and are not: an **empty** label means "the correct answer is
no match", a first-class outcome; an **absent** source id means unlabelled, and is
excluded from every metric. Conflating them inflates no-match recall silently.

ID normalization and alias expansion are user-supplied callables. The paper repo owned
these as named hooks inside the library (`normalize_ncbi_gene_prediction`,
`expand_ctd_eval_ids`); here they live in your code and the library never learns your
identifier scheme.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Callable, Iterable, Mapping

Normalizer = Callable[[str], str | None]
Expander = Callable[[frozenset[str]], AbstractSet[str]]


@dataclass(frozen=True)
class GoldSet:
    labels: Mapping[str, frozenset[str]]

    def get(self, source_id: str) -> frozenset[str] | None:
        return self.labels.get(source_id)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self) -> Iterable[str]:
        return iter(self.labels)

    @property
    def no_match_ids(self) -> frozenset[str]:
        """Source ids whose correct answer is explicitly 'no match'."""
        return frozenset(sid for sid, ids in self.labels.items() if not ids)

    def is_correct(self, source_id: str, predicted: str | None) -> bool | None:
        """None when unlabelled. Otherwise: does the prediction match the label?"""
        expected = self.get(source_id)
        if expected is None:
            return None
        if predicted is None:
            return not expected
        return predicted in expected


def _finalise(
    raw: Iterable[str],
    normalize: Normalizer | None,
    expand: Expander | None,
) -> frozenset[str]:
    ids = {i.strip() for i in raw if i.strip()}
    if normalize is not None:
        ids = {n for n in (normalize(i) for i in ids) if n}
    if not ids:
        return frozenset()  # a no-match label needs no alias expansion
    if expand is not None:
        ids = set(expand(frozenset(ids)))
    return frozenset(ids)


def load_gold_csv(
    path: str | Path,
    *,
    source_column: str = "source_id",
    gold_column: str = "gold_ids",
    separator: str = "|",
    encoding: str = "utf-8",
    normalize: Normalizer | None = None,
    expand: Expander | None = None,
) -> GoldSet:
    labels: dict[str, frozenset[str]] = {}
    path = Path(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or source_column not in reader.fieldnames:
            raise ValueError(
                f"{source_column!r} not found in {path}; columns are {reader.fieldnames!r}"
            )
        if gold_column not in reader.fieldnames:
            raise ValueError(
                f"{gold_column!r} not found in {path}; columns are {reader.fieldnames!r}"
            )
        for line_no, row in enumerate(reader, start=2):
            source_id = (row[source_column] or "").strip()
            if not source_id:
                raise ValueError(f"{path}: row {line_no} has a blank {source_column!r}")
            if source_id in labels:
                raise ValueError(f"{path}: row {line_no} is a duplicate source id {source_id!r}")
            cell = row[gold_column] or ""
            labels[source_id] = _finalise(cell.split(separator), normalize, expand)
    return GoldSet(labels)


def load_gold_jsonl(
    path: str | Path,
    *,
    source_field: str = "source_id",
    gold_field: str = "gold_ids",
    encoding: str = "utf-8",
    normalize: Normalizer | None = None,
    expand: Expander | None = None,
) -> GoldSet:
    labels: dict[str, frozenset[str]] = {}
    path = Path(path)
    with path.open("r", encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no} is not valid JSON: {exc}") from exc
            source_id = str(obj.get(source_field, "")).strip()
            if not source_id:
                raise ValueError(f"{path}: line {line_no} has a blank {source_field!r}")
            if source_id in labels:
                raise ValueError(f"{path}: line {line_no} duplicates source id {source_id!r}")
            raw = obj.get(gold_field, [])
            values = [raw] if isinstance(raw, str) else [str(v) for v in raw]
            labels[source_id] = _finalise(values, normalize, expand)
    return GoldSet(labels)
```

`src/xwalk/evaluate/__init__.py`:

```python
from xwalk.evaluate.gold import GoldSet, load_gold_csv, load_gold_jsonl

__all__ = ["GoldSet", "load_gold_csv", "load_gold_jsonl"]
```

- [ ] **Step 5: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_gold.py -v      # expect 18 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/evaluate tests/test_gold.py tests/fixtures/gold_tiny.csv
git commit -m "feat: gold label loading with user-supplied normalization and expansion"
```

---

## Task 2: Latency instrumentation and the operational metric suite

**Files:**
- Modify: `src/xwalk/records.py` (add `elapsed_seconds` to `Attempt` and `MatchResult`)
- Modify: `src/xwalk/matcher.py` (populate it)
- Modify: `src/xwalk/serde.py`, `src/xwalk/batch.py` (carry and export it)
- Create: `src/xwalk/evaluate/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `GoldSet` (Task 1), `MatchResult`/`MatchStatus`/`DecisionReason` (Phase 1), `Ledger` (Phase 1).
- Produces: `EvalReport` (frozen dataclass, see below); `evaluate_results(results, gold, *, duplicate_targets=None) -> EvalReport`; `evaluate_run(ledger, run_fingerprint, gold) -> EvalReport`; `threshold_curve(results, gold, *, steps=21) -> list[ThresholdPoint]`; `calibration_warning(results, gold) -> str | None`.

**Why these metrics and not accuracy:** accuracy on a matching task hides the decisions that actually matter operationally — how much got matched *automatically*, how much precision that automation bought, and how much human time the review bucket costs. A run at 85% accuracy with a 60% review rate is a worse product than one at 80% accuracy with a 5% review rate, and a single accuracy number cannot tell you which you have.

**Metric definitions** (these go verbatim into the module docstring):

| Metric | Definition |
|---|---|
| `accepted_precision` | of results with status `MATCHED` **and** a gold label, the fraction whose `matched_id` is in the gold set |
| `automatic_coverage` | `MATCHED` / all labelled results |
| `review_rate` | `NEEDS_REVIEW` / all labelled results |
| `unmatched_rate` | `UNMATCHED` / all labelled results |
| `error_rate` | `FAILED` / all labelled results |
| `unresolved_rate` | results whose `reason` is `UNRESOLVED_OUTPUT` / all labelled results |
| `recall_at_any_status` | of labelled results with a non-empty gold set, the fraction where `matched_id` is in the gold set regardless of status — the ceiling automation could reach with perfect thresholds |
| `no_match_precision` | of results predicted no-match (`matched_id is None`, status not `FAILED`), the fraction whose gold set is empty |
| `no_match_recall` | of results whose gold set is empty, the fraction predicted no-match |
| `duplicate_target_conflicts` | count of target ids selected by more than one source record |
| `mean_llm_calls` / `mean_tokens` / `mean_seconds` | per **completed** (non-`FAILED`) result |

- [x] **Step 1: Add `elapsed_seconds` to the result types — ALREADY DONE in `515f792`**

> Shipped ahead of this plan, together with `finish_reason` (a truncated provider answer is
> otherwise indistinguishable from a badly-answered one). **Do not re-apply** — the fields
> already exist and re-adding them breaks `serde`. Verify instead:
>
> ```bash
> python -m pytest -q -m "not integration"   # 406 passed
> grep -n "elapsed_seconds" src/xwalk/records.py src/xwalk/matcher.py src/xwalk/serde.py src/xwalk/batch.py
> ```
>
> The original instructions are kept below for the record.

In `src/xwalk/records.py`, add `elapsed_seconds: float` to `Attempt` (after `usage`) and to `MatchResult` (after `usage`).

In `src/xwalk/matcher.py`, wrap each attempt and the whole match:

```python
import time

# in _attempt, at the top:
started = time.perf_counter()
# ... and pass elapsed_seconds=time.perf_counter() - started into every `build(...)` call
# by adding it to the `build` closure's returned Attempt.

# in match():
match_started = time.perf_counter()
# ... and pass elapsed_seconds=time.perf_counter() - match_started into MatchResult.
```

In `src/xwalk/serde.py`, add `"elapsed_seconds"` to `_attempt_to_dict`/`_attempt_from_dict` and `result_to_dict`/`result_from_dict`.

In `src/xwalk/batch.py`, add `"elapsed_seconds"` to `MAPPING_COLUMNS` and write `round(result.elapsed_seconds, 3)`.

Update `tests/test_serde.py`'s `sample_result()` and `tests/test_policy.py`'s `attempt()` helper to pass `elapsed_seconds=0.0`, and `tests/test_batch.py::test_mapping_csv_columns_are_the_documented_set` to expect the new column.

Run: `python -m pytest -q -m "not integration"` — Expected: all green.

- [ ] **Step 2: Write the failing metrics test**

Create `tests/test_metrics.py`:

```python
import pytest

from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import calibration_warning, evaluate_results, threshold_curve
from xwalk.records import (
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    Usage,
)


def result(
    source_id,
    matched_id,
    status=MatchStatus.MATCHED,
    reason=DecisionReason.ACCEPT_THRESHOLD,
    confidence=0.9,
    calls=2,
    seconds=1.0,
):
    return MatchResult(
        result_key=f"rk-{source_id}",
        source_id=source_id,
        source_hash="h",
        matched_id=matched_id,
        matched_record=None if matched_id is None else Record(id=matched_id, fields={}),
        confidence=confidence,
        status=status,
        reason=reason,
        explanation="",
        candidates=(),
        attempts=(),
        usage=Usage(prompt_tokens=100, completion_tokens=20, calls=calls),
        run_fingerprint="fp1",
        elapsed_seconds=seconds,
    )


GOLD = GoldSet(
    {
        "s1": frozenset({"T1"}),
        "s2": frozenset({"T2"}),
        "s3": frozenset({"T3"}),
        "s4": frozenset(),  # correct answer is no match
    }
)


# --- accepted precision and coverage ---------------------------------------------

def test_accepted_precision_counts_only_matched_results():
    results = [result("s1", "T1"), result("s2", "TX")]
    report = evaluate_results(results, GOLD)
    assert report.accepted_precision == 0.5


def test_a_review_row_does_not_count_against_accepted_precision():
    results = [
        result("s1", "T1"),
        result("s2", "TX", status=MatchStatus.NEEDS_REVIEW,
               reason=DecisionReason.BELOW_ACCEPT_THRESHOLD),
    ]
    report = evaluate_results(results, GOLD)
    assert report.accepted_precision == 1.0


def test_automatic_coverage_is_matched_over_labelled():
    results = [
        result("s1", "T1"),
        result("s2", None, status=MatchStatus.NEEDS_REVIEW,
               reason=DecisionReason.UNRESOLVED_OUTPUT),
    ]
    assert evaluate_results(results, GOLD).automatic_coverage == 0.5


def test_unlabelled_results_are_excluded_entirely():
    results = [result("s1", "T1"), result("s99", "TX")]
    report = evaluate_results(results, GOLD)
    assert report.labelled == 1
    assert report.accepted_precision == 1.0


def test_multi_gold_counts_any_listed_id_as_correct():
    gold = GoldSet({"s1": frozenset({"T1", "T1-alias"})})
    assert evaluate_results([result("s1", "T1-alias")], gold).accepted_precision == 1.0


# --- rates ------------------------------------------------------------------------

def test_review_rate():
    results = [
        result("s1", "T1"),
        result("s2", "T2", status=MatchStatus.NEEDS_REVIEW,
               reason=DecisionReason.BELOW_ACCEPT_THRESHOLD),
    ]
    assert evaluate_results(results, GOLD).review_rate == 0.5


def test_error_rate_counts_failed_results():
    results = [
        result("s1", None, status=MatchStatus.FAILED,
               reason=DecisionReason.PROVIDER_FAILURE),
        result("s2", "T2"),
    ]
    assert evaluate_results(results, GOLD).error_rate == 0.5


def test_unresolved_rate_is_reported_separately_from_review_rate():
    results = [
        result("s1", None, status=MatchStatus.NEEDS_REVIEW,
               reason=DecisionReason.UNRESOLVED_OUTPUT),
        result("s2", "T2", status=MatchStatus.NEEDS_REVIEW,
               reason=DecisionReason.BELOW_ACCEPT_THRESHOLD),
    ]
    report = evaluate_results(results, GOLD)
    assert report.review_rate == 1.0 and report.unresolved_rate == 0.5


def test_recall_at_any_status_ignores_the_threshold():
    """The ceiling automation could reach with perfectly tuned thresholds."""
    results = [
        result("s1", "T1", status=MatchStatus.NEEDS_REVIEW,
               reason=DecisionReason.BELOW_ACCEPT_THRESHOLD),
        result("s2", "TX"),
    ]
    report = evaluate_results(results, GOLD)
    assert report.recall_at_any_status == 0.5
    assert report.automatic_coverage == 0.5


# --- no-match ---------------------------------------------------------------------

def test_no_match_precision_and_recall():
    results = [
        result("s4", None, status=MatchStatus.UNMATCHED,
               reason=DecisionReason.SELECTOR_ABSTAINED),
        result("s1", None, status=MatchStatus.UNMATCHED,
               reason=DecisionReason.SELECTOR_ABSTAINED),
    ]
    report = evaluate_results(results, GOLD)
    assert report.no_match_precision == 0.5  # s4 right, s1 wrong
    assert report.no_match_recall == 1.0     # the only gold no-match was found


def test_no_match_metrics_are_none_when_no_such_labels_exist():
    gold = GoldSet({"s1": frozenset({"T1"})})
    report = evaluate_results([result("s1", "T1")], gold)
    assert report.no_match_recall is None


def test_a_failed_result_is_not_counted_as_a_no_match_prediction():
    """A provider 500 is not evidence of a non-match."""
    results = [result("s4", None, status=MatchStatus.FAILED,
                      reason=DecisionReason.PROVIDER_FAILURE)]
    report = evaluate_results(results, GOLD)
    assert report.no_match_precision is None


# --- duplicates, cost, latency ----------------------------------------------------

def test_duplicate_target_conflicts_are_counted():
    results = [result("s1", "T1"), result("s2", "T1")]
    assert evaluate_results(results, GOLD).duplicate_target_conflicts == 1


def test_cost_and_latency_are_per_completed_record():
    results = [
        result("s1", "T1", calls=2, seconds=1.0),
        result("s2", "T2", calls=4, seconds=3.0),
        result("s3", None, status=MatchStatus.FAILED,
               reason=DecisionReason.PROVIDER_FAILURE, calls=1, seconds=10.0),
    ]
    report = evaluate_results(results, GOLD)
    assert report.mean_llm_calls == 3.0     # failures excluded
    assert report.mean_seconds == 2.0


def test_an_empty_result_set_does_not_divide_by_zero():
    report = evaluate_results([], GOLD)
    assert report.labelled == 0
    assert report.accepted_precision is None


# --- threshold curve and calibration ----------------------------------------------

def test_threshold_curve_covers_zero_to_one():
    results = [result("s1", "T1", confidence=0.9)]
    points = threshold_curve(results, GOLD, steps=11)
    assert points[0].threshold == 0.0 and points[-1].threshold == 1.0


def test_raising_the_threshold_never_increases_coverage():
    results = [
        result("s1", "T1", confidence=0.95),
        result("s2", "TX", confidence=0.45),
    ]
    points = threshold_curve(results, GOLD, steps=11)
    coverages = [p.coverage for p in points]
    assert coverages == sorted(coverages, reverse=True)


def test_a_higher_threshold_improves_precision_on_separated_confidences():
    results = [
        result("s1", "T1", confidence=0.95),
        result("s2", "TX", confidence=0.45),
    ]
    points = {round(p.threshold, 2): p for p in threshold_curve(results, GOLD, steps=11)}
    assert points[0.0].precision == 0.5
    assert points[0.9].precision == 1.0


def test_calibration_warning_fires_when_bands_overlap():
    """Correct and incorrect confidences that look the same mean the score is useless
    as a threshold, and every threshold recommendation from it is noise."""
    correct = [result(f"c{i}", "T1", confidence=c) for i, c in enumerate([0.80, 0.82, 0.78])]
    wrong = [result(f"w{i}", "TX", confidence=c) for i, c in enumerate([0.81, 0.79, 0.83])]
    gold = GoldSet(
        {**{f"c{i}": frozenset({"T1"}) for i in range(3)},
         **{f"w{i}": frozenset({"T9"}) for i in range(3)}}
    )
    assert calibration_warning(correct + wrong, gold) is not None


def test_no_calibration_warning_when_bands_separate():
    correct = [result(f"c{i}", "T1", confidence=c) for i, c in enumerate([0.95, 0.92, 0.97])]
    wrong = [result(f"w{i}", "TX", confidence=c) for i, c in enumerate([0.20, 0.15, 0.25])]
    gold = GoldSet(
        {**{f"c{i}": frozenset({"T1"}) for i in range(3)},
         **{f"w{i}": frozenset({"T9"}) for i in range(3)}}
    )
    assert calibration_warning(correct + wrong, gold) is None


def test_calibration_warning_needs_three_of_each_class():
    """Two correct and one incorrect is not evidence of anything."""
    results = [
        result("s1", "T1", confidence=0.95),
        result("s2", "TX", confidence=0.20),
        result("s3", "T3", confidence=0.92),
    ]
    assert calibration_warning(results, GOLD) is None


def test_calibration_warning_is_none_without_enough_data():
    assert calibration_warning([result("s1", "T1")], GOLD) is None


# --- from a ledger ------------------------------------------------------------------

async def test_evaluate_run_reads_the_ledger(tmp_path):
    from xwalk.evaluate.metrics import evaluate_run
    from xwalk.ledger import Ledger

    ledger = Ledger.open(tmp_path / "l.sqlite")
    await ledger.put_result(result("s1", "T1"))
    await ledger.put_result(result("s2", "TX"))
    report = evaluate_run(ledger, "fp1", GOLD)
    ledger.close()
    assert report.accepted_precision == 0.5


def test_report_as_dict_is_json_safe():
    import json

    report = evaluate_results([result("s1", "T1")], GOLD)
    json.dumps(report.as_dict())
```

- [ ] **Step 3: Write `src/xwalk/evaluate/metrics.py`**

```python
"""Operational metrics.

Accuracy alone hides the decisions that matter: how much was matched *automatically*,
what precision that automation bought, and how much human time the review bucket costs.
A run at 85% accuracy with a 60% review rate is a worse product than one at 80% with a
5% review rate, and one accuracy number cannot tell you which you have.

Definitions, all computed over **labelled** results only (a source id absent from the
gold file is excluded entirely):

- accepted_precision  : of MATCHED results, the fraction whose matched_id is in the gold set
- automatic_coverage  : MATCHED / labelled
- review_rate         : NEEDS_REVIEW / labelled
- unmatched_rate      : UNMATCHED / labelled
- error_rate          : FAILED / labelled
- unresolved_rate     : reason == UNRESOLVED_OUTPUT / labelled
- recall_at_any_status: of labelled results with a non-empty gold set, the fraction whose
                        matched_id is in the gold set at *any* status — the ceiling that
                        perfect threshold tuning could reach
- no_match_precision  : of results predicted no-match (matched_id is None, not FAILED),
                        the fraction whose gold set is empty
- no_match_recall     : of results whose gold set is empty, the fraction predicted no-match
- duplicate_target_conflicts : target ids selected by more than one source record
- mean_llm_calls / mean_tokens / mean_seconds : per completed (non-FAILED) result
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Sequence

from xwalk.evaluate.gold import GoldSet
from xwalk.ledger import Ledger
from xwalk.records import DecisionReason, MatchResult, MatchStatus

_MIN_CALIBRATION_SAMPLES = 3
_MIN_SEPARATION = 0.10


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    coverage: float
    precision: float | None
    accepted: int
    correct: int


@dataclass(frozen=True)
class EvalReport:
    total: int
    labelled: int
    accepted_precision: float | None
    automatic_coverage: float | None
    review_rate: float | None
    unmatched_rate: float | None
    error_rate: float | None
    unresolved_rate: float | None
    recall_at_any_status: float | None
    no_match_precision: float | None
    no_match_recall: float | None
    duplicate_target_conflicts: int
    mean_llm_calls: float | None
    mean_tokens: float | None
    mean_seconds: float | None
    status_counts: dict[str, int]
    reason_counts: dict[str, int]
    calibration_warning: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "labelled": self.labelled,
            "accepted_precision": self.accepted_precision,
            "automatic_coverage": self.automatic_coverage,
            "review_rate": self.review_rate,
            "unmatched_rate": self.unmatched_rate,
            "error_rate": self.error_rate,
            "unresolved_rate": self.unresolved_rate,
            "recall_at_any_status": self.recall_at_any_status,
            "no_match_precision": self.no_match_precision,
            "no_match_recall": self.no_match_recall,
            "duplicate_target_conflicts": self.duplicate_target_conflicts,
            "mean_llm_calls": self.mean_llm_calls,
            "mean_tokens": self.mean_tokens,
            "mean_seconds": self.mean_seconds,
            "status_counts": self.status_counts,
            "reason_counts": self.reason_counts,
            "calibration_warning": self.calibration_warning,
        }


def evaluate_results(results: Sequence[MatchResult], gold: GoldSet) -> EvalReport:
    labelled = [r for r in results if r.source_id in gold]

    matched = [r for r in labelled if r.status is MatchStatus.MATCHED]
    matched_correct = sum(1 for r in matched if gold.is_correct(r.source_id, r.matched_id))

    with_gold = [r for r in labelled if gold.get(r.source_id)]
    recalled = sum(
        1 for r in with_gold if r.matched_id is not None and gold.is_correct(r.source_id, r.matched_id)
    )

    predicted_no_match = [
        r for r in labelled if r.matched_id is None and r.status is not MatchStatus.FAILED
    ]
    no_match_correct = sum(1 for r in predicted_no_match if not gold.get(r.source_id))
    gold_no_match = [r for r in labelled if gold.get(r.source_id) == frozenset()]
    no_match_found = sum(
        1 for r in gold_no_match if r.matched_id is None and r.status is not MatchStatus.FAILED
    )

    target_counts = Counter(r.matched_id for r in results if r.matched_id is not None)
    duplicates = sum(1 for count in target_counts.values() if count > 1)

    completed = [r for r in results if r.status is not MatchStatus.FAILED]

    n = len(labelled)
    return EvalReport(
        total=len(results),
        labelled=n,
        accepted_precision=_ratio(matched_correct, len(matched)),
        automatic_coverage=_ratio(len(matched), n),
        review_rate=_ratio(
            sum(1 for r in labelled if r.status is MatchStatus.NEEDS_REVIEW), n
        ),
        unmatched_rate=_ratio(
            sum(1 for r in labelled if r.status is MatchStatus.UNMATCHED), n
        ),
        error_rate=_ratio(sum(1 for r in labelled if r.status is MatchStatus.FAILED), n),
        unresolved_rate=_ratio(
            sum(1 for r in labelled if r.reason is DecisionReason.UNRESOLVED_OUTPUT), n
        ),
        recall_at_any_status=_ratio(recalled, len(with_gold)),
        no_match_precision=_ratio(no_match_correct, len(predicted_no_match)),
        no_match_recall=_ratio(no_match_found, len(gold_no_match)),
        duplicate_target_conflicts=duplicates,
        mean_llm_calls=mean(r.usage.calls for r in completed) if completed else None,
        mean_tokens=mean(r.usage.total_tokens for r in completed) if completed else None,
        mean_seconds=mean(r.elapsed_seconds for r in completed) if completed else None,
        status_counts=dict(Counter(r.status.value for r in results)),
        reason_counts=dict(Counter(r.reason.value for r in results)),
        calibration_warning=calibration_warning(results, gold),
    )


def evaluate_run(ledger: Ledger, run_fingerprint: str, gold: GoldSet) -> EvalReport:
    return evaluate_results(list(ledger.iter_results(run_fingerprint)), gold)


def threshold_curve(
    results: Iterable[MatchResult],
    gold: GoldSet,
    *,
    steps: int = 21,
) -> list[ThresholdPoint]:
    """What precision and coverage would be if `accept_at` were set differently.

    Re-derived from recorded confidences, so it costs nothing and needs no re-run.
    """
    scored = [
        (r.confidence, gold.is_correct(r.source_id, r.matched_id))
        for r in results
        if r.source_id in gold and r.confidence is not None and r.matched_id is not None
    ]
    labelled_total = sum(1 for r in results if r.source_id in gold)

    points: list[ThresholdPoint] = []
    for i in range(steps):
        threshold = i / (steps - 1) if steps > 1 else 0.0
        accepted = [correct for confidence, correct in scored if confidence >= threshold]
        correct = sum(1 for c in accepted if c)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                coverage=(len(accepted) / labelled_total) if labelled_total else 0.0,
                precision=_ratio(correct, len(accepted)),
                accepted=len(accepted),
                correct=correct,
            )
        )
    return points


def calibration_warning(results: Iterable[MatchResult], gold: GoldSet) -> str | None:
    """Flag confidences that do not separate correct from incorrect.

    When they overlap, every threshold recommendation derived from them is noise, and a
    user tuning `accept_at` is tuning nothing.
    """
    correct: list[float] = []
    wrong: list[float] = []
    for result in results:
        if result.confidence is None or result.matched_id is None:
            continue
        verdict = gold.is_correct(result.source_id, result.matched_id)
        if verdict is None:
            continue
        (correct if verdict else wrong).append(result.confidence)

    if len(correct) < _MIN_CALIBRATION_SAMPLES or len(wrong) < _MIN_CALIBRATION_SAMPLES:
        return None
    separation = mean(correct) - mean(wrong)
    if separation >= _MIN_SEPARATION:
        return None
    return (
        f"confidence scores barely separate correct from incorrect matches "
        f"(mean correct {mean(correct):.2f} vs mean incorrect {mean(wrong):.2f}, "
        f"separation {separation:.2f} < {_MIN_SEPARATION}); threshold tuning will not help "
        f"until the scoring prompt discriminates better"
    )
```

Add `EvalReport`, `evaluate_results`, `evaluate_run`, `threshold_curve`, `calibration_warning` to `src/xwalk/evaluate/__init__.py`.

- [ ] **Step 4: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_metrics.py -v   # expect 24 passed
python -m pytest -q -m "not integration"
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk tests/
git commit -m "feat: latency instrumentation and operational metric suite"
```

---

## Task 3: Three-way retrieval-ceiling decomposition

**Files:**
- Create: `src/xwalk/evaluate/ceiling.py`
- Test: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: `GoldSet` (Task 1), `MatchResult`/`Attempt`/`Candidate` (Phase 1).
- Produces: `CeilingBucket` enum (`FOUND`, `NEVER_RETRIEVED`, `TRUNCATED`, `MISJUDGED`, `NO_GOLD`, `UNLABELLED`); `classify_ceiling(result, gold) -> CeilingBucket`; `CeilingReport` with `buckets`, `by_retriever`, `by_attempt`, `as_dict()`; `ceiling_report(results, gold) -> CeilingReport`.

**Why three buckets and not two:** the selector budget creates a genuinely distinct failure. If the gold record was retrieved but cut by `max_candidates`, that is a *budget* miss and the fix is a larger budget — not a better retriever, and not a better prompt. Collapsing it into "retrieval failure" sends users to fix the wrong thing. This is nearly free because the loop already records `evidence` on every candidate and `issued_keys` on every attempt.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ceiling.py`:

```python
import pytest

from xwalk.evaluate.ceiling import CeilingBucket, ceiling_report, classify_ceiling
from xwalk.evaluate.gold import GoldSet
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    Usage,
)

GOLD = GoldSet({"s1": frozenset({"T1"}), "s2": frozenset(), "s3": frozenset({"T3"})})


def cand(record_id, retrievers=("bm25",)):
    return Candidate(
        record=Record(id=record_id, fields={}),
        fused_score=0.5,
        evidence=tuple(
            RetrievalHit(record_id=record_id, retriever=name, raw_score=1.0, rank=i)
            for i, name in enumerate(retrievers, start=1)
        ),
    )


def attempt(index, candidates, issued, truncated=0):
    return Attempt(
        index=index, query="q", proposal=None, candidates=tuple(candidates),
        candidate_count=len(candidates), candidates_truncated=truncated,
        issued_keys=issued, raw_selection=None, chosen_id=None, resolution="abstain",
        primary_score=None, explanation="", verifier_decision=None, verifier_score=None,
        verifier_preferred_id=None, audited=False, dropped_proposals=(), reason=None,
        error=None, usage=Usage.zero(), elapsed_seconds=0.0, finish_reason="stop",
    )


def result(source_id, matched_id, attempts, status=MatchStatus.MATCHED):
    return MatchResult(
        result_key=f"rk-{source_id}", source_id=source_id, source_hash="h",
        matched_id=matched_id,
        matched_record=None if matched_id is None else Record(id=matched_id, fields={}),
        confidence=0.9, status=status, reason=DecisionReason.ACCEPT_THRESHOLD,
        explanation="", candidates=tuple(attempts[-1].candidates) if attempts else (),
        attempts=tuple(attempts), usage=Usage.zero(), run_fingerprint="fp1",
        elapsed_seconds=0.0,
    )


# --- the four outcomes -----------------------------------------------------------

def test_a_correct_match_is_found():
    r = result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])
    assert classify_ceiling(r, GOLD) is CeilingBucket.FOUND


def test_gold_absent_from_every_candidate_list_is_never_retrieved():
    r = result("s1", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})])
    assert classify_ceiling(r, GOLD) is CeilingBucket.NEVER_RETRIEVED


def test_gold_retrieved_but_not_issued_a_key_is_truncated():
    """A budget miss, not a retrieval miss. Raise max_candidates, not the retriever."""
    r = result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9"}, truncated=1)])
    assert classify_ceiling(r, GOLD) is CeilingBucket.TRUNCATED


def test_gold_presented_but_not_chosen_is_misjudged():
    r = result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9", "C02": "T1"})])
    assert classify_ceiling(r, GOLD) is CeilingBucket.MISJUDGED


def test_the_three_failure_buckets_are_mutually_exclusive():
    cases = [
        result("s1", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]),
        result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9"}, truncated=1)]),
        result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9", "C02": "T1"})]),
    ]
    assert len({classify_ceiling(c, GOLD) for c in cases}) == 3


# --- edges ------------------------------------------------------------------------

def test_a_no_match_gold_label_gets_its_own_bucket():
    r = result("s2", None, [attempt(0, [], {})], status=MatchStatus.UNMATCHED)
    assert classify_ceiling(r, GOLD) is CeilingBucket.NO_GOLD


def test_an_unlabelled_result_gets_its_own_bucket():
    r = result("s99", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])
    assert classify_ceiling(r, GOLD) is CeilingBucket.UNLABELLED


def test_gold_retrieved_in_a_later_attempt_still_counts_as_retrieved():
    """Reformulation recovered it; retrieval is not the bottleneck."""
    r = result(
        "s1", "T9",
        [
            attempt(0, [cand("T9")], {"C01": "T9"}),
            attempt(1, [cand("T1"), cand("T9")], {"C01": "T1", "C02": "T9"}),
        ],
    )
    assert classify_ceiling(r, GOLD) is CeilingBucket.MISJUDGED


def test_a_result_with_no_attempts_is_never_retrieved():
    r = result("s1", None, [], status=MatchStatus.FAILED)
    assert classify_ceiling(r, GOLD) is CeilingBucket.NEVER_RETRIEVED


def test_multi_gold_counts_any_listed_id_as_retrieved():
    gold = GoldSet({"s1": frozenset({"T1", "T2"})})
    r = result("s1", "T9", [attempt(0, [cand("T2")], {"C01": "T2"})])
    assert classify_ceiling(r, gold) is CeilingBucket.MISJUDGED


# --- the report -------------------------------------------------------------------

def test_report_counts_every_bucket():
    results = [
        result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})]),
        result("s3", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]),
    ]
    report = ceiling_report(results, GOLD)
    assert report.buckets[CeilingBucket.FOUND] == 1
    assert report.buckets[CeilingBucket.NEVER_RETRIEVED] == 1


def test_report_attributes_retrieval_per_retriever():
    """Which retriever surfaced the gold record, when any did."""
    results = [
        result("s1", "T1", [attempt(0, [cand("T1", ("bm25", "dense"))], {"C01": "T1"})]),
        result("s3", "T3", [attempt(0, [cand("T3", ("dense",))], {"C01": "T3"})]),
    ]
    report = ceiling_report(results, GOLD)
    assert report.by_retriever["dense"] == 2
    assert report.by_retriever["bm25"] == 1


def test_report_records_which_attempt_first_surfaced_the_gold():
    results = [
        result("s1", "T1", [attempt(0, [cand("T9")], {"C01": "T9"}),
                            attempt(1, [cand("T1")], {"C01": "T1"})]),
    ]
    assert ceiling_report(results, GOLD).by_attempt[1] == 1


def test_report_excludes_unlabelled_rows_from_the_failure_totals():
    results = [result("s99", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])]
    report = ceiling_report(results, GOLD)
    assert report.evaluable == 0


def test_report_as_dict_is_json_safe():
    import json

    results = [result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])]
    json.dumps(ceiling_report(results, GOLD).as_dict())


def test_report_recommends_the_budget_when_truncation_dominates():
    results = [
        result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9"}, truncated=1)]),
        result("s3", "T9", [attempt(0, [cand("T9"), cand("T3")], {"C01": "T9"}, truncated=1)]),
    ]
    assert "budget" in ceiling_report(results, GOLD).recommendation.lower()


def test_report_recommends_retrieval_when_never_retrieved_dominates():
    results = [
        result("s1", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]),
        result("s3", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]),
    ]
    assert "retriev" in ceiling_report(results, GOLD).recommendation.lower()


def test_report_recommends_prompts_when_misjudgement_dominates():
    results = [
        result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9", "C02": "T1"})]),
        result("s3", "T9", [attempt(0, [cand("T9"), cand("T3")], {"C01": "T9", "C02": "T3"})]),
    ]
    assert "prompt" in ceiling_report(results, GOLD).recommendation.lower()
```

- [ ] **Step 2: Write `src/xwalk/evaluate/ceiling.py`**

```python
"""Where the failures actually are.

Three buckets, never two. The selector budget creates a genuinely distinct failure: if
the gold record was retrieved but cut before the model saw it, the fix is a larger
budget — not a better retriever and not a better prompt. Collapsing that into "retrieval
failure" sends users to fix the wrong thing.

Nearly free: the loop already records `evidence` on every candidate and `issued_keys`
on every attempt, so this reads a completed run and calls nothing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from xwalk.evaluate.gold import GoldSet
from xwalk.records import MatchResult


class CeilingBucket(Enum):
    FOUND = "found"
    NEVER_RETRIEVED = "never_retrieved"
    TRUNCATED = "truncated"
    MISJUDGED = "misjudged"
    NO_GOLD = "no_gold"          # the correct answer is no match; not a ceiling question
    UNLABELLED = "unlabelled"    # excluded from every total


def _retrieved_ids(result: MatchResult) -> set[str]:
    return {c.id for attempt in result.attempts for c in attempt.candidates}


def _issued_ids(result: MatchResult) -> set[str]:
    return {rid for attempt in result.attempts for rid in attempt.issued_keys.values()}


def classify_ceiling(result: MatchResult, gold: GoldSet) -> CeilingBucket:
    expected = gold.get(result.source_id)
    if expected is None:
        return CeilingBucket.UNLABELLED
    if not expected:
        return CeilingBucket.NO_GOLD
    if result.matched_id is not None and result.matched_id in expected:
        return CeilingBucket.FOUND
    if expected & _issued_ids(result):
        return CeilingBucket.MISJUDGED
    if expected & _retrieved_ids(result):
        return CeilingBucket.TRUNCATED
    return CeilingBucket.NEVER_RETRIEVED


@dataclass(frozen=True)
class CeilingReport:
    evaluable: int
    buckets: dict[CeilingBucket, int]
    by_retriever: dict[str, int]
    by_attempt: dict[int, int]
    recommendation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluable": self.evaluable,
            "buckets": {b.value: n for b, n in self.buckets.items()},
            "by_retriever": self.by_retriever,
            "by_attempt": {str(k): v for k, v in self.by_attempt.items()},
            "recommendation": self.recommendation,
        }


def ceiling_report(results: Sequence[MatchResult], gold: GoldSet) -> CeilingReport:
    buckets: Counter[CeilingBucket] = Counter()
    by_retriever: Counter[str] = Counter()
    by_attempt: Counter[int] = Counter()

    for result in results:
        bucket = classify_ceiling(result, gold)
        buckets[bucket] += 1
        if bucket in (CeilingBucket.UNLABELLED, CeilingBucket.NO_GOLD):
            continue

        expected = gold.get(result.source_id) or frozenset()
        seen_retrievers: set[str] = set()
        first_attempt: int | None = None
        for attempt in result.attempts:
            for candidate in attempt.candidates:
                if candidate.id not in expected:
                    continue
                if first_attempt is None:
                    first_attempt = attempt.index
                seen_retrievers.update(hit.retriever for hit in candidate.evidence)
        for name in seen_retrievers:
            by_retriever[name] += 1
        if first_attempt is not None:
            by_attempt[first_attempt] += 1

    evaluable = sum(
        count
        for bucket, count in buckets.items()
        if bucket not in (CeilingBucket.UNLABELLED, CeilingBucket.NO_GOLD)
    )
    failures = {
        CeilingBucket.NEVER_RETRIEVED: buckets[CeilingBucket.NEVER_RETRIEVED],
        CeilingBucket.TRUNCATED: buckets[CeilingBucket.TRUNCATED],
        CeilingBucket.MISJUDGED: buckets[CeilingBucket.MISJUDGED],
    }
    total_failures = sum(failures.values())

    if total_failures == 0:
        recommendation = "no ceiling failures on labelled records"
    else:
        worst = max(failures, key=lambda b: failures[b])
        share = failures[worst] / total_failures
        recommendation = {
            CeilingBucket.NEVER_RETRIEVED: (
                f"{share:.0%} of failures never retrieved the gold record — spend effort on "
                f"retrieval: the doc template, the retriever set, or retrieval depth"
            ),
            CeilingBucket.TRUNCATED: (
                f"{share:.0%} of failures retrieved the gold record but the selector budget "
                f"cut it before the model saw it — raise SelectorPolicy.max_candidates"
            ),
            CeilingBucket.MISJUDGED: (
                f"{share:.0%} of failures presented the gold record and the model chose "
                f"otherwise — spend effort on prompts, not retrieval"
            ),
        }[worst]

    return CeilingReport(
        evaluable=evaluable,
        buckets=dict(buckets),
        by_retriever=dict(by_retriever),
        by_attempt=dict(by_attempt),
        recommendation=recommendation,
    )
```

Add `CeilingBucket`, `CeilingReport`, `ceiling_report`, `classify_ceiling` to `src/xwalk/evaluate/__init__.py`.

- [ ] **Step 3: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_ceiling.py -v   # expect 18 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/evaluate tests/test_ceiling.py
git commit -m "feat: three-way retrieval/budget/judgement ceiling decomposition"
```

---

## Task 4: Deterministic three-way partitioning

**Files:**
- Create: `src/xwalk/evaluate/partition.py`
- Test: `tests/test_partition.py`

**Interfaces:**
- Consumes: nothing beyond the standard library.
- Produces: `Partition` enum (`PROMPT_TRAIN`, `VALIDATION`, `TEST`); `Partitioner(fractions=(0.5, 0.25, 0.25), salt="xwalk")` with `assign(source_id) -> Partition`; `split(source_ids) -> dict[Partition, list[str]]`; `load_partition_file(path) -> dict[str, Partition]`; `write_partition_file(assignment, path)`.

**Why three and not two:** revising a prompt from dev failures *and* selecting the retained round on dev performance makes dev part of training. The roles must be separate:

| Partition | Role |
|---|---|
| **prompt-train** | failures shown to the optimising model |
| **validation** | chooses the retained round and the stopping point |
| **test** | evaluated **once**, after the final prompt is selected |

- [ ] **Step 1: Write the failing test**

Create `tests/test_partition.py`:

```python
import pytest

from xwalk.evaluate.partition import (
    Partition,
    Partitioner,
    load_partition_file,
    write_partition_file,
)

IDS = [f"s{i}" for i in range(1000)]


def test_assignment_is_deterministic():
    p = Partitioner()
    assert [p.assign(i) for i in IDS] == [Partitioner().assign(i) for i in IDS]


def test_a_different_salt_produces_a_different_split():
    a = Partitioner(salt="one")
    b = Partitioner(salt="two")
    assert [a.assign(i) for i in IDS] != [b.assign(i) for i in IDS]


def test_every_id_lands_in_exactly_one_partition():
    split = Partitioner().split(IDS)
    total = sum(len(v) for v in split.values())
    assert total == len(IDS)
    assert len(set().union(*(set(v) for v in split.values()))) == len(IDS)


def test_fractions_are_approximately_honoured():
    split = Partitioner(fractions=(0.5, 0.25, 0.25)).split(IDS)
    assert 0.45 < len(split[Partition.PROMPT_TRAIN]) / 1000 < 0.55
    assert 0.20 < len(split[Partition.VALIDATION]) / 1000 < 0.30
    assert 0.20 < len(split[Partition.TEST]) / 1000 < 0.30


def test_custom_fractions_are_honoured():
    split = Partitioner(fractions=(0.8, 0.1, 0.1)).split(IDS)
    assert 0.75 < len(split[Partition.PROMPT_TRAIN]) / 1000 < 0.85


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        Partitioner(fractions=(0.5, 0.5, 0.5))


def test_a_zero_fraction_is_rejected():
    """An empty validation set silently disables round selection."""
    with pytest.raises(ValueError, match="positive"):
        Partitioner(fractions=(0.8, 0.2, 0.0))


def test_split_output_is_sorted_for_reproducible_reporting():
    split = Partitioner().split(["s3", "s1", "s2"] * 10)
    for ids in split.values():
        assert ids == sorted(set(ids))


def test_adding_records_does_not_move_existing_ones():
    """Growing the labelled set must not reshuffle the test partition."""
    p = Partitioner()
    before = {i: p.assign(i) for i in IDS[:500]}
    after = {i: p.assign(i) for i in IDS}
    assert all(after[i] is before[i] for i in before)


def test_a_partition_file_round_trips(tmp_path):
    assignment = {i: Partitioner().assign(i) for i in IDS[:20]}
    path = tmp_path / "partitions.csv"
    write_partition_file(assignment, path)
    assert load_partition_file(path) == assignment


def test_an_explicit_partition_file_overrides_hashing(tmp_path):
    path = tmp_path / "partitions.csv"
    path.write_text("source_id,partition\ns1,test\ns2,validation\n", encoding="utf-8")
    loaded = load_partition_file(path)
    assert loaded["s1"] is Partition.TEST


def test_an_unknown_partition_name_is_rejected(tmp_path):
    path = tmp_path / "partitions.csv"
    path.write_text("source_id,partition\ns1,dev\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dev"):
        load_partition_file(path)
```

- [ ] **Step 2: Write `src/xwalk/evaluate/partition.py`**

```python
"""Three partitions with distinct roles.

Revising a prompt from dev failures *and* selecting the retained round on dev
performance makes dev part of training. Hence:

  prompt-train : failures shown to the optimising model
  validation   : chooses the retained round and the stopping point
  test         : evaluated once, after the final prompt is selected

Assignment is hash-based rather than random so that adding labelled records never
reshuffles the existing split — a test partition that moves between runs is not a test
partition.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class Partition(Enum):
    PROMPT_TRAIN = "prompt_train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class Partitioner:
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25)
    salt: str = "xwalk"

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-9:
            raise ValueError(f"fractions must sum to 1, got {sum(self.fractions)}")
        if any(f <= 0 for f in self.fractions):
            raise ValueError("every fraction must be positive; an empty partition disables its role")

    def assign(self, source_id: str) -> Partition:
        digest = hashlib.sha256(f"{self.salt}\x00{source_id}".encode("utf-8")).digest()
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        train, validation, _ = self.fractions
        if draw < train:
            return Partition.PROMPT_TRAIN
        if draw < train + validation:
            return Partition.VALIDATION
        return Partition.TEST

    def split(self, source_ids: Iterable[str]) -> dict[Partition, list[str]]:
        out: dict[Partition, set[str]] = {p: set() for p in Partition}
        for source_id in source_ids:
            out[self.assign(source_id)].add(source_id)
        return {partition: sorted(ids) for partition, ids in out.items()}


def write_partition_file(assignment: Mapping[str, Partition], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "partition"])
        for source_id in sorted(assignment):
            writer.writerow([source_id, assignment[source_id].value])


def load_partition_file(path: str | Path) -> dict[str, Partition]:
    """An explicit file always wins over hashing — for reproducing a published split."""
    out: dict[str, Partition] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            source_id = (row.get("source_id") or "").strip()
            name = (row.get("partition") or "").strip().lower()
            if not source_id:
                continue
            try:
                out[source_id] = Partition(name)
            except ValueError as exc:
                raise ValueError(
                    f"row {line_no}: unknown partition {name!r}; "
                    f"expected one of {[p.value for p in Partition]}"
                ) from exc
    return out


def partition_of(
    source_id: str,
    partitioner: Partitioner,
    overrides: Mapping[str, Partition] | None = None,
) -> Partition:
    if overrides and source_id in overrides:
        return overrides[source_id]
    return partitioner.assign(source_id)


def ids_in(
    source_ids: Sequence[str],
    partition: Partition,
    partitioner: Partitioner,
    overrides: Mapping[str, Partition] | None = None,
) -> list[str]:
    return sorted(
        {sid for sid in source_ids if partition_of(sid, partitioner, overrides) is partition}
    )
```

- [ ] **Step 3: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_partition.py -v   # expect 12 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/evaluate/partition.py tests/test_partition.py
git commit -m "feat: deterministic three-way partitioning with stable assignment"
```

---

## Task 5: Role-specific failure selection

**Files:**
- Create: `src/xwalk/evaluate/failures.py`
- Test: `tests/test_failures.py`

**Interfaces:**
- Consumes: `GoldSet` (Task 1), `CeilingBucket`/`classify_ceiling` (Task 3), `MatchResult` (Phase 1).
- Produces: `PromptRole` enum (`SELECTOR`, `SCORER`, `REWRITER`, `DOC_TEMPLATE`); `FailureCase` (frozen: `source_id`, `gold_ids`, `chosen_id`, `confidence`, `bucket`, `presented`, `queries`, `explanation`); `select_failures(results, gold, role, *, limit=None) -> list[FailureCase]`; `render_failure(case) -> str`.

**Why per role:** "only selection failures feed the optimiser" is correct for the selector and wrong for everything else. Showing a selector-optimisation model a case where the gold record was never retrieved teaches it nothing — the model never saw the right answer.

| Role | Useful cases |
|---|---|
| Selector | gold was retrieved **and presented**, and a different candidate was chosen |
| Scorer / gate | incorrect automatic accepts (**including where gold was never retrieved** — abstaining there is the scorer's job), and unnecessary abstentions or suppressions where gold *was* presented |
| Rewriter | gold absent from the first attempt but present in a later one, or absent throughout while retrievable |
| Doc template | gold never surfaced in any attempt despite being in the target |

- [ ] **Step 1: Write the failing test**

Create `tests/test_failures.py`:

```python
import pytest

from tests.test_ceiling import GOLD, attempt, cand, result
from xwalk.evaluate.failures import PromptRole, render_failure, select_failures
from xwalk.records import DecisionReason, MatchResult, MatchStatus


def with_status(r, status, reason, confidence=0.9):
    return MatchResult(
        **{**r.__dict__, "status": status, "reason": reason, "confidence": confidence}
    )


MISJUDGED = result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")],
                                        {"C01": "T9", "C02": "T1"})])
NEVER = result("s1", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})])
TRUNCATED = result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9"},
                                        truncated=1)])
CORRECT = result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])


# --- selector ---------------------------------------------------------------------

def test_selector_takes_misjudged_cases():
    cases = select_failures([MISJUDGED], GOLD, PromptRole.SELECTOR)
    assert [c.source_id for c in cases] == ["s1"]


def test_selector_ignores_never_retrieved_cases():
    """The model never saw the right answer; there is nothing for it to learn."""
    assert select_failures([NEVER], GOLD, PromptRole.SELECTOR) == []


def test_selector_ignores_truncated_cases():
    """A budget miss is not a prompt problem."""
    assert select_failures([TRUNCATED], GOLD, PromptRole.SELECTOR) == []


def test_selector_ignores_correct_results():
    assert select_failures([CORRECT], GOLD, PromptRole.SELECTOR) == []


# --- scorer -----------------------------------------------------------------------

def test_scorer_takes_incorrect_automatic_accepts():
    accepted_wrong = with_status(MISJUDGED, MatchStatus.MATCHED,
                                 DecisionReason.ACCEPT_THRESHOLD)
    cases = select_failures([accepted_wrong], GOLD, PromptRole.SCORER)
    assert [c.source_id for c in cases] == ["s1"]


def test_scorer_takes_unnecessary_abstentions_where_gold_was_presented():
    abstained = MatchResult(
        **{
            **result("s1", None, [attempt(0, [cand("T1")], {"C01": "T1"})]).__dict__,
            "status": MatchStatus.UNMATCHED,
            "reason": DecisionReason.SELECTOR_ABSTAINED,
        }
    )
    assert [c.source_id for c in select_failures([abstained], GOLD, PromptRole.SCORER)] == ["s1"]


def test_scorer_takes_correct_matches_pushed_below_the_floor():
    below = with_status(CORRECT, MatchStatus.UNMATCHED,
                        DecisionReason.BELOW_REVIEW_FLOOR, confidence=0.2)
    assert len(select_failures([below], GOLD, PromptRole.SCORER)) == 1


def test_scorer_ignores_a_correct_confident_accept():
    good = with_status(CORRECT, MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD)
    assert select_failures([good], GOLD, PromptRole.SCORER) == []


def test_scorer_takes_a_wrong_accept_even_when_gold_was_never_retrieved():
    """The canonical scorer failure. When nothing correct was retrieved, the right
    behaviour is to abstain; a confident accept instead is exactly what the rubric
    exists to prevent. Excluding these would blind the optimiser to abstention."""
    assert len(select_failures([NEVER], GOLD, PromptRole.SCORER)) == 1


# --- rewriter ---------------------------------------------------------------------

def test_rewriter_takes_cases_recovered_by_a_later_attempt():
    recovered = result(
        "s1", "T1",
        [attempt(0, [cand("T9")], {"C01": "T9"}), attempt(1, [cand("T1")], {"C01": "T1"})],
    )
    assert [c.source_id for c in select_failures([recovered], GOLD, PromptRole.REWRITER)] == ["s1"]


def test_rewriter_takes_cases_never_recovered():
    assert len(select_failures([NEVER], GOLD, PromptRole.REWRITER)) == 1


def test_rewriter_ignores_cases_found_on_the_first_attempt():
    assert select_failures([CORRECT], GOLD, PromptRole.REWRITER) == []


# --- doc template -----------------------------------------------------------------

def test_doc_template_takes_only_never_retrieved_cases():
    cases = select_failures([NEVER, MISJUDGED, TRUNCATED], GOLD, PromptRole.DOC_TEMPLATE)
    assert [c.bucket.value for c in cases] == ["never_retrieved"]


# --- shared behaviour -------------------------------------------------------------

def test_unlabelled_results_are_never_selected():
    unlabelled = result("s99", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})])
    for role in PromptRole:
        assert select_failures([unlabelled], GOLD, role) == []


def test_no_match_gold_labels_are_never_selected_for_retrieval_roles():
    no_match = result("s2", None, [attempt(0, [], {})])
    assert select_failures([no_match], GOLD, PromptRole.DOC_TEMPLATE) == []


def test_the_limit_is_honoured():
    many = [
        MatchResult(**{**MISJUDGED.__dict__, "source_id": f"x{i}", "result_key": f"rk{i}"})
        for i in range(10)
    ]
    gold_many = type(GOLD)({f"x{i}": frozenset({"T1"}) for i in range(10)})
    assert len(select_failures(many, gold_many, PromptRole.SELECTOR, limit=3)) == 3


def test_selection_is_deterministic_under_a_limit():
    many = [
        MatchResult(**{**MISJUDGED.__dict__, "source_id": f"x{i}", "result_key": f"rk{i}"})
        for i in range(10)
    ]
    gold_many = type(GOLD)({f"x{i}": frozenset({"T1"}) for i in range(10)})
    first = select_failures(many, gold_many, PromptRole.SELECTOR, limit=3)
    second = select_failures(list(reversed(many)), gold_many, PromptRole.SELECTOR, limit=3)
    assert [c.source_id for c in first] == [c.source_id for c in second]


def test_render_failure_shows_gold_chosen_and_what_was_presented():
    [case] = select_failures([MISJUDGED], GOLD, PromptRole.SELECTOR)
    text = render_failure(case)
    assert "T1" in text and "T9" in text


def test_render_failure_never_contains_a_python_repr():
    [case] = select_failures([MISJUDGED], GOLD, PromptRole.SELECTOR)
    assert "frozenset" not in render_failure(case)
```

- [ ] **Step 2: Write `src/xwalk/evaluate/failures.py`**

```python
"""Choosing which failures to show the optimising model.

"Only selection failures feed the optimiser" is correct for the selector and wrong for
everything else. A selector-optimisation prompt shown a case where the gold record was
never retrieved teaches nothing — the model never saw the right answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from xwalk.evaluate.ceiling import CeilingBucket, classify_ceiling
from xwalk.evaluate.gold import GoldSet
from xwalk.records import DecisionReason, MatchResult, MatchStatus


class PromptRole(Enum):
    SELECTOR = "selector"
    SCORER = "scorer"
    REWRITER = "rewriter"
    DOC_TEMPLATE = "doc_template"


@dataclass(frozen=True)
class FailureCase:
    source_id: str
    source_fields: dict[str, str]
    gold_ids: tuple[str, ...]
    chosen_id: str | None
    confidence: float | None
    status: str
    reason: str
    bucket: CeilingBucket
    presented: tuple[str, ...]
    queries: tuple[str, ...]
    explanation: str


def _to_case(result: MatchResult, gold: GoldSet, bucket: CeilingBucket) -> FailureCase:
    presented: list[str] = []
    for attempt in result.attempts:
        for record_id in attempt.issued_keys.values():
            if record_id not in presented:
                presented.append(record_id)
    return FailureCase(
        source_id=result.source_id,
        source_fields={},  # populated by the optimizer, which has the source records
        gold_ids=tuple(sorted(gold.get(result.source_id) or ())),
        chosen_id=result.matched_id,
        confidence=result.confidence,
        status=result.status.value,
        reason=result.reason.value,
        bucket=bucket,
        presented=tuple(presented),
        queries=tuple(a.query for a in result.attempts),
        explanation=result.explanation,
    )


def select_failures(
    results: Sequence[MatchResult],
    gold: GoldSet,
    role: PromptRole,
    *,
    limit: int | None = None,
) -> list[FailureCase]:
    """Cases worth showing the model for this role, in a deterministic order."""
    cases: list[FailureCase] = []

    for result in results:
        bucket = classify_ceiling(result, gold)
        if bucket in (CeilingBucket.UNLABELLED, CeilingBucket.NO_GOLD):
            continue
        expected = gold.get(result.source_id) or frozenset()
        correct = result.matched_id is not None and result.matched_id in expected

        keep = False
        if role is PromptRole.SELECTOR:
            # gold was presented and something else was chosen
            keep = bucket is CeilingBucket.MISJUDGED

        elif role is PromptRole.SCORER:
            gold_presented = bool(
                expected & {rid for a in result.attempts for rid in a.issued_keys.values()}
            )
            wrong_accept = result.status is MatchStatus.MATCHED and not correct
            needless_abstain = (
                gold_presented
                and result.matched_id is None
                and result.reason
                in (DecisionReason.SELECTOR_ABSTAINED, DecisionReason.BELOW_REVIEW_FLOOR)
            )
            suppressed = correct and result.status in (
                MatchStatus.UNMATCHED,
                MatchStatus.NEEDS_REVIEW,
            )
            keep = wrong_accept or needless_abstain or suppressed

        elif role is PromptRole.REWRITER:
            first = result.attempts[0] if result.attempts else None
            found_first = bool(first and expected & {c.id for c in first.candidates})
            keep = not found_first  # either recovered later, or never — both are its job

        elif role is PromptRole.DOC_TEMPLATE:
            keep = bucket is CeilingBucket.NEVER_RETRIEVED

        if keep:
            cases.append(_to_case(result, gold, bucket))

    cases.sort(key=lambda c: c.source_id)  # deterministic under a limit
    return cases if limit is None else cases[:limit]


def render_failure(case: FailureCase) -> str:
    """A compact, prose-safe rendering for inclusion in an optimisation prompt."""
    lines = [f"Source record {case.source_id}"]
    for key, value in case.source_fields.items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  correct answer: {', '.join(case.gold_ids) or 'no match'}")
    lines.append(f"  model chose:    {case.chosen_id or 'no match'}")
    if case.confidence is not None:
        lines.append(f"  confidence:     {case.confidence:.2f}")
    lines.append(f"  outcome:        {case.status} / {case.reason}")
    if case.presented:
        lines.append(f"  candidates shown: {', '.join(case.presented)}")
    if case.queries:
        lines.append(f"  queries tried:    {', '.join(case.queries)}")
    if case.explanation:
        lines.append(f"  model said:     {case.explanation}")
    return "\n".join(lines)
```

- [ ] **Step 3: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_failures.py -v   # expect 18 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/evaluate/failures.py tests/test_failures.py
git commit -m "feat: role-specific failure selection for prompt optimisation"
```

---

## Task 6: LLM-assisted prompt drafting

**Files:**
- Create: `src/xwalk/prompts/author.py`
- Test: `tests/test_author.py`

**Interfaces:**
- Consumes: `LLMClient`/`LLMRequest`/`parse_json_object` (Phase 1), `PromptSlots`/`PromptSet`/`validate_contract` (Phase 1 Task 10), `Record` (Phase 1).
- Produces: `SLOTS_SCHEMA`; `DraftResult(slots, warnings, usage, raw)`; `async draft_slots(llm, *, description, source_samples, target_samples, existing=None, max_samples=8) -> DraftResult`; `slots_diff(before, after) -> str`; `write_slots(slots, path)`.

**Design note:** the model returns **slots only**, never raw prompt text. A bad draft can produce a poor rubric but never a broken prompt, because Phase 1's skeleton owns every part of the machine-readable contract and `validate_contract` runs before anything is written to disk.

- [ ] **Step 1: Write the failing test**

Create `tests/test_author.py`:

```python
import json

import pytest
import yaml

from xwalk.llm.fake import FakeLLM
from xwalk.prompts.author import draft_slots, slots_diff, write_slots
from xwalk.prompts.contract import PromptSlots, load_slots
from xwalk.records import Record

SOURCES = [
    Record(id="s1", fields={"mention": "aspirin"}),
    Record(id="s2", fields={"mention": "paracetamol"}),
]
TARGETS = [
    Record(id="T1", fields={"label": "acetylsalicylic acid", "synonyms": ["aspirin"]}),
    Record(id="T2", fields={"label": "paracetamol", "synonyms": ["acetaminophen"]}),
]

GOOD = {
    "entity_noun": "drug mention",
    "target_noun": "chemical compound record",
    "domain_brief": "pharmaceutical nomenclature",
    "rubric": [
        {"score": 1.0, "name": "Certain", "when": "exact label or synonym match",
         "example": "aspirin -> acetylsalicylic acid"},
        {"score": 0.6, "name": "Plausible", "when": "same active ingredient, different salt",
         "example": "..."},
        {"score": 0.3, "name": "Speculative", "when": "same drug class only", "example": "..."},
    ],
    "hard_rules": ["A salt or ester is a distinct entity from its parent compound"],
    "disambiguation_steps": "",
}


async def test_returns_validated_slots():
    llm = FakeLLM([json.dumps(GOOD)])
    draft = await draft_slots(llm, description="matching drugs to compounds",
                              source_samples=SOURCES, target_samples=TARGETS)
    assert isinstance(draft.slots, PromptSlots)
    assert draft.slots.entity_noun == "drug mention"


async def test_the_description_reaches_the_model():
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(llm, description="matching legal citations to court decisions",
                      source_samples=SOURCES, target_samples=TARGETS)
    assert "legal citations" in llm.requests[0].user


async def test_sample_records_from_both_sides_reach_the_model():
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)
    prompt = llm.requests[0].user
    assert "aspirin" in prompt and "acetylsalicylic acid" in prompt


async def test_samples_are_capped():
    llm = FakeLLM([json.dumps(GOOD)])
    many = [Record(id=f"s{i}", fields={"mention": f"drug{i}"}) for i in range(50)]
    await draft_slots(llm, description="d", source_samples=many, target_samples=TARGETS,
                      max_samples=3)
    prompt = llm.requests[0].user
    assert "drug0" in prompt and "drug40" not in prompt


async def test_the_model_is_never_asked_for_prompt_text():
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)
    prompt = llm.requests[0].user.lower()
    assert "chosen_key" not in prompt  # the output contract is not the model's business


async def test_an_out_of_order_rubric_is_repaired_and_warned_about():
    payload = {**GOOD, "rubric": list(reversed(GOOD["rubric"]))}
    llm = FakeLLM([json.dumps(payload)])
    draft = await draft_slots(llm, description="d", source_samples=SOURCES,
                              target_samples=TARGETS)
    assert [r.score for r in draft.slots.rubric] == [1.0, 0.6, 0.3]
    assert any("rubric" in w for w in draft.warnings)


async def test_an_out_of_range_score_is_clamped_and_warned_about():
    payload = {**GOOD, "rubric": [{**GOOD["rubric"][0], "score": 1.4}, *GOOD["rubric"][1:]]}
    llm = FakeLLM([json.dumps(payload)])
    draft = await draft_slots(llm, description="d", source_samples=SOURCES,
                              target_samples=TARGETS)
    assert draft.slots.rubric[0].score == 1.0
    assert any("clamp" in w.lower() for w in draft.warnings)


async def test_a_single_row_rubric_is_rejected_with_a_clear_error():
    payload = {**GOOD, "rubric": GOOD["rubric"][:1]}
    llm = FakeLLM([json.dumps(payload)])
    with pytest.raises(ValueError, match="rubric"):
        await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)


async def test_malformed_output_is_a_clear_error_not_a_broken_prompt():
    llm = FakeLLM(["I'm not sure what you want."])
    with pytest.raises(ValueError, match="could not"):
        await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)


async def test_a_draft_that_fails_contract_validation_is_rejected():
    """Belt and braces: even valid slots must render a contract-satisfying prompt."""
    payload = {**GOOD, "entity_noun": "   "}
    llm = FakeLLM([json.dumps(payload)])
    with pytest.raises(ValueError):
        await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)


async def test_existing_slots_are_shown_when_refining():
    existing = PromptSlots(**GOOD)
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS,
                      existing=existing)
    assert "pharmaceutical nomenclature" in llm.requests[0].user


def test_slots_diff_shows_changed_fields():
    before = PromptSlots(**GOOD)
    after = before.model_copy(update={"domain_brief": "veterinary pharmacology"})
    diff = slots_diff(before, after)
    assert "domain_brief" in diff and "veterinary pharmacology" in diff


def test_slots_diff_is_empty_for_identical_slots():
    before = PromptSlots(**GOOD)
    assert slots_diff(before, before) == ""


def test_write_slots_produces_a_loadable_file(tmp_path):
    path = tmp_path / "slots.yaml"
    write_slots(PromptSlots(**GOOD), path)
    assert load_slots(path).entity_noun == "drug mention"


def test_write_slots_produces_readable_yaml(tmp_path):
    path = tmp_path / "slots.yaml"
    write_slots(PromptSlots(**GOOD), path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data["rubric"], list)
    assert "!!python" not in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Write `src/xwalk/prompts/author.py`**

```python
"""Drafting domain slots with an LLM.

The model returns **slots only** — never prompt text. Phase 1's skeleton owns every part
of the machine-readable contract, so a bad draft can produce a poor rubric but never a
broken prompt. `validate_contract` runs before anything reaches disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_json_object
from xwalk.prompts.contract import PromptSet, PromptSlots, validate_contract
from xwalk.records import Record, Usage

SLOTS_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "entity_noun": {"type": "string"},
        "target_noun": {"type": "string"},
        "domain_brief": {"type": "string"},
        "rubric": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "name": {"type": "string"},
                    "when": {"type": "string"},
                    "example": {"type": "string"},
                },
                "required": ["score", "name", "when"],
            },
        },
        "hard_rules": {"type": "array", "items": {"type": "string"}},
        "disambiguation_steps": {"type": "string"},
    },
    "required": ["entity_noun", "target_noun", "domain_brief", "rubric"],
}

_SYSTEM = (
    "You design the domain-specific vocabulary for a record-matching system. "
    "You return JSON only. No prose, no code fences."
)

_INSTRUCTIONS = """\
Describe the matching task below as a set of slots.

- entity_noun: what one source record is, as a noun phrase ("chemical entity mention")
- target_noun: what one target record is ("ChEBI ontology term")
- domain_brief: one clause naming the domain and its naming conventions
- rubric: 3 to 5 confidence bands, scores strictly decreasing, each with a short `when`
  condition and a concrete `example` drawn from the samples where possible
- hard_rules: absolute rules a careful domain expert would insist on; omit if none apply
- disambiguation_steps: free text, only if this domain has a specific procedure
  (for example, using surrounding context to decide the organism for a gene symbol)

Write for a careful annotator who knows nothing about this domain."""


def _sample_block(records: Sequence[Record], limit: int) -> str:
    lines = []
    for record in list(records)[:limit]:
        rendered = "; ".join(f"{k}={v}" for k, v in record.fields.items())
        lines.append(f"- {record.id}: {rendered}")
    return "\n".join(lines)


@dataclass(frozen=True)
class DraftResult:
    slots: PromptSlots
    warnings: tuple[str, ...]
    usage: Usage
    raw: str


def _repair(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fix what is mechanically fixable; everything else raises through PromptSlots."""
    warnings: list[str] = []
    rows = payload.get("rubric") or []

    clamped = []
    for row in rows:
        score = row.get("score")
        if isinstance(score, (int, float)) and not 0.0 <= float(score) <= 1.0:
            warnings.append(f"clamped rubric score {score} into [0, 1]")
            row = {**row, "score": max(0.0, min(1.0, float(score)))}
        clamped.append(row)

    scores = [r.get("score") for r in clamped]
    if scores != sorted(scores, reverse=True):
        warnings.append("rubric was not in strictly decreasing score order; reordered")
        clamped = sorted(clamped, key=lambda r: -float(r.get("score", 0.0)))

    return {**payload, "rubric": clamped}, warnings


async def draft_slots(
    llm: LLMClient,
    *,
    description: str,
    source_samples: Sequence[Record],
    target_samples: Sequence[Record],
    existing: PromptSlots | None = None,
    max_samples: int = 8,
    max_tokens: int = 2048,
) -> DraftResult:
    parts = [
        _INSTRUCTIONS,
        f"\n## The task\n{description}",
        f"\n## Sample source records\n{_sample_block(source_samples, max_samples)}",
        f"\n## Sample target records\n{_sample_block(target_samples, max_samples)}",
    ]
    if existing is not None:
        parts.append(
            "\n## Current slots, to refine rather than replace\n"
            + json.dumps(json.loads(existing.model_dump_json()), indent=2)
        )

    response = await llm.complete(
        LLMRequest(
            system=_SYSTEM,
            user="\n".join(parts),
            schema=SLOTS_SCHEMA,
            schema_name="slots",
            max_tokens=max_tokens,
        )
    )

    try:
        payload = parse_json_object(response.text)
    except ParseError as exc:
        raise ValueError(
            f"could not read slots from the drafting model: {exc}\n"
            f"raw response:\n{response.text[:1000]}"
        ) from exc

    repaired, warnings = _repair(payload)
    slots = PromptSlots.model_validate(repaired)  # raises with a clear message on bad input
    validate_contract(PromptSet.from_slots(slots))  # never write a prompt that cannot parse

    return DraftResult(
        slots=slots, warnings=tuple(warnings), usage=response.usage, raw=response.text
    )


def slots_diff(before: PromptSlots, after: PromptSlots) -> str:
    """A field-level diff, shown before anything is written."""
    old = json.loads(before.model_dump_json())
    new = json.loads(after.model_dump_json())
    lines: list[str] = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) == new.get(key):
            continue
        lines.append(f"- {key}:")
        lines.append(f"    before: {json.dumps(old.get(key), ensure_ascii=False)}")
        lines.append(f"    after:  {json.dumps(new.get(key), ensure_ascii=False)}")
    return "\n".join(lines)


def write_slots(slots: PromptSlots, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            json.loads(slots.model_dump_json()), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
```

- [ ] **Step 3: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_author.py -v   # expect 15 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/prompts/author.py tests/test_author.py
git commit -m "feat: LLM-assisted slot drafting with repair, validation, and diff"
```

---

## Task 7: The three-partition prompt optimizer

**Files:**
- Create: `src/xwalk/prompts/optimize.py`
- Test: `tests/test_optimize.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6, plus `Matcher`/`run_batch`/`Ledger` (Phase 1).
- Produces: `OptimizeConfig` (frozen: `role`, `rounds=4`, `patience=2`, `failures_per_round=12`, `max_calls=None`, `objective="accepted_precision"`); `RoundResult` (frozen: `round_index`, `slots`, `validation`, `improved`, `warnings`, `usage`); `OptimizeReport` (frozen: `best_slots`, `baseline`, `rounds`, `test_report`, `total_usage`, `stopped_because`); `estimate_calls(config, n_prompt_train, n_validation, n_test) -> int`; `async optimize_prompt(...) -> OptimizeReport`.

**The rule this task exists to enforce:** test is evaluated **once**, after the final prompt is selected. Not per round. A test set observed every round is a second validation set, which is the same mistake one level up.

**Signature** — the optimizer needs to re-run matching per round, so it takes a factory rather than a matcher:

```python
async def optimize_prompt(
    *,
    matcher_factory: Callable[[PromptSet], Matcher],
    source_records: Sequence[Record],
    gold: GoldSet,
    initial: PromptSlots,
    optimiser_llm: LLMClient,
    partitioner: Partitioner = Partitioner(),
    partition_overrides: Mapping[str, Partition] | None = None,
    config: OptimizeConfig = OptimizeConfig(),
    work_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> OptimizeReport: ...
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_optimize.py`:

```python
import json
from pathlib import Path

import pytest

from tests.test_matcher import PROMPTS, STORE, TEMPLATES, ScriptedRetriever, score_reply, select_reply
from xwalk.evaluate.failures import PromptRole
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.partition import Partition, Partitioner
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.prompts.optimize import OptimizeConfig, estimate_calls, optimize_prompt
from xwalk.records import Record
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector

SLOTS = PromptSlots(
    entity_noun="mention", target_noun="term", domain_brief="v0",
    rubric=[{"score": 1.0, "name": "Certain", "when": "exact"},
            {"score": 0.4, "name": "Weak", "when": "vague"}],
)
SOURCES = [Record(id=f"s{i:02d}", fields={"mention": "glucose"}) for i in range(30)]
GOLD = GoldSet({f"s{i:02d}": frozenset({"T1"}) for i in range(30)})
RETRIEVER_MAP = {"glucose": ["T1", "T2"]}


def matcher_llm(accuracy: float):
    """Answers correctly for the first `accuracy` share of each block of ten records.

    Deterministic and independent of how many matchers get built, which is the point:
    the earlier version keyed correctness on factory-call order, so a "successful"
    prompt-train run produced no failures and the optimiser exited before round one.
    """
    state = {"n": 0}

    def handler(request):
        if "## Candidates" in request.user:
            index = state["n"]
            state["n"] += 1
            return select_reply("C01" if (index % 10) < round(accuracy * 10) else "C02")
        return score_reply(0.95)

    return FakeLLM(handler=handler)


def make_factory(accuracy_by_brief, default=0.5):
    """Correctness is a property of the prompt under test, not of call ordering."""

    def factory(prompts: PromptSet) -> Matcher:
        llm = matcher_llm(accuracy_by_brief.get(prompts.slots.domain_brief, default))
        return Matcher(
            templates=TEMPLATES, retrievers=[ScriptedRetriever(RETRIEVER_MAP)], store=STORE,
            selector=Selector(llm, prompts, TEMPLATES),
            scorer=Scorer(llm, prompts, TEMPLATES),
            verifier=Verifier(llm, prompts, TEMPLATES),
            rewriter=QueryRewriter(llm, prompts, TEMPLATES),
            policy=MatchPolicy(max_attempts=1),
            run_fingerprint=prompts.slots.domain_brief,
        )

    return factory


def optimiser_llm(*briefs):
    replies = [
        json.dumps(
            {
                "entity_noun": "mention", "target_noun": "term", "domain_brief": brief,
                "rubric": [{"score": 1.0, "name": "Certain", "when": "exact"},
                           {"score": 0.4, "name": "Weak", "when": "vague"}],
                "hard_rules": [], "disambiguation_steps": "",
            }
        )
        for brief in briefs
    ]
    return FakeLLM(replies)


# --- partition discipline ---------------------------------------------------------

async def test_the_test_partition_is_evaluated_exactly_once(tmp_path):
    """The single most important property of this module."""
    seen: list[str] = []

    report = await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0, "v2": 1.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("v1", "v2"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=2),
        work_dir=tmp_path, progress=seen.append,
    )
    assert sum(1 for line in seen if "test partition" in line.lower()) == 1
    assert report.test_report is not None


async def test_test_scores_are_absent_from_every_round_result(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    for round_result in report.rounds:
        assert not hasattr(round_result, "test")


async def test_only_prompt_train_failures_reach_the_optimising_model(tmp_path):
    llm = optimiser_llm("v1")
    partitioner = Partitioner(fractions=(0.34, 0.33, 0.33), salt="fixed")
    test_ids = {
        s.id for s in SOURCES if partitioner.assign(s.id) is Partition.TEST
    }
    await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS, optimiser_llm=llm,
        partitioner=partitioner,
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1), work_dir=tmp_path,
    )
    prompt = llm.requests[0].user
    assert test_ids and not any(tid in prompt for tid in test_ids)


async def test_partition_overrides_are_honoured(tmp_path):
    overrides = {s.id: Partition.PROMPT_TRAIN for s in SOURCES[:10]}
    overrides.update({s.id: Partition.VALIDATION for s in SOURCES[10:20]})
    overrides.update({s.id: Partition.TEST for s in SOURCES[20:]})
    report = await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"), partition_overrides=overrides,
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1), work_dir=tmp_path,
    )
    assert report.test_report.labelled == 10


# --- round selection --------------------------------------------------------------

async def test_an_improving_round_is_retained(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({"better": 1.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("better"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1), work_dir=tmp_path,
    )
    assert report.best_slots.domain_brief == "better"
    assert report.rounds[0].improved is True


async def test_a_regressing_round_is_discarded(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({"worse": 0.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("worse"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1), work_dir=tmp_path,
    )
    assert report.best_slots.domain_brief == "v0"
    assert report.rounds[0].improved is False


async def test_the_baseline_is_measured_before_any_round(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({}, default=1.0),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1), work_dir=tmp_path,
    )
    assert report.baseline.accepted_precision == 1.0


async def test_optimisation_stops_early_after_patience_rounds_without_improvement(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("a", "b", "c", "d"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4, patience=2),
        work_dir=tmp_path,
    )
    assert len(report.rounds) == 2
    assert "patience" in report.stopped_because


async def test_optimisation_stops_when_there_are_no_failures_to_learn_from(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({}, default=1.0),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("unused"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4), work_dir=tmp_path,
    )
    assert report.rounds == []
    assert "no failures" in report.stopped_because.lower()


# --- safety -----------------------------------------------------------------------

async def test_a_round_producing_invalid_slots_is_skipped_not_fatal(tmp_path):
    llm = FakeLLM(["not json at all", json.dumps(
        {"entity_noun": "mention", "target_noun": "term", "domain_brief": "recovered",
         "rubric": [{"score": 1.0, "name": "C", "when": "x"},
                    {"score": 0.4, "name": "W", "when": "y"}],
         "hard_rules": [], "disambiguation_steps": ""}
    )])
    report = await optimize_prompt(
        matcher_factory=make_factory({"recovered": 1.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS, optimiser_llm=llm,
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=2), work_dir=tmp_path,
    )
    assert any(r.warnings for r in report.rounds)
    assert report.best_slots.domain_brief == "recovered"


async def test_the_call_budget_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="max_calls"):
        await optimize_prompt(
            matcher_factory=make_factory({}),
            source_records=SOURCES, gold=GOLD, initial=SLOTS,
            optimiser_llm=optimiser_llm("v1"),
            config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4, max_calls=5),
            work_dir=tmp_path,
        )


def test_estimate_calls_scales_with_rounds_and_records():
    small = estimate_calls(OptimizeConfig(rounds=1), 10, 10, 10)
    large = estimate_calls(OptimizeConfig(rounds=4), 10, 10, 10)
    assert large > small


def test_estimate_calls_counts_the_test_partition_once():
    """Not once per round — that would be both wasteful and methodologically wrong.

    Compared as the test partition's *contribution*: the per-round optimiser call means
    the totals themselves differ with `rounds`, so equating them would be simply false.
    """
    one = OptimizeConfig(rounds=1)
    four = OptimizeConfig(rounds=4)
    contribution_one = estimate_calls(one, 0, 0, 100) - estimate_calls(one, 0, 0, 0)
    contribution_four = estimate_calls(four, 0, 0, 100) - estimate_calls(four, 0, 0, 0)
    assert contribution_one == contribution_four == 200


async def test_every_round_is_persisted_for_inspection(tmp_path):
    await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1), work_dir=tmp_path,
    )
    assert (Path(tmp_path) / "round_01" / "slots.yaml").exists()
    assert (Path(tmp_path) / "best" / "slots.yaml").exists()
    assert (Path(tmp_path) / "report.json").exists()


async def test_the_final_report_is_json_serialisable(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES, gold=GOLD, initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1), work_dir=tmp_path,
    )
    json.dumps(report.as_dict())
```

- [ ] **Step 2: Write `src/xwalk/prompts/optimize.py`**

```python
"""Label-driven prompt optimisation over three partitions.

  prompt-train : failures shown to the optimising model
  validation   : chooses the retained round and the stopping point
  test         : evaluated ONCE, after the final prompt is selected

Reporting test per round would turn it into a second validation set — the same mistake
one level up — so this module never evaluates it inside the loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from xwalk.evaluate.failures import PromptRole, render_failure, select_failures
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import EvalReport, evaluate_results
from xwalk.evaluate.partition import Partition, Partitioner, partition_of
from xwalk.llm.base import LLMClient, LLMRequest
from xwalk.matcher import Matcher
from xwalk.prompts.author import SLOTS_SCHEMA, write_slots
from xwalk.prompts.contract import PromptSet, PromptSlots, validate_contract
from xwalk.llm.parsing import parse_json_object
from xwalk.records import MatchResult, Record, Usage

_SYSTEM = (
    "You improve the domain vocabulary of a record-matching system by revising its "
    "slots. You return JSON only. No prose, no code fences."
)

_ROLE_BRIEF: Mapping[PromptRole, str] = {
    PromptRole.SELECTOR: (
        "These are cases where the correct target record WAS shown to the model and the "
        "model picked a different one. Revise the slots so the distinction is stated "
        "explicitly — usually in `hard_rules` or `disambiguation_steps`."
    ),
    PromptRole.SCORER: (
        "These are confidence-calibration failures: confident wrong accepts, and needless "
        "abstentions or low scores on correct matches. Revise the `rubric` so its bands "
        "separate these cases, and add `hard_rules` where a band is being misapplied."
    ),
    PromptRole.REWRITER: (
        "These are cases where the first search query did not surface the correct record. "
        "Revise `domain_brief` and `disambiguation_steps` to describe how names vary in "
        "this domain — abbreviations, formal names, qualifiers worth dropping."
    ),
    PromptRole.DOC_TEMPLATE: (
        "These are cases where the correct record was never retrieved at all. Revise "
        "`domain_brief` to describe which target fields carry the searchable names, so a "
        "human can fix the indexing template."
    ),
}


@dataclass(frozen=True)
class OptimizeConfig:
    role: PromptRole = PromptRole.SELECTOR
    rounds: int = 4
    patience: int = 2
    failures_per_round: int = 12
    max_calls: int | None = None
    objective: str = "accepted_precision"
    max_tokens: int = 2048


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    slots: PromptSlots
    validation: EvalReport
    improved: bool
    warnings: tuple[str, ...]
    usage: Usage

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "slots": json.loads(self.slots.model_dump_json()),
            "validation": self.validation.as_dict(),
            "improved": self.improved,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class OptimizeReport:
    best_slots: PromptSlots
    baseline: EvalReport
    rounds: tuple[RoundResult, ...]
    test_report: EvalReport | None
    total_usage: Usage
    stopped_because: str
    partition_sizes: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_slots": json.loads(self.best_slots.model_dump_json()),
            "baseline_validation": self.baseline.as_dict(),
            "rounds": [r.as_dict() for r in self.rounds],
            "test": None if self.test_report is None else self.test_report.as_dict(),
            "stopped_because": self.stopped_because,
            "partition_sizes": dict(self.partition_sizes),
            "usage": {
                "prompt_tokens": self.total_usage.prompt_tokens,
                "completion_tokens": self.total_usage.completion_tokens,
                "calls": self.total_usage.calls,
            },
        }


def estimate_calls(
    config: OptimizeConfig,
    n_prompt_train: int,
    n_validation: int,
    n_test: int,
    *,
    calls_per_record: int = 2,
) -> int:
    """Printed before spending anything. Test is counted once, not once per round."""
    per_round = (n_prompt_train + n_validation) * calls_per_record + 1  # +1 optimiser call
    baseline = n_validation * calls_per_record  # the baseline runs validation only
    return baseline + config.rounds * per_round + n_test * calls_per_record


def _objective(report: EvalReport, name: str) -> float:
    value = getattr(report, name, None)
    return -1.0 if value is None else float(value)


async def _run(
    matcher: Matcher, records: Sequence[Record]
) -> tuple[list[MatchResult], Usage]:
    results = [await matcher.match(record) for record in records]
    return results, sum((r.usage for r in results), Usage.zero())


def _render_optimiser_prompt(
    slots: PromptSlots,
    role: PromptRole,
    failures: Sequence[Any],
) -> str:
    cases = "\n\n".join(render_failure(case) for case in failures)
    return "\n".join(
        [
            _ROLE_BRIEF[role],
            "",
            "## Current slots",
            json.dumps(json.loads(slots.model_dump_json()), indent=2, ensure_ascii=False),
            "",
            f"## Failing cases ({len(failures)})",
            cases,
            "",
            "Return the complete revised slots object. Keep every field; change only what "
            "the failures justify. Rubric scores must remain strictly decreasing and within "
            "[0, 1].",
        ]
    )


async def optimize_prompt(
    *,
    matcher_factory: Callable[[PromptSet], Matcher],
    source_records: Sequence[Record],
    gold: GoldSet,
    initial: PromptSlots,
    optimiser_llm: LLMClient,
    work_dir: str | Path,
    partitioner: Partitioner = Partitioner(),
    partition_overrides: Mapping[str, Partition] | None = None,
    config: OptimizeConfig = OptimizeConfig(),
    progress: Callable[[str], None] | None = None,
) -> OptimizeReport:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    labelled = [r for r in source_records if r.id in gold]
    by_partition: dict[Partition, list[Record]] = {p: [] for p in Partition}
    for record in labelled:
        by_partition[partition_of(record.id, partitioner, partition_overrides)].append(record)

    train = by_partition[Partition.PROMPT_TRAIN]
    validation = by_partition[Partition.VALIDATION]
    test = by_partition[Partition.TEST]
    sizes = {p.value: len(by_partition[p]) for p in Partition}
    say(f"partitions: {sizes}")

    estimated = estimate_calls(config, len(train), len(validation), len(test))
    say(f"estimated LLM calls: ~{estimated}")
    if config.max_calls is not None and estimated > config.max_calls:
        raise ValueError(
            f"estimated {estimated} calls exceeds max_calls={config.max_calls}; "
            f"reduce rounds, shrink the labelled set, or raise the budget"
        )

    total_usage = Usage.zero()

    # --- baseline, on validation only ---
    best_slots = initial
    baseline_matcher = matcher_factory(PromptSet.from_slots(best_slots))
    baseline_results, usage = await _run(baseline_matcher, validation)
    total_usage = total_usage + usage
    baseline = evaluate_results(baseline_results, gold)
    best_score = _objective(baseline, config.objective)
    say(f"baseline {config.objective}: {best_score:.3f}")

    rounds: list[RoundResult] = []
    stopped_because = f"completed {config.rounds} rounds"
    since_improvement = 0

    for round_index in range(1, config.rounds + 1):
        # failures come from prompt-train, run under the CURRENT best slots
        train_matcher = matcher_factory(PromptSet.from_slots(best_slots))
        train_results, usage = await _run(train_matcher, train)
        total_usage = total_usage + usage

        failures = select_failures(
            train_results, gold, config.role, limit=config.failures_per_round
        )
        # attach source fields so the rendering is informative
        fields = {r.id: {k: str(v) for k, v in r.fields.items()} for r in train}
        failures = [
            type(case)(**{**case.__dict__, "source_fields": fields.get(case.source_id, {})})
            for case in failures
        ]

        if not failures:
            stopped_because = f"no failures for role {config.role.value} on prompt-train"
            say(stopped_because)
            break

        response = await optimiser_llm.complete(
            LLMRequest(
                system=_SYSTEM,
                user=_render_optimiser_prompt(best_slots, config.role, failures),
                schema=SLOTS_SCHEMA,
                schema_name="slots",
                max_tokens=config.max_tokens,
            )
        )
        total_usage = total_usage + response.usage

        warnings: list[str] = []
        try:
            payload = parse_json_object(response.text)
            candidate_slots = PromptSlots.model_validate(payload)
            validate_contract(PromptSet.from_slots(candidate_slots))
        except Exception as exc:  # noqa: BLE001 - a bad round must not end the run
            warnings.append(f"round {round_index} produced unusable slots: {exc}")
            say(warnings[-1])
            rounds.append(
                RoundResult(
                    round_index=round_index, slots=best_slots, validation=baseline,
                    improved=False, warnings=tuple(warnings), usage=response.usage,
                )
            )
            since_improvement += 1
            if since_improvement >= config.patience:
                stopped_because = f"patience {config.patience} exhausted"
                say(stopped_because)
                break
            continue

        candidate_matcher = matcher_factory(PromptSet.from_slots(candidate_slots))
        candidate_results, usage = await _run(candidate_matcher, validation)
        total_usage = total_usage + usage
        candidate_report = evaluate_results(candidate_results, gold)
        candidate_score = _objective(candidate_report, config.objective)

        improved = candidate_score > best_score
        say(
            f"round {round_index}: validation {config.objective} "
            f"{candidate_score:.3f} ({'kept' if improved else 'discarded'})"
        )

        round_dir = work_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        write_slots(candidate_slots, round_dir / "slots.yaml")
        (round_dir / "validation.json").write_text(
            json.dumps(candidate_report.as_dict(), indent=2), encoding="utf-8"
        )

        rounds.append(
            RoundResult(
                round_index=round_index, slots=candidate_slots, validation=candidate_report,
                improved=improved, warnings=tuple(warnings), usage=response.usage,
            )
        )

        if improved:
            best_slots = candidate_slots
            best_score = candidate_score
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                stopped_because = f"patience {config.patience} exhausted"
                say(stopped_because)
                break

    # --- test, once, after the final prompt is chosen ---
    say("evaluating the selected prompt on the test partition (once)")
    test_matcher = matcher_factory(PromptSet.from_slots(best_slots))
    test_results, usage = await _run(test_matcher, test)
    total_usage = total_usage + usage
    test_report = evaluate_results(test_results, gold)

    best_dir = work_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    write_slots(best_slots, best_dir / "slots.yaml")

    report = OptimizeReport(
        best_slots=best_slots,
        baseline=baseline,
        rounds=tuple(rounds),
        test_report=test_report,
        total_usage=total_usage,
        stopped_because=stopped_because,
        partition_sizes=sizes,
    )
    (work_dir / "report.json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
```

- [ ] **Step 3: Run the tests**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: 16 passed.

`test_the_test_partition_is_evaluated_exactly_once` is the one that must never be weakened. If it fails, the fix is in `optimize_prompt`, not in the test.

The `type(case)(**{...})` reconstruction for `source_fields` is awkward; replace it with `dataclasses.replace(case, source_fields=fields.get(case.source_id, {}))` — `FailureCase` is a frozen dataclass, so that is the idiomatic form.

- [ ] **Step 4: Run everything, lint, type-check, commit**

```bash
python -m pytest -q -m "not integration"
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/prompts/optimize.py tests/test_optimize.py
git commit -m "feat: three-partition prompt optimiser with role-specific failure selection"
```

---

## Task 8: The evaluation entry point and a worked example

**Files:**
- Create: `src/xwalk/evaluate/report.py`
- Modify: `src/xwalk/evaluate/__init__.py`, `README.md`
- Create: `examples/chemistry/README.md`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `FullReport` (frozen: `metrics`, `ceiling`, `thresholds`, `partition_sizes`); `evaluate(ledger, run_fingerprint, gold, *, partitioner=None, partition_overrides=None) -> FullReport`; `render_report(report) -> str`; `write_report(report, path)`.

**Design note:** this is what a user actually calls. One function, one printable summary, and the ceiling recommendation front and centre — the whole point of the phase is answering "where should I spend effort?", and that answer must not be buried three attribute accesses deep.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_report.py`:

```python
import json

import pytest

from tests.test_ceiling import GOLD, attempt, cand, result
from xwalk.evaluate.report import evaluate, render_report, write_report
from xwalk.ledger import Ledger


@pytest.fixture
async def ledger(tmp_path):
    led = Ledger.open(tmp_path / "l.sqlite")
    await led.put_result(result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})]))
    await led.put_result(result("s3", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]))
    yield led
    led.close()


async def test_evaluate_returns_metrics_and_ceiling(ledger):
    report = evaluate(ledger, "fp1", GOLD)
    assert report.metrics.labelled == 2
    assert report.ceiling.evaluable == 2


async def test_evaluate_includes_a_threshold_curve(ledger):
    assert len(evaluate(ledger, "fp1", GOLD).thresholds) == 21


async def test_the_rendered_report_leads_with_the_recommendation(ledger):
    text = render_report(evaluate(ledger, "fp1", GOLD))
    assert "Where to spend effort" in text
    assert "retriev" in text.lower()


async def test_the_rendered_report_names_every_headline_metric(ledger):
    text = render_report(evaluate(ledger, "fp1", GOLD))
    for label in ("accepted precision", "automatic coverage", "review rate"):
        assert label in text.lower()


async def test_the_rendered_report_states_when_a_metric_is_undefined(ledger):
    """A blank is ambiguous; 'n/a (no gold no-match labels)' is not."""
    text = render_report(evaluate(ledger, "fp1", GOLD))
    assert "n/a" in text.lower()


async def test_a_calibration_warning_appears_prominently_when_present(ledger, tmp_path):
    from xwalk.records import MatchResult

    led = Ledger.open(tmp_path / "cal.sqlite")
    base = result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])
    rows = [(f"c{i}", "T1", c) for i, c in enumerate([0.80, 0.82, 0.78])]
    rows += [(f"w{i}", "T9", c) for i, c in enumerate([0.81, 0.79, 0.83])]
    for i, (sid, mid, conf) in enumerate(rows):
        await led.put_result(
            MatchResult(**{**base.__dict__, "result_key": f"rk{i}", "source_id": sid,
                           "matched_id": mid, "confidence": conf})
        )
    gold = type(GOLD)(
        {**{f"c{i}": frozenset({"T1"}) for i in range(3)},
         **{f"w{i}": frozenset({"T8"}) for i in range(3)}}
    )
    text = render_report(evaluate(led, "fp1", gold))
    led.close()
    assert "calibration" in text.lower()


async def test_partition_sizes_are_reported_when_a_partitioner_is_given(ledger):
    from xwalk.evaluate.partition import Partitioner

    report = evaluate(ledger, "fp1", GOLD, partitioner=Partitioner())
    assert sum(report.partition_sizes.values()) == 2


async def test_partition_sizes_are_empty_without_a_partitioner(ledger):
    assert evaluate(ledger, "fp1", GOLD).partition_sizes == {}


async def test_write_report_produces_json_and_text(ledger, tmp_path):
    report = evaluate(ledger, "fp1", GOLD)
    write_report(report, tmp_path / "eval")
    assert (tmp_path / "eval.json").exists()
    assert (tmp_path / "eval.txt").exists()
    json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))


async def test_evaluate_calls_no_llm_and_no_retriever(ledger):
    """Evaluation reads the ledger. It must be free, repeatable, and offline."""
    report = evaluate(ledger, "fp1", GOLD)
    assert report.metrics.total == 2  # completed without any client being constructed
```

- [ ] **Step 2: Write `src/xwalk/evaluate/report.py`**

```python
"""The evaluation entry point.

One call, one printable summary, with the ceiling recommendation first — "where should I
spend effort?" is the question this phase exists to answer, and it must not be buried.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from xwalk.evaluate.ceiling import CeilingBucket, CeilingReport, ceiling_report
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import EvalReport, ThresholdPoint, evaluate_results, threshold_curve
from xwalk.evaluate.partition import Partition, Partitioner, partition_of
from xwalk.ledger import Ledger


@dataclass(frozen=True)
class FullReport:
    metrics: EvalReport
    ceiling: CeilingReport
    thresholds: tuple[ThresholdPoint, ...]
    partition_sizes: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.as_dict(),
            "ceiling": self.ceiling.as_dict(),
            "thresholds": [
                {
                    "threshold": p.threshold,
                    "coverage": p.coverage,
                    "precision": p.precision,
                    "accepted": p.accepted,
                    "correct": p.correct,
                }
                for p in self.thresholds
            ],
            "partition_sizes": dict(self.partition_sizes),
        }


def evaluate(
    ledger: Ledger,
    run_fingerprint: str,
    gold: GoldSet,
    *,
    partitioner: Partitioner | None = None,
    partition_overrides: Mapping[str, Partition] | None = None,
    threshold_steps: int = 21,
) -> FullReport:
    results = list(ledger.iter_results(run_fingerprint))

    sizes: dict[str, int] = {}
    if partitioner is not None:
        counts = {p.value: 0 for p in Partition}
        for result in results:
            if result.source_id not in gold:
                continue
            counts[partition_of(result.source_id, partitioner, partition_overrides).value] += 1
        sizes = counts

    return FullReport(
        metrics=evaluate_results(results, gold),
        ceiling=ceiling_report(results, gold),
        thresholds=tuple(threshold_curve(results, gold, steps=threshold_steps)),
        partition_sizes=sizes,
    )


def _pct(value: float | None, *, reason: str = "") -> str:
    if value is None:
        return f"n/a{f' ({reason})' if reason else ''}"
    return f"{value:.1%}"


def render_report(report: FullReport) -> str:
    m = report.metrics
    c = report.ceiling

    lines = [
        "## Where to spend effort",
        f"  {c.recommendation}",
        "",
        "## Headline",
        f"  accepted precision   : {_pct(m.accepted_precision, reason='nothing matched')}",
        f"  automatic coverage   : {_pct(m.automatic_coverage)}",
        f"  review rate          : {_pct(m.review_rate)}",
        f"  unmatched rate       : {_pct(m.unmatched_rate)}",
        f"  error rate           : {_pct(m.error_rate)}",
        f"  unresolved rate      : {_pct(m.unresolved_rate)}",
        f"  recall at any status : {_pct(m.recall_at_any_status)}",
        "",
        "## No-match handling",
        f"  no-match precision   : "
        f"{_pct(m.no_match_precision, reason='nothing predicted no-match')}",
        f"  no-match recall      : "
        f"{_pct(m.no_match_recall, reason='no gold no-match labels')}",
        "",
        "## Failure decomposition",
    ]
    for bucket in (
        CeilingBucket.FOUND,
        CeilingBucket.MISJUDGED,
        CeilingBucket.TRUNCATED,
        CeilingBucket.NEVER_RETRIEVED,
    ):
        lines.append(f"  {bucket.value:<18}: {c.buckets.get(bucket, 0)}")
    if c.by_retriever:
        lines.append("  gold surfaced by   : " + ", ".join(
            f"{name} ({count})" for name, count in sorted(c.by_retriever.items())
        ))

    lines += [
        "",
        "## Cost",
        f"  llm calls / record   : {m.mean_llm_calls if m.mean_llm_calls is not None else 'n/a'}",
        f"  tokens / record      : {m.mean_tokens if m.mean_tokens is not None else 'n/a'}",
        f"  seconds / record     : {m.mean_seconds if m.mean_seconds is not None else 'n/a'}",
        f"  duplicate targets    : {m.duplicate_target_conflicts}",
    ]

    if report.partition_sizes:
        lines += ["", "## Partitions", "  " + json.dumps(dict(report.partition_sizes))]

    if m.calibration_warning:
        lines += ["", "## Calibration warning", f"  {m.calibration_warning}"]

    return "\n".join(lines)


def write_report(report: FullReport, path_stem: str | Path) -> None:
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stem.with_suffix(".txt").write_text(render_report(report), encoding="utf-8")
```

Export `FullReport`, `evaluate`, `render_report`, `write_report`, plus `Partition`, `Partitioner`, `PromptRole`, `select_failures` from `src/xwalk/evaluate/__init__.py`.

- [ ] **Step 3: Extend the README**

Append to `README.md`:

````markdown
## Evaluation

```python
from xwalk.evaluate import evaluate, load_gold_csv, render_report
from xwalk.ledger import Ledger

gold = load_gold_csv("gold.csv")            # source_id,gold_ids  (empty = no match)
ledger = Ledger.open("run/ledger.sqlite")
report = evaluate(ledger, run_fingerprint, gold)
print(render_report(report))
```

The report leads with where to spend effort, decomposed three ways:

- **never retrieved** — fix the `doc` template, the retriever set, or retrieval depth
- **truncated** — the gold record was retrieved but the selector budget cut it; raise
  `SelectorPolicy.max_candidates`
- **misjudged** — the gold record was presented and the model chose otherwise; fix prompts

Alias expansion and ID normalization are yours, not the library's:

```python
gold = load_gold_csv(
    "gold.csv",
    normalize=lambda i: f"NCBIGene:{i}" if i.isdigit() else i,
    expand=lambda ids: frozenset().union(*(alias_map.get(i, {i}) for i in ids)),
)
```

## Prompt optimisation

```python
from xwalk.evaluate.failures import PromptRole
from xwalk.prompts.optimize import OptimizeConfig, optimize_prompt

report = await optimize_prompt(
    matcher_factory=lambda prompts: build_matcher(prompts),
    source_records=records, gold=gold, initial=slots,
    optimiser_llm=strong_model,
    config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4, max_calls=5_000),
    work_dir="opt/",
)
print(report.stopped_because, report.test_report.accepted_precision)
```

Three partitions: **prompt-train** supplies the failures shown to the optimising model,
**validation** chooses the retained round, and **test** is evaluated exactly once at the
end. Test is never reported per round — that would make it a second validation set.
````

- [ ] **Step 4: Write `examples/chemistry/README.md`**

```markdown
# Chemistry example

Matching chemical entity mentions to ChEBI ontology terms. Demonstrates the simplest
case: a one-field source, an ontology-shaped target, and domain knowledge living
entirely in `slots.yaml`.

## Files

- `slots.yaml` — the only domain-specific artefact
- `templates.yaml` — the four templates (query, context, doc, candidate)
- `gold.csv` — `source_id,gold_ids`; an empty cell means the answer is no match

## Run

```bash
python -m examples.chemistry.run --out run/
python -m examples.chemistry.evaluate --run run/ --gold examples/chemistry/gold.csv
```

Swapping this to a different domain means editing `slots.yaml` and `templates.yaml`.
No library code changes.
```

- [ ] **Step 5: Run everything, lint, type-check, commit**

```bash
python -m pytest -q -m "not integration"
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/evaluate README.md examples tests/test_eval_report.py
git commit -m "feat: evaluation entry point with ceiling-first reporting"
```

---

## Definition of done for Phase 2

- [ ] `python -m pytest -q -m "not integration"` — all green, including every Phase 1 test.
- [ ] `python -m ruff check src tests && python -m ruff format --check src tests` — clean.
- [ ] `python -m mypy` — clean under `--strict`.
- [ ] Base install still pulls no torch, sentence-transformers, faiss, rdflib, sqlalchemy, numpy, or sklearn.
- [ ] `grep -rn "test" src/xwalk/prompts/optimize.py` — every reference to the test partition is outside the round loop.
- [ ] `render_report` on the Phase 1 fixture run prints a recommendation naming one of retrieval / budget / prompts.
- [ ] `git log --oneline` shows one commit per task.

## Plan self-review notes

Spec coverage for Phase 2's three build-order items:

| Spec item | Task |
|---|---|
| Gold loading + user-supplied normalization/alias expansion | 1 |
| Operational metric suite (all nine metrics plus threshold curves and calibration warning) | 2 |
| Three-way ceiling decomposition, per retriever and per attempt | 3 |
| LLM-assisted prompt drafting into slots | 6 |
| Three partitions with role-specific failure selection | 4, 5, 7 |
| Cost estimate printed before starting, `--max-calls` honoured | 7 |

Deferred to Phase 3, deliberately: `evaluate/ablate.py`, `evaluate/compare.py`, and the
CLI surface (`xwalk eval`, `xwalk prompts draft`, `xwalk prompts optimize`) — all three
are wrappers over what this phase builds, and wrapping is cheaper once the shapes have
survived contact with real data.

One Phase 1 modification was required and is called out explicitly in Task 2 Step 1:
`elapsed_seconds` on `Attempt` and `MatchResult`. Cost and latency per record is a listed
metric, and it cannot be reconstructed after the fact.

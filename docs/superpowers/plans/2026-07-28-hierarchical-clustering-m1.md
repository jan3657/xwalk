# Hierarchical LLM-Adjudicated Clustering — Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Milestone 1 reference implementation of the clustering engine specified in `docs/superpowers/specs/2026-07-28-hierarchical-clustering-design.md`: equivalence clustering and single-parent roll-up, deterministic `batch_size=1`, BM25 + one generic dense head, streaming assign-or-mint with retrieval exhaustion, retry/consolidation/comparative-reassignment/oscillation/finalize phases, ledger/resume, review overlay, structural diagnostics, exports.

**Architecture:** A new `src/xwalk/cluster/` package alongside `Matcher`, reusing the existing generic seams verbatim: `stages/keying.py` (`assign_keys`/`resolve_key` — the opaque-key invariant), `stages/select.py` `Selector` (via a duck-typed prompts protocol), `retrieval/fusion.py` (`reciprocal_rank_fusion`), `llm/*` clients including `FakeLLM`, `fingerprint.py` hashing, and `TemplateSet`. New pieces: cluster value types, a `ClusterPolicy`/`LevelSpec` config model, a standalone `ClusterLedger` (own tables, same SQLite file convention), a growable in-RAM `Pool` (Tantivy in-memory + brute-force dense), six verifier stages + namer + reformulator with their own prompt skeletons and contract validator, the streaming `LevelEngine`, refinement phases, a cluster review overlay, diagnostics, and exports.

**Tech Stack:** Python 3.10+ (match repo), tantivy (already a hard dep), pydantic + Jinja2 (already deps, used by prompts/config), pytest, `FakeLLM` for all engine tests. Dense head via the existing `Encoder` protocol from `retrieval/dense.py` — no new dependencies.

## Global Constraints

- The spec is the authority: `docs/superpowers/specs/2026-07-28-hierarchical-clustering-design.md`. Every invariant listed in its "Invariants" section must hold.
- Milestone 1 only: `Cardinality.MANY` raises `NotImplementedError` in the engine (the enum and spec fields exist); optimistic batching, LLM-judged diagnostics, stability probes, calibration tooling are OUT (Milestone 2 plan).
- **Documented deviation:** `ClusterPolicy` has no `audit_rate` field in M1 — every assignment already gets a verifier call, so audit sampling is meaningless until M2's batched mode; the spec's `audit_rate` lands there.
- All thresholds are provisional defaults (0.7 accept / 0.5 review floor everywhere) and each threshold dataclass docstring must say "provisional — calibrate on an adjudicated development set before production use".
- Existing behavior must not change: no edits to `matcher.py`, `policy.py`, `ledger.py`, `records.py`, `gate.py`. The only permitted edit to existing code is the `SelectPrompts` Protocol in `stages/select.py` (Task 4) and package exports/docs (Task 12).
- Style: match repo conventions — `from __future__ import annotations`, frozen dataclasses, module docstrings that state the why, `mypy --strict`-clean typing, flat `tests/test_cluster_*.py` files.
- Run the full existing suite (`pytest`) before the final commit of every task; nothing may regress.
- Determinism: no wall-clock, no `random`. Any sampling or tie-break is seeded from fingerprints via `hashlib`, following `policy.should_audit`'s pattern.

## File Structure

```
src/xwalk/cluster/
    __init__.py        # public API re-exports
    records.py         # enums, ClusterEntity, Decision, Assignment, id/key helpers
    policy.py          # ClusterPolicy, LevelSpec + sub-configs, fingerprint payloads
    ledger.py          # ClusterLedger: own additive schema, same SQLite file convention
    prompts.py         # ClusterPromptSlots, ClusterPromptSet, schemas, contract validator
    skeletons/         # select.j2, verify_assign.j2, verify_novelty.j2, merge.j2,
                       # comparative.j2, name.j2, name_verify.j2, reformulate.j2
    pool.py            # growable in-RAM index of taxonomy nodes + minted clusters
    stages.py          # AssignmentVerifier, NoveltyVerifier, MergeVerifier,
                       # ComparativeVerifier, Namer, NameVerifier, Reformulator
    engine.py          # LevelEngine streaming pass, ClusterRun, build_cluster_fingerprint
    refine.py          # R1 retry, R2 consolidate, R3 reassign, R4 loop+oscillation, R5 finalize
    review.py          # cluster review overlay (export/read/apply/adjudicated)
    exports.py         # hierarchy.csv, clusters.csv
    diagnostics.py     # token-free structural diagnostics from the ledger
tests/
    test_cluster_records.py, test_cluster_policy.py, test_cluster_ledger.py,
    test_cluster_prompts.py, test_cluster_pool.py, test_cluster_stages.py,
    test_cluster_engine.py, test_cluster_refine.py, test_cluster_run.py,
    test_cluster_review.py, test_cluster_diagnostics.py
```

Dependency order: records → policy → ledger → prompts → pool → stages → engine → refine → review → exports/diagnostics → docs.

---

### Task 1: Cluster value types (`records.py`)

**Files:**
- Create: `src/xwalk/cluster/__init__.py` (empty for now), `src/xwalk/cluster/records.py`
- Test: `tests/test_cluster_records.py`

**Interfaces:**
- Consumes: `xwalk.fingerprint.hash_value`, `xwalk.records.Record`
- Produces (used by every later task):
  - Enums (all `str`-valued): `Relation{EQUIVALENCE,SUBSUMPTION}`, `Cardinality{ONE,MANY}`, `Provenance{TAXONOMY,MINTED}`, `ClusterState{PROVISIONAL,FINAL,RETIRED}`, `MintProvenance{ORDINARY,EXHAUSTED}`, `StopReason{CONVERGED,MAX_ITERATIONS,OSCILLATION}`, `AssignStatus{ASSIGNED,DEFERRED,NEEDS_REVIEW,FAILED}`, `Phase{STREAM,RETRY,CONSOLIDATE,REASSIGN,FINALIZE}`
  - `ClusterEntity(cluster_id, level, name, gloss, aliases: tuple[str,...], provenance, state, seed_member_id: str|None, exemplars: tuple[str,...], created_snapshot: int, mint_provenance: MintProvenance|None=None, merged_into: str|None=None)` frozen dataclass
  - `Decision(decision_key, run_fingerprint, level, phase, iteration, order_index, batch_id, snapshot_id, subject_id, subject_hash, kind, model, prompt, response, resolved: Mapping, confidence: float|None, outcome: str)` frozen dataclass
  - `Assignment(level, member_id, member_hash, cluster_id: str|None, status: AssignStatus, confidence: float|None, phase: Phase, iteration: int, via_mint: bool)` frozen dataclass
  - `make_cluster_id(run_fingerprint: str, level: str, seed_member_id: str) -> str`
  - `decision_key(run_fingerprint, level, phase: Phase, iteration: int, subject_id: str, subject_hash: str, kind: str, seq: int = 0) -> str`
  - `entity_to_record(entity: ClusterEntity) -> Record` with fields `{"name","gloss","aliases","exemplars_count","provenance"}`
  - `entity_hash(entity: ClusterEntity) -> str` — digest of identity-relevant content (name, gloss, aliases, exemplars), used by review snapshot guards

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_cluster_records.py"""
from xwalk.cluster.records import (
    ClusterEntity, ClusterState, MintProvenance, Phase, Provenance,
    decision_key, entity_hash, entity_to_record, make_cluster_id,
)


def _entity(**overrides):
    base = dict(
        cluster_id="k1", level="canonical", name="dark chocolate",
        gloss="chocolate with high cocoa content", aliases=("Dark Chocolate",),
        provenance=Provenance.MINTED, state=ClusterState.PROVISIONAL,
        seed_member_id="m1", exemplars=("m1",), created_snapshot=1,
        mint_provenance=MintProvenance.ORDINARY,
    )
    base.update(overrides)
    return ClusterEntity(**base)


def test_cluster_id_is_deterministic_and_name_independent():
    a = make_cluster_id("fp", "canonical", "m1")
    assert a == make_cluster_id("fp", "canonical", "m1")
    assert a != make_cluster_id("fp", "canonical", "m2")
    assert a != make_cluster_id("fp", "family", "m1")


def test_decision_key_separates_kind_phase_iteration_seq():
    keys = {
        decision_key("fp", "l", Phase.STREAM, 0, "m1", "h", "select"),
        decision_key("fp", "l", Phase.STREAM, 0, "m1", "h", "verify_assign"),
        decision_key("fp", "l", Phase.REASSIGN, 1, "m1", "h", "select"),
        decision_key("fp", "l", Phase.STREAM, 0, "m1", "h", "select", seq=1),
    }
    assert len(keys) == 4


def test_entity_to_record_and_hash():
    record = entity_to_record(_entity())
    assert record.id == "k1"
    assert record.fields["name"] == "dark chocolate"
    assert entity_hash(_entity()) != entity_hash(_entity(name="milk chocolate"))
    # identity-irrelevant fields do not change the hash
    assert entity_hash(_entity()) == entity_hash(_entity(created_snapshot=9))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cluster_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xwalk.cluster'`

- [ ] **Step 3: Implement `records.py`**

```python
"""Value types for the clustering engine. Everything in xwalk.cluster is built from these."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from xwalk.fingerprint import hash_value
from xwalk.records import Record


class Relation(Enum):
    EQUIVALENCE = "equivalence"
    SUBSUMPTION = "subsumption"


class Cardinality(Enum):
    ONE = "one"
    MANY = "many"  # defined by the spec; the M1 engine rejects it


class Provenance(Enum):
    TAXONOMY = "taxonomy"
    MINTED = "minted"


class ClusterState(Enum):
    PROVISIONAL = "provisional"
    FINAL = "final"
    RETIRED = "retired"


class MintProvenance(Enum):
    """ORDINARY: exhaustion surfaced nothing beyond ordinary retrieval.
    EXHAUSTED: exhaustion surfaced additional candidates, all rejected by the
    novelty verifier. Either way, minting is only reachable through exhaustion."""

    ORDINARY = "ordinary"
    EXHAUSTED = "exhausted"


class StopReason(Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    OSCILLATION = "oscillation"


class AssignStatus(Enum):
    ASSIGNED = "assigned"
    DEFERRED = "deferred"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class Phase(Enum):
    STREAM = "stream"
    RETRY = "retry"
    CONSOLIDATE = "consolidate"
    REASSIGN = "reassign"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class ClusterEntity:
    cluster_id: str
    level: str
    name: str
    gloss: str
    aliases: tuple[str, ...]
    provenance: Provenance
    state: ClusterState
    seed_member_id: str | None
    exemplars: tuple[str, ...]
    created_snapshot: int
    mint_provenance: MintProvenance | None = None
    merged_into: str | None = None


@dataclass(frozen=True)
class Decision:
    decision_key: str
    run_fingerprint: str
    level: str
    phase: Phase
    iteration: int
    order_index: int
    batch_id: int
    snapshot_id: int
    subject_id: str
    subject_hash: str
    kind: str
    model: str
    prompt: str
    response: str
    resolved: Mapping[str, Any]
    confidence: float | None
    outcome: str


@dataclass(frozen=True)
class Assignment:
    level: str
    member_id: str
    member_hash: str
    cluster_id: str | None
    status: AssignStatus
    confidence: float | None
    phase: Phase
    iteration: int
    via_mint: bool


def make_cluster_id(run_fingerprint: str, level: str, seed_member_id: str) -> str:
    """Stable id: deterministic, independent of generated names and glosses."""
    return "K" + hash_value(["cluster", run_fingerprint, level, seed_member_id])


def decision_key(
    run_fingerprint: str,
    level: str,
    phase: Phase,
    iteration: int,
    subject_id: str,
    subject_hash: str,
    kind: str,
    seq: int = 0,
) -> str:
    """Resume identity of one LLM decision. Parts go in as a list, never concatenated."""
    return hash_value(
        [run_fingerprint, level, phase.value, iteration, subject_id, subject_hash, kind, seq]
    )


def entity_to_record(entity: ClusterEntity) -> Record:
    """Render a pool entry as a Record so keying and fusion reuse works unchanged."""
    return Record(
        id=entity.cluster_id,
        fields={
            "name": entity.name,
            "gloss": entity.gloss,
            "aliases": list(entity.aliases),
            "exemplars_count": len(entity.exemplars),
            "provenance": entity.provenance.value,
        },
    )


def entity_hash(entity: ClusterEntity) -> str:
    """Digest of identity-relevant content only — review snapshot guards use this."""
    return hash_value(
        {
            "cluster_id": entity.cluster_id,
            "name": entity.name,
            "gloss": entity.gloss,
            "aliases": list(entity.aliases),
            "exemplars": list(entity.exemplars),
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cluster_records.py -v` — Expected: PASS. Then `pytest` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/cluster/__init__.py src/xwalk/cluster/records.py tests/test_cluster_records.py
git commit -m "feat(cluster): value types, stable ids, decision keys"
```

---

### Task 2: Policy and level configuration (`policy.py`)

**Files:**
- Create: `src/xwalk/cluster/policy.py`
- Test: `tests/test_cluster_policy.py`

**Interfaces:**
- Consumes: Task 1 enums.
- Produces:
  - `ClusterPolicy(assign_accept_at=0.7, assign_review_floor=0.5, novelty_accept_at=0.7, novelty_review_floor=0.5, merge_accept_at=0.7, merge_review_floor=0.5, comparative_accept_at=0.7, name_verify_accept_at=0.7)` — frozen, validated (`0 <= floor <= accept <= 1` per pair)
  - `CandidateEligibility(taxonomy=True, minted=True, provisional=True)`
  - `ClusterRepresentation(exemplars=5, include_gloss=True)`
  - `MintingPolicy(allowed=True, exhaust_limit=80, reformulate=True)`
  - `TaxonomyConstraints(restrict_to: frozenset[str] | None = None)`
  - `LevelSpec(name, relation, granularity: str, cardinality=Cardinality.ONE, eligibility=..., taxonomy_constraints=..., representation=..., minting=..., policy=..., max_refine_iterations=3, base_k=20)` — frozen; `fingerprint_payload() -> dict` returning every field as JSON-safe primitives

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_cluster_policy.py"""
import pytest

from xwalk.cluster.policy import ClusterPolicy, LevelSpec
from xwalk.cluster.records import Relation
from xwalk.fingerprint import hash_value


def test_threshold_pairs_validated_independently():
    with pytest.raises(ValueError, match="novelty"):
        ClusterPolicy(novelty_accept_at=0.4, novelty_review_floor=0.6)
    with pytest.raises(ValueError, match="merge"):
        ClusterPolicy(merge_accept_at=1.4)
    ClusterPolicy()  # provisional defaults are self-consistent


def test_level_spec_fingerprint_payload_is_hashable_and_complete():
    spec = LevelSpec(name="canonical", relation=Relation.EQUIVALENCE,
                     granularity="a canonical concept merging spelling variants")
    payload = spec.fingerprint_payload()
    hash_value(payload)  # must not raise
    assert payload["granularity"].startswith("a canonical")
    assert payload["policy"]["assign_accept_at"] == 0.7
    changed = LevelSpec(name="canonical", relation=Relation.EQUIVALENCE,
                        granularity="something else")
    assert hash_value(payload) != hash_value(changed.fingerprint_payload())
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cluster_policy.py -v` — Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `policy.py`**

```python
"""Cluster policy: task-specific thresholds and per-level configuration.

Assignment, novelty, merge, comparative switching, and final-name verification are
different tasks; each gets its own thresholds. Every default below is PROVISIONAL —
calibrate on an adjudicated development set before a production run. Nothing here is
inherited from MatchPolicy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xwalk.cluster.records import Cardinality, Relation


def _check_pair(label: str, floor: float, accept: float) -> None:
    if not 0.0 <= floor <= accept <= 1.0:
        raise ValueError(f"need 0 <= {label}_review_floor ({floor}) <= {label}_accept_at ({accept}) <= 1")


@dataclass(frozen=True)
class ClusterPolicy:
    assign_accept_at: float = 0.7
    assign_review_floor: float = 0.5
    novelty_accept_at: float = 0.7
    novelty_review_floor: float = 0.5
    merge_accept_at: float = 0.7
    merge_review_floor: float = 0.5
    comparative_accept_at: float = 0.7
    name_verify_accept_at: float = 0.7

    def __post_init__(self) -> None:
        _check_pair("assign", self.assign_review_floor, self.assign_accept_at)
        _check_pair("novelty", self.novelty_review_floor, self.novelty_accept_at)
        _check_pair("merge", self.merge_review_floor, self.merge_accept_at)
        for label, value in (("comparative_accept_at", self.comparative_accept_at),
                             ("name_verify_accept_at", self.name_verify_accept_at)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {value}")

    def payload(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in (
            "assign_accept_at", "assign_review_floor", "novelty_accept_at",
            "novelty_review_floor", "merge_accept_at", "merge_review_floor",
            "comparative_accept_at", "name_verify_accept_at")}


@dataclass(frozen=True)
class CandidateEligibility:
    taxonomy: bool = True
    minted: bool = True
    provisional: bool = True


@dataclass(frozen=True)
class ClusterRepresentation:
    exemplars: int = 5
    include_gloss: bool = True


@dataclass(frozen=True)
class MintingPolicy:
    allowed: bool = True
    exhaust_limit: int = 80
    reformulate: bool = True


@dataclass(frozen=True)
class TaxonomyConstraints:
    restrict_to: frozenset[str] | None = None


@dataclass(frozen=True)
class LevelSpec:
    name: str
    relation: Relation
    granularity: str
    cardinality: Cardinality = Cardinality.ONE
    eligibility: CandidateEligibility = field(default_factory=CandidateEligibility)
    taxonomy_constraints: TaxonomyConstraints = field(default_factory=TaxonomyConstraints)
    representation: ClusterRepresentation = field(default_factory=ClusterRepresentation)
    minting: MintingPolicy = field(default_factory=MintingPolicy)
    policy: ClusterPolicy = field(default_factory=ClusterPolicy)
    max_refine_iterations: int = 3
    base_k: int = 20

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("LevelSpec.name must be non-empty")
        if not self.granularity.strip():
            raise ValueError("LevelSpec.granularity must be non-empty")
        if self.max_refine_iterations < 1:
            raise ValueError("max_refine_iterations must be at least 1")
        if self.base_k < 1:
            raise ValueError("base_k must be at least 1")

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relation": self.relation.value,
            "granularity": self.granularity,
            "cardinality": self.cardinality.value,
            "eligibility": {"taxonomy": self.eligibility.taxonomy,
                            "minted": self.eligibility.minted,
                            "provisional": self.eligibility.provisional},
            "restrict_to": sorted(self.taxonomy_constraints.restrict_to)
            if self.taxonomy_constraints.restrict_to is not None else None,
            "representation": {"exemplars": self.representation.exemplars,
                               "include_gloss": self.representation.include_gloss},
            "minting": {"allowed": self.minting.allowed,
                        "exhaust_limit": self.minting.exhaust_limit,
                        "reformulate": self.minting.reformulate},
            "policy": self.policy.payload(),
            "max_refine_iterations": self.max_refine_iterations,
            "base_k": self.base_k,
        }
```

- [ ] **Step 4: Run tests, then full suite** — `pytest tests/test_cluster_policy.py -v && pytest`
- [ ] **Step 5: Commit** — `git add src/xwalk/cluster/policy.py tests/test_cluster_policy.py && git commit -m "feat(cluster): task-specific thresholds and LevelSpec"`

---

### Task 3: Cluster ledger (`ledger.py`)

**Files:**
- Create: `src/xwalk/cluster/ledger.py`
- Test: `tests/test_cluster_ledger.py`

**Interfaces:**
- Consumes: Task 1 dataclasses/enums. Follows the style of `src/xwalk/ledger.py` (WAL, `CREATE TABLE IF NOT EXISTS`, row factory) but is a **standalone class** — the existing `Ledger` is untouched; both may open the same file since table names are disjoint (`cluster_*`, `pool_snapshots`).
- Produces (exact signatures the engine and refine phases rely on):
  - `ClusterLedger.open(path) -> ClusterLedger`, `.close()`
  - `put_manifest(run_fp, mapping)`, `get_manifest(run_fp) -> dict | None`
  - `put_cluster(run_fp, entity: ClusterEntity)` (upsert), `get_cluster(run_fp, cluster_id) -> ClusterEntity | None`, `iter_clusters(run_fp, level, states: Sequence[ClusterState] | None = None) -> list[ClusterEntity]`
  - `put_decision(decision: Decision)`, `has_decision(key) -> bool`, `get_decision(key) -> Decision | None`, `iter_decisions(run_fp, level=None, phase=None, kind=None) -> list[Decision]`
  - `put_assignment(run_fp, assignment: Assignment) -> int` (rowid), `supersede(assignment_id: int, by: int)`, `current_assignments(run_fp, level) -> dict[str, tuple[int, Assignment]]` (member_id → (rowid, latest non-superseded))
  - `put_merge(run_fp, level, loser_id, winner_id, phase: Phase, iteration: int, confidence: float | None, outcome: str)`, `iter_merges(run_fp, level) -> list[dict]`
  - `put_snapshot(run_fp, level, snapshot_id, phase: Phase, iteration: int)`
  - `put_level_state(run_fp, level, stop_reason: StopReason, best_iteration: int, state_hashes: Sequence[str])`, `get_level_state(run_fp, level) -> dict | None`
  - `put_review_row(mapping)`, `iter_review_rows(run_fp) -> list[dict]`

All writes are plain synchronous methods: the M1 engine is a single sequential writer (`batch_size=1`), so the async write-lock machinery of the match ledger is not needed; say so in the module docstring.

- [ ] **Step 1: Write the failing tests** (`tests/test_cluster_ledger.py`)

```python
from xwalk.cluster.ledger import ClusterLedger
from xwalk.cluster.records import (
    Assignment, AssignStatus, ClusterEntity, ClusterState, Decision,
    MintProvenance, Phase, Provenance, StopReason,
)


def _entity(cid="k1", state=ClusterState.PROVISIONAL, name="dark chocolate"):
    return ClusterEntity(cluster_id=cid, level="canonical", name=name,
                         gloss="g", aliases=("Dark Chocolate",), provenance=Provenance.MINTED,
                         state=state, seed_member_id="m1", exemplars=("m1",),
                         created_snapshot=1, mint_provenance=MintProvenance.ORDINARY)


def test_cluster_roundtrip_and_upsert(tmp_path):
    ledger = ClusterLedger.open(tmp_path / "ledger.sqlite")
    ledger.put_cluster("fp", _entity())
    ledger.put_cluster("fp", _entity(name="Dark Chocolate (renamed)"))
    stored = ledger.get_cluster("fp", "k1")
    assert stored is not None and stored.name == "Dark Chocolate (renamed)"
    assert [c.cluster_id for c in ledger.iter_clusters("fp", "canonical")] == ["k1"]
    assert ledger.iter_clusters("fp", "canonical", states=[ClusterState.FINAL]) == []


def test_decision_resume_key(tmp_path):
    ledger = ClusterLedger.open(tmp_path / "ledger.sqlite")
    d = Decision(decision_key="dk", run_fingerprint="fp", level="canonical",
                 phase=Phase.STREAM, iteration=0, order_index=3, batch_id=3,
                 snapshot_id=2, subject_id="m1", subject_hash="h", kind="select",
                 model="fake", prompt="p", response="r", resolved={"chosen_key": "C01"},
                 confidence=0.9, outcome="selected")
    assert not ledger.has_decision("dk")
    ledger.put_decision(d)
    assert ledger.has_decision("dk")
    back = ledger.get_decision("dk")
    assert back is not None and back.resolved["chosen_key"] == "C01"
    assert back.phase is Phase.STREAM


def test_assignments_supersede_chain(tmp_path):
    ledger = ClusterLedger.open(tmp_path / "ledger.sqlite")
    first = ledger.put_assignment("fp", Assignment(
        level="canonical", member_id="m1", member_hash="h", cluster_id="k1",
        status=AssignStatus.ASSIGNED, confidence=0.8, phase=Phase.STREAM,
        iteration=0, via_mint=False))
    second = ledger.put_assignment("fp", Assignment(
        level="canonical", member_id="m1", member_hash="h", cluster_id="k2",
        status=AssignStatus.ASSIGNED, confidence=0.9, phase=Phase.REASSIGN,
        iteration=1, via_mint=False))
    ledger.supersede(first, by=second)
    current = ledger.current_assignments("fp", "canonical")
    rowid, latest = current["m1"]
    assert rowid == second and latest.cluster_id == "k2"


def test_level_state_records_stop_reason(tmp_path):
    ledger = ClusterLedger.open(tmp_path / "ledger.sqlite")
    ledger.put_level_state("fp", "canonical", StopReason.OSCILLATION, 1, ["a", "b", "a"])
    state = ledger.get_level_state("fp", "canonical")
    assert state == {"stop_reason": "oscillation", "best_iteration": 1,
                     "state_hashes": ["a", "b", "a"]}
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_cluster_ledger.py -v`
- [ ] **Step 3: Implement `ClusterLedger`.** Schema (executescript, all `IF NOT EXISTS`):

```sql
CREATE TABLE IF NOT EXISTS cluster_manifests (run_fingerprint TEXT PRIMARY KEY, blob TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clusters (
    run_fingerprint TEXT NOT NULL, cluster_id TEXT NOT NULL, level TEXT NOT NULL,
    name TEXT NOT NULL, gloss TEXT NOT NULL, aliases TEXT NOT NULL,
    provenance TEXT NOT NULL, state TEXT NOT NULL, seed_member_id TEXT,
    exemplars TEXT NOT NULL, created_snapshot INTEGER NOT NULL,
    mint_provenance TEXT, merged_into TEXT,
    PRIMARY KEY (run_fingerprint, cluster_id));
CREATE INDEX IF NOT EXISTS clusters_by_level ON clusters(run_fingerprint, level, state);
CREATE TABLE IF NOT EXISTS cluster_decisions (
    decision_key TEXT PRIMARY KEY, run_fingerprint TEXT NOT NULL, level TEXT NOT NULL,
    phase TEXT NOT NULL, iteration INTEGER NOT NULL, order_index INTEGER NOT NULL,
    batch_id INTEGER NOT NULL, snapshot_id INTEGER NOT NULL,
    subject_id TEXT NOT NULL, subject_hash TEXT NOT NULL, kind TEXT NOT NULL,
    model TEXT NOT NULL, prompt TEXT NOT NULL, response TEXT NOT NULL,
    resolved TEXT NOT NULL, confidence REAL, outcome TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS decisions_by_run ON cluster_decisions(run_fingerprint, level, phase, kind);
CREATE TABLE IF NOT EXISTS cluster_assignments (
    assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_fingerprint TEXT NOT NULL, level TEXT NOT NULL, member_id TEXT NOT NULL,
    member_hash TEXT NOT NULL, cluster_id TEXT, status TEXT NOT NULL,
    confidence REAL, phase TEXT NOT NULL, iteration INTEGER NOT NULL,
    via_mint INTEGER NOT NULL DEFAULT 0, superseded_by INTEGER);
CREATE INDEX IF NOT EXISTS assignments_current
    ON cluster_assignments(run_fingerprint, level, member_id, superseded_by);
CREATE TABLE IF NOT EXISTS cluster_merges (
    merge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_fingerprint TEXT NOT NULL, level TEXT NOT NULL, loser_id TEXT NOT NULL,
    winner_id TEXT NOT NULL, phase TEXT NOT NULL, iteration INTEGER NOT NULL,
    confidence REAL, outcome TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pool_snapshots (
    run_fingerprint TEXT NOT NULL, level TEXT NOT NULL, snapshot_id INTEGER NOT NULL,
    phase TEXT NOT NULL, iteration INTEGER NOT NULL,
    PRIMARY KEY (run_fingerprint, level, snapshot_id));
CREATE TABLE IF NOT EXISTS cluster_levels (
    run_fingerprint TEXT NOT NULL, level TEXT NOT NULL, stop_reason TEXT NOT NULL,
    best_iteration INTEGER NOT NULL, state_hashes TEXT NOT NULL,
    PRIMARY KEY (run_fingerprint, level));
CREATE TABLE IF NOT EXISTS cluster_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_fingerprint TEXT NOT NULL, level TEXT NOT NULL, subject_id TEXT NOT NULL,
    subject_hash TEXT NOT NULL, verb TEXT NOT NULL, target_cluster_id TEXT,
    new_name TEXT, reviewer TEXT NOT NULL, review_note TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL);
```

Python side mirrors `xwalk/ledger.py`: `open()` does `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `row_factory = sqlite3.Row`, `executescript(_SCHEMA)`. Serialize `aliases`/`exemplars`/`resolved`/`state_hashes` as JSON text (`json.dumps(..., ensure_ascii=False)`); enums by `.value`; rebuild dataclasses on read. `current_assignments` query: `SELECT * FROM cluster_assignments WHERE run_fingerprint=? AND level=? AND superseded_by IS NULL` — build the dict keyed by `member_id`; if two non-superseded rows exist for one member, raise `RuntimeError` (invariant violation: engine bug, `failed` not review).

- [ ] **Step 4: Run tests, then full suite** — both green.
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): standalone cluster ledger with decisions, assignments, merges, level state"`

---

### Task 4: Prompts, schemas, contract (`prompts.py`, `skeletons/`, `SelectPrompts` protocol)

**Files:**
- Create: `src/xwalk/cluster/prompts.py`, `src/xwalk/cluster/skeletons/{select,verify_assign,verify_novelty,merge,comparative,name,name_verify,reformulate}.j2`
- Modify: `src/xwalk/stages/select.py` (add `SelectPrompts` Protocol; annotate `Selector.__init__`'s `prompts` with it — runtime behavior unchanged, existing `PromptSet` satisfies it structurally)
- Test: `tests/test_cluster_prompts.py`

**Interfaces:**
- Consumes: `jinja2` (`StrictUndefined`, same env settings as `prompts/contract.py`), `pydantic`, `xwalk.fingerprint.hash_value`, `xwalk.llm.parsing.parse_json_object`.
- Produces:
  - Schemas (module constants): `VERIFY_ASSIGN_SCHEMA` (`decision: enum[support,disagree]`, `confidence_score`, `explanation`; required all three), `VERIFY_NOVELTY_SCHEMA` (`candidates_all_invalid: boolean`, `warrants_new_cluster: boolean`, `confidence_score`, `explanation`; required all), `MERGE_SCHEMA` (`decision: enum[merge,keep_separate]`, `confidence_score`, `explanation`), `COMPARATIVE_SCHEMA` (`preferred: enum[incumbent,challenger]`, `confidence_score`, `explanation`), `NAME_SCHEMA` (`name`, `gloss`, `explanation`), `NAME_VERIFY_SCHEMA` (`decision: enum[fits,does_not_fit]`, `confidence_score`, `explanation`), `REFORMULATE_SCHEMA` (`query`, `explanation`). All `additionalProperties: False`.
  - `ClusterPromptSlots(BaseModel)`: `member_noun: str`, `parent_noun: str`, `domain_brief: str`, `equivalence_brief: str`, `subsumption_brief: str`, `hard_rules: list[str] = []` — non-empty validation on the four str fields; `load_cluster_slots(path) -> ClusterPromptSlots`
  - `ClusterPromptSet` frozen dataclass (`slots`, `skeletons: Mapping[str, str]`), `from_slots(slots, base_dir=None)`, `fingerprint` property, and renders — **exact keyword signatures**:
    - `render_select(*, source_fields, context, candidate_block) -> str` (must match what `Selector` calls)
    - `render_verify_assign(*, member_fields, context, chosen_block, other_candidates) -> str`
    - `render_verify_novelty(*, member_fields, context, candidate_block) -> str`
    - `render_merge(*, left_block, right_block, context) -> str`
    - `render_comparative(*, member_fields, context, incumbent_block, challenger_block) -> str`
    - `render_name(*, member_labels: Sequence[str], context) -> str`
    - `render_name_verify(*, name, gloss, member_labels, context) -> str`
    - `render_reformulate(*, member_fields, context, previous_query) -> str`
  - `validate_cluster_contract(prompts: ClusterPromptSet) -> None` mirroring `prompts/contract.py::validate_contract`: renders each skeleton with fixture inputs; asserts required sections appear exactly once; asserts each rendered input block survives exactly once; asserts the select skeleton contains the literal `"Never answer with an identifier"`; asserts the `## Output` example parses and covers the schema's `required` keys.
  - `default_cluster_slots() -> ClusterPromptSlots` — generic, domain-neutral wording, used by tests and as a starting point.
  - In `stages/select.py`: `class SelectPrompts(Protocol): def render_select(self, *, source_fields: Mapping[str, Any], context: str, candidate_block: str) -> str: ...`

**Context convention (all tasks):** the `context` string passed to every render is built by the engine as `f"{spec.granularity}\n\n{relation_brief}"` where `relation_brief` is `slots.equivalence_brief` or `slots.subsumption_brief` per `spec.relation`. Skeletons render it under a `## Level definition` heading.

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_cluster_prompts.py"""
import pytest

from xwalk.cluster.prompts import (
    ClusterPromptSet, default_cluster_slots, validate_cluster_contract,
)


@pytest.fixture()
def prompts():
    return ClusterPromptSet.from_slots(default_cluster_slots())


def test_default_skeletons_pass_contract(prompts):
    validate_cluster_contract(prompts)  # must not raise


def test_select_skeleton_keeps_key_instruction(prompts):
    text = prompts.render_select(
        source_fields={"label": "dark chocolate"},
        context="a canonical concept\n\nsame concept, different spelling",
        candidate_block="[C01] name: chocolate",
    )
    assert "Never answer with an identifier" in text
    assert "[C01] name: chocolate" in text


def test_contract_rejects_lost_section(prompts):
    broken = ClusterPromptSet(
        slots=prompts.slots,
        skeletons={**prompts.skeletons,
                   "verify_novelty": prompts.skeletons["verify_novelty"].replace(
                       "## Candidates", "## Stuff")},
    )
    with pytest.raises(Exception, match="verify_novelty"):
        validate_cluster_contract(broken)


def test_fingerprint_changes_with_slots(prompts):
    other = default_cluster_slots().model_copy(update={"domain_brief": "food products"})
    assert prompts.fingerprint != ClusterPromptSet.from_slots(other).fingerprint
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_cluster_prompts.py -v`
- [ ] **Step 3: Implement.** `prompts.py` follows `prompts/contract.py` structurally (schemas → slots model → PromptSet → `_REQUIRED_SECTIONS`/`_REQUIRED_INPUTS` fixtures → validator). Required sections per skeleton:
  - `select`: `## Member record`, `## Level definition`, `## Candidates`, `## Output`
  - `verify_assign`: `## Member record`, `## Level definition`, `## Proposed parent`, `## Output`
  - `verify_novelty`: `## Member record`, `## Level definition`, `## Candidates`, `## Output` — body must instruct the model to answer BOTH questions: are all listed candidates invalid parents, and does the member warrant a new cluster at this granularity.
  - `merge`: `## Left cluster`, `## Right cluster`, `## Level definition`, `## Output`
  - `comparative`: `## Member record`, `## Incumbent`, `## Challenger`, `## Level definition`, `## Output` — body must say: prefer the incumbent unless the challenger is clearly better (anti-churn instruction).
  - `name`: `## Members`, `## Level definition`, `## Output` (name + one-to-two-sentence gloss)
  - `name_verify`: `## Proposed name`, `## Members`, `## Output`
  - `reformulate`: `## Member record`, `## Previous query`, `## Output` (one alternative search query)

  Each skeleton ends with an `## Output` section containing a literal JSON example matching its schema (the validator parses it). Write the eight `.j2` files with generic wording built from `{{ slots.* }}` variables; keep them short (10–25 lines each). The `select` skeleton is adapted from `src/xwalk/prompts/base/select.j2` — read it first and preserve its output contract (`chosen_key`/`confidence_score`/`explanation`, null-for-abstain, the key instruction sentence verbatim).

  In `stages/select.py`, add the Protocol (imports: `typing.Protocol`, `collections.abc.Mapping`, `typing.Any`) and change the annotation `prompts: PromptSet` → `prompts: SelectPrompts` in `Selector.__init__`. No behavioral change; run `pytest tests/test_select.py` to confirm.
- [ ] **Step 4: Run tests + full suite** — green, including all existing prompt-contract tests.
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): prompt skeletons, schemas, contract validator; SelectPrompts protocol"`

---

### Task 5: The growable pool (`pool.py`)

**Files:**
- Create: `src/xwalk/cluster/pool.py`
- Test: `tests/test_cluster_pool.py`

**Interfaces:**
- Consumes: `tantivy` (in-RAM index: `tantivy.Index(schema)` with **no path**), `xwalk.retrieval.bm25.sanitise_query` and `_build_schema`-equivalent (write a local `_pool_schema()`; do not import the private one), `xwalk.retrieval.fusion.reciprocal_rank_fusion` (read `src/xwalk/retrieval/fusion.py` first for its exact signature and store requirements), `xwalk.retrieval.dense.Encoder` protocol (optional dense head, brute-force cosine — the pool is ≤ a few thousand entries), Task 1/2 types.
- Produces:

```python
class Pool:
    def __init__(self, *, level: str, representation: ClusterRepresentation,
                 eligibility: CandidateEligibility,
                 taxonomy: Sequence[Record] = (),
                 restrict_to: frozenset[str] | None = None,
                 encoder: Encoder | None = None,
                 exemplar_labels: Mapping[str, str] | None = None) -> None
    # exemplar_labels: member_id -> display label, for rendering exemplars in docs
    @property
    def snapshot_id(self) -> int
    @property
    def fingerprint(self) -> str   # config + taxonomy digest ONLY (content grows by design)
    def add(self, entity: ClusterEntity) -> None          # buffered until commit()
    def replace(self, entity: ClusterEntity) -> None      # refreshed representation
    def apply_merge(self, loser_id: str, winner: ClusterEntity) -> None
        # marks loser RETIRED with merged_into=winner.cluster_id, replaces winner rep
    def commit(self) -> int                               # rebuild indexes, bump snapshot
    def get(self, record_id: str) -> Record               # TargetStore protocol (for fusion)
    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]
    def entity(self, cluster_id: str) -> ClusterEntity    # KeyError if absent
    def entities(self, states: Sequence[ClusterState] | None = None) -> list[ClusterEntity]
    def doc_text(self, entity: ClusterEntity) -> str
    async def search(self, query: str, k: int,
                     exclude: frozenset[str] = frozenset()) -> list[Candidate]
```

Behavioral contract:
- Taxonomy records are wrapped at construction into `ClusterEntity(provenance=TAXONOMY, state=FINAL, name=<label field or str(fields.get("label", id))>, gloss=str(fields.get("gloss",""))…, created_snapshot=0)`; `restrict_to` filters which taxonomy ids enter at all. Taxonomy entities are immutable: `replace`/`apply_merge` with a taxonomy loser or a taxonomy `replace` target raises `ValueError`.
- `doc_text` = name + gloss (if `include_gloss`) + aliases + up to `representation.exemplars` exemplar labels, space-joined — this is what BM25 indexes and the encoder embeds.
- `commit()` rebuilds the in-RAM Tantivy index from scratch (delete-by-update is not worth the complexity at this size), re-encodes only entities whose `doc_text` changed since the last commit (cache by hash), and returns the incremented snapshot id. Retired entities are not indexed.
- `search` runs BM25 and (if an encoder is present) dense cosine, fuses with `reciprocal_rank_fusion(groups, self, k=60)` (the pool itself is the store), then filters: eligibility (`taxonomy`/`minted`/`provisional` flags against each entity), `exclude` ids, retired entities. Deterministic given the index state.
- Searching with uncommitted buffered adds raises `RuntimeError` — the engine must control snapshot boundaries explicitly.

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_cluster_pool.py"""
import pytest

from xwalk.cluster.policy import CandidateEligibility, ClusterRepresentation
from xwalk.cluster.pool import Pool
from xwalk.cluster.records import ClusterEntity, ClusterState, MintProvenance, Provenance
from xwalk.records import Record


def _mint(cid, name, aliases=()):
    return ClusterEntity(cluster_id=cid, level="canonical", name=name, gloss=f"{name} gloss",
                         aliases=tuple(aliases), provenance=Provenance.MINTED,
                         state=ClusterState.PROVISIONAL, seed_member_id="m-" + cid,
                         exemplars=(), created_snapshot=1,
                         mint_provenance=MintProvenance.ORDINARY)


def _pool(**kw):
    defaults = dict(level="canonical", representation=ClusterRepresentation(),
                    eligibility=CandidateEligibility(),
                    taxonomy=[Record(id="T1", fields={"label": "chocolate"})])
    defaults.update(kw)
    return Pool(**defaults)


@pytest.mark.asyncio
async def test_taxonomy_searchable_and_minted_appear_after_commit():
    pool = _pool()
    pool.commit()
    hits = await pool.search("chocolate", k=5)
    assert [c.id for c in hits] == ["T1"]
    pool.add(_mint("K1", "dark chocolate"))
    pool.commit()
    hits = await pool.search("dark chocolate", k=5)
    assert {c.id for c in hits} == {"T1", "K1"}


@pytest.mark.asyncio
async def test_search_before_commit_of_buffered_adds_raises():
    pool = _pool()
    pool.commit()
    pool.add(_mint("K1", "dark chocolate"))
    with pytest.raises(RuntimeError):
        await pool.search("chocolate", k=5)


def test_taxonomy_is_immutable():
    pool = _pool()
    pool.commit()
    with pytest.raises(ValueError):
        pool.apply_merge("T1", pool.entity("T1"))


@pytest.mark.asyncio
async def test_merge_retires_loser_and_excludes_it_from_search():
    pool = _pool()
    pool.add(_mint("K1", "dark chocolate"))
    pool.add(_mint("K2", "chocolate dark"))
    pool.commit()
    winner = pool.entity("K1")
    pool.apply_merge("K2", winner)
    pool.commit()
    hits = await pool.search("chocolate dark", k=10)
    assert "K2" not in {c.id for c in hits}
    assert pool.entity("K2").state is ClusterState.RETIRED
    assert pool.entity("K2").merged_into == "K1"
```

(Repo test setup: check `tests/conftest.py` for the asyncio plugin/pattern used by `test_matcher.py` — mirror it; if the repo uses `asyncio.run` helpers instead of `pytest.mark.asyncio`, follow the repo.)

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_cluster_pool.py -v`
- [ ] **Step 3: Implement `pool.py`.** In-RAM Tantivy: local `_pool_schema()` builds `record_id` (stored) + `text` fields, `tantivy.Index(schema)` with no path; `commit()` recreates the index object, adds one document per live eligible-storable entity (all non-retired entities are indexed; eligibility filtering happens at search time so it can differ per level without reindex), `writer.commit()`, `index.reload()`. Track a `_dirty` flag set by `add`/`replace`/`apply_merge`, cleared by `commit`; `search` raises `RuntimeError("pool has uncommitted changes")` when dirty. Dense: `self._vectors: dict[str, list[float]]`, re-encode changed doc_texts at commit, cosine = dot product (encoder returns unit vectors), rank top-k. Build `RetrievalHit` groups (`retriever="pool-bm25"` / `"pool-dense"`) and call `reciprocal_rank_fusion`. Wrap Tantivy search in `asyncio.to_thread` like `BM25Retriever.search` does.
- [ ] **Step 4: Run tests + full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): growable in-RAM pool with snapshot discipline and RRF search"`

---

### Task 6: Verifier stages (`stages.py`)

**Files:**
- Create: `src/xwalk/cluster/stages.py`
- Test: `tests/test_cluster_stages.py`

**Interfaces:**
- Consumes: `xwalk.llm.base.{LLMClient, LLMRequest, ParseError}`, `xwalk.llm.parsing.parse_json_object`, `xwalk.stages.keying.KeyedCandidates`, Task 4 schemas + `ClusterPromptSet`. Reuse `_SYSTEM = "You return JSON only. No prose, no code fences."` and the `_clamp`/`_blocks` patterns from `stages/gate.py` (copy the small helpers locally; do not import private names).
- Produces (all constructors: `(llm: LLMClient, prompts: ClusterPromptSet, *, max_tokens: int = 512)`; every result dataclass carries `raw: str`, `usage: Usage`, `error: str | None`, `finish_reason: str | None`):
  - `AssignmentVerifier.verify(member: Record, context: str, keyed: KeyedCandidates, chosen_key: str) -> AssignVerdict(decision: Literal["support","disagree"], confidence: float | None, explanation: str, ...)` — parse failure or unknown decision ⇒ `decision="disagree"`, `confidence=None` (fails toward review).
  - `NoveltyVerifier.verify(member: Record, context: str, keyed: KeyedCandidates) -> NoveltyVerdict(candidates_all_invalid: bool, warrants_new_cluster: bool, confidence: float | None, ...)` — parse failure ⇒ both flags `False`, `confidence=None`.
  - `MergeVerifier.verify(left: Record, right: Record, context: str) -> MergeVerdict(decision: Literal["merge","keep_separate"], confidence, ...)` — parse failure ⇒ `"keep_separate"`, `confidence=None`.
  - `ComparativeVerifier.verify(member: Record, context: str, incumbent: Record, challenger: Record) -> ComparativeVerdict(preferred: Literal["incumbent","challenger"], confidence, ...)` — parse failure ⇒ `"incumbent"` (no churn on malformed output).
  - `Namer.name(member_labels: Sequence[str], context: str) -> NameResult(name: str | None, gloss: str | None, ...)` — parse failure or empty name ⇒ `name=None`.
  - `NameVerifier.verify(name: str, gloss: str, member_labels: Sequence[str], context: str) -> NameVerdict(decision: Literal["fits","does_not_fit"], confidence, ...)` — parse failure ⇒ `"does_not_fit"`.
  - `Reformulator.reformulate(member: Record, context: str, previous_query: str) -> str | None` — parse failure or query equal (case-folded) to previous ⇒ `None`.
  - Candidate/record blocks: module function `render_block(record: Record) -> str` producing `"name: … | gloss: … | aliases: …"` from the record's fields — used for `chosen_block`, `left_block`, incumbent/challenger blocks so all verifiers describe entities identically.

- [ ] **Step 1: Write the failing tests** — use `FakeLLM` with fixed scripts; four load-bearing cases:

```python
"""tests/test_cluster_stages.py"""
import asyncio
import json

from xwalk.cluster.prompts import ClusterPromptSet, default_cluster_slots
from xwalk.cluster.stages import (
    AssignmentVerifier, ComparativeVerifier, NoveltyVerifier, render_block,
)
from xwalk.llm.fake import FakeLLM
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.keying import assign_keys
from xwalk.templates import TemplateSet


TEMPLATES = TemplateSet(query="{{ label }}", context="{{ label }}",
                        doc="{{ name }}", candidate="name: {{ name }}")
PROMPTS = ClusterPromptSet.from_slots(default_cluster_slots())


def _keyed():
    record = Record(id="K1", fields={"name": "dark chocolate", "gloss": "g",
                                     "aliases": [], "exemplars_count": 0,
                                     "provenance": "minted"})
    cand = Candidate(record=record, fused_score=1.0,
                     evidence=(RetrievalHit(record_id="K1", retriever="pool-bm25",
                                            raw_score=1.0, rank=1),))
    return assign_keys([cand], TEMPLATES)


def test_assignment_verifier_parse_failure_fails_toward_review():
    verifier = AssignmentVerifier(FakeLLM(script=["not json"]), PROMPTS)
    verdict = asyncio.run(verifier.verify(Record(id="m", fields={"label": "x"}),
                                          "ctx", _keyed(), "C01"))
    assert verdict.decision == "disagree" and verdict.confidence is None


def test_assignment_verifier_support():
    payload = json.dumps({"decision": "support", "confidence_score": 0.92,
                          "explanation": "same concept"})
    verifier = AssignmentVerifier(FakeLLM(script=[payload]), PROMPTS)
    verdict = asyncio.run(verifier.verify(Record(id="m", fields={"label": "x"}),
                                          "ctx", _keyed(), "C01"))
    assert verdict.decision == "support" and verdict.confidence == 0.92


def test_novelty_verifier_two_part_output():
    payload = json.dumps({"candidates_all_invalid": True, "warrants_new_cluster": True,
                          "confidence_score": 0.85, "explanation": "novel"})
    verifier = NoveltyVerifier(FakeLLM(script=[payload]), PROMPTS)
    verdict = asyncio.run(verifier.verify(Record(id="m", fields={"label": "x"}),
                                          "ctx", _keyed()))
    assert verdict.candidates_all_invalid and verdict.warrants_new_cluster


def test_comparative_parse_failure_keeps_incumbent():
    verifier = ComparativeVerifier(FakeLLM(script=["garbage"]), PROMPTS)
    verdict = asyncio.run(verifier.verify(
        Record(id="m", fields={"label": "x"}), "ctx",
        incumbent=Record(id="K1", fields={"name": "a"}),
        challenger=Record(id="K2", fields={"name": "b"})))
    assert verdict.preferred == "incumbent"
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement `stages.py`** following `gate.py`'s shape: build prompt via the Task 4 render, `LLMRequest(system=_SYSTEM, user=prompt, schema=<TASK_SCHEMA>, schema_name=..., max_tokens=...)`, `parse_json_object`, clamp confidence, and the fail-toward-safety defaults listed in Interfaces. Each verifier's failure default is a comment-worthy constraint: state it in the class docstring.
- [ ] **Step 4: Run tests + full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): verifier stages with fail-toward-review defaults"`

---### Task 7: Streaming engine (`engine.py` — stream pass only)

**Files:**
- Create: `src/xwalk/cluster/engine.py`
- Test: `tests/test_cluster_engine.py`

**Interfaces:**
- Consumes: everything above, plus `xwalk.stages.select.Selector`, `xwalk.stages.keying.{assign_keys, resolve_key}` (via Selector), `xwalk.fingerprint.{hash_record, hash_value}`, `xwalk.templates.TemplateSet`.
- Produces:

```python
@dataclass(frozen=True)
class StreamReport:
    level: str
    assigned: int
    minted: int
    deferred: int
    needs_review: int
    failed: int

class LevelEngine:
    def __init__(self, *, spec: LevelSpec, pool: Pool, templates: TemplateSet,
                 selector: Selector, assignment_verifier: AssignmentVerifier,
                 novelty_verifier: NoveltyVerifier, merge_verifier: MergeVerifier,
                 comparative_verifier: ComparativeVerifier, namer: Namer,
                 name_verifier: NameVerifier, reformulator: Reformulator | None,
                 ledger: ClusterLedger, run_fingerprint: str) -> None
    @property
    def context(self) -> str            # granularity + relation brief (Task 4 convention)
    def ordered(self, members: Sequence[Record]) -> list[Record]
        # stable sort by (casefolded whitespace-normalized query render, member id)
    async def run_stream(self, members: Sequence[Record]) -> StreamReport
    async def decide(self, member: Record, *, phase: Phase, iteration: int,
                     order_index: int, allow_mint: bool) -> Assignment
        # the shared decision path: stream and retry both call this
```

`decide` control flow (spec section "The streaming pass", implemented exactly):
1. Resume check: if a `cluster_decisions` row exists for `(fp, level, phase, iteration, member, hash, "outcome")`, reconstruct the recorded `Assignment` from `current_assignments` and return without LLM calls.
2. Ordinary retrieval: `pool.search(query, k=spec.base_k)`.
3. `selector.select(member, self.context, candidates)` → record a `select` Decision (prompt from `FakeLLM`-visible request is not retrievable through Selector, so record `keyed.rendered` + raw response + resolved choice; the full rendered prompt is recorded for the stages we own — for the reused Selector record `resolved={"chosen_key":…, "resolution":…}` and `prompt=keyed.rendered`; note this limitation in a comment).
4. Candidate chosen → `assignment_verifier.verify` → record `verify_assign` Decision → route on `spec.policy`: `support` and `conf >= assign_accept_at` ⇒ ASSIGNED; `support` and `conf >= assign_review_floor` ⇒ NEEDS_REVIEW; anything else (including `disagree` and `conf=None`) ⇒ DEFERRED (stream) / NEEDS_REVIEW (retry — `allow_mint` still True but deferral is no longer available; pass `phase` to decide the fallback).
5. Null → retrieval exhaustion (only when `spec.minting.allowed` and `allow_mint`): double `k` until `spec.minting.exhaust_limit`, re-search; if `reformulator` is set, one reformulation query, union by id, stable order (fused score desc, id). If the expanded candidate set is strictly larger, re-run `selector.select` (record with `seq=1`); a choice there goes to step 4.
6. Still null → `novelty_verifier.verify` against the **exhausted** keyed candidates → record `verify_novelty` Decision. Routing: both flags true and `conf >= novelty_accept_at` ⇒ MINT; `conf` in `[novelty_review_floor, accept)` ⇒ DEFERRED; below floor or `candidates_all_invalid is False` ⇒ DEFERRED (with `outcome="re_select_hint"` recorded on the decision).
7. MINT: `namer.name([query_label], self.context)` → name+gloss (namer failure ⇒ NEEDS_REVIEW instead of mint — a cluster that cannot be named cannot enter the pool); `make_cluster_id`; `mint_provenance = ORDINARY` if exhaustion surfaced no new candidates else `EXHAUSTED`; `ClusterEntity(state=PROVISIONAL, seed_member_id=member.id, aliases=(label,), exemplars=(member.id,), created_snapshot=pool.snapshot_id + 1)`; `pool.add`, `pool.commit()`, `ledger.put_snapshot`, `ledger.put_cluster`; member ASSIGNED to the new cluster with `via_mint=True`.
8. Every terminal outcome writes an `Assignment` row plus a final `outcome`-kind Decision (kind=`"outcome"`, resolved=`{"status":…, "cluster_id":…}`) so resume can reconstruct without re-deriving.
9. LLM provider errors (`LLMError`): status FAILED, recorded; `LLMFatalError` propagates (stop the run, like `Matcher`).

`run_stream`: iterate `ordered(members)` with `order_index = batch_id = enumerate index` (batch_size=1 ⇒ batch_id == order_index), commit taxonomy pool once up front (`pool.commit()`, snapshot recorded), call `decide(..., phase=Phase.STREAM, iteration=0, allow_mint=True)` per member sequentially.

- [ ] **Step 1: Write the failing tests.** Use a `FakeLLM(handler=...)` that dispatches on marker strings in the request (`"## Candidates"` + which skeleton heading is present) — this is the pattern for all engine tests:

```python
"""tests/test_cluster_engine.py"""
import asyncio
import json

import pytest

from xwalk.cluster.engine import LevelEngine
from xwalk.cluster.ledger import ClusterLedger
from xwalk.cluster.policy import LevelSpec
from xwalk.cluster.pool import Pool
from xwalk.cluster.prompts import ClusterPromptSet, default_cluster_slots
from xwalk.cluster.records import AssignStatus, ClusterState, Phase, Relation
from xwalk.cluster.stages import (
    AssignmentVerifier, ComparativeVerifier, MergeVerifier, Namer,
    NameVerifier, NoveltyVerifier, Reformulator,
)
from xwalk.llm.fake import FakeLLM
from xwalk.records import Record
from xwalk.stages.select import Selector
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(query="{{ label }}", context="{{ label }}",
                        doc="{{ name }}", candidate="name: {{ name }}")
PROMPTS = ClusterPromptSet.from_slots(default_cluster_slots())


def scripted_handler(request):
    """Route by skeleton headings; assert-fail on anything unexpected."""
    user = request.user
    if "## Proposed parent" in user:          # verify_assign
        return json.dumps({"decision": "support", "confidence_score": 0.9,
                           "explanation": "same"})
    if "## Candidates" in user and "warrant" in user.lower():   # verify_novelty
        return json.dumps({"candidates_all_invalid": True,
                           "warrants_new_cluster": True,
                           "confidence_score": 0.9, "explanation": "new"})
    if "## Members" in user and "## Proposed name" not in user:  # name
        return json.dumps({"name": "dark chocolate", "gloss": "high-cocoa chocolate",
                           "explanation": ""})
    if "## Candidates" in user:               # select
        # abstain when nothing matches the member label textually
        if "dark chocolate" in user.lower() and "[c" in user.lower():
            return json.dumps({"chosen_key": "C01", "confidence_score": 0.9,
                               "explanation": "match"})
        return json.dumps({"chosen_key": None, "confidence_score": 0.0,
                           "explanation": "nothing fits"})
    raise AssertionError(f"unexpected prompt:\n{user[:200]}")


def _engine(tmp_path, llm, members_level="canonical", taxonomy=()):
    spec = LevelSpec(name=members_level, relation=Relation.EQUIVALENCE,
                     granularity="a canonical concept; merge spelling variants only")
    pool = Pool(level=members_level, representation=spec.representation,
                eligibility=spec.eligibility, taxonomy=list(taxonomy))
    ledger = ClusterLedger.open(tmp_path / "ledger.sqlite")
    return LevelEngine(
        spec=spec, pool=pool, templates=TEMPLATES,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        assignment_verifier=AssignmentVerifier(llm, PROMPTS),
        novelty_verifier=NoveltyVerifier(llm, PROMPTS),
        merge_verifier=MergeVerifier(llm, PROMPTS),
        comparative_verifier=ComparativeVerifier(llm, PROMPTS),
        namer=Namer(llm, PROMPTS), name_verifier=NameVerifier(llm, PROMPTS),
        reformulator=None, ledger=ledger, run_fingerprint="fp-test",
    )


def test_first_member_mints_second_joins(tmp_path):
    engine = _engine(tmp_path, FakeLLM(handler=scripted_handler))
    members = [Record(id="m1", fields={"label": "Dark Chocolate"}),
               Record(id="m2", fields={"label": "chocolate Dark"})]
    report = asyncio.run(engine.run_stream(members))
    assert report.minted == 1 and report.assigned == 2
    clusters = engine.ledger.iter_clusters("fp-test", "canonical")
    minted = [c for c in clusters if c.state is ClusterState.PROVISIONAL]
    assert len(minted) == 1 and minted[0].name == "dark chocolate"
    current = engine.ledger.current_assignments("fp-test", "canonical")
    assert current["m1"][1].cluster_id == current["m2"][1].cluster_id
    assert current["m1"][1].via_mint and not current["m2"][1].via_mint


def test_minting_goes_through_exhaustion_and_is_recorded(tmp_path):
    engine = _engine(tmp_path, FakeLLM(handler=scripted_handler))
    asyncio.run(engine.run_stream([Record(id="m1", fields={"label": "Dark Chocolate"})]))
    kinds = [d.kind for d in engine.ledger.iter_decisions("fp-test")]
    assert "verify_novelty" in kinds        # mint was verified, not assumed
    minted = [c for c in engine.ledger.iter_clusters("fp-test", "canonical")
              if c.seed_member_id == "m1"]
    assert minted[0].mint_provenance is not None


def test_resume_skips_completed_members(tmp_path):
    llm = FakeLLM(handler=scripted_handler)
    engine = _engine(tmp_path, llm)
    members = [Record(id="m1", fields={"label": "Dark Chocolate"})]
    asyncio.run(engine.run_stream(members))
    calls_after_first = len(llm.requests)
    engine2 = _engine(tmp_path, llm)      # same ledger file, same fingerprint
    asyncio.run(engine2.run_stream(members))
    assert len(llm.requests) == calls_after_first   # zero new LLM calls


def test_deterministic_ordering_is_recorded(tmp_path):
    engine = _engine(tmp_path, FakeLLM(handler=scripted_handler))
    members = [Record(id="m2", fields={"label": "zebra cake"}),
               Record(id="m1", fields={"label": "apple pie"})]
    asyncio.run(engine.run_stream(members))
    selects = [d for d in engine.ledger.iter_decisions("fp-test", kind="outcome")]
    by_order = sorted(selects, key=lambda d: d.order_index)
    assert [d.subject_id for d in by_order] == ["m1", "m2"]   # sorted by label
```

(`LevelEngine` must expose `.ledger` as a public attribute for tests and refine phases.) Resume in `test_resume_skips_completed_members` requires reloading minted clusters into the pool at engine start: `LevelEngine.__init__` loads `iter_clusters(fp, level)` into the pool (non-retired, minted only — taxonomy came from the constructor) and commits once.

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement the streaming engine** per the control flow above. Keep `decide` under ~120 lines by delegating: `_record_decision(...)`, `_route_assign(verdict) -> AssignStatus`, `_exhaust(query, member, first_candidates) -> tuple[list[Candidate], bool surfaced_new]`, `_mint(member, label, surfaced_new) -> Assignment`.
- [ ] **Step 4: Run tests + full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): streaming assign-or-mint engine with exhaustion, resume, deterministic order"`

---

### Task 8: Refinement R1–R2 (`refine.py`: retry + consolidation)

**Files:**
- Create: `src/xwalk/cluster/refine.py`
- Test: `tests/test_cluster_refine.py`

**Interfaces:**
- Consumes: `LevelEngine` (uses `engine.decide`, `engine.pool`, `engine.ledger`, `engine.spec`, `engine.context`, verifier stages via `engine.<verifier>` — make the verifiers public attributes on `LevelEngine`).
- Produces:
  - `async def retry_deferred(engine: LevelEngine, members_by_id: Mapping[str, Record]) -> int` — re-runs `engine.decide(phase=Phase.RETRY, iteration=0, allow_mint=True)` for every member whose current assignment is DEFERRED, in original order-index order; a still-unresolved member (DEFERRED outcome in retry) is written as NEEDS_REVIEW. Returns count resolved.
  - `async def consolidate(engine: LevelEngine, iteration: int) -> int` — sequential agglomeration; returns merges applied.
  - `class MergeLog` is not needed — merges go straight to `ledger.put_merge` + `pool.apply_merge`.

Consolidation algorithm (spec R2, exactly):
1. Candidates for merging: provisional clusters, ordered by `(-membership_size, cluster_id)` where membership size comes from `ledger.current_assignments`.
2. Track `consumed: set[str]` (losers) and `frozen: set[str]` (winners whose post-merge reverification failed — take no further merges this round).
3. For each provisional cluster A not consumed/frozen: `pool.search(doc_text(A), k=spec.base_k, exclude={A.cluster_id})`; iterate fused candidates B (skip consumed; skip taxonomy targets when `A` merging *into* taxonomy is fine — allowed — but taxonomy as *loser* never occurs since A iterates provisional only):
   - `merge_verifier.verify(entity_to_record(A), entity_to_record(B), engine.context)` → record Decision (kind `"merge"`, subject A, seq = candidate index).
   - `conf >= merge_accept_at` and decision `"merge"` ⇒ apply: winner = B if B is taxonomy, else the larger-membership of (A, B), ties → smaller cluster_id; loser = the other (never taxonomy — invariant check with explicit `raise` if violated). Merged winner entity: union aliases, exemplars = first `representation.exemplars` member ids of the union sorted by (assignment confidence desc, member id); `pool.apply_merge(loser, winner_entity)`; reassign loser's members to winner (`put_assignment` + `supersede`); `ledger.put_merge(..., outcome="applied")`; `pool.commit()` + snapshot. Then **reverify** the merged representation: `name_verifier.verify(winner.name, winner.gloss, member_labels, context)`; below `name_verify_accept_at` ⇒ `frozen.add(winner_id)` and a review row is appended (verb `"reverify_merge"` via `ledger.put_review_row`) — the merge stays applied, the *winner* just takes no further merges this round. Mark A consumed if A was the loser; break to next A either way (one merge per subject per round).
   - `conf` in `[merge_review_floor, accept)` ⇒ `ledger.put_merge(..., outcome="review")` + review row; continue to next candidate.
   - below floor or `"keep_separate"` ⇒ `ledger.put_merge(..., outcome="rejected")`; **no review row**; continue.
4. No unverified transitive merges: the loop never merges two clusters without a fresh verifier call on their current representations; committed merges land in `cluster_merges` (that table is the union-find record for edge normalization in Task 9's reassignment).

- [ ] **Step 1: Write the failing tests** — script with a `FakeLLM(handler=...)` like Task 7's, extended with a `## Left cluster` branch. Cases:

```python
def test_retry_resolves_deferred_after_pool_completes(tmp_path):
    # member m3 defers during stream (novelty confidence in review band),
    # then assigns to the cluster minted later by m4; assert retry resolves it
    # to ASSIGNED and writes phase=RETRY assignment superseding the DEFERRED row.
    ...

def test_consolidate_merges_duplicate_mints_and_reassigns_members(tmp_path):
    # two near-identical minted clusters K_a ("dark chocolate", 2 members),
    # K_b ("chocolate dark", 1 member); merge handler answers merge/0.9.
    # Assert: loser retired with merged_into, members reassigned + superseded,
    # cluster_merges outcome="applied", pool search no longer returns loser.
    ...

def test_rejected_merge_creates_no_review_work(tmp_path):
    # merge handler answers keep_separate/0.2 ⇒ outcome="rejected",
    # iter_review_rows() stays empty.
    ...

def test_taxonomy_never_loses_a_merge(tmp_path):
    # provisional A vs taxonomy T candidate, merge accepted ⇒ winner is ALWAYS T;
    # A retired with merged_into=T; T's entity unchanged (name, gloss intact).
    ...
```

Write these four tests out fully (same fixture style as Task 7 — `_engine` helper is imported from a shared `tests/cluster_helpers.py` you create now by extracting `scripted_handler`, `TEMPLATES`, `PROMPTS`, `_engine` from `tests/test_cluster_engine.py`; update that file's imports in the same commit).

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement `retry_deferred` and `consolidate`.**
- [ ] **Step 4: Run tests + full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): retry pass and sequential-agglomeration consolidation"`

---

### Task 9: Refinement R3–R5 + orchestration (`refine.py`, `engine.py`)

**Files:**
- Modify: `src/xwalk/cluster/refine.py`, `src/xwalk/cluster/engine.py`
- Test: `tests/test_cluster_refine.py` (extend), `tests/test_cluster_run.py`

**Interfaces:**
- Produces in `refine.py`:
  - `async def reassign(engine, members_by_id, iteration) -> int` (switches applied). Per assigned member, in order: retrieve from completed pool; **inject the incumbent** into the candidate list if retrieval missed it (build a `Candidate` with `fused_score=0.0`, empty evidence — read `records.py:Candidate` validation first: evidence may be empty); `selector.select`; route: incumbent chosen ⇒ record decision (kind `"reassign_select"`), keep, zero further calls; challenger chosen ⇒ `comparative_verifier.verify(member, context, incumbent_record, challenger_record)`; switch iff `preferred == "challenger"` and `conf >= comparative_accept_at` (new assignment row, supersede; count it); else keep and record. Null ⇒ `assignment_verifier.verify` on the incumbent (keyed list of just the incumbent, chosen_key its key); support+`conf >= assign_accept_at` ⇒ keep; else NEEDS_REVIEW (supersede with a review-status row). **Never mints.** Edges to merged clusters are normalized first: any current assignment pointing at a retired cluster is rewritten (follow `merged_into` chains to the live root — iterative loop with a visited-set; a cycle raises `RuntimeError`).
  - `def state_hash(ledger, run_fp, level) -> str` — `hash_value` of sorted `(member_id, cluster_id, status)` triples from `current_assignments` plus the sorted live (non-retired) cluster ids.
  - `async def refine(engine, members_by_id) -> StopReason` — the R4 loop: compute `state_hash` after retry; iterate `iteration = 1..spec.max_refine_iterations`: `merges = consolidate(...)`; `switches = reassign(...)`; new hash; if `merges == 0 and switches == 0` ⇒ CONVERGED; if hash seen before ⇒ OSCILLATION (flag unstable); else continue; loop end ⇒ MAX_ITERATIONS. Persist via `ledger.put_level_state(fp, level, stop_reason, best_iteration, hashes)` where `best_iteration` = the iteration whose stored assignment state had the fewest NEEDS_REVIEW rows (ties → earlier). On OSCILLATION/MAX_ITERATIONS the final table state stays at the last completed iteration; `best_iteration` records where the best state lives — assignment rows carry `iteration`, so the best state is reconstructable (document this in the docstring; M1 does not roll back rows).
  - `async def finalize(engine) -> int` — per live provisional cluster with ≥1 member: `namer.name(member_labels, context)` → `name_verifier.verify` → accepted ⇒ new name/gloss, `state=FINAL`, `pool.replace` + `put_cluster`; rejected or namer failure ⇒ **keep prior name**, still `state=FINAL`, review row (verb `"rename"`). Provisional clusters with zero members ⇒ `state=RETIRED`. Cluster ids never change (assert: entity passed to `pool.replace` keeps its `cluster_id`). Returns finalized count.
- Produces in `engine.py`:
  - `async def run_level(engine, members: Sequence[Record]) -> LevelReport` — stream → retry → refine → finalize; `LevelReport(level, stop_reason, assigned, minted, merged, needs_review, finalized)` frozen dataclass.
  - `def build_cluster_fingerprint(*, templates, prompts, level_specs: Sequence[LevelSpec], taxonomies: Mapping[str, Sequence[Record]], llm, verifier_llm, encoder_name: str | None, library_version: str = __version__) -> str` — `hash_value` over: library version, `templates.fingerprint`, `prompts.fingerprint`, each `spec.fingerprint_payload()`, per-level sorted taxonomy record hashes, `llm.fingerprint`, `verifier_llm.fingerprint if verifier_llm else None`, encoder name, and the ordering rule name (`"label_casefold"`).
  - `class ClusterRun` — `__init__(*, level_specs, templates, prompts, llm, verifier_llm=None, encoder=None, taxonomies=None, out: Path)`; `async def run(self, source: Iterable[Record]) -> list[LevelReport]`: computes the fingerprint, opens `ClusterLedger` at `out / "ledger.sqlite"`, writes a manifest (fingerprint, library version, level names, model identity), then per level: build `Pool` (+ taxonomy for that level), all stages (verifier stages use `verifier_llm or llm` — the spec's separate-model support), `LevelEngine`, `run_level`; the next level's members are the finalized clusters of this level rendered as `Record(id=cluster_id, fields={"label": name, "gloss": gloss, "exemplars": [...]})`. Taxonomy-assigned members do not produce next-level members from their taxonomy parents (taxonomy nodes are already part of the next level's pool if configured there).

- [ ] **Step 1: Write the failing tests.** In `test_cluster_refine.py` add:

```python
def test_reassign_requires_comparative_margin(tmp_path):
    # incumbent K1; selector now prefers challenger K2; comparative answers
    # challenger/0.55 with comparative_accept_at=0.7 ⇒ NO switch, decision recorded.
    ...

def test_reassign_null_reverifies_incumbent(tmp_path):
    # selector returns null on reassign; assignment verifier supports incumbent
    # at 0.9 ⇒ assignment kept, no review row.
    ...

def test_oscillation_detection_stops_and_flags(tmp_path):
    # handler alternates member m1 between K1 and K2 on successive reassign
    # iterations (route on iteration marker in the prompt or a closure counter)
    # with comparative always 0.95 ⇒ state hash repeats ⇒ StopReason.OSCILLATION,
    # level state persisted with best_iteration recorded.
    ...

def test_finalize_verifies_names_and_never_changes_ids(tmp_path):
    # name_verify answers does_not_fit/0.2 ⇒ prior name kept, review row "rename",
    # state=FINAL, cluster_id unchanged.
    ...
```

In `test_cluster_run.py` write the end-to-end acceptance test:

```python
def test_two_level_run_end_to_end(tmp_path):
    """The spec's chocolate example, driven entirely by FakeLLM.

    Level 'canonical' (equivalence): Dark Chocolate + chocolate Dark merge;
    Milk Chocolate mints separately.
    Level 'family' (subsumption): both canonical clusters roll up under the
    taxonomy node 'Chocolate'.
    Asserts: hierarchy of assignments across levels, all decisions in the
    ledger, deterministic re-run produces identical assignment tables, and a
    resumed (interrupted) run produces a ledger byte-identical in decisions
    to an uninterrupted one.
    """
    ...

def test_fingerprint_shifts_on_granularity_change(tmp_path):
    # same inputs, one LevelSpec granularity word changed => different fingerprint
    ...
```

Write these fully using the shared `tests/cluster_helpers.py` handler, extended with `## Incumbent` (comparative) and `## Proposed name` (name_verify) branches.

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement R3–R5, `run_level`, `build_cluster_fingerprint`, `ClusterRun`.**
- [ ] **Step 4: Run tests + full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): comparative reassignment, oscillation handling, finalize, ClusterRun"`

---

### Task 10: Review overlay (`review.py`)

**Files:**
- Create: `src/xwalk/cluster/review.py`
- Test: `tests/test_cluster_review.py`

**Interfaces:**
- Consumes: `ClusterLedger`, Task 1 types, `csv` — modeled directly on `src/xwalk/review.py` (read it first; keep the three-layer doctrine and all-or-nothing apply).
- Produces:
  - `class ClusterReviewVerb(Enum)`: `ASSIGN_TO = "assign_to"`, `CONFIRM_MINT = "confirm_mint"`, `REJECT_MINT = "reject_mint"`, `APPROVE_MERGE = "approve_merge"`, `REJECT_MERGE = "reject_merge"`, `RENAME = "rename"`
  - `COLUMNS = ("run_fingerprint","level","subject_id","subject_hash","subject_kind","proposed_cluster_id","verb","target_cluster_id","new_name","reviewer","review_note","reviewed_at")`
  - `export_cluster_review(ledger, run_fp, level, out_path) -> int` — one row per NEEDS_REVIEW assignment (`subject_kind="member"`), per merge with `outcome="review"` (`subject_kind="merge"`, subject_id=loser), per `"rename"`/`"reverify_merge"` review row from finalize/consolidate (`subject_kind="cluster"`); decision columns blank.
  - `read_cluster_review(path) -> list[ClusterReviewRow]` — blank verbs skipped; unknown verb raises with row number; `ASSIGN_TO`/`APPROVE_MERGE` require `target_cluster_id`; `RENAME` requires `new_name`; empty reviewer raises.
  - `apply_cluster_review(ledger, rows) -> ApplyReport` — validation first, all-or-nothing: every row's `subject_hash` must match current state (member hash from current assignment; cluster hash via `entity_hash` from the clusters table), every referenced `target_cluster_id` must exist and not be RETIRED, `RENAME` targets must be MINTED (taxonomy immutable — a rename of a taxonomy node raises `SnapshotMismatch`-style error before anything is written); then append all rows to `cluster_reviews`.
  - `adjudicated_assignments(ledger, run_fp, level) -> Iterator[AdjudicatedAssignment]` — model answer + latest review verb per member; `ASSIGN_TO` overrides cluster and status→ASSIGNED, `REJECT_MINT` on a member's mint sends that member to status NEEDS_REVIEW with cluster None, verbs on non-member subjects don't appear in this view (they affect `adjudicated_clusters(ledger, run_fp, level)` — name overrides from RENAME).

- [ ] **Step 1: Write four failing tests** (full code, fixture = a small ledger built by hand with `put_cluster`/`put_assignment`): round-trip export→edit→read; all-or-nothing on one stale `subject_hash`; taxonomy rename refused; adjudicated view shows `ASSIGN_TO` override while preserving the model's answer.
- [ ] **Step 2–4: fail → implement → pass + full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): snapshot-guarded review overlay with clustering verbs"`

---

### Task 11: Exports and diagnostics (`exports.py`, `diagnostics.py`)

**Files:**
- Create: `src/xwalk/cluster/exports.py`, `src/xwalk/cluster/diagnostics.py`
- Test: `tests/test_cluster_diagnostics.py` (covers both)

**Interfaces:**
- `export_hierarchy_csv(ledger, run_fp, levels: Sequence[str], path, *, use_review: bool = False) -> int` — columns: `member_id`, then per level `"{level}_cluster_id","{level}_cluster_name","{level}_status"`; one row per level-0 member, later levels resolved by following the member's cluster into the next level's assignments; `use_review=True` reads the adjudicated views.
- `export_clusters_csv(ledger, run_fp, path) -> int` — columns `cluster_id,level,name,gloss,provenance,state,seed_member_id,member_count,mint_provenance,merged_into`.
- `class LevelDiagnostics` frozen dataclass with exactly the spec's token-free metrics: `singleton_rate: float | None`, `size_distribution: dict[str, float]` (p50/p90/max), `late_parent_rate: float | None`, `streaming_retrieval_recall: float | None`, `completed_pool_retrieval_recall: float | None`, `duplicate_mint_rate: float | None`, `reassignment_curve: list[int]`, `review_rate: float | None`, `defer_rate: float | None`, `stop_reason: str | None`. `None` when the denominator is zero (`_ratio` pattern from `evaluate/metrics.py`).
- `def diagnose_level(ledger, run_fp, level) -> LevelDiagnostics`. The three-way retrieval split is computed from recorded decisions: for each finally-assigned member, its final parent cluster's `created_snapshot` vs the `snapshot_id` on the member's STREAM `select` decision determines **late-parent** (parent did not exist yet); otherwise membership of the final parent id in the STREAM select decision's `resolved["candidate_ids"]` gives **streaming retrieval recall**; membership in the REASSIGN select decision's candidate ids gives **completed-pool retrieval recall**. ⇒ Task 7/9 must record `resolved["candidate_ids"]` (the fused candidate id list) on every select-kind decision — add that requirement to those tasks' Decision writes (it is listed here so the implementer of 7/9 includes it; if you are implementing this task and it's missing, add it there first with a test).
- `def render_diagnostics(diags: Mapping[str, LevelDiagnostics]) -> str` — plain-text report in the style of `evaluate/report.py`: metric, value, and one "what to fix" line per anomalous metric (high duplicate-mint → retrieval depth or representation; high late-parent → ordering; low streaming recall with high completed recall → ordering artifact, not retrieval failure; low both → doc rendering/retriever).

- [ ] **Step 1: Failing tests** (full code): build a small ledger by hand covering: one late parent, one streaming miss recovered at reassign, one duplicate mint (merged), one singleton; assert each rate exactly; assert `export_hierarchy_csv` row content across two levels; assert `render_diagnostics` mentions "ordering" when late-parent rate is high and streaming recall is low while completed recall is high.
- [ ] **Step 2–4: fail → implement → pass + full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(cluster): hierarchy/cluster exports and structural diagnostics"`

---

### Task 12: Public API, docs, changelog

**Files:**
- Modify: `src/xwalk/cluster/__init__.py` (re-export: `ClusterRun`, `LevelSpec`, `ClusterPolicy`, `Relation`, `Cardinality`, `ClusterPromptSet`, `default_cluster_slots`, `load_cluster_slots`, `build_cluster_fingerprint`, `ClusterLedger`, review + export + diagnostics functions), `CHANGELOG.md`, `docs/components.md`, `docs/README.md`
- Create: `docs/guide/clustering.md`
- Test: `tests/test_docs.py` already validates docs structure — read it first and satisfy it.

- [ ] **Step 1: Write `docs/guide/clustering.md`** — a runnable end-to-end example (two levels, `FakeLLM`-free: shows `OpenAICompatClient` usage mirroring the README example), the phase diagram from the spec, the threshold-calibration warning (provisional defaults; adjudicated dev set), the diagnostics table with the "what to fix" column, and the M1 limitations list (no batching, single-parent only, no LLM-judged metrics — Milestone 2).
- [ ] **Step 2: Update `docs/components.md`** (one section: what each `cluster/` module does), `docs/README.md` (link), `CHANGELOG.md` (Unreleased: "Added: hierarchical LLM-adjudicated clustering (Milestone 1) — see docs/guide/clustering.md").
- [ ] **Step 3: Run the full suite** — `pytest` green, plus `mypy src/xwalk/cluster --strict` if the repo runs mypy in CI (check `pyproject.toml`/CI config; match whatever gate exists).
- [ ] **Step 4: Commit** — `git commit -m "docs(cluster): public API, guide, changelog for clustering M1"`

---

## Acceptance criteria (Milestone 1 done means)

1. `test_cluster_run.py::test_two_level_run_end_to_end` passes: chocolate example produces the spec's hierarchy with zero real LLM calls.
2. Determinism: running the same `ClusterRun` twice into fresh directories yields identical `cluster_assignments` and `clusters` tables (ignoring autoincrement ids).
3. Resume: interrupting after member N and re-running produces decisions identical to an uninterrupted run, with zero repeated LLM calls for completed decisions.
4. Every spec invariant has at least one test asserting it (map: 1 → existing keying tests + Task 7 select routing; 2 → Tasks 6–9 routing tests; 3 → Tasks 5/8/10 taxonomy tests; 4 → Task 7 exhaustion test; 5 → Task 8 sequential-agglomeration tests; 6 → Task 9 finalize test; 7 → Task 9 margin + never-mints tests; 8 → Task 9 oscillation test; 9 → Tasks 3/7 resume tests).
5. The full pre-existing test suite passes unmodified except the one `Selector` annotation change.

## Self-review notes (already applied)

- Spec coverage gaps found and fixed: `resolved["candidate_ids"]` recording (needed by diagnostics) pushed into Tasks 7/9; `tests/cluster_helpers.py` extraction scheduled in Task 8 where the duplication first appears; verifier-model separation (`verifier_llm`) wired in Task 9's `ClusterRun` rather than left to M2 (spec requires the config hook now, even though M1 tests use one FakeLLM).
- Deliberate M1 exclusions restated: `audit_rate` (documented deviation, Global Constraints), `Cardinality.MANY` engine support, optimistic batching, LLM-judged diagnostics, B-cubed/gold metrics, stability probes, CLI/YAML config (`xwalk cluster …` lands with M2 once the Python API has settled).

"""A job spec is serialized constructor arguments -- nothing more.

Every field maps to a parameter the README passes by hand. No config field gates
behaviour the SDK cannot express, and credentials are named environment variables,
never values in the file.

Paths inside a job file resolve against **the job file's own directory**, so a job
directory is movable as a unit.

Validation is strict (docs/claude-upgrade/CONTRACTS.md section 8): every spec forbids
unknown fields, so a misspelled key fails instead of silently falling back to a default;
numeric settings are range-checked; a field that does not apply to the declared `kind` is
rejected rather than ignored. `load_job` raises `JobValidationError`, whose `issues` name
each problem with a dotted location and, for an unknown key, the nearest valid one.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from xwalk._yaml import load_strict_yaml, yaml_error_message
from xwalk.batch import decision_run_fingerprint_components, run_fingerprint_components
from xwalk.decide.base import DecisionClient
from xwalk.decide.jev import JevClient
from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.decide.questions import QuestionSet
from xwalk.fingerprint import hash_value
from xwalk.llm.base import LLMClient
from xwalk.llm.openai_compat import CAPABILITY_PROFILES, OpenAICompatClient
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet, load_slots
from xwalk.records import Record
from xwalk.retrieval.base import Retriever
from xwalk.retrieval.bm25 import MAX_FUZZY_DISTANCE, BM25Retriever
from xwalk.retrieval.dense import Encoder
from xwalk.sources.tabular import csv_source, jsonl_source
from xwalk.stages.choose import Chooser
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.property_gate import PropertyGate
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.screen import Screener
from xwalk.stages.select import Selector, SelectorPolicy
from xwalk.stores.base import TargetStore
from xwalk.stores.memory import MemoryStore
from xwalk.templates import TemplateSet


class CredentialMissingError(ValueError):
    """The environment variable named by an `api_key_env` is unset. Its value, when
    set, is never printed or stored. `where` is the block's path in the job file."""

    def __init__(self, variable: str, where: str = "llm") -> None:
        self.variable = variable
        self.where = where
        super().__init__(
            f"environment variable {variable} is not set; "
            f"export it or change {where}.api_key_env in the job file"
        )


class _Spec(BaseModel):
    """Every job spec forbids unknown keys: a typo must fail, not become a default."""

    model_config = ConfigDict(extra="forbid")

    def _reject_inapplicable(self, fields: Sequence[str], *, kind: str) -> None:
        stray = [name for name in fields if name in self.model_fields_set]
        if stray:
            raise ValueError(f"{', '.join(stray)} does not apply to kind {kind!r}")


class TemplateSpec(_Spec):
    query: str | None = None
    queries: list[str] = Field(default_factory=list)
    context: str = ""
    doc: str
    candidate: str

    @model_validator(mode="after")
    def _need_a_query(self) -> TemplateSpec:
        if not self.query and not self.queries:
            raise ValueError("templates need a query or a non-empty queries list")
        return self


_FILE_KINDS = ("csv", "tsv", "jsonl", "obo", "owl")
_TABULAR_ONLY = ("multivalue_columns", "multivalue_sep")
_SQL_ONLY = ("url", "query")
_ONTOLOGY_ONLY = ("id_prefix", "include_obsolete")


class RecordSpec(_Spec):
    kind: Literal["csv", "tsv", "jsonl", "obo", "owl", "sql"]
    path: str | None = None
    id_column: str = "id"
    id_field: str | None = None
    multivalue_columns: list[str] = Field(default_factory=list)
    multivalue_sep: str = "|"
    # sql
    url: str | None = None
    query: str | None = None
    # ontology
    id_prefix: str | None = None
    include_obsolete: bool = False

    @model_validator(mode="after")
    def _check(self) -> RecordSpec:
        if self.kind in _FILE_KINDS and not self.path:
            raise ValueError(f"a {self.kind} collection needs a path")
        if self.kind == "sql":
            if not (self.url and self.query):
                raise ValueError("a sql collection needs url and query")
            self._reject_inapplicable(("path", *_ONTOLOGY_ONLY), kind=self.kind)
        else:
            self._reject_inapplicable(_SQL_ONLY, kind=self.kind)
        if self.kind not in ("obo", "owl"):
            self._reject_inapplicable(_ONTOLOGY_ONLY, kind=self.kind)
        if self.kind in ("jsonl", "obo", "owl"):
            self._reject_inapplicable(_TABULAR_ONLY, kind=self.kind)
        if self.kind != "jsonl":
            self._reject_inapplicable(("id_field",), kind=self.kind)
        if not self.id_column:
            raise ValueError("id_column must not be empty")
        if not self.multivalue_sep:
            raise ValueError("multivalue_sep must not be empty")
        return self

    @property
    def required_extra(self) -> tuple[str, str] | None:
        """`(extra, module)` this collection needs beyond the base install."""
        if self.kind == "owl":
            return ("ontology", "rdflib")
        if self.kind == "sql":
            return ("sql", "sqlalchemy")
        return None

    def _resolve(self, base_dir: Path) -> Path:
        if self.path is None:
            raise ValueError(f"a {self.kind} source needs a path")
        return base_dir / self.path

    def build(self, base_dir: Path) -> Iterator[Record]:
        if self.kind in ("csv", "tsv"):
            return csv_source(
                self._resolve(base_dir),
                id_column=self.id_column,
                delimiter="\t" if self.kind == "tsv" else ",",
                multivalue_columns=self.multivalue_columns,
                multivalue_sep=self.multivalue_sep,
            )
        if self.kind == "jsonl":
            return jsonl_source(self._resolve(base_dir), id_field=self.id_field or self.id_column)
        if self.kind == "obo":
            from xwalk.sources.ontology import obo_source

            return obo_source(
                self._resolve(base_dir),
                id_prefix=self.id_prefix,
                include_obsolete=self.include_obsolete,
            )
        if self.kind == "owl":
            from xwalk.sources.ontology import owl_source

            return owl_source(
                self._resolve(base_dir),
                id_prefix=self.id_prefix,
                include_obsolete=self.include_obsolete,
            )
        from xwalk.sources.sql import sql_source

        if not (self.url and self.query):
            raise ValueError("a sql source needs url and query")
        return sql_source(
            self.url,
            self.query,
            id_column=self.id_column,
            multivalue_columns=self.multivalue_columns,
            multivalue_sep=self.multivalue_sep,
        )


_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_DENSE_ONLY = ("model", "device", "query_prefix", "doc_prefix", "revision", "normalize")


class RetrieverSpec(_Spec):
    kind: Literal["bm25", "dense"]
    name: str | None = None
    limit: int = Field(default=20, ge=1)
    # bm25 only
    exact_fields: list[str] | None = None
    analyzer: Literal["default", "en_stem"] = "default"
    fuzzy_distance: int = Field(default=0, ge=0, le=MAX_FUZZY_DISTANCE)
    # dense only
    model: str | None = None
    device: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""
    revision: str | None = None  # a pinned model revision; unpinned is recorded "unknown"
    normalize: bool = True

    @model_validator(mode="after")
    def _check(self) -> RetrieverSpec:
        if self.name is not None and not self.name.strip():
            raise ValueError("a retriever name must not be blank")
        if self.name is not None and not _SAFE_NAME.fullmatch(self.name):
            # The name is the index subdirectory; it may not leave the index directory.
            raise ValueError(
                f"retriever name {self.name!r} must start with a letter or digit and use only "
                "letters, digits, '.', '_' and '-' (it names the index directory)"
            )
        if self.kind == "dense":
            if not self.model:
                raise ValueError("a dense retriever needs a model name")
            self._reject_inapplicable(
                ("exact_fields", "analyzer", "fuzzy_distance"), kind=self.kind
            )
        else:
            self._reject_inapplicable(_DENSE_ONLY, kind=self.kind)
        return self

    @property
    def effective_exact_fields(self) -> tuple[str, ...]:
        """bm25 `exact_fields`, defaulting to label and synonyms."""
        if self.exact_fields is None:
            return ("label", "synonyms")
        return tuple(self.exact_fields)

    @property
    def index_name(self) -> str:
        """The index subdirectory: `name`, or the kind when unnamed."""
        return self.name or self.kind

    @property
    def required_extra(self) -> tuple[str, str] | None:
        return ("dense", "sentence_transformers") if self.kind == "dense" else None


class LLMSpec(_Spec):
    kind: Literal["openai_compat", "litellm"] = "openai_compat"
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None  # present only so it can be rejected
    profile: str = "unknown"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    seed: int | None = None

    @model_validator(mode="after")
    def _no_inline_secrets(self) -> LLMSpec:
        if self.api_key is not None:
            raise ValueError(
                "api_key must not appear in a job file; use api_key_env with the name of "
                "an environment variable"
            )
        if self.kind == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat needs a base_url")
        if self.profile not in CAPABILITY_PROFILES:
            raise ValueError(
                f"unknown profile {self.profile!r}; choose from {sorted(CAPABILITY_PROFILES)}"
            )
        return self


class RewriteSpec(_Spec):
    """An LLM that proposes alternative queries after a screen miss. It never decides."""

    llm: LLMSpec
    max_queries: int = Field(default=3, ge=1, le=5)


class DeciderSpec(_Spec):
    kind: Literal["jev"] = "jev"
    model: str = Field(min_length=1)
    base_url: str = Field(default="https://openrouter.ai/api/alpha/decisions", min_length=1)
    api_key_env: str | None = None
    api_key: str | None = None  # present only so it can be rejected
    timeout: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=5, ge=0)
    rewrite: RewriteSpec | None = None

    @model_validator(mode="after")
    def _no_inline_secrets(self) -> DeciderSpec:
        if self.api_key is not None:
            raise ValueError(
                "api_key must not appear in a job file; use api_key_env with the name of "
                "an environment variable"
            )
        return self


class DecisionPolicySpec(_Spec):
    """`policy:` of a decider job (the YAML key is shared; the path decides the model)."""

    screen_floor: float = Field(default=0.30, ge=0.0, le=1.0)
    shortlist_size: int = Field(default=15, ge=1)
    shortlist_floor: float = Field(default=0.20, ge=0.0, le=1.0)
    none_at: float = Field(default=0.70, ge=0.0, le=1.0)
    choose_at: float = Field(default=0.50, ge=0.0, le=1.0)
    accept_at: float = Field(default=0.85, ge=0.0, le=1.0)
    rubric_floor: float | None = Field(default=None, ge=0.0)  # a rubric level, not a probability
    property_floor: float = Field(default=0.50, ge=0.0, le=1.0)
    chunk_size: int = Field(default=50, ge=1)
    max_candidates: int = Field(default=300, ge=1)
    concurrency: int = Field(default=32, ge=1)
    retriever_timeout: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> DecisionPolicySpec:
        if self.shortlist_floor > self.accept_at:
            raise ValueError(
                f"shortlist_floor ({self.shortlist_floor}) must not exceed "
                f"accept_at ({self.accept_at})"
            )
        return self


class PolicySpec(_Spec):
    max_attempts: int = Field(default=4, ge=1)
    accept_at: float = Field(default=0.6, ge=0.0, le=1.0)
    review_floor: float = Field(default=0.4, ge=0.0, le=1.0)
    verify_band: tuple[float, float] | None = (0.6, 0.8)
    audit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    concurrency: int = Field(default=32, ge=1)
    legacy_id_resolution: bool = False
    retriever_timeout: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> PolicySpec:
        if self.review_floor > self.accept_at:
            raise ValueError(
                f"review_floor ({self.review_floor}) must not exceed accept_at ({self.accept_at})"
            )
        if self.verify_band is not None:
            low, high = self.verify_band
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(
                    f"verify_band must be an ordered pair within [0, 1], got [{low}, {high}]"
                )
        return self


class SelectorSpec(_Spec):
    max_candidates: int = Field(default=30, ge=1)
    max_candidate_tokens: int = Field(default=8_000, ge=1)


class PromptSpec(_Spec):
    slots: str


class JobSpec(_Spec):
    name: str = Field(min_length=1)
    templates: TemplateSpec
    target: RecordSpec
    source: RecordSpec
    retrievers: list[RetrieverSpec] = Field(min_length=1)
    llm: LLMSpec | None = None
    decider: DeciderSpec | None = None
    prompts: PromptSpec
    policy: PolicySpec = PolicySpec()
    decision_policy: DecisionPolicySpec = DecisionPolicySpec()
    selector: SelectorSpec = SelectorSpec()

    base_dir: Path = Path(".")

    @model_validator(mode="before")
    @classmethod
    def _route_policy(cls, data: Any) -> Any:
        # One YAML key, `policy:`, on both paths. Which model parses it depends on the
        # path; `decision_policy` is internal and not a job-file key.
        if isinstance(data, dict) and data.get("decider") is not None and "policy" in data:
            data = dict(data)
            data["decision_policy"] = data.pop("policy")
        return data

    @model_validator(mode="after")
    def _check(self) -> JobSpec:
        if (self.llm is None) == (self.decider is None):
            raise ValueError("a job needs exactly one of `llm:` or `decider:`")
        if self.decider is not None and "selector" in self.model_fields_set:
            raise ValueError("selector does not apply to a decider job")
        if self.llm is not None and len(self.templates.queries) > 1:
            # The LLM matcher searches one query per attempt (the rewriter supplies the
            # next); extra queries would be fingerprinted and silently never searched.
            raise ValueError(
                "templates.queries with more than one entry applies to decider jobs; "
                "an llm job searches templates.query (or a single queries entry)"
            )
        seen: set[str] = set()
        for spec in self.retrievers:
            if spec.index_name in seen:
                raise ValueError(
                    f"two retrievers share the name {spec.index_name!r}; each needs its own "
                    f"index directory, so give them distinct names"
                )
            seen.add(spec.index_name)
        return self

    # --- builders ---------------------------------------------------------------

    def build_templates(self) -> TemplateSet:
        return TemplateSet(
            query=self.templates.query or self.templates.queries[0],
            queries=tuple(self.templates.queries),
            context=self.templates.context,
            doc=self.templates.doc,
            candidate=self.templates.candidate,
        )

    def build_target_records(self) -> Iterator[Record]:
        return self.target.build(self.base_dir)

    def build_source_records(self) -> Iterator[Record]:
        return self.source.build(self.base_dir)

    def build_store(self) -> MemoryStore:
        return MemoryStore.from_source(self.build_target_records())

    def build_retrievers(
        self,
        records: Sequence[Record],
        templates: TemplateSet,
        index_dir: str | Path,
    ) -> list[Retriever]:
        index_dir = Path(index_dir)
        built: list[Retriever] = []
        for spec in self.retrievers:
            target = index_dir / spec.index_name
            if spec.kind == "bm25":
                built.append(
                    BM25Retriever.build(
                        records,
                        templates,
                        target,
                        name=spec.name or "bm25",
                        exact_fields=spec.effective_exact_fields,
                        default_limit=spec.limit,
                        analyzer=spec.analyzer,
                        fuzzy_distance=spec.fuzzy_distance,
                    )
                )
            else:
                from xwalk.retrieval.dense import DenseRetriever

                built.append(
                    DenseRetriever.build(
                        records,
                        templates,
                        target,
                        self.build_encoder(spec),
                        name=spec.name,
                        default_limit=spec.limit,
                    )
                )
        return built

    @staticmethod
    def build_encoder(spec: RetrieverSpec) -> Encoder:
        """The encoder a dense retriever spec describes. Loads the model (requires
        xwalk[dense]); never encodes anything."""
        from xwalk.retrieval.dense import SentenceTransformerEncoder

        return SentenceTransformerEncoder(
            spec.model or "",
            device=spec.device,
            normalize=spec.normalize,
            query_prefix=spec.query_prefix,
            doc_prefix=spec.doc_prefix,
            revision=spec.revision,
        )

    def build_llm(self) -> LLMClient:
        if self.llm is None:
            raise ValueError("this job has no llm: block")
        return _build_llm_client(self.llm, where="llm")

    def build_rewrite_llm(self) -> LLMClient:
        if self.decider is None or self.decider.rewrite is None:
            raise ValueError("this job has no decider.rewrite: block")
        return _build_llm_client(self.decider.rewrite.llm, where="decider.rewrite.llm")

    @property
    def is_decider(self) -> bool:
        """True for a job that decides with a decision model (`decider:`), not an LLM."""
        return self.decider is not None

    def build_prompts(self) -> PromptSet:
        return PromptSet.from_slots(load_slots(self.base_dir / self.prompts.slots))

    def build_decider(self) -> DecisionClient:
        if self.decider is None:
            raise ValueError("this job has no decider: block")
        api_key = None
        if self.decider.api_key_env:
            api_key = os.environ.get(self.decider.api_key_env)
            if not api_key:
                raise CredentialMissingError(self.decider.api_key_env, "decider")
        return JevClient(
            self.decider.base_url,
            self.decider.model,
            api_key=api_key,
            timeout=self.decider.timeout,
            max_retries=self.decider.max_retries,
        )

    def build_questions(self) -> QuestionSet:
        return QuestionSet.from_slots(load_slots(self.base_dir / self.prompts.slots))

    def build_decision_policy(self) -> DecisionPolicy:
        return DecisionPolicy(**self.decision_policy.model_dump())

    def build_policy(self) -> MatchPolicy:
        return MatchPolicy(**self.policy.model_dump())

    def build_selector_policy(self) -> SelectorPolicy:
        return SelectorPolicy(**self.selector.model_dump())

    def run_fingerprint_components(
        self,
        *,
        store: TargetStore,
        retrievers: Sequence[Retriever],
        llm: LLMClient,
        prompts: PromptSet | None = None,
    ) -> dict[str, Any]:
        return run_fingerprint_components(
            templates=self.build_templates(),
            prompts=prompts or self.build_prompts(),
            store=store,
            retrievers=retrievers,
            llm=llm,
            policy=self.build_policy(),
            selector_policy=self.build_selector_policy(),
            # The same fallback depth `build_matcher` hands the Matcher.
            retriever_limit=self._retriever_limit(),
        )

    def run_fingerprint(
        self,
        *,
        store: TargetStore,
        retrievers: Sequence[Retriever],
        llm: LLMClient,
        prompts: PromptSet | None = None,
    ) -> str:
        return hash_value(
            self.run_fingerprint_components(
                store=store, retrievers=retrievers, llm=llm, prompts=prompts
            )
        )

    def _retriever_limit(self) -> int:
        return max(spec.limit for spec in self.retrievers)

    def build_matcher(
        self,
        *,
        store: TargetStore,
        retrievers: Sequence[Retriever],
        llm: LLMClient,
        prompts: PromptSet | None = None,
    ) -> Matcher:
        """`prompts` overrides the job's own slots.

        The prompt optimiser needs this: without it every round silently re-runs the
        original slots and the optimiser measures nothing.
        """
        templates = self.build_templates()
        resolved = prompts or self.build_prompts()
        policy = self.build_policy()
        return Matcher(
            templates=templates,
            retrievers=list(retrievers),
            store=store,
            selector=Selector(
                llm,
                resolved,
                templates,
                policy=self.build_selector_policy(),
                legacy_id_resolution=policy.legacy_id_resolution,
            ),
            scorer=Scorer(llm, resolved, templates, review_floor=policy.review_floor),
            verifier=Verifier(llm, resolved, templates),
            rewriter=QueryRewriter(llm, resolved, templates),
            policy=policy,
            run_fingerprint=self.run_fingerprint(
                store=store, retrievers=retrievers, llm=llm, prompts=resolved
            ),
            retriever_limit=self._retriever_limit(),
        )

    def decision_run_fingerprint_components(
        self, *, store: TargetStore, retrievers: Sequence[Retriever], decider: DecisionClient
    ) -> dict[str, Any]:
        return decision_run_fingerprint_components(
            templates=self.build_templates(),
            questions=self.build_questions(),
            store=store,
            retrievers=retrievers,
            decider=decider,
            policy=self.build_decision_policy(),
            rewrite=self._rewrite_fingerprint(),
        )

    def decision_run_fingerprint(
        self, *, store: TargetStore, retrievers: Sequence[Retriever], decider: DecisionClient
    ) -> str:
        return hash_value(
            self.decision_run_fingerprint_components(
                store=store, retrievers=retrievers, decider=decider
            )
        )

    def _rewrite_fingerprint(self) -> str | None:
        """What the rewrite would propose depends on its model, its prompt, and how many
        queries it may return; credentials and the env var name do not change that."""
        if self.decider is None or self.decider.rewrite is None:
            return None
        spec = self.decider.rewrite
        return hash_value(
            {
                "llm": spec.llm.model_dump(exclude={"api_key", "api_key_env"}),
                "max_queries": spec.max_queries,
                "prompts": self.build_prompts().fingerprint,
            }
        )

    def build_decision_matcher(
        self,
        *,
        store: TargetStore,
        retrievers: Sequence[Retriever],
        decider: DecisionClient,
        rewrite_llm: LLMClient | None = None,
    ) -> DecisionMatcher:
        """`rewrite_llm` is the client for `decider.rewrite`; it is built from the job
        file when the block is present and none is passed."""
        templates = self.build_templates()
        questions = self.build_questions()
        policy = self.build_decision_policy()
        rewrite = None if self.decider is None else self.decider.rewrite
        rewriter: QueryRewriter | None = None
        if rewrite is None:
            if rewrite_llm is not None:
                raise ValueError("a rewrite LLM was passed but the job has no decider.rewrite:")
        else:
            rewriter = QueryRewriter(
                rewrite_llm or self.build_rewrite_llm(),
                self.build_prompts(),
                templates,
                max_queries=rewrite.max_queries,
            )
        return DecisionMatcher(
            templates=templates,
            retrievers=list(retrievers),
            store=store,
            screener=Screener(
                decider,
                questions,
                templates,
                chunk_size=policy.chunk_size,
                shortlist_size=policy.shortlist_size,
                shortlist_floor=policy.shortlist_floor,
            ),
            chooser=Chooser(decider, questions, templates),
            gate=PropertyGate(decider, questions, templates),
            policy=policy,
            run_fingerprint=self.decision_run_fingerprint(
                store=store, retrievers=retrievers, decider=decider
            ),
            rewriter=rewriter,
        )


def _build_llm_client(spec: LLMSpec, *, where: str) -> LLMClient:
    """`where` is the spec's path in the job file, for the missing-key error."""
    api_key = None
    if spec.api_key_env:
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise CredentialMissingError(spec.api_key_env, where)
    if spec.kind == "openai_compat":
        return OpenAICompatClient(
            base_url=spec.base_url or "",
            model=spec.model,
            api_key=api_key,
            profile=spec.profile,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            seed=spec.seed,
        )
    from xwalk.llm.litellm import LiteLLMClient

    return LiteLLMClient(
        spec.model,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        seed=spec.seed,
    )


@dataclass(frozen=True)
class JobIssue:
    """One problem with a job file. `loc` is a dotted path such as `policy.accept_at`."""

    code: str
    loc: str
    message: str

    def __str__(self) -> str:
        return f"{self.loc}: {self.message}" if self.loc else self.message


class JobValidationError(ValueError):
    """The job file is missing, unreadable or invalid. `issues` lists every problem."""

    def __init__(self, path: str | Path | None, issues: Sequence[JobIssue]) -> None:
        self.path = None if path is None else Path(path)
        self.issues = tuple(issues)
        where = f"{self.path}: " if self.path is not None else ""
        lines = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"{where}invalid job file\n{lines}")


def _model_at(
    loc: Sequence[int | str], root: type[BaseModel] | None = None
) -> type[BaseModel] | None:
    """The spec class a pydantic error location points into (for "did you mean")."""
    model: type[BaseModel] = root or JobSpec
    for part in loc:
        if isinstance(part, int):
            continue
        field = model.model_fields.get(str(part))
        if field is None:
            return None
        annotation = field.annotation
        inner = getattr(annotation, "__args__", None) or (annotation,)
        nested = [a for a in inner if isinstance(a, type) and issubclass(a, BaseModel)]
        if not nested:
            return None
        model = nested[0]
    return model


def _issues_from(error: ValidationError, root: type[BaseModel] | None = None) -> list[JobIssue]:
    issues: list[JobIssue] = []
    for item in error.errors():
        loc = [p for p in item["loc"] if not (isinstance(p, str) and p.startswith("function-"))]
        # A decider job's `policy:` is parsed as `decision_policy`; report it as written.
        shown = ["policy" if i == 0 and p == "decision_policy" else p for i, p in enumerate(loc)]
        dotted = ".".join(str(p) for p in shown)
        if item["type"] == "extra_forbidden":
            parent = _model_at(loc[:-1], root)
            known = [
                n
                for n in (parent.model_fields if parent else {})
                if n not in ("base_dir", "decision_policy")
            ]
            close = difflib.get_close_matches(str(loc[-1]), known, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            issues.append(JobIssue("unknown_field", dotted, f"unknown field{hint}"))
        elif item["type"] == "missing":
            issues.append(JobIssue("missing_field", dotted, "required field is missing"))
        else:
            message = str(item["msg"]).removeprefix("Value error, ")
            given = item.get("input")
            if isinstance(given, (str, int, float, bool)):
                message = f"{message} (got {given!r})"
            issues.append(JobIssue("invalid_value", dotted, message))
    return issues


def parse_job(data: Any, *, path: str | Path | None = None) -> JobSpec:
    """Validate a job mapping strictly. Raises `JobValidationError`."""
    if not isinstance(data, dict):
        raise JobValidationError(path, [JobIssue("invalid_value", "", "a job must be a mapping")])
    for internal in ("base_dir", "decision_policy"):
        if internal in data:
            hint = (
                "set by load_job; not a job-file field"
                if internal == "base_dir"
                else "not a job-file field; a decider job's thresholds go under policy:"
            )
            raise JobValidationError(path, [JobIssue("unknown_field", internal, hint)])
    if data.get("kind") == "cluster":
        raise JobValidationError(
            path,
            [
                JobIssue(
                    "wrong_job_kind",
                    "kind",
                    "this is a clustering job (kind: cluster); run it with `xwalk cluster`",
                )
            ],
        )
    neither = "llm" not in data and "decider" not in data
    try:
        return JobSpec.model_validate(data)
    except ValidationError as exc:
        issues = _issues_from(exc)
        if neither:
            # Named at `llm`, with the alternative; the model-level "exactly one" check
            # says the same thing and is dropped.
            issues = [
                JobIssue(
                    "missing_field",
                    "llm",
                    "required field is missing; a job needs exactly one of `llm:` or `decider:`",
                ),
                *(i for i in issues if "exactly one of" not in i.message),
            ]
        raise JobValidationError(path, issues) from None


def load_job_yaml(path: Path, text: str) -> Any:
    """Parse a job file's text strictly (no duplicate keys). Raises `JobValidationError`."""
    try:
        return load_strict_yaml(text)
    except yaml.YAMLError as exc:
        raise JobValidationError(
            path, [JobIssue("job_yaml_invalid", "", yaml_error_message(exc))]
        ) from None


def load_job(path: str | Path) -> JobSpec:
    """Read and strictly validate a job file. Raises `JobValidationError`."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        reason = "not found" if isinstance(exc, FileNotFoundError) else str(exc)
        raise JobValidationError(
            path, [JobIssue("job_not_found", "", f"cannot read {path}: {reason}")]
        ) from None
    data = load_job_yaml(path, text)
    spec = parse_job(data, path=path)
    return spec.model_copy(update={"base_dir": path.parent.resolve()})

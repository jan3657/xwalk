"""A job spec is serialized constructor arguments -- nothing more.

Every field maps to a parameter the README passes by hand. No config field gates
behaviour the SDK cannot express, and credentials are named environment variables,
never values in the file.

Paths inside a job file resolve against **the job file's own directory**, so a job
directory is movable as a unit.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from xwalk.batch import build_decision_run_fingerprint, build_run_fingerprint
from xwalk.decide.base import DecisionClient
from xwalk.decide.jev import JevClient
from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.decide.questions import QuestionSet
from xwalk.llm.base import LLMClient
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet, load_slots
from xwalk.records import Record
from xwalk.retrieval.base import Retriever
from xwalk.retrieval.bm25 import MAX_FUZZY_DISTANCE, BM25Retriever
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


class TemplateSpec(BaseModel):
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


class RecordSpec(BaseModel):
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


class RetrieverSpec(BaseModel):
    kind: Literal["bm25", "dense"]
    name: str | None = None
    limit: int = 20
    # bm25 only
    exact_fields: list[str] | None = None
    analyzer: Literal["default", "en_stem"] = "default"
    fuzzy_distance: int = Field(default=0, ge=0, le=MAX_FUZZY_DISTANCE)
    # dense only
    model: str | None = None
    device: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""

    @model_validator(mode="after")
    def _check(self) -> RetrieverSpec:
        if self.kind == "dense" and not self.model:
            raise ValueError("a dense retriever needs a model name")
        return self


class LLMSpec(BaseModel):
    kind: Literal["openai_compat", "litellm"] = "openai_compat"
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None  # present only so it can be rejected
    profile: str = "unknown"
    temperature: float = 0.0
    max_tokens: int = 1024

    @model_validator(mode="after")
    def _no_inline_secrets(self) -> LLMSpec:
        if self.api_key is not None:
            raise ValueError(
                "api_key must not appear in a job file; use api_key_env with the name of "
                "an environment variable"
            )
        if self.kind == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat needs a base_url")
        return self


class DeciderSpec(BaseModel):
    kind: Literal["jev"] = "jev"
    model: str
    base_url: str = "https://openrouter.ai/api/alpha/decisions"
    api_key_env: str | None = None
    api_key: str | None = None  # present only so it can be rejected
    timeout: float = 60.0
    max_retries: int = 5

    @model_validator(mode="after")
    def _no_inline_secrets(self) -> DeciderSpec:
        if self.api_key is not None:
            raise ValueError(
                "api_key must not appear in a job file; use api_key_env with the name of "
                "an environment variable"
            )
        return self


class DecisionPolicySpec(BaseModel):
    screen_floor: float = 0.30
    shortlist_size: int = 15
    shortlist_floor: float = 0.20
    none_at: float = 0.70
    choose_at: float = 0.50
    accept_at: float = 0.85
    rubric_floor: float | None = None
    property_floor: float = 0.50
    chunk_size: int = 50
    max_candidates: int = 300
    concurrency: int = 32
    retriever_timeout: float = 60.0


class PolicySpec(BaseModel):
    max_attempts: int = 4
    accept_at: float = 0.6
    review_floor: float = 0.4
    verify_band: tuple[float, float] | None = (0.6, 0.8)
    audit_rate: float = 0.0
    concurrency: int = 32
    legacy_id_resolution: bool = False
    retriever_timeout: float = 60.0


class SelectorSpec(BaseModel):
    max_candidates: int = 30
    max_candidate_tokens: int = 8_000


class PromptSpec(BaseModel):
    slots: str


class JobSpec(BaseModel):
    name: str
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
        # One YAML key, `policy:`, on both paths. Which model parses it depends on the path.
        if isinstance(data, dict) and data.get("decider") is not None and "policy" in data:
            data = dict(data)
            data["decision_policy"] = data.pop("policy")
        return data

    @model_validator(mode="after")
    def _exactly_one_decision_maker(self) -> JobSpec:
        if (self.llm is None) == (self.decider is None):
            raise ValueError("a job needs exactly one of `llm:` or `decider:`")
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
            target = index_dir / (spec.name or spec.kind)
            if spec.kind == "bm25":
                built.append(
                    BM25Retriever.build(
                        records,
                        templates,
                        target,
                        name=spec.name or "bm25",
                        exact_fields=tuple(
                            spec.exact_fields
                            if spec.exact_fields is not None
                            else ("label", "synonyms")
                        ),
                        default_limit=spec.limit,
                        analyzer=spec.analyzer,
                        fuzzy_distance=spec.fuzzy_distance,
                    )
                )
            else:
                from xwalk.retrieval.dense import DenseRetriever, SentenceTransformerEncoder

                encoder = SentenceTransformerEncoder(
                    spec.model or "",
                    device=spec.device,
                    query_prefix=spec.query_prefix,
                    doc_prefix=spec.doc_prefix,
                )
                built.append(
                    DenseRetriever.build(
                        records,
                        templates,
                        target,
                        encoder,
                        name=spec.name,
                        default_limit=spec.limit,
                    )
                )
        return built

    def build_llm(self) -> LLMClient:
        if self.llm is None:
            raise ValueError("this job has no llm: block")
        api_key = None
        if self.llm.api_key_env:
            api_key = os.environ.get(self.llm.api_key_env)
            if not api_key:
                raise ValueError(
                    f"environment variable {self.llm.api_key_env} is not set; "
                    f"export it or change llm.api_key_env in the job file"
                )
        if self.llm.kind == "openai_compat":
            return OpenAICompatClient(
                base_url=self.llm.base_url or "",
                model=self.llm.model,
                api_key=api_key,
                profile=self.llm.profile,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
            )
        from xwalk.llm.litellm import LiteLLMClient

        return LiteLLMClient(
            self.llm.model,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        )

    def build_prompts(self) -> PromptSet:
        return PromptSet.from_slots(load_slots(self.base_dir / self.prompts.slots))

    def build_decider(self) -> DecisionClient:
        if self.decider is None:
            raise ValueError("this job has no decider: block")
        api_key = None
        if self.decider.api_key_env:
            api_key = os.environ.get(self.decider.api_key_env)
            if not api_key:
                raise ValueError(
                    f"environment variable {self.decider.api_key_env} is not set; "
                    f"export it or change decider.api_key_env in the job file"
                )
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

    def run_fingerprint(
        self,
        *,
        store: TargetStore,
        retrievers: Sequence[Retriever],
        llm: LLMClient,
        prompts: PromptSet | None = None,
    ) -> str:
        return build_run_fingerprint(
            templates=self.build_templates(),
            prompts=prompts or self.build_prompts(),
            store=store,
            retrievers=retrievers,
            llm=llm,
            policy=self.build_policy(),
            selector_policy=self.build_selector_policy(),
        )

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
            retriever_limit=max(spec.limit for spec in self.retrievers),
        )

    def decision_run_fingerprint(
        self, *, store: TargetStore, retrievers: Sequence[Retriever], decider: DecisionClient
    ) -> str:
        return build_decision_run_fingerprint(
            templates=self.build_templates(),
            questions=self.build_questions(),
            store=store,
            retrievers=retrievers,
            decider=decider,
            policy=self.build_decision_policy(),
        )

    def build_decision_matcher(
        self, *, store: TargetStore, retrievers: Sequence[Retriever], decider: DecisionClient
    ) -> DecisionMatcher:
        templates = self.build_templates()
        questions = self.build_questions()
        policy = self.build_decision_policy()
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
        )


def load_job(path: str | Path) -> JobSpec:
    path = Path(path)
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = JobSpec.model_validate(data)
    return spec.model_copy(update={"base_dir": path.parent.resolve()})

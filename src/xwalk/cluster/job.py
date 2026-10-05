"""The clustering job file (`kind: cluster`).

Validated as strictly as a matching job (CONTRACTS.md section 8): unknown keys fail with a
"did you mean", numbers are range-checked, and validation never calls a model. Record,
LLM and dense-encoder sections are the matching job's own specs, so they read the same.

    kind: cluster
    name: food-labels
    source: {kind: csv, path: labels.csv}
    templates:
      query: "{{ label }}"          # retrieval text and the ordering key
      context: "{{ label }} ({{ category }})"   # the record as the model sees it
      candidate: "{{ label }}"      # one member line inside a cluster
    llm: {kind: openai_compat, base_url: ..., model: ..., api_key_env: ...}
    relation: "Two labels are equivalent when ..."   # optional domain brief
    order: label                    # or input
    pool: {retrieve_limit: 20, shown_limit: 8, ...}
    policy: {assign_accept_at: 0.7, max_refine_iterations: 2, ...}
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from xwalk.cluster.prompts import DEFAULT_RELATION, ClusterPrompts
from xwalk.cluster.settings import ClusterPolicy, ClusterSettings, PoolSettings
from xwalk.config import (
    JobIssue,
    JobSpec,
    JobValidationError,
    LLMSpec,
    RecordSpec,
    RetrieverSpec,
    _issues_from,
    _Spec,
    load_job_yaml,
)
from xwalk.llm.base import LLMClient
from xwalk.records import Record
from xwalk.retrieval.dense import Encoder
from xwalk.templates import TemplateSet


class ClusterTemplateSpec(_Spec):
    query: str
    context: str = ""
    candidate: str = ""

    def build(self) -> TemplateSet:
        context = self.context or self.query
        return TemplateSet(
            query=self.query,
            context=context,
            doc=self.query,
            candidate=self.candidate or context,
        )


_POOL_DEFAULTS = PoolSettings()
_POLICY_DEFAULTS = ClusterPolicy()


class ClusterPoolSpec(_Spec):
    retrieve_limit: int = Field(default=_POOL_DEFAULTS.retrieve_limit, ge=1)
    shown_limit: int = Field(default=_POOL_DEFAULTS.shown_limit, ge=1, le=50)
    expand_limit: int = Field(default=_POOL_DEFAULTS.expand_limit, ge=1)
    max_expansion_pages: int = Field(default=_POOL_DEFAULTS.max_expansion_pages, ge=0, le=20)
    scan_below: int = Field(default=_POOL_DEFAULTS.scan_below, ge=0)
    member_evidence: int = Field(default=_POOL_DEFAULTS.member_evidence, ge=1, le=50)
    evidence_chars: int = Field(default=_POOL_DEFAULTS.evidence_chars, ge=20)
    mint_requires: Literal["pool_exhausted", "retrieval_exhausted", "bounded"] = (
        _POOL_DEFAULTS.mint_requires  # type: ignore[assignment]
    )

    @model_validator(mode="after")
    def _check(self) -> ClusterPoolSpec:
        if self.expand_limit < self.retrieve_limit:
            raise ValueError("expand_limit must be >= retrieve_limit")
        return self


class ClusterPolicySpec(_Spec):
    verify_assignments: bool = _POLICY_DEFAULTS.verify_assignments
    assign_accept_at: float = Field(default=_POLICY_DEFAULTS.assign_accept_at, ge=0.0, le=1.0)
    assign_review_floor: float = Field(default=_POLICY_DEFAULTS.assign_review_floor, ge=0.0, le=1.0)
    novelty_accept_at: float = Field(default=_POLICY_DEFAULTS.novelty_accept_at, ge=0.0, le=1.0)
    merge_accept_at: float = Field(default=_POLICY_DEFAULTS.merge_accept_at, ge=0.0, le=1.0)
    merge_review_floor: float = Field(default=_POLICY_DEFAULTS.merge_review_floor, ge=0.0, le=1.0)
    comparative_accept_at: float = Field(
        default=_POLICY_DEFAULTS.comparative_accept_at, ge=0.0, le=1.0
    )
    max_refine_iterations: int = Field(default=_POLICY_DEFAULTS.max_refine_iterations, ge=0, le=20)
    consolidate: bool = _POLICY_DEFAULTS.consolidate
    reassign: bool = _POLICY_DEFAULTS.reassign

    @model_validator(mode="after")
    def _check(self) -> ClusterPolicySpec:
        if self.assign_review_floor > self.assign_accept_at:
            raise ValueError("assign_review_floor must not exceed assign_accept_at")
        if self.merge_review_floor > self.merge_accept_at:
            raise ValueError("merge_review_floor must not exceed merge_accept_at")
        return self


class ClusterJobSpec(_Spec):
    kind: Literal["cluster"]
    name: str = Field(min_length=1)
    source: RecordSpec
    templates: ClusterTemplateSpec
    llm: LLMSpec
    relation: str = Field(default=DEFAULT_RELATION, min_length=1)
    order: Literal["label", "input"] = "label"
    pool: ClusterPoolSpec = ClusterPoolSpec()
    policy: ClusterPolicySpec = ClusterPolicySpec()
    dense: RetrieverSpec | None = None

    base_dir: Path = Path(".")

    @model_validator(mode="after")
    def _check(self) -> ClusterJobSpec:
        if self.dense is not None and self.dense.kind != "dense":
            raise ValueError("dense must describe a dense encoder (kind: dense)")
        return self

    def build_templates(self) -> TemplateSet:
        return self.templates.build()

    def build_source_records(self) -> Iterator[Record]:
        return self.source.build(self.base_dir)

    def build_settings(self) -> ClusterSettings:
        return ClusterSettings(
            order=self.order,
            pool=PoolSettings(**self.pool.model_dump()),
            policy=ClusterPolicy(**self.policy.model_dump()),
        )

    def build_prompts(self) -> ClusterPrompts:
        return ClusterPrompts(relation=self.relation)

    def build_encoder(self) -> Encoder | None:
        return None if self.dense is None else JobSpec.build_encoder(self.dense)

    def build_llm(self) -> LLMClient:
        """The job's client. Reuses the matching job's construction (credentials too)."""
        shim = JobSpec.model_construct(llm=self.llm)
        return shim.build_llm()


def parse_cluster_job(data: Any, *, path: str | Path | None = None) -> ClusterJobSpec:
    if not isinstance(data, dict):
        raise JobValidationError(path, [JobIssue("invalid_value", "", "a job must be a mapping")])
    if "base_dir" in data:
        raise JobValidationError(
            path, [JobIssue("unknown_field", "base_dir", "set by the loader; not a job-file field")]
        )
    try:
        return ClusterJobSpec.model_validate(data)
    except ValidationError as exc:
        raise JobValidationError(path, _issues_from(exc, ClusterJobSpec)) from None


def is_cluster_job(path: str | Path) -> bool:
    """True when the file is readable YAML declaring `kind: cluster`. Never raises."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(data, dict) and data.get("kind") == "cluster"


def load_cluster_job(path: str | Path) -> ClusterJobSpec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        reason = "not found" if isinstance(exc, FileNotFoundError) else str(exc)
        raise JobValidationError(
            path, [JobIssue("job_not_found", "", f"cannot read {path}: {reason}")]
        ) from None
    data = load_job_yaml(path, text)
    spec = parse_cluster_job(data, path=path)
    return spec.model_copy(update={"base_dir": path.parent.resolve()})


__all__ = [
    "ClusterJobSpec",
    "ClusterPolicySpec",
    "ClusterPoolSpec",
    "ClusterTemplateSpec",
    "is_cluster_job",
    "load_cluster_job",
    "parse_cluster_job",
]

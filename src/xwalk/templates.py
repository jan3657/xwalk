"""The four templates that carry the entire domain mapping.

| Template    | Input         | Produces                        |
|-------------|---------------|---------------------------------|
| `query`     | source record | retrieval query string          |
| `context`   | source record | context block shown to the LLM  |
| `doc`       | target record | indexed text                    |
| `candidate` | target record | one entry in the candidate list |
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, Template, Undefined
from jinja2 import TemplateSyntaxError as JinjaSyntaxError

from xwalk.fingerprint import hash_value
from xwalk.records import Record


class TemplateError(Exception):
    """A template failed to compile or to render."""


class _EmptyUndefined(Undefined):
    """Missing fields render as an empty string.

    Source collections are sparse in practice. Aborting a 100k-row run because one row
    lacks an optional column is the wrong trade; a silently-empty slot is visible in the
    rendered output and in the attempt trace.
    """

    def __str__(self) -> str:
        return ""


_BLANK_LINES = re.compile(r"\n\s*\n+")
_SPACES = re.compile(r"[ \t]+")


def _tidy(text: str) -> str:
    """Collapse the whitespace that conditionals leave behind."""
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    joined = "\n".join(line for line in lines if line)
    return _BLANK_LINES.sub("\n", joined).strip()


@dataclass(frozen=True)
class TemplateSet:
    """Holds template *source*; compiles on construction so typos fail fast."""

    query: str
    context: str
    doc: str
    candidate: str
    _compiled: dict[str, Template] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        env = Environment(undefined=_EmptyUndefined, keep_trailing_newline=False)
        for name in ("query", "context", "doc", "candidate"):
            source = getattr(self, name)
            try:
                self._compiled[name] = env.from_string(source)
            except JinjaSyntaxError as exc:
                raise TemplateError(f"{name} template failed to compile: {exc}") from exc

    def _render(self, name: str, record: Record) -> str:
        variables: dict[str, Any] = dict(record.fields)
        variables["id"] = record.id  # record identity always wins over a field named "id"
        try:
            return _tidy(self._compiled[name].render(**variables))
        except Exception as exc:  # noqa: BLE001 - any Jinja runtime error is a template error
            raise TemplateError(f"{name} template failed on record {record.id!r}: {exc}") from exc

    def render_query(self, record: Record) -> str:
        return self._render("query", record)

    def render_context(self, record: Record) -> str:
        return self._render("context", record)

    def render_doc(self, record: Record) -> str:
        return self._render("doc", record)

    def render_candidate(self, record: Record) -> str:
        return self._render("candidate", record)

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "query": self.query,
                "context": self.context,
                "doc": self.doc,
                "candidate": self.candidate,
            }
        )

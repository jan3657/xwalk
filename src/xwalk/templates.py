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
from functools import lru_cache
from typing import Any

from jinja2 import Environment, Template, Undefined, nodes
from jinja2 import TemplateSyntaxError as JinjaSyntaxError
from jinja2.exceptions import SecurityError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from xwalk.fingerprint import hash_value
from xwalk.records import Record

# Bumped whenever the same template source can render differently for the same record.
# 2: null fields render as missing ones (they used to render as "None" or raise), in
#    these templates, in the base prompts' source block and in the decider's source state.
TEMPLATE_RENDER_VERSION = 2


def has_null_field(record: Record) -> bool:
    """Whether `record` renders differently under render version 2 than before it.

    Index identities add `template_render_version` only when some target has a null
    field, so indexes whose text cannot have changed keep the identity they had.
    """
    return any(value is None for value in record.fields.values())


class TemplateError(Exception):
    """A template failed to compile or to render."""


class UnsafeTemplateError(TemplateError):
    """A template reaches for Python internals (a dunder attribute) and is refused."""


class _EmptyUndefined(Undefined):
    """Missing fields render as an empty string.

    Source collections are sparse in practice. Aborting a 100k-row run because one row
    lacks an optional column is the wrong trade; a silently-empty slot is visible in the
    rendered output and in the attempt trace.

    The sandbox reports an attribute it refuses as an undefined too; that one raises
    instead of rendering as an empty slot, so a refused template is never mistaken for
    a sparse row.
    """

    def __str__(self) -> str:
        if self._undefined_exception is SecurityError:
            self._fail_with_undefined_error()
        return ""


def checked_parse(env: Environment, source: str) -> nodes.Template:
    """Parse `source`, refusing a constant dunder in attribute or subscript position.

    Templates come from job files, slots directories and the workbench: untrusted
    input. Every environment that compiles them is a sandbox, which refuses unsafe
    attribute access when the template runs. This check is the earlier, clearer half:
    `''.__class__` never reads a record field, so `xwalk validate` names it before any
    record is rendered, even in a branch no record would take. A computed name
    (`attr(...)`, `x[key]`) is left to the sandbox.
    """
    tree = env.parse(source)
    reads: list[tuple[object, int]] = [(n.attr, n.lineno) for n in tree.find_all(nodes.Getattr)]
    reads += [
        (n.arg.value, n.lineno)
        for n in tree.find_all(nodes.Getitem)
        if isinstance(n.arg, nodes.Const)
    ]
    for name, line in reads:
        if isinstance(name, str) and name.startswith("__"):
            raise UnsafeTemplateError(
                f"it reads {name!r} on line {line}; templates may not reach Python internals"
            )
    return tree


# Immutable: a template renders text, so it has no business calling `list.append` or
# `dict.update` on a record's fields.
_ENV = ImmutableSandboxedEnvironment(undefined=_EmptyUndefined, keep_trailing_newline=False)


@lru_cache(maxsize=256)
def _compile(source: str) -> Template:
    # A job's templates are built more than once per process (validate, then the run;
    # every workbench request), and compiling is the expensive part of a render.
    return _ENV.from_string(checked_parse(_ENV, source))


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
    # Replaces `query` when configured; Matcher preserves the historical single-entry
    # alternate alongside a distinct `query`.
    queries: tuple[str, ...] = ()
    _compiled: dict[str, Template] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        named = [
            (name, name, getattr(self, name)) for name in ("query", "context", "doc", "candidate")
        ]
        named += [(f"queries.{i}", f"queries[{i}]", q) for i, q in enumerate(self.queries)]
        for key, label, source in named:
            try:
                self._compiled[key] = _compile(source)
            except JinjaSyntaxError as exc:
                raise TemplateError(f"{label} template failed to compile: {exc}") from exc
            except UnsafeTemplateError as exc:
                raise UnsafeTemplateError(f"{label} template is unsafe: {exc}") from None

    def _render(self, name: str, record: Record) -> str:
        # A null field is a missing field: omitted, so it reaches the template as
        # undefined and `default`, `is defined` and `join` treat it as absent instead of
        # rendering the text "None" or raising. Bumps TEMPLATE_RENDER_VERSION.
        variables: dict[str, Any] = {k: v for k, v in record.fields.items() if v is not None}
        variables["id"] = record.id  # record identity always wins over a field named "id"
        try:
            return _tidy(self._compiled[name].render(**variables))
        except Exception as exc:  # noqa: BLE001 - any Jinja runtime error is a template error
            raise TemplateError(f"{name} template failed on record {record.id!r}: {exc}") from exc

    def render_query(self, record: Record) -> str:
        return self._render("query", record)

    def render_queries(self, record: Record) -> list[str]:
        """Every declared query for this record, in order, non-empty, de-duplicated.

        Falls back to the single `query` template so the two paths share one call site.
        """
        if not self.queries:
            return [self.render_query(record)]
        rendered: list[str] = []
        for index in range(len(self.queries)):
            text = self._render(f"queries.{index}", record)
            if text and text not in rendered:
                rendered.append(text)
        return rendered

    def render_context(self, record: Record) -> str:
        return self._render("context", record)

    def render_doc(self, record: Record) -> str:
        return self._render("doc", record)

    def render_candidate(self, record: Record) -> str:
        return self._render("candidate", record)

    @property
    def fingerprint(self) -> str:
        identity: dict[str, Any] = {
            # Rendering semantics are part of what a template means: results rendered
            # under an older version must not be resumed as if nothing changed.
            "render_version": TEMPLATE_RENDER_VERSION,
            "query": self.query,
            "context": self.context,
            "doc": self.doc,
            "candidate": self.candidate,
        }
        # Only when declared, so a template set without extra queries keeps the
        # fingerprint (and the run identity) it had before they existed.
        if self.queries:
            identity["queries"] = list(self.queries)
        return hash_value(identity)

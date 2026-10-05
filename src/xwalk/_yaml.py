"""Strict YAML loading shared by job files and prompt slots.

`StrictLoader` is a `yaml.SafeLoader` that refuses a mapping key given twice, and
`yaml_error_message` reports a YAML error by position without quoting the file.
"""

from __future__ import annotations

from typing import Any

import yaml


class StrictLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` that refuses a mapping with the same key twice.

    Plain YAML loading keeps the last value, so `accept_at` written twice would load
    silently with whichever came last -- the same failure `extra="forbid"` prevents for
    a misspelled key. Keys brought in by a `<<` merge may still be overridden.
    """


def _construct_unique_mapping(loader: StrictLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    first_line: dict[Any, int] = {}
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=True)
        try:
            earlier = first_line.get(key)
        except TypeError:  # an unhashable key; construct_mapping reports it
            continue
        if earlier is not None:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"duplicate key {key!r} (first given on line {earlier})",
                key_node.start_mark,
            )
        first_line[key] = key_node.start_mark.line + 1
    return loader.construct_mapping(node, deep=True)


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def yaml_error_message(exc: yaml.YAMLError) -> str:
    """The problem and its position, without echoing the file's content back.

    PyYAML's own message quotes the offending line; for a path that is not a job file
    (an MCP client can name any path) that would disclose part of an unrelated file.
    """
    if isinstance(exc, yaml.MarkedYAMLError) and exc.problem:
        mark = exc.problem_mark
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        context = f" ({exc.context})" if exc.context else ""
        return f"{exc.problem}{where}{context}"
    return type(exc).__name__


def load_strict_yaml(text: str) -> Any:
    """Parse YAML safely, refusing duplicate mapping keys. Raises `yaml.YAMLError`."""
    return yaml.load(text, Loader=StrictLoader)  # noqa: S506 - a SafeLoader subclass


__all__ = ["StrictLoader", "load_strict_yaml", "yaml_error_message"]

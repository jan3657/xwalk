"""Turning whatever the model said into a JSON object.

Three behaviours port over from the paper repo because they were learned the hard way:
thinking-block stripping (including the dangling-closing-tag case), fenced-block
extraction, and modest repair. Anything these cannot salvage becomes UNRESOLVED_OUTPUT
and routes to human review — a malformed answer is a signal, not a non-match.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from xwalk.llm.base import ParseError

_TAGS = ("think", "thinking", "reasoning")
_PAIRED = re.compile(r"<(" + "|".join(_TAGS) + r")\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_DANGLING_CLOSE = re.compile(r"^.*?</(?:" + "|".join(_TAGS) + r")\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_OPEN = re.compile(r"<(?:" + "|".join(_TAGS) + r")\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks, closed or not."""
    out = _PAIRED.sub("", text)
    if re.search(r"</(?:" + "|".join(_TAGS) + r")\s*>", out, re.IGNORECASE):
        # A closing tag with no opener: everything up to and including it was reasoning.
        out = _DANGLING_CLOSE.sub("", out)
    out = _UNCLOSED_OPEN.sub("", out)  # truncated output: an opener with no close
    return out.strip()


def extract_json_object(text: str) -> str | None:
    """Find the first balanced JSON object, preferring a fenced block."""
    fenced = _FENCE.search(text)
    haystack = fenced.group(1) if fenced else text

    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(haystack):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                return haystack[start : i + 1]
    return None


def repair_json(text: str) -> str:
    """Only repairs that cannot change meaning. Trailing commas, and nothing more."""
    return _TRAILING_COMMA.sub(r"\1", text)


def parse_json_object(raw: str) -> dict[str, Any]:
    """strip thinking -> direct parse -> extract -> repair. Raise ParseError if all fail."""
    cleaned = strip_thinking(raw)
    for attempt in (cleaned, extract_json_object(cleaned), repair_json(cleaned)):
        if not attempt:
            continue
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            raise ParseError(f"expected a JSON object, got {type(value).__name__}", raw=raw)
        return value

    extracted = extract_json_object(cleaned)
    if extracted:
        try:
            repaired = json.loads(repair_json(extracted))
        except json.JSONDecodeError:
            repaired = None
        if isinstance(repaired, dict):
            return repaired
    raise ParseError("could not extract a JSON object from the response", raw=raw)


def parse_confidence(value: object) -> tuple[float | None, str | None]:
    """Validate a model-reported `confidence_score`. Returns `(score, None)` or `(None, why)`.

    Only a finite JSON number in [0, 1] is a confidence. Everything else is invalid
    output, and invalid output is a signal for review, never a certainty:

    - missing or null -> invalid
    - a string (even "0.9"), a boolean, a list -> invalid; nothing is coerced
    - NaN, Infinity, -Infinity (which Python's json module accepts) -> invalid
    - a finite number outside [0, 1] -> invalid, not clamped: 1.7 says the model did
      not follow the scale, and clamping it would turn that into a perfect score
    """
    if value is None:
        return None, "confidence_score was missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"confidence_score must be a number, got {type(value).__name__}"
    try:
        number = float(value)
    except OverflowError:
        return None, "confidence_score is not a finite number"
    if not math.isfinite(number):
        return None, f"confidence_score is not a finite number ({number!r})"
    if not 0.0 <= number <= 1.0:
        return None, f"confidence_score {number!r} is outside [0, 1]"
    return number, None

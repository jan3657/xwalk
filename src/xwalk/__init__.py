"""xwalk — LLM-RAG record matching between two collections."""

# __version__ is defined FIRST, before any submodule import. `batch.py` does
# `from xwalk import __version__`, so if a batch name were re-exported below while
# __version__ was still unbound, importing xwalk would raise
# "cannot import name '__version__' from partially initialized module".
#
# The version lives in pyproject.toml and is read back from installed distribution
# metadata. A literal here as well is a second source of truth, and two of them is how a
# release ends up mislabelled. The fallback covers a source checkout that was never
# installed -- in which case there is no metadata to read.
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("xwalk")
except PackageNotFoundError:  # pragma: no cover - only in an uninstalled checkout
    __version__ = "0.0.0.dev0+unknown"

from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    RetryProposal,
    Usage,
)
from xwalk.templates import TemplateSet

__all__ = [
    "Attempt",
    "Candidate",
    "DecisionMatcher",
    "DecisionPolicy",
    "DecisionReason",
    "MatchPolicy",
    "MatchResult",
    "MatchStatus",
    "Matcher",
    "Record",
    "RetrievalHit",
    "RetryProposal",
    "TemplateSet",
    "Usage",
    "__version__",
]

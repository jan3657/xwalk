"""xwalk — LLM-RAG record matching between two collections."""

# __version__ is defined FIRST, before any submodule import. `batch.py` does
# `from xwalk import __version__`, so if a batch name were re-exported below while
# __version__ was still unbound, importing xwalk would raise
# "cannot import name '__version__' from partially initialized module".
__version__ = "0.1.0.dev0"

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

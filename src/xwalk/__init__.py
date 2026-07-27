"""xwalk — LLM-RAG record matching between two collections."""

# Defined before any submodule import and kept that way: `batch.py` (Task 18) does
# `from xwalk import __version__`, which fails if a submodule import above this line
# ever pulls `xwalk.batch` back in while __version__ is still unbound.
__version__ = "0.1.0.dev0"

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

__all__ = [
    "Attempt",
    "Candidate",
    "DecisionReason",
    "MatchPolicy",
    "MatchResult",
    "MatchStatus",
    "Record",
    "RetrievalHit",
    "RetryProposal",
    "Usage",
    "__version__",
]

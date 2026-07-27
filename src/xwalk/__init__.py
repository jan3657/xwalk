"""xwalk — LLM-RAG record matching between two collections."""

# Defined before any submodule import and kept that way: `batch.py` (Task 18) does
# `from xwalk import __version__`, which fails if a submodule import above this line
# ever pulls `xwalk.batch` back in while __version__ is still unbound.
__version__ = "0.1.0.dev0"

from xwalk.records import Candidate, Record, RetrievalHit, Usage

__all__ = ["Candidate", "Record", "RetrievalHit", "Usage", "__version__"]

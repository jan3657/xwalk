"""Reading a completed run and answering "is this good, and where should I spend effort?".

Nothing in this package calls an LLM or a retriever. It reads the ledger, so evaluation
is free, repeatable, and safe to run on a machine with no credentials.
"""

from xwalk.evaluate.gold import GoldSet, load_gold_csv, load_gold_jsonl

__all__ = ["GoldSet", "load_gold_csv", "load_gold_jsonl"]

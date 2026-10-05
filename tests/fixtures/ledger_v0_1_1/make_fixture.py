"""Regenerate the v0.1.1 ledger fixture. Run it against the 0.1.1 code, never HEAD.

    git worktree add /tmp/xwalk-v011 f737099
    cd /tmp/xwalk-v011
    PYTHONPATH=src:. python <repo>/tests/fixtures/ledger_v0_1_1/make_fixture.py <out_dir>

The fixture is the point of the exercise: a ledger written by released code, kept
byte-for-byte, so the current code is tested against what users actually have on disk.
Scenario (run fingerprint ``v011fp``, one attempt per record):

- s1 "glucose" is matched to T1, then edited (a ``note`` field) and resumed, so the
  ledger holds two rows for s1: the old version and the new one.
- s2 "sugar" scores 0.5 and needs review; a reviewer accepts T2.
- s3 "unobtainium" fails: every selector call raises a retryable provider error.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from tests.test_matcher import (
    PROMPTS,
    STORE,
    TEMPLATES,
    ScriptedRetriever,
    score_reply,
    select_reply,
)
from xwalk.batch import run_batch
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import Record
from xwalk.review import ReviewDecision, ReviewRow, apply_review
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector

RETRIEVER = {"glucose": ["T1"], "sugar": ["T2"], "unobtainium": ["T3"]}


def handler(request):
    if "unobtainium" in request.user:
        raise LLMRetryableError("provider unavailable")
    if "## Candidates" in request.user:
        return select_reply("C01")
    return score_reply(0.5 if "sugar" in request.user else 0.95)


def matcher() -> Matcher:
    llm = FakeLLM(handler=handler)
    return Matcher(
        templates=TEMPLATES,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=MatchPolicy(max_attempts=1, verify_band=None, concurrency=1),
        run_fingerprint="v011fp",
    )


async def main(out: Path) -> None:
    run = out / "rundir"
    if run.exists():
        shutil.rmtree(run)
    sources = [
        Record(id="s1", fields={"mention": "glucose"}),
        Record(id="s2", fields={"mention": "sugar"}),
        Record(id="s3", fields={"mention": "unobtainium"}),
    ]
    await run_batch(matcher(), sources, out=run, manifest_extra={"job": "fixture"})

    ledger = Ledger.open(run / "ledger.sqlite")
    s2 = next(r for r in ledger.iter_results("v011fp") if r.source_id == "s2")
    apply_review(
        ledger,
        [
            ReviewRow(
                result_key=s2.result_key,
                run_fingerprint="v011fp",
                source_id="s2",
                source_hash=s2.source_hash,
                proposed_target_id=s2.matched_id,
                decision=ReviewDecision.ACCEPT,
                corrected_target_id=None,
                reviewer="jan",
                review_note="sugar here means glucose-free syrup",
                reviewed_at="2026-09-01T10:00:00Z",
            )
        ],
        target_store_fingerprint=STORE.fingerprint,
    )
    ledger.close()

    edited = [Record(id="s1", fields={"mention": "glucose", "note": "edited"}), *sources[1:]]
    await run_batch(matcher(), edited, out=run, manifest_extra={"job": "fixture"})


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))

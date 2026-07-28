"""Match chemical entity mentions to ChEBI-style ontology terms.

    python -m examples.chemistry.run --out run/

Needs a live provider. Set XWALK_TEST_API_KEY, XWALK_TEST_BASE_URL, and
XWALK_TEST_MODEL (any OpenAI-compatible endpoint) before running.

Everything domain-specific lives in `slots.yaml` and `templates.yaml`. Pointing this at
a different domain means editing those two files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from xwalk import Matcher, MatchPolicy, TemplateSet
from xwalk.batch import build_run_fingerprint, run_batch_sync
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.prompts.contract import PromptSet, load_slots
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.sources.tabular import csv_source
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy
from xwalk.stores.memory import MemoryStore

HERE = Path(__file__).parent


def load_templates(path: Path) -> TemplateSet:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TemplateSet(**data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="run/", help="output directory")
    parser.add_argument("--index", default=None, help="BM25 index directory")
    parser.add_argument("--resume", action="store_true", help="skip completed records")
    args = parser.parse_args(argv)

    api_key = os.environ.get("XWALK_TEST_API_KEY")
    if not api_key:
        print(
            "XWALK_TEST_API_KEY is not set. This example calls a real provider.",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    templates = load_templates(HERE / "templates.yaml")
    targets = list(
        csv_source(HERE / "targets.csv", id_column="id", multivalue_columns=["synonyms"])
    )
    store = MemoryStore.from_source(targets)
    retriever = BM25Retriever.build(
        targets,
        templates,
        Path(args.index) if args.index else out / "index",
        exact_fields=("label", "synonyms"),
    )

    llm = OpenAICompatClient(
        base_url=os.environ.get("XWALK_TEST_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("XWALK_TEST_MODEL", "gpt-4o-mini"),
        api_key=api_key,
    )
    prompts = PromptSet.from_slots(load_slots(HERE / "slots.yaml"))
    policy, selector_policy = MatchPolicy(), SelectorPolicy()

    matcher = Matcher(
        templates=templates,
        retrievers=[retriever],
        store=store,
        selector=Selector(llm, prompts, templates, policy=selector_policy),
        scorer=Scorer(llm, prompts, templates, review_floor=policy.review_floor),
        verifier=Verifier(llm, prompts, templates),
        rewriter=QueryRewriter(llm, prompts, templates),
        policy=policy,
        run_fingerprint=build_run_fingerprint(
            templates=templates,
            prompts=prompts,
            store=store,
            retrievers=[retriever],
            llm=llm,
            policy=policy,
            selector_policy=selector_policy,
        ),
    )

    report = run_batch_sync(
        matcher,
        csv_source(HERE / "sources.csv", id_column="mention_id"),
        out=out,
        resume=args.resume,
    )
    print(f"run fingerprint : {report.run_fingerprint}")
    print(f"by status       : {report.by_status()}")
    print(f"mapping written : {out / 'mapping.csv'}")
    print()
    print("Now score it:")
    print(
        f"  python -m examples.chemistry.evaluate_run --run {out} "
        f"--fingerprint {report.run_fingerprint}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())

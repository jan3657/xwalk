#!/usr/bin/env python3
"""Focused offline diagnostic for xwalk's audited v0.1.1 source.

This AST harness runs selected real functions while replacing external providers
and dependencies. It is not the normal package test suite or a quality benchmark.
See README.md for scope and interpretation.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo", type=Path, help="Path to a compatible xwalk checkout")
repo_path = parser.parse_args().repo.resolve()

sources = {path: (repo_path / path).read_text() for path in ('src/xwalk/records.py', 'src/xwalk/fingerprint.py', 'src/xwalk/llm/base.py', 'src/xwalk/stages/keying.py', 'src/xwalk/stages/proposals.py', 'src/xwalk/policy.py', 'src/xwalk/retrieval/fusion.py', 'src/xwalk/stages/gate.py', 'src/xwalk/stages/rewrite.py', 'src/xwalk/llm/parsing.py', 'src/xwalk/matcher.py', 'src/xwalk/batch.py')}

import ast
from types import SimpleNamespace
from dataclasses import asdict
import tempfile
import json

SearchRequest = lambda **kwargs: SimpleNamespace(**kwargs)
__version__ = "0.1.1"
module_envs = {}
for path, source in sources.items():
    tree = ast.parse(source, filename=path)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("xwalk"))]
    env = dict(globals())
    exec(compile(tree, path, "exec"), env)
    module_envs[path] = env
    globals().update({k:v for k,v in env.items() if k not in ("__name__", "__builtins__", "sources", "module_envs")})

SearchRequest = lambda **kwargs: SimpleNamespace(**kwargs)
__version__ = "0.1.1"
unit = Usage(prompt_tokens=100, completion_tokens=20, calls=1)
for name in ("serde.py", "ledger.py"):
    path = "src/xwalk/" + name
    source = Path(repo_path, path).read_text()
    tree = ast.parse(source, filename=path)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("xwalk"))]
    env = dict(globals())
    exec(compile(tree, path, "exec"), env)
    module_envs[path] = env
    globals().update({k:v for k,v in env.items() if k not in ("__name__", "__builtins__", "sources", "module_envs")})
module_envs["src/xwalk/batch.py"].update(Ledger=Ledger, result_to_dict=result_to_dict)
target = Record("T1", {"label":"glucose"})
source_record = Record("s1", {"mention":"glucose"})

class TestTemplates:
    def render_query(self, source): return source.fields["mention"]
    def render_context(self, source): return ""
    def render_candidate(self, record): return record.fields["label"]

class TestStore:
    fingerprint = "test-store"
    def get(self, key): return target
    def get_many(self, keys): return [target for key in keys if key=="T1"]

class TestRetriever:
    name = "test"
    default_limit = 10
    async def search(self, request):
        return [RetrievalHit("T1","test",1.0,1)]

class TestSelector:
    async def select(self, source, context, candidates):
        keyed = assign_keys(candidates, TestTemplates())
        return SimpleNamespace(
            usage=unit, finish_reason="stop", keyed=keyed, truncated=0,
            choice=ResolvedChoice("T1", Resolution.EXACT_KEY, "C01"),
            raw="C01", explanation="", error=None,
        )

class TestScorer:
    def __init__(self, scores): self.scores = iter(scores)
    async def score(self, *args):
        return ScoreOutcome(next(self.scores), "", (), "", unit, finish_reason="stop")

class TestVerifier:
    def __init__(self, error=None): self.error=error
    async def verify(self, *args):
        if self.error: raise self.error
        return VerifierVerdict("support", None, 1.0, "", "", unit, finish_reason="stop")

class TestRewriter:
    def __init__(self, error=None): self.error=error
    async def rewrite(self, *args):
        if self.error: raise self.error
        return RewriteOutcome((RetryProposal("query","dextrose","rewriter"),), "", "", unit)

def build(scores, verifier_error=None, rewrite_error=None):
    return Matcher(
        templates=TestTemplates(), retrievers=[TestRetriever()], store=TestStore(),
        selector=TestSelector(), scorer=TestScorer(scores),
        verifier=TestVerifier(verifier_error), rewriter=TestRewriter(rewrite_error),
        policy=MatchPolicy(max_attempts=2), run_fingerprint="probe",
    )

async def probes():
    results = {}
    raw_score = parse_json_object('{"confidence_score": NaN}')["confidence_score"]
    score = _clamp(raw_score)
    result = await build([score]).match(source_record)
    results["non_finite_confidence"] = {"clamped_nan":score,"result_status":result.status.value,"confidence":result.confidence}

    result = await build([0.3, 0.95]).match(source_record)
    results["rewrite_cost_omitted"] = {"expected_calls":5,"reported_calls":result.usage.calls,"expected_tokens":600,"reported_tokens":result.usage.total_tokens,"attempts":len(result.attempts)}

    try:
        await build([0.7], verifier_error=LLMRetryableError("429 exhausted")).match(source_record)
    except Exception as exc:
        results["verifier_failure_escapes"] = {"exception_type":type(exc).__name__,"exception":str(exc)}

    try:
        await build([0.3], rewrite_error=LLMFatalError("invalid api key")).match(source_record)
    except Exception as exc:
        results["rewriter_failure_escapes"] = {"exception_type":type(exc).__name__,"exception":str(exc)}

    keyed = KeyedCandidates(("C01","C02"), {"C01":object(),"C02":object()}, {"C01":"T1","C02":"T2"}, "[C01] label A\n\ncritical detail A\n\n[C02] label B")
    chosen, others = _blocks(keyed, "C01")
    results["multiline_candidate_corrupted"] = {"chosen":chosen,"others":others,"critical_detail_in_chosen":"critical detail A" in chosen}

    class CandidateRetryScorer:
        def __init__(self): self.calls=0
        async def score(self, *args):
            self.calls += 1
            if self.calls == 1:
                return ScoreOutcome(0.3, "", (RetryProposal("candidate","C02","scorer"),), "", unit, finish_reason="stop")
            return ScoreOutcome(0.95, "", (), "", unit, finish_reason="stop")
    class TwoCandidateStore:
        fingerprint="test-two-store"
        values={"T1":target, "T2":Record("T2", {"label":"dextrose"})}
        def get(self, key): return self.values[key]
        def get_many(self, keys): return [self.values[key] for key in keys]
    class TwoCandidateRetriever(TestRetriever):
        async def search(self, request): return [RetrievalHit("T1","test",1.0,1),RetrievalHit("T2","test",0.9,2)]
    trace_results = {}
    for keep in (True, False):
        scorer = CandidateRetryScorer()
        matcher = Matcher(templates=TestTemplates(), retrievers=[TwoCandidateRetriever()], store=TwoCandidateStore(), selector=TestSelector(), scorer=scorer, verifier=TestVerifier(), rewriter=TestRewriter(), policy=MatchPolicy(max_attempts=2), run_fingerprint="trace", keep_candidates_in_trace=keep)
        result = await matcher.match(source_record)
        trace_results[str(keep)]={"status":result.status.value,"matched_id":result.matched_id,"scorer_calls":scorer.calls,"attempt_reasons":[a.reason.value if a.reason else None for a in result.attempts]}
    results["trace_flag_changes_matching"] = trace_results

    with tempfile.TemporaryDirectory() as d:
        matcher = build([0.95, 0.95])
        await run_batch(matcher, [source_record], out=d)
        changed = Record("s1", {"mention":"glucose", "new_field":"changed"})
        report = await run_batch(matcher, [changed], out=d)
        with Path(d, "mapping.csv").open() as f:
            rows = list(csv.DictReader(f))
        results["changed_source_resume_duplicates"] = {"expected_current_rows":1,"reported_total":report.total,"exported_source_ids":[r["source_id"] for r in rows],"duplicate_targets":report.duplicate_targets()}

    ledger_events = []
    class ProbeLedger:
        closed = False
        @classmethod
        def open(cls, path): return cls()
        def put_manifest(self, *args): pass
        def has_result(self, key): return False
        async def put_result(self, result):
            ledger_events.append({"event":"write","after_close":self.closed})
            if self.closed: raise RuntimeError("closed ledger")
        def close(self):
            self.closed = True
            ledger_events.append({"event":"close"})
    module_envs["src/xwalk/batch.py"]["Ledger"] = ProbeLedger
    class BatchMatcher:
        run_fingerprint="batchprobe"
        store_fingerprint="store"
        policy=MatchPolicy(concurrency=2)
        async def match(self, record):
            if record.id=="fails":
                await asyncio.sleep(0.005)
                raise LLMRetryableError("429 exhausted")
            await asyncio.sleep(0.02)
            return SimpleNamespace(usage=unit)
    with tempfile.TemporaryDirectory() as d:
        try:
            await run_batch(BatchMatcher(), [Record("fails",{}),Record("slow",{})], out=d)
        except LLMRetryableError:
            pass
        await asyncio.sleep(0.04)
    results["batch_write_after_close"] = ledger_events
    print(json.dumps(results, indent=2))

asyncio.run(probes())

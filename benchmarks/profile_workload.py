"""Profile one representative workload: `xwalk.ops.run` on a pilot variant.

    python -m benchmarks.profile_workload                      # cProfile + timings
    python -m benchmarks.profile_workload --llm trivial        # library cost only
    python -m benchmarks.profile_workload --experiment prompt-template-cache --llm trivial

The model is either the SYNTHETIC judge (`--llm synthetic`, the default) or a trivial
FakeLLM that always picks the first candidate at 0.95 (`--llm trivial`), which leaves
almost nothing but xwalk's own code on the profile. Nothing calls the network. Every run
gets a fresh output directory: nothing is resumed, and every run builds its index.

`--experiment prompt-template-cache` measures, without editing the library, what caching
the compiled prompt templates in `xwalk.prompts.contract.PromptSet._render` would save:
it alternates baseline and patched runs and checks that both send byte-identical
prompts. See docs/benchmarks.md, "Performance".
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import io
import json
import pstats
import statistics
import tempfile
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from benchmarks.data import load_manifest, materialise
from benchmarks.synthetic_llm import SyntheticJudgeLLM
from xwalk import ops
from xwalk.llm.base import LLMClient, LLMRequest, LLMResponse
from xwalk.llm.fake import FakeLLM
from xwalk.prompts import contract


def _trivial(request: LLMRequest) -> str:
    if "## Candidates" in request.user:
        return json.dumps({"chosen_key": "C01", "confidence_score": 0.95, "explanation": ""})
    return json.dumps({"confidence_score": 0.95, "explanation": ""})


class Recording:
    """Wraps a client and keeps every request it forwards (for prompt equality checks)."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.prompts: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def capabilities(self) -> Any:
        return self._inner.capabilities

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.prompts.append((request.system, request.user))
        return await self._inner.complete(request)


def make_llm(kind: str, query_field: str, latency: float) -> Recording:
    if kind == "trivial":
        return Recording(FakeLLM(handler=_trivial))
    return Recording(SyntheticJudgeLLM(query_field=query_field, latency_s=latency))


def one_run(job: Path, out: Path, llm: Recording) -> dict[str, Any]:
    started = time.perf_counter()
    result = ops.run(job, out, llm=llm)
    return {
        "wall_seconds": time.perf_counter() - started,
        "calls": (result.usage or {}).get("calls"),
        "exit_code": result.exit_code,
    }


# --- the experiment: cache compiled prompt templates ---------------------------------


@functools.lru_cache(maxsize=64)
def _compiled(source: str) -> Template:
    env = Environment(
        loader=FileSystemLoader(str(contract.BASE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )
    return env.from_string(source)


def _cached_render(self: contract.PromptSet, name: str, **variables: Any) -> str:
    return _compiled(self.skeletons[name]).render(slots=self.slots, **variables).strip()


@contextmanager
def prompt_template_cache() -> Iterator[None]:
    original: Callable[..., str] = contract.PromptSet._render
    contract.PromptSet._render = _cached_render  # type: ignore[method-assign]
    try:
        yield
    finally:
        contract.PromptSet._render = original  # type: ignore[method-assign]


def experiment(job: Path, work: Path, args: argparse.Namespace, query_field: str) -> dict[str, Any]:
    baseline: list[float] = []
    patched: list[float] = []
    prompts: dict[str, Counter[tuple[str, str]]] = {}
    for i in range(args.repeat):
        llm = make_llm(args.llm, query_field, args.latency)
        baseline.append(one_run(job, work / f"base{i}", llm)["wall_seconds"])
        prompts["baseline"] = Counter(llm.prompts)
        with prompt_template_cache():
            llm = make_llm(args.llm, query_field, args.latency)
            patched.append(one_run(job, work / f"patched{i}", llm)["wall_seconds"])
            prompts["patched"] = Counter(llm.prompts)
    return {
        "experiment": "prompt-template-cache",
        "baseline_wall_seconds": [round(s, 3) for s in baseline],
        "patched_wall_seconds": [round(s, 3) for s in patched],
        "baseline_median": round(statistics.median(baseline), 3),
        "patched_median": round(statistics.median(patched), 3),
        "identical_prompts": prompts["baseline"] == prompts["patched"],
        "prompts_per_run": sum(prompts["baseline"].values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.profile_workload")
    parser.add_argument("--dataset", default="ncbi_disease")
    parser.add_argument("--variant")
    parser.add_argument("--llm", choices=("synthetic", "trivial"), default="synthetic")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--experiment", choices=("prompt-template-cache",))
    parser.add_argument("--json", type=Path, help="write the timing summary here")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="xwalk-profile-") as tmp:
        work = Path(tmp)
        name = args.variant or args.dataset
        (variant,) = materialise(load_manifest(args.dataset), work / "data", variants=[name])
        job, field = variant.job_path, variant.query_field
        # One unmeasured run first: imports and first-use initialisation.
        one_run(job, work / "warmup", make_llm(args.llm, field, args.latency))

        if args.experiment:
            summary = experiment(job, work, args, field)
        else:
            profiler = cProfile.Profile()
            profiler.enable()
            one_run(job, work / "profiled", make_llm(args.llm, field, args.latency))
            profiler.disable()
            buffer = io.StringIO()
            pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(args.top)
            print(buffer.getvalue())

            runs = [
                one_run(job, work / f"run{i}", make_llm(args.llm, field, args.latency))
                for i in range(args.repeat)
            ]
            tracemalloc.start()
            one_run(job, work / "traced", make_llm(args.llm, field, args.latency))
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            seconds = [r["wall_seconds"] for r in runs]
            summary = {
                "wall_seconds": [round(s, 3) for s in seconds],
                "wall_seconds_median": round(statistics.median(seconds), 3),
                "calls": runs[0]["calls"],
                "peak_python_mib": round(peak / 2**20, 2),
            }
    summary = {
        "dataset": args.dataset,
        "variant": name,
        "llm": args.llm,
        "latency_s": args.latency,
        "repeat": args.repeat,
        **summary,
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

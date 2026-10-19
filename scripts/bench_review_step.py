"""Time one review step in `xwalk ui` on a generated run: offline, FakeLLM, no paid calls.

The workbench's Review screen calls `GET /api/explain` for every record it shows and
`POST /api/review` for every decision. This builds a mapping run of about `--rows`
sources (the ChEBI sample's labels and synonyms, repeated as needed, mapped onto that
sample) in a temporary workspace, with a stand-in model whose 0.9 lands every row in
`needs_review`, then times those routes in-process through `dispatch` (no HTTP).

    python scripts/bench_review_step.py --rows 2000 --steps 15
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from xwalk import ops
from xwalk.llm.base import LLMRequest
from xwalk.llm.fake import FakeLLM
from xwalk.ui.api import Workspace, dispatch


def _uncertain(request: LLMRequest) -> str:
    """The first candidate at 0.9: below accept_at 0.95, so every row needs review."""
    return json.dumps(
        {"chosen_key": "C01", "confidence_score": 0.9, "decision": "support", "queries": []}
    )


def _terms(rows: int) -> list[str]:
    path = resources.files("xwalk.ui") / "ontologies" / "chebi-sample.jsonl"
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = json.loads(line)
        names += [str(term["label"]), *(str(s) for s in term.get("synonyms") or [])]
    out: list[str] = []
    while len(out) < rows:
        out += [f"{name} {len(out) // len(names)}" if out else name for name in names]
    return out[:rows]


def build(root: Path, rows: int) -> tuple[Workspace, str, str]:
    ws = Workspace(root, max_calls_cap=rows * 4)
    project = dispatch(
        ws,
        "POST",
        "/api/projects",
        {
            "mode": "map",
            "name": "bench",
            "source": {"text": "\n".join(_terms(rows))},
            "target": {"libraries": ["chebi-sample"]},
            "policy": {"accept_at": 0.95, "review_floor": 0.4},
        },
    )
    assert isinstance(project, dict)
    job = str(project["job_path"])
    run = "projects/bench/runs/bench"
    ops.run(root / job, root / run, llm=FakeLLM(handler=_uncertain))
    ws._remember_job(root / run, root / job)
    return ws, job, run


def _time(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    value = fn()
    return (time.perf_counter() - start) * 1000, value


def _ok(response: Any) -> Mapping[str, Any]:
    assert isinstance(response, Mapping), response
    assert response.get("status") != "error", response.get("errors")
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=15, help="decisions to time")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        build_ms, (ws, job, run) = _time(lambda: build(root, args.rows))
        sheet_ms, sheet = _time(lambda: _ok(dispatch(ws, "GET", "/api/review", {"dir": run})))
        queue = sheet["data"]["rows"]
        print(f"run of {len(queue)} needs_review rows built in {build_ms / 1000:.1f} s")

        timings: dict[str, list[float]] = {"explain": [], "apply": [], "reviews": []}
        for row in queue[: args.steps]:
            explain = {"dir": run, "source_id": row["source_id"]}
            decision = {
                "dir": run,
                "job": job,
                "reviewer": "bench",
                "decisions": [{"result_key": row["result_key"], "decision": "accept"}],
            }
            for name, method, path, params in (
                ("explain", "GET", "/api/explain", explain),
                ("apply", "POST", "/api/review", decision),
                ("reviews", "GET", "/api/reviews", {"dir": run}),
            ):
                start = time.perf_counter()
                _ok(dispatch(ws, method, path, params))
                timings[name].append((time.perf_counter() - start) * 1000)

        print(f"GET /api/review (worksheet, once per queue load): {sheet_ms:.0f} ms")
        for name, values in timings.items():
            print(
                f"{name:>8}: median {statistics.median(values):6.1f} ms, "
                f"max {max(values):6.1f} ms over {len(values)}"
            )
        step = statistics.median(timings["apply"]) + statistics.median(timings["explain"])
        print(f"save and next (apply + explain of the next record): {step:.0f} ms")


if __name__ == "__main__":
    main()

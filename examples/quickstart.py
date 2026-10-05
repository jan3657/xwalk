"""The high-level Python route: validate, match, inspect and export one job.

    python -m examples.quickstart [OUT_DIR]

Runs offline against the bundled quickstart job (`xwalk init` copies the same files).
The scripted `offline_model` stands in for a real endpoint; drop `llm=` to use the
model the job file names (its credential must then be set).
"""

from __future__ import annotations

import json
import sys

from xwalk import ops
from xwalk.llm.base import LLMRequest
from xwalk.llm.fake import FakeLLM


def offline_model(request: LLMRequest) -> str:
    """One answer every stage understands: pick the top candidate, score it 0.9."""
    return json.dumps(
        {"chosen_key": "C01", "confidence_score": 0.9, "decision": "support", "queries": []}
    )


def main(out: str = "runs/quickstart") -> int:
    job = ops.bundled_example() / "job.yaml"  # or the path to your own job.yaml
    check = ops.validate(job, check_credentials=False)  # strict, offline, zero model calls
    if not check.ok:
        raise SystemExit(f"invalid job: {[e.message for e in check.errors]}")
    result = ops.run(job, out, llm=FakeLLM(handler=offline_model), max_calls=100)
    print(result.run, result.counts, result.usage)  # one row per source in mapping.csv
    print(ops.explain(out, "s2").data["decision"])  # why s2 got its answer
    ops.export(out, "reviewed", f"{out}/reviewed.csv")  # review overlay applied
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))

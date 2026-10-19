# Pre-batch-5 single-query oracle

Captured on 19 October 2026 from commit
`587176ec777400d57f27db5568c9e1db98a4e8b3`, using Python 3.12.9 and httpx 0.28.1.
The capture ran from an archive of that commit with `PYTHONPATH` pointing at its
`src`. Only the new capture harness was copied into the baseline's tests directory;
all xwalk code and the shared matcher constants came from the baseline.

Each `.body` file is the exact `httpx.Request.content` bytes, with no parsing,
reserialization, or added newline. The numbered files preserve request order:

- `confident`: select, score.
- `verified`: select, score, verify.
- `rewritten`: select, score, rewrite, select, score.

Each case fits both the old and new candidate budgets. All three query configurations
have identical request bytes; the capture asserts this before sharing their fixtures.
The three identity JSON files contain complete run components and all four stage
identities for no extra query, the same query, and the historical single alternate.
The tests allow only two additions: `run.selector_policy.budget_version = 2` and
`stages.selector.budget_version = 2`, as documented by the rendered-budget change.
All other components must match exactly.

No network or paid calls: every request is handled by `httpx.MockTransport`.

To reproduce from the repository root (use a fresh scratch directory):

```sh
baseline_dir=$(mktemp -d /private/tmp/xwalk-baseline-XXXXXX)
git archive 587176e | tar -x -C "$baseline_dir"
cp tests/test_single_query_compat.py "$baseline_dir/tests/"
cd "$baseline_dir"
PYTHONPATH="$baseline_dir/src" /Users/jandrole/projects/xwalk/.venv/bin/python - <<'PY'
import asyncio
import json
from pathlib import Path
import xwalk
from tests.test_single_query_compat import QUERY_CASES, SCENARIOS, capture_single_query

assert Path(xwalk.__file__).resolve().is_relative_to(Path.cwd() / 'src')
output = Path('tests/fixtures/single_query_587176e')
output.mkdir(parents=True, exist_ok=True)

async def main():
    for scenario in SCENARIOS:
        directory = output / scenario
        directory.mkdir(exist_ok=True)
        for query_case in QUERY_CASES:
            bodies, identity = await capture_single_query(scenario, query_case)
            for index, body in enumerate(bodies, 1):
                path = directory / f'{index:02}.body'
                if path.exists():
                    assert path.read_bytes() == body
                else:
                    path.write_bytes(body)
            path = output / f'{query_case}.identity.json'
            text = json.dumps(identity, indent=2, sort_keys=True) + '\n'
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)

asyncio.run(main())
PY
```

The generated files under the scratch directory can then be compared byte-for-byte
with this directory. Never regenerate the oracle from the current implementation.

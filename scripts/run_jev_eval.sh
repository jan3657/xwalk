#!/usr/bin/env bash
# Run the four gold samples through the decider path and evaluate each.
#
# Idempotent: an index directory that already exists is reused, and a job_jev.yaml
# that already exists is never regenerated. Needs XWALK_TEST_API_KEY in .env.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
source .env
set +a

for ex in cafeteria_fcd chebi ncbi_disease nlm_gene; do
  job="examples/$ex/job_jev.yaml"
  if [ ! -f "$job" ]; then
    # Only cafeteria_fcd ships one; derive the rest from the LLM job by swapping the block.
    .venv/bin/python - "$ex" <<'EOF'
import pathlib
import sys

import yaml

ex = sys.argv[1]
src = pathlib.Path(f"examples/{ex}/job.yaml")
data = yaml.safe_load(src.read_text())
data.pop("llm")
data.pop("selector", None)
# The decider path retrieves with templates.queries; the LLM twin declares a single
# templates.query. Promote it so both paths issue exactly the same query text.
templates = data["templates"]
if not templates.get("queries"):
    templates["queries"] = [templates.pop("query")]
data["decider"] = {
    "kind": "jev",
    "model": "~typesafe/jev-latest",
    "base_url": "https://openrouter.ai/api/alpha/decisions",
    "api_key_env": "XWALK_TEST_API_KEY",
}
data["policy"] = {
    "accept_at": 0.85,
    "screen_floor": 0.30,
    "chunk_size": 50,
    "max_candidates": 300,
    "concurrency": 16,
}
for r in data["retrievers"]:
    r["limit"] = 200
data["name"] = f"{ex}_jev"
pathlib.Path(f"examples/{ex}/job_jev.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
EOF
  fi
  echo "=== $ex ==="
  if [ ! -d "runs/jev_eval/$ex/index" ]; then
    .venv/bin/xwalk index --job "$job" --out "runs/jev_eval/$ex/index"
  else
    echo "index runs/jev_eval/$ex/index already built; reusing"
  fi
  .venv/bin/xwalk match --job "$job" --out "runs/jev_eval/$ex" --index "runs/jev_eval/$ex/index" || true
  .venv/bin/xwalk eval  --run "runs/jev_eval/$ex" --gold "examples/$ex/sample/gold.csv" --out "runs/jev_eval/$ex/eval"
  # fit exits 1 when no grid point reaches the target precision; that is a result, not a crash.
  { .venv/bin/xwalk fit --run "runs/jev_eval/$ex" --gold "examples/$ex/sample/gold.csv" \
      --job "$job" --precision 0.95 || true; } | tail -3
done

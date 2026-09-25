#!/usr/bin/env bash
set -euo pipefail
cd /ceph/grid/home/jd3099/projects/xwalk
source ./_env.sh

echo "========================================================="
echo "Stage 1: Full Qwen Run (2,030 records, concurrency 8)..."
echo "========================================================="
.venv/bin/xwalk match \
  --job examples/ref_zivila/jobs/foodon/job_qwen.yaml \
  --index runs/ref_zivila/foodon/index \
  --out runs/ref_zivila/foodon_qwen \
  --resume || true

echo "========================================================="
echo "Stage 2: Full Nex Run (2,030 records, concurrency 4)..."
echo "========================================================="
.venv/bin/xwalk match \
  --job examples/ref_zivila/jobs/foodon/job_nex.yaml \
  --index runs/ref_zivila/foodon/index \
  --out runs/ref_zivila/foodon_nex \
  --resume || true

echo "========================================================="
echo "Stage 3: Comparing and Ensembling Runs..."
echo "========================================================="
python scripts/compare_runs.py \
  --run-a runs/ref_zivila/foodon_qwen/mapping.csv \
  --run-b runs/ref_zivila/foodon_nex/mapping.csv \
  --name-a Qwen \
  --name-b Nex \
  --ensemble-out runs/ref_zivila/ensemble_mapping.csv

echo "========================================================="
echo "Stage 4: Merging Ensemble into Ref_zivila_mapped.csv..."
echo "========================================================="
python examples/ref_zivila/merge_results.py \
  --input data/Ref_zivila.csv \
  --mapping runs/ref_zivila/ensemble_mapping.csv \
  --out data/Ref_zivila_mapped.csv \
  --long-out data/ref_zivila/mapping_long.csv

echo "========================================================="
echo "Stage 5: Line and Schema Validation"
echo "========================================================="
wc -l data/Ref_zivila_mapped.csv data/Ref_zivila.csv
echo "ALL STAGES COMPLETED SUCCESSFULLY!"

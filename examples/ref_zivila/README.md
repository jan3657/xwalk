# Ref_zivila FoodOn pilot

This directory contains the first executable slice of the Ref_zivila crosswalk plan.
Downloaded ontology data, generated source/target CSVs, and runs remain under the ignored
`data/` and `runs/` directories.

The FoodOn snapshot used for the current pilot is release `2025-12-30`, downloaded from
`http://purl.obolibrary.org/obo/foodon.owl`. Its checksum and extraction counts are written
to `data/ref_zivila/manifests/foodon.json`.

```bash
.venv/bin/python examples/ref_zivila/prepare_source.py
.venv/bin/python examples/ref_zivila/prepare_foodon.py

set -a
source .env
set +a

.venv/bin/xwalk index \
  --job examples/ref_zivila/jobs/foodon/job.yaml \
  --out runs/ref_zivila/foodon/index

.venv/bin/xwalk match \
  --job examples/ref_zivila/jobs/foodon/job.yaml \
  --index runs/ref_zivila/foodon/index \
  --out runs/ref_zivila/foodon
```

The 12-row pilot is a fixed diagnostic slice, not an accuracy sample. It intentionally
contains difficult states, missing English, curation warnings, an additive, a dish, and a
conflicting translation.

"""Gold ID normalization and alias expansion for NCBI Disease / CTD.

This is what replaced the paper repo's `normalize_ctd_prediction` and
`expand_ctd_eval_ids` hooks. It lives here, in the example, because it is knowledge
about *this dataset* -- not about matching. xwalk never learns anyone's identifier
scheme; `load_gold_csv` takes `normalize` and `expand` callables and that is the whole
extension point.

    from examples.ncbi_disease.gold_normalize import make_expander, normalize
    from xwalk.evaluate import load_gold_csv

    gold = load_gold_csv(
        "examples/ncbi_disease/sample/gold.csv",
        normalize=normalize,
        expand=make_expander("examples/ncbi_disease/sample/targets.tsv"),
    )
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path

MESH_SHORT = re.compile(r"^[CD]\d+$")


def normalize(raw: str) -> str | None:
    """Put a bare MeSH accession behind its prefix; leave anything else alone."""
    rid = raw.strip()
    if not rid:
        return None
    upper = rid.upper()
    if upper.startswith("MESH:"):
        return "MESH:" + rid.split(":", 1)[1]
    if upper.startswith("OMIM:"):
        return "OMIM:" + rid.split(":", 1)[1]
    if MESH_SHORT.match(upper):
        return f"MESH:{upper}"
    return rid


@cache
def _alias_map(tsv_path: str) -> dict[str, frozenset[str]]:
    """Map every identifier a term is known by to the identifiers it stands for.

    Built from the target slice's own `id` and `synonyms` columns, so an answer given
    under a term's alternate accession still counts as correct.
    """
    aliases: dict[str, set[str]] = {}
    with Path(tsv_path).open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            canonical = normalize(row.get("id", ""))
            if not canonical:
                continue
            aliases.setdefault(canonical, set()).add(canonical)
            for alt in (row.get("synonyms") or "").split("|"):
                alt_id = normalize(alt)
                if alt_id and (alt_id.startswith("MESH:") or alt_id.startswith("OMIM:")):
                    aliases.setdefault(alt_id, set()).add(canonical)
    return {k: frozenset(v) for k, v in aliases.items()}


def make_expander(tsv_path: str) -> Callable[[frozenset[str]], frozenset[str]]:
    """Return an `expand` callable for `load_gold_csv`."""
    aliases = _alias_map(tsv_path)

    def expand(ids: frozenset[str]) -> frozenset[str]:
        out: set[str] = set(ids)
        for i in ids:
            out |= aliases.get(i, frozenset())
        return frozenset(out)

    return expand

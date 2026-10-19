"""Clustering track: metrics over a predicted flat partition.

The metric code lives in the package, `xwalk.evaluate.clustering`, so that `xwalk
cluster eval` and this benchmark score with the same functions; it is re-exported here
for the benchmark runners and `python -m benchmarks.cluster_eval`. See that module for
the two conventions (`as_singletons`, `accepted_only`) and the metrics.
"""

from __future__ import annotations

from xwalk.evaluate.clustering import (
    ACCEPTED,
    OUTCOMES,
    Assignment,
    evaluate_clustering,
    order_sensitivity,
    partition_scores,
    read_assignments,
    write_assignments,
)

__all__ = [
    "ACCEPTED",
    "OUTCOMES",
    "Assignment",
    "evaluate_clustering",
    "order_sensitivity",
    "partition_scores",
    "read_assignments",
    "write_assignments",
]

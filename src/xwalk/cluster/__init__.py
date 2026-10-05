"""Flat equivalence clustering (experimental).

Groups one collection's records into clusters of equivalent records, deciding every
assignment, new cluster and merge with the LLM under bounded context and call limits,
with every decision stored and the run resumable. See docs/guide/clustering.md and
docs/claude-upgrade/CLUSTERING_DECISIONS.md.

Not built on `Matcher`: the candidate set (the cluster pool) changes as the run
proceeds, so it has its own state, identity and store.
"""

from xwalk.cluster.engine import ClusterEngine, EngineReport
from xwalk.cluster.exports import EXPORT_OUTCOMES, write_exports
from xwalk.cluster.pool import PoolIndex
from xwalk.cluster.prompts import DEFAULT_RELATION, ClusterPrompts
from xwalk.cluster.run import (
    ClusterReport,
    ClusterRunError,
    fingerprint_components,
    order_sources,
    run_clustering,
)
from xwalk.cluster.settings import ClusterPolicy, ClusterSettings, PoolSettings
from xwalk.cluster.store import ClusterStore

__all__ = [
    "DEFAULT_RELATION",
    "EXPORT_OUTCOMES",
    "ClusterEngine",
    "ClusterPolicy",
    "ClusterPrompts",
    "ClusterReport",
    "ClusterRunError",
    "ClusterSettings",
    "ClusterStore",
    "EngineReport",
    "PoolIndex",
    "PoolSettings",
    "fingerprint_components",
    "order_sources",
    "run_clustering",
    "write_exports",
]

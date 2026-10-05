"""Flat equivalence clustering: stream, then bounded refinement.

The run is a deterministic sequence of steps (docs/claude-upgrade/CLUSTERING_DECISIONS.md):

    stream/<source>                 assign to a cluster, mint one, defer, or fail
    end/0                           state revision 0
    retry/<i>/<source>              deferred and failed sources, against the grown pool
    consolidate/<i>/<cluster>       propose one neighbour, verify the merge
    reassign/<i>/<source>           incumbent kept unless a comparative verdict at margin
    end/<i>                         state revision i; stop when converged, oscillating,
                                    or at max_refine_iterations

Each step commits in one transaction. A phase's work list is stored when the phase
starts. Resume rebuilds the state from the store and skips completed steps, so an
interrupted run that is resumed reproduces a clean run row for row.

Stored source outcomes: `assigned`, `seed` (minted the cluster), `deferred`, `failed`.
Accepted membership is the clusters' member lists: every member is `assigned` or `seed`,
and a source is in at most one cluster. `deferred` and `failed` sources are in none.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xwalk.cluster.pool import PoolHit, PoolIndex
from xwalk.cluster.prompts import SYSTEM, ClusterPrompts, cluster_block
from xwalk.cluster.settings import PROVENANCE_STRENGTH, ClusterSettings
from xwalk.cluster.stages import Reply, ask, issue_keys, keyed_clusters, read_choice, read_label
from xwalk.cluster.store import ClusterStore, StepWrites
from xwalk.fingerprint import hash_value
from xwalk.llm.base import LLMClient
from xwalk.records import Record, Usage
from xwalk.retrieval.dense import Encoder
from xwalk.templates import TemplateSet

ASSIGNED = "assigned"
SEED = "seed"
DEFERRED = "deferred"
FAILED = "failed"
ACCEPTED = (ASSIGNED, SEED)
UNRESOLVED = (DEFERRED, FAILED)

STOP_REASONS = ("converged", "max_iterations", "oscillation")

Progress = Callable[[str, str], None]


@dataclass
class Cluster:
    cluster_id: str
    seed_id: str
    created_seq: int
    members: list[str]
    revision: int
    live: bool
    merged_into: str | None
    provenance: str


@dataclass
class Member:
    outcome: str
    cluster_id: str | None
    reason: str
    proposal: str | None
    confidence: float | None
    decision_id: int | None


@dataclass(frozen=True)
class EngineReport:
    stop_reason: str
    selected_revision: int
    last_revision: int
    pool_updates: int


def _usage_dict(usage: Usage) -> dict[str, int]:
    return {
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "unknown_calls": usage.unknown_calls,
    }


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def cluster_id_for(run_fingerprint: str, seed_id: str, nth: int = 0) -> str:
    """Stable run-local cluster identity. `nth` > 0 only when the same source seeds a
    second cluster (it was moved out of its first one and later minted again)."""
    parts: list[Any] = ["cluster", run_fingerprint, seed_id]
    if nth:
        parts.append(nth)
    return "K" + hash_value(parts)


def state_hash(members: Mapping[str, Member], clusters: Mapping[str, Cluster]) -> str:
    return hash_value(
        {
            "members": sorted([sid, m.outcome, m.cluster_id or ""] for sid, m in members.items()),
            "clusters": sorted(c.cluster_id for c in clusters.values() if c.live and c.members),
        }
    )


class ClusterEngine:
    def __init__(
        self,
        *,
        store: ClusterStore,
        llm: LLMClient,
        templates: TemplateSet,
        settings: ClusterSettings,
        prompts: ClusterPrompts,
        run_fingerprint: str,
        encoder: Encoder | None = None,
        progress: Progress | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.templates = templates
        self.settings = settings
        self.prompts = prompts
        self.run_fingerprint = run_fingerprint
        self.pool = PoolIndex(encoder=encoder)
        self._progress = progress
        self.records: dict[str, Record] = {}
        self.order: list[str] = []
        self.clusters: dict[str, Cluster] = {}
        self.members: dict[str, Member] = {}
        self._next_decision = 0
        self._completed: set[str] = set()
        self._frozen: set[str] = set()
        self._query_cache: dict[str, str] = {}
        self._line_cache: dict[str, str] = {}

    # --- state ------------------------------------------------------------------------

    def load(self) -> None:
        """Rebuild the in-memory state from the store. The pool index is built once here
        from the live clusters; afterwards it is only updated incrementally."""
        self.records.clear()
        self.order.clear()
        for _, sid, _, fields in self.store.sources():
            self.records[sid] = Record(id=sid, fields=fields)
            self.order.append(sid)
        self.clusters = {
            cid: Cluster(
                cluster_id=cid,
                seed_id=row["seed_id"],
                created_seq=row["created_seq"],
                members=list(row["members"]),
                revision=row["revision"],
                live=row["live"],
                merged_into=row["merged_into"],
                provenance=row["mint_provenance"],
            )
            for cid, row in self.store.latest_revisions().items()
        }
        self.members = {
            sid: Member(
                outcome=row["outcome"],
                cluster_id=row["cluster_id"],
                reason=row["reason"],
                proposal=row["proposal_cluster_id"],
                confidence=row["confidence"],
                decision_id=row["decision_id"],
            )
            for sid, row in self.store.current_assignments().items()
        }
        for cluster in sorted(self.clusters.values(), key=lambda c: c.created_seq):
            if cluster.live and cluster.members:
                self.pool.upsert(cluster.cluster_id, self._doc_text(cluster))
        self._next_decision = self.store.max_decision_id()
        self._completed = self.store.completed_steps()

    def live_ids(self) -> list[str]:
        live = [c for c in self.clusters.values() if c.live and c.members]
        return [c.cluster_id for c in sorted(live, key=lambda c: c.created_seq)]

    # --- rendering --------------------------------------------------------------------

    def _query(self, sid: str) -> str:
        if sid not in self._query_cache:
            self._query_cache[sid] = self.templates.render_query(self.records[sid])
        return self._query_cache[sid]

    def subject(self, sid: str) -> str:
        record = self.records[sid]
        return self.templates.render_context(record) or self._query(sid)

    def member_line(self, sid: str) -> str:
        if sid not in self._line_cache:
            text = self.templates.render_candidate(self.records[sid]) or self._query(sid)
            self._line_cache[sid] = _clip(" ".join(text.split()), self.settings.pool.evidence_chars)
        return self._line_cache[sid]

    def block(
        self, members: Sequence[str], *, key: str | None = None, exclude: str | None = None
    ) -> str:
        """A cluster as shown to the model: bounded member evidence, seed first."""
        visible = [m for m in members if m != exclude]
        lines = [self.member_line(m) for m in visible[: self.settings.pool.member_evidence]]
        return cluster_block(lines, len(visible), key=key)

    def _doc_text(self, cluster: Cluster) -> str:
        return "\n".join(self._query(m) for m in cluster.members)

    # --- recording --------------------------------------------------------------------

    def _hit(self, hit: PoolHit) -> dict[str, Any]:
        return {
            "cluster_id": hit.cluster_id,
            "revision": self.clusters[hit.cluster_id].revision,
            "rank": hit.rank,
            "score": round(hit.score, 12),
            "heads": dict(sorted(hit.heads.items())),
        }

    def _shown(
        self, cluster_ids: Sequence[str], *, keyed: bool = True, exclude: Mapping[str, str] = {}
    ) -> tuple[list[str], list[dict[str, Any]]]:
        keys: list[str | None] = (
            list(issue_keys(len(cluster_ids))) if keyed else [None] * len(cluster_ids)
        )
        blocks: list[str] = []
        entries: list[dict[str, Any]] = []
        for key, cid in zip(keys, cluster_ids, strict=True):
            cluster = self.clusters[cid]
            blocks.append(self.block(cluster.members, key=key, exclude=exclude.get(cid)))
            entries.append(
                {
                    "key": key,
                    "cluster_id": cid,
                    "revision": cluster.revision,
                    "exclude": exclude.get(cid),
                }
            )
        return blocks, entries

    def _record(
        self,
        w: StepWrites,
        *,
        kind: str,
        subject_id: str,
        retrieved: Sequence[Mapping[str, Any]] = (),
        shown: Sequence[Mapping[str, Any]] = (),
        reply: Reply | None = None,
        result: Mapping[str, Any],
    ) -> int:
        self._next_decision += 1
        w.decisions.append(
            {
                "decision_id": self._next_decision,
                "subject_id": subject_id,
                "kind": kind,
                "retrieved": list(retrieved),
                "shown": list(shown),
                "system": SYSTEM if reply is not None else None,
                "prompt": reply.user if reply is not None else None,
                "raw": reply.raw if reply is not None else None,
                "result": dict(result),
                "confidence": reply.confidence if reply is not None else None,
                "error": reply.error if reply is not None else None,
                "usage": _usage_dict(reply.usage if reply is not None else Usage.zero()),
            }
        )
        return self._next_decision

    def _revise(self, w: StepWrites, cluster: Cluster, decision_id: int | None) -> None:
        cluster.revision += 1
        w.revisions.append(
            {
                "cluster_id": cluster.cluster_id,
                "revision": cluster.revision,
                "seed_id": cluster.seed_id,
                "created_seq": cluster.created_seq,
                "members": list(cluster.members),
                "representation": self.block(cluster.members) if cluster.members else "",
                "live": cluster.live,
                "merged_into": cluster.merged_into,
                "mint_provenance": cluster.provenance,
                "decision_id": decision_id,
            }
        )
        if cluster.live and cluster.members:
            self.pool.upsert(cluster.cluster_id, self._doc_text(cluster))
        else:
            self.pool.remove(cluster.cluster_id)

    def _set(
        self,
        w: StepWrites,
        sid: str,
        outcome: str,
        cluster_id: str | None,
        reason: str,
        *,
        proposal: str | None = None,
        confidence: float | None = None,
        decision_id: int | None = None,
    ) -> None:
        self.members[sid] = Member(outcome, cluster_id, reason, proposal, confidence, decision_id)
        w.assignments.append(
            {
                "source_id": sid,
                "outcome": outcome,
                "cluster_id": cluster_id,
                "reason": reason,
                "proposal_cluster_id": proposal,
                "confidence": confidence,
                "decision_id": decision_id,
            }
        )
        if self._progress is not None:
            self._progress(sid, outcome)

    # --- mutations --------------------------------------------------------------------

    def _mint(self, w: StepWrites, sid: str, provenance: str, decision_id: int) -> None:
        nth = 0
        cid = cluster_id_for(self.run_fingerprint, sid)
        while cid in self.clusters:
            nth += 1
            cid = cluster_id_for(self.run_fingerprint, sid, nth)
        cluster = Cluster(cid, sid, len(self.clusters), [sid], 0, True, None, provenance)
        self.clusters[cid] = cluster
        self._revise(w, cluster, decision_id)
        self._set(w, sid, SEED, cid, f"minted:{provenance}", decision_id=decision_id)

    def _join(
        self, w: StepWrites, sid: str, cid: str, reason: str, confidence: float | None, did: int
    ) -> None:
        cluster = self.clusters[cid]
        cluster.members.append(sid)
        self._revise(w, cluster, did)
        self._set(w, sid, ASSIGNED, cid, reason, confidence=confidence, decision_id=did)

    # --- steps ------------------------------------------------------------------------

    async def _step(
        self,
        key: str,
        phase: str,
        iteration: int,
        subject: str | None,
        work: Callable[..., Awaitable[None]],
        *args: Any,
    ) -> None:
        """Run `work(writes, *args)` and commit everything it wrote with the step marker."""
        if key in self._completed:
            return
        writes = StepWrites()
        await work(writes, *args)
        self.store.commit_step(key, phase, iteration, subject, writes)
        self._completed.add(key)

    def _plan(self, phase: str, iteration: int, compute: Callable[[], list[str]]) -> list[str]:
        stored = self.store.get_plan(phase, iteration)
        if stored is not None:
            return stored
        items = compute()
        self.store.put_plan(phase, iteration, items)
        return items

    async def run(self) -> EngineReport:
        self.load()
        policy = self.settings.policy
        for sid in self._plan("stream", 0, lambda: list(self.order)):
            await self._step(f"stream/{sid}", "stream", 0, sid, self._decide, sid)
        await self._step("end/0", "end", 0, None, self._end, 0)

        for it in range(1, policy.max_refine_iterations + 1):
            if self.store.get_meta("stop_reason") is not None:
                break
            for sid in self._plan("retry", it, self._retry_plan):
                await self._step(f"retry/{it}/{sid}", "retry", it, sid, self._decide, sid)
            if policy.consolidate:
                self._frozen = set()
                for merge in self.store.merges(it):
                    if merge["outcome"] == "applied":
                        self._frozen |= {merge["left_id"], merge["right_id"]}
                for cid in self._plan("consolidate", it, self._consolidate_plan):
                    await self._step(
                        f"consolidate/{it}/{cid}", "consolidate", it, cid, self._consolidate, cid
                    )
            if policy.reassign:
                for sid in self._plan("reassign", it, self._reassign_plan):
                    await self._step(
                        f"reassign/{it}/{sid}", "reassign", it, sid, self._reassign, sid
                    )
            await self._step(f"end/{it}", "end", it, None, self._end, it)

        stop = self.store.get_meta("stop_reason")
        selected = self.store.get_meta("selected_revision")
        last = self.store.get_meta("last_revision")
        assert stop in STOP_REASONS and selected is not None and last is not None
        return EngineReport(str(stop), int(selected), int(last), self.pool.updates)

    def _retry_plan(self) -> list[str]:
        return [s for s in self.order if self.members[s].outcome in UNRESOLVED]

    def _consolidate_plan(self) -> list[str]:
        live = [self.clusters[c] for c in self.live_ids()]
        return [c.cluster_id for c in sorted(live, key=lambda c: (-len(c.members), c.created_seq))]

    def _reassign_plan(self) -> list[str]:
        plan = []
        for sid in self.order:
            member = self.members[sid]
            cid = member.cluster_id
            if member.outcome in ACCEPTED and cid and len(self.clusters[cid].members) >= 2:
                plan.append(sid)
        return plan

    # --- stream and retry: assign, mint, defer, fail ----------------------------------

    async def _decide(self, w: StepWrites, sid: str) -> None:
        pool = self.settings.pool
        live = self.live_ids()
        if not live:
            did = self._record(w, kind="mint", subject_id=sid, result={"provenance": "empty_pool"})
            self._mint(w, sid, "empty_pool", did)
            return

        subject = self.subject(sid)
        query = self._query(sid)
        shown: list[str] = []
        first_page: list[str] = []
        depth = pool.retrieve_limit
        hits: list[PoolHit] = []
        page_index = 0
        while page_index <= pool.max_expansion_pages:
            hits = self.pool.search(query, depth)
            seen = set(shown)
            page = [h.cluster_id for h in hits if h.cluster_id not in seen][: pool.shown_limit]
            scanned = 0
            if page_index > 0 and len(page) < pool.shown_limit and len(live) <= pool.scan_below:
                for cid in live:
                    if len(page) == pool.shown_limit:
                        break
                    if cid not in seen and cid not in page:
                        page.append(cid)
                        scanned += 1
            if not page:
                if page_index == 0:
                    page_index += 1  # nothing retrieved: go straight to expansion
                    depth = min(depth * 2, pool.expand_limit)
                    continue
                break
            blocks, entries = self._shown(page)
            keyed = keyed_clusters(page, blocks)
            reply = await ask(self.llm, "select", self.prompts.select(subject, "\n\n".join(blocks)))
            choice = read_choice(reply, keyed)
            did = self._record(
                w,
                kind="select",
                subject_id=sid,
                retrieved=[self._hit(h) for h in hits],
                shown=entries,
                reply=reply,
                result={
                    "chosen": choice.cluster_id,
                    "resolution": choice.resolution,
                    "page": page_index,
                    "depth": depth,
                    "scanned": scanned,
                },
            )
            shown += page
            if page_index == 0:
                first_page = list(page)
            if choice.error is not None:
                self._set(w, sid, FAILED, None, choice.error.split(":", 1)[0], decision_id=did)
                return
            if choice.cluster_id is not None:
                await self._verify_join(w, sid, subject, choice.cluster_id, reply.confidence, did)
                return
            page_index += 1
            depth = min(depth * 2, pool.expand_limit)

        shown_set = set(shown)
        if set(live) <= shown_set:
            provenance = "pool_exhausted"
        elif len(hits) < depth and {h.cluster_id for h in hits} <= shown_set:
            provenance = "retrieval_exhausted"
        else:
            provenance = "bounded"
        if PROVENANCE_STRENGTH[provenance] < PROVENANCE_STRENGTH[pool.mint_requires]:
            self._set(w, sid, DEFERRED, None, f"expansion_incomplete:{provenance}")
            return
        await self._novelty(w, sid, subject, (first_page or shown)[: pool.shown_limit], provenance)

    async def _verify_join(
        self, w: StepWrites, sid: str, subject: str, cid: str, select_conf: float | None, sel: int
    ) -> None:
        policy = self.settings.policy
        if not policy.verify_assignments:
            if select_conf is not None and select_conf >= policy.assign_accept_at:
                self._join(w, sid, cid, "selected", select_conf, sel)
            else:
                self._set(
                    w,
                    sid,
                    DEFERRED,
                    None,
                    "below_accept_threshold",
                    proposal=cid,
                    confidence=select_conf,
                    decision_id=sel,
                )
            return
        cluster = self.clusters[cid]
        blocks, entries = self._shown([cid], keyed=False)
        reply = await ask(self.llm, "verify", self.prompts.verify(subject, blocks[0]))
        decision = read_label(reply, "decision", ("equivalent", "not_equivalent"))
        did = self._record(
            w,
            kind="verify",
            subject_id=sid,
            shown=entries,
            reply=reply,
            result={"decision": decision, "cluster_id": cluster.cluster_id},
        )
        conf = reply.confidence
        if decision is None:
            reason = (reply.error or "invalid_output").split(":", 1)[0]
            self._set(w, sid, FAILED, None, reason, proposal=cid, confidence=conf, decision_id=did)
        elif decision == "equivalent" and conf is not None and conf >= policy.assign_accept_at:
            self._join(w, sid, cid, "verified", conf, did)
        else:
            if decision != "equivalent":
                reason = "verifier_rejected"
            elif conf is not None and conf >= policy.assign_review_floor:
                reason = "below_accept_threshold"
            else:
                reason = "below_review_floor"
            self._set(
                w, sid, DEFERRED, None, reason, proposal=cid, confidence=conf, decision_id=did
            )

    async def _novelty(
        self, w: StepWrites, sid: str, subject: str, nearest: Sequence[str], provenance: str
    ) -> None:
        policy = self.settings.policy
        blocks, entries = self._shown(nearest)
        reply = await ask(self.llm, "novelty", self.prompts.novelty(subject, "\n\n".join(blocks)))
        novel = read_label(reply, "novel", (True, False))
        did = self._record(
            w,
            kind="novelty",
            subject_id=sid,
            shown=entries,
            reply=reply,
            result={"novel": novel, "provenance": provenance},
        )
        conf = reply.confidence
        if novel is None:
            reason = (reply.error or "invalid_output").split(":", 1)[0]
            self._set(w, sid, FAILED, None, reason, confidence=conf, decision_id=did)
        elif novel is True and conf is not None and conf >= policy.novelty_accept_at:
            self._mint(w, sid, provenance, did)
        else:
            proposal = nearest[0] if nearest else None
            reason = "novelty_rejected" if novel is False else "novelty_below_threshold"
            self._set(
                w, sid, DEFERRED, None, reason, proposal=proposal, confidence=conf, decision_id=did
            )

    # --- consolidation: verified cluster-vs-cluster merges -----------------------------

    async def _consolidate(self, w: StepWrites, cid: str) -> None:
        pool, policy = self.settings.pool, self.settings.policy
        cluster = self.clusters[cid]
        if not (cluster.live and cluster.members) or cid in self._frozen:
            w.note = "skipped"
            return
        hits = self.pool.search(
            self._doc_text(cluster), pool.retrieve_limit, exclude={cid} | self._frozen
        )
        page = [h.cluster_id for h in hits[: pool.shown_limit]]
        if not page:
            w.note = "no_neighbours"
            return
        blocks, entries = self._shown(page)
        subject_block = self.block(cluster.members)
        reply = await ask(
            self.llm,
            "merge_select",
            self.prompts.select(subject_block, "\n\n".join(blocks), subject_is_cluster=True),
        )
        choice = read_choice(reply, keyed_clusters(page, blocks))
        own = {"key": None, "cluster_id": cid, "revision": cluster.revision, "exclude": None}
        self._record(
            w,
            kind="merge_select",
            subject_id=cid,
            retrieved=[self._hit(h) for h in hits],
            shown=[own, *entries],
            reply=reply,
            result={"chosen": choice.cluster_id, "resolution": choice.resolution},
        )
        if choice.cluster_id is None:
            return
        other = self.clusters[choice.cluster_id]
        _, pair = self._shown([cid, other.cluster_id], keyed=False)
        verdict = await ask(
            self.llm, "merge", self.prompts.merge(subject_block, self.block(other.members))
        )
        decision = read_label(verdict, "decision", ("same", "different"))
        did = self._record(
            w,
            kind="merge",
            subject_id=cid,
            shown=pair,
            reply=verdict,
            result={"decision": decision, "left": cid, "right": other.cluster_id},
        )
        conf = verdict.confidence
        outcome = "rejected"
        winner_id: str | None = None
        if decision is None:
            outcome = "error"
        elif decision == "same" and conf is not None and conf >= policy.merge_accept_at:
            outcome = "applied"
            winner_id = self._merge(w, cluster, other, conf, did)
        elif decision == "same" and conf is not None and conf >= policy.merge_review_floor:
            outcome = "review"
        w.merges.append(
            {
                "left_id": cid,
                "right_id": other.cluster_id,
                "outcome": outcome,
                "winner_id": winner_id,
                "decision_id": did,
            }
        )

    def _merge(self, w: StepWrites, left: Cluster, right: Cluster, conf: float, did: int) -> str:
        winner, loser = sorted((left, right), key=lambda c: (-len(c.members), c.created_seq))
        moved = list(loser.members)
        winner.members.extend(moved)
        loser.members = []
        loser.live = False
        loser.merged_into = winner.cluster_id
        self._revise(w, winner, did)
        self._revise(w, loser, did)
        for sid in moved:
            self._set(
                w, sid, ASSIGNED, winner.cluster_id, "merged", confidence=conf, decision_id=did
            )
        # Neither takes part in another merge this iteration: the merged representation
        # is judged afresh next iteration, never by chaining this verdict.
        self._frozen |= {winner.cluster_id, loser.cluster_id}
        return winner.cluster_id

    # --- reassignment: incumbent always shown, switch only at margin ------------------

    async def _reassign(self, w: StepWrites, sid: str) -> None:
        pool, policy = self.settings.pool, self.settings.policy
        member = self.members[sid]
        if member.outcome not in ACCEPTED or member.cluster_id is None:
            w.note = "skipped"
            return
        incumbent = self.clusters[member.cluster_id]
        if len(incumbent.members) < 2:
            w.note = "skipped"
            return
        hits = self.pool.search(self._query(sid), pool.retrieve_limit)
        page = [h.cluster_id for h in hits][: pool.shown_limit]
        injected = incumbent.cluster_id not in page
        if injected:
            if len(page) >= pool.shown_limit:
                page[-1] = incumbent.cluster_id  # the incumbent survives truncation
            else:
                page.append(incumbent.cluster_id)
        if page == [incumbent.cluster_id]:
            w.note = "no_challenger"
            return
        subject = self.subject(sid)
        exclude = {incumbent.cluster_id: sid}
        blocks, entries = self._shown(page, exclude=exclude)
        reply = await ask(self.llm, "select", self.prompts.select(subject, "\n\n".join(blocks)))
        choice = read_choice(reply, keyed_clusters(page, blocks))
        self._record(
            w,
            kind="select",
            subject_id=sid,
            retrieved=[self._hit(h) for h in hits],
            shown=entries,
            reply=reply,
            result={
                "chosen": choice.cluster_id,
                "resolution": choice.resolution,
                "incumbent": incumbent.cluster_id,
                "incumbent_injected": injected,
            },
        )
        if choice.error is not None or choice.cluster_id == incumbent.cluster_id:
            w.note = "kept"
            return
        incumbent_block = self.block(incumbent.members, exclude=sid)
        if choice.cluster_id is not None:
            challenger = self.clusters[choice.cluster_id]
            _, pair = self._shown(
                [incumbent.cluster_id, challenger.cluster_id], keyed=False, exclude=exclude
            )
            verdict = await ask(
                self.llm,
                "compare",
                self.prompts.compare(subject, incumbent_block, self.block(challenger.members)),
            )
            preferred = read_label(verdict, "preferred", ("current", "alternative"))
            did = self._record(
                w,
                kind="compare",
                subject_id=sid,
                shown=pair,
                reply=verdict,
                result={
                    "preferred": preferred,
                    "incumbent": incumbent.cluster_id,
                    "challenger": challenger.cluster_id,
                },
            )
            conf = verdict.confidence
            if (
                preferred == "alternative"
                and conf is not None
                and conf >= policy.comparative_accept_at
            ):
                incumbent.members.remove(sid)
                self._revise(w, incumbent, did)
                challenger.members.append(sid)
                self._revise(w, challenger, did)
                self._set(
                    w,
                    sid,
                    ASSIGNED,
                    challenger.cluster_id,
                    "reassigned",
                    confidence=conf,
                    decision_id=did,
                )
                w.note = "switched"
            else:
                w.note = "kept"
            return
        # Null selection: the incumbent must survive a fresh verification.
        _, own = self._shown([incumbent.cluster_id], keyed=False, exclude=exclude)
        verdict = await ask(self.llm, "reverify", self.prompts.verify(subject, incumbent_block))
        decision = read_label(verdict, "decision", ("equivalent", "not_equivalent"))
        did = self._record(
            w,
            kind="reverify",
            subject_id=sid,
            shown=own,
            reply=verdict,
            result={"decision": decision, "cluster_id": incumbent.cluster_id},
        )
        conf = verdict.confidence
        if decision is None or (
            decision == "equivalent" and conf is not None and conf >= policy.assign_accept_at
        ):
            w.note = "kept"  # a provider error keeps the incumbent; it is recorded above
            return
        incumbent.members.remove(sid)
        self._revise(w, incumbent, did)
        self._set(
            w,
            sid,
            DEFERRED,
            None,
            "reverify_failed",
            proposal=incumbent.cluster_id,
            confidence=conf,
            decision_id=did,
        )
        w.note = "removed"

    # --- iteration end: state revision and stopping ------------------------------------

    def snapshot(self) -> tuple[dict[str, list[Any]], dict[str, int]]:
        assignments = {
            sid: [m.outcome, m.cluster_id, m.reason, m.proposal, m.confidence, m.decision_id]
            for sid, m in sorted(self.members.items())
        }
        clusters = {cid: c.revision for cid, c in sorted(self.clusters.items())}
        return assignments, clusters

    async def _end(self, w: StepWrites, iteration: int) -> None:
        assignments, clusters = self.snapshot()
        current = state_hash(self.members, self.clusters)
        unresolved = sum(1 for m in self.members.values() if m.outcome in UNRESOLVED)
        w.state_revision = {
            "revision": iteration,
            "iteration": iteration,
            "state_hash": current,
            "unresolved": unresolved,
            "assignments": assignments,
            "clusters": clusters,
        }
        previous = self.store.state_revisions()
        stop: str | None = None
        if iteration > 0 and previous and previous[-1]["state_hash"] == current:
            stop = "converged"
        elif iteration > 0 and any(r["state_hash"] == current for r in previous):
            stop = "oscillation"
        elif iteration >= self.settings.policy.max_refine_iterations:
            stop = "max_iterations"
        if stop is None:
            return
        if stop == "converged":
            selected = iteration
        else:
            ranked = [(r["unresolved"], r["revision"]) for r in previous]
            ranked.append((unresolved, iteration))
            selected = min(ranked)[1]
        w.meta = {"stop_reason": stop, "selected_revision": selected, "last_revision": iteration}
        w.note = stop


__all__ = [
    "ACCEPTED",
    "ASSIGNED",
    "DEFERRED",
    "FAILED",
    "SEED",
    "STOP_REASONS",
    "UNRESOLVED",
    "Cluster",
    "ClusterEngine",
    "EngineReport",
    "Member",
    "cluster_id_for",
    "state_hash",
]

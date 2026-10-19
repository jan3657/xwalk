# Clustering one collection (experimental)

`xwalk cluster` groups the records of **one** collection into clusters of equivalent
records: spelling variants, word-order variants and synonyms of the same concept end up
together; broader, narrower and merely related concepts stay apart. Every decision is
made by the LLM over a bounded list of candidate clusters, every decision is stored, and
a run is resumable.

> **Experimental.** The workflow is tested end to end with a scripted judge (FakeLLM),
> not evaluated on a real model (`xwalk cluster eval` is the tool for doing so). Its thresholds are provisional, processing is
> sequential, and there is no review-apply step for clusters yet. Use it on a sample you
> can check by hand before you rely on it. The design decisions are recorded in
> [the clustering decision record](../claude-upgrade/CLUSTERING_DECISIONS.md).

## A clustering job

A clustering job is a YAML file with `kind: cluster`. Validation is as strict as for a
matching job: unknown keys fail with a suggestion, numbers are range-checked, and
`xwalk validate --job cluster.yaml` checks it offline without calling a model.

```yaml
kind: cluster
name: food-labels
source:                       # any collection kind a matching job accepts
  kind: csv
  path: labels.csv            # relative to this file
templates:
  query: "{{ label }}"        # retrieval text, and the ordering key
  context: "{{ label }}"      # the record as the model sees it (default: query)
  candidate: "{{ label }}"    # one member line inside a cluster (default: context)
llm:                          # exactly as in a matching job
  kind: openai_compat
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
relation: >-                  # optional: what "equivalent" means in your domain
  Two labels are equivalent when they name the same food product.
order: label                  # label (default) or input
pool: {}                      # see below; defaults shown in the table
policy: {}
# dense: {kind: dense, model: sentence-transformers/all-MiniLM-L6-v2}  # optional, xwalk[dense]
```

| Key | Default | Meaning |
|---|---|---|
| `order` | `label` | `label`: stable sort on the case-folded query text, then id, so a shuffled input file is the same run. `input`: file order; then order is part of the run identity |
| `pool.retrieve_limit` | `20` | clusters retrieved per search |
| `pool.shown_limit` | `8` | clusters shown to the model per call (keyed `C01`..) |
| `pool.member_evidence` | `5` | members listed per cluster shown; the rest are counted, not shown |
| `pool.evidence_chars` | `200` | longest member line |
| `pool.max_expansion_pages` | `2` | further pages shown after the model rejects the first |
| `pool.expand_limit` | `80` | deepest retrieval during expansion |
| `pool.scan_below` | `40` | while the pool has at most this many clusters, expansion pages also show clusters retrieval did not find |
| `pool.mint_requires` | `bounded` | the weakest search that may precede a new cluster: `bounded` (all expansion pages shown), `retrieval_exhausted` (no further retrievable cluster), `pool_exhausted` (every cluster shown). Stricter settings send the record to review instead |
| `policy.verify_assignments` | `true` | a second call verifies each assignment against the cluster's members |
| `policy.assign_accept_at` / `assign_review_floor` | `0.7` / `0.5` | verification confidence to join a cluster |
| `policy.novelty_accept_at` | `0.7` | confidence that a record is a new concept |
| `policy.merge_accept_at` / `merge_review_floor` | `0.8` / `0.6` | confidence to merge two clusters; between the two the merge is recorded for review |
| `policy.comparative_accept_at` | `0.8` | confidence to move a member to another cluster during refinement |
| `policy.max_refine_iterations` | `2` | refinement rounds after the first pass |
| `policy.consolidate` / `policy.reassign` | `true` | turn either refinement phase off |

All thresholds are provisional: a verifier's confidence is a routing signal, not proof.
Calibrate on a small set you have labelled before trusting a run.

## Running it

```console
$ xwalk validate --job cluster.yaml
$ xwalk cluster --job cluster.yaml --out runs/food --max-calls 2000
clustered 1200 records into 431 clusters in runs/food
  run state     : complete
  stop reason   : converged (exported revision 2)
  assigned      : 742
  singleton     : 401
  needs_review  : 57
  failed        : 0
  pending       : 0
  tokens        : ... in ... calls
```

(The numbers above are illustrative.) The run directory follows the same rules as
`match`: the same job and the same source collection resume; any change to either is a
different run and is refused, so pick a new `--out`. Adding or removing `dense:`, or changing its encoder,
is such a change. A directory written before format 2 is refused with
`run_format_outdated`: its identity did not record the retrieval heads. `--max-calls` caps the upstream
requests of one invocation; when it is reached the run stops at a step boundary (exit
`3`, error `call_limit_reached`) and the next `xwalk cluster` continues from there.
Because a step is stored only when it finishes and one record's decision may need
`2 + pool.max_expansion_pages` calls (4 by default), a smaller positive `--max-calls` is
refused with exit `2`; `--max-calls 0` makes no calls. A complete run repeated makes no
calls and rewrites identical exports.

A finished run with `failed` records retries them when it is resumed, as `match` does.
The retry round starts from the exported state, decides each failed record afresh against
it, and stores the result as a new state revision, which becomes the exported one
(`manifest.json`: `selected_revision`, `last_revision`). Nothing is rewritten: the
earlier decisions, assignments and state revisions stay in the store and in
`decisions.jsonl`. Records that fail again stay `failed` (exit `3`) and are retried on
the next resume. Other records keep their exported outcome; to reconsider everything,
use a new `--out`.

From Python:

```python
from xwalk import ops

result = ops.cluster("cluster.yaml", out="runs/food", max_calls=2000)
print(result.exit_code, result.counts, result.data["stop_reason"])
```

`ops.cluster(spec_or_path, out, llm=client)` accepts your own client (no credential
needed), and `xwalk.cluster.run_clustering(records, out=..., llm=..., templates=...)` is
the lower-level route for records you already hold.

## What happens

1. **Stream.** Records are processed one at a time in the recorded order. Each record
   retrieves candidate clusters from the pool, and the model picks one or none. A pick
   is verified against the cluster's listed members (all of them, not one exemplar). If
   the model picks none, further pages are shown (expansion). Only after the configured
   expansion may a new cluster be created, and only if a novelty check agrees. The first
   record, with nothing to compare against, starts a cluster without a model call.
2. **Retry** records that were deferred or failed, against the grown pool.
3. **Consolidate.** Each cluster may propose one neighbour; a separate verification of
   the two clusters' members decides a merge. A cluster takes part in at most one merge
   per round, so merges never chain on an earlier verdict: A~B and B~C never make A~C
   without a decision of its own.
4. **Reassign.** Members of clusters with two or more members are reconsidered. Their
   current cluster is always among the candidates shown; a member moves only when a
   comparative verdict prefers the other cluster at `comparative_accept_at`. If the model
   picks nothing, the current cluster must pass a fresh verification or the member goes
   to review. Reassignment never creates clusters.
5. Steps 2-4 repeat until nothing changes (`converged`), a state repeats
   (`oscillation`), or `max_refine_iterations` is reached. Each round's state is stored.
   The export is the final state when converged, otherwise the stored state with the
   fewest unresolved records (ties: earliest).

## Outputs

| File | Contents |
|---|---|
| `members.csv` | every source record exactly once: `order_index, source_id, outcome, cluster_id, cluster_size, reason, proposal_cluster_id, confidence, decision_id` |
| `clusters.csv` | accepted clusters: `cluster_id, revision, size, status (verified or singleton), seed_id, mint_provenance, member_ids, representation`. `seed_id` is the first member: the record that started the cluster, or the earliest remaining member once that record has moved out |
| `unresolved.csv` | records outside every cluster (`needs_review`, `failed`, `pending`) with the best proposal |
| `decisions.jsonl` | every decision: the clusters retrieved and the clusters shown (with their revisions), the exact prompt, the raw answer, the parsed result and usage |
| `manifest.json` | run fingerprint and components, run state, stop reason, selected and exported revision, counts, usage |
| `cluster.sqlite` | the store everything above is regenerated from |

Outcomes: `assigned` (a verified member of a cluster with other members), `singleton`
(a record alone in its cluster: it started a cluster no other record joined, or every
other member moved out during refinement, reason `sole_remaining_member`; *not* a
verified equivalence),
`needs_review` (no confident decision: a rejected or low-confidence verification, a
failed reverification, or a search that did not reach `mint_requires`), `failed` (a
provider error or unparseable answer, retried in every round and on resume), `pending` (not reached
before the run stopped). Accepted clusters are a partition: no record is in two.

## Scoring a run against gold

With a gold partition (`item_id,cluster_id` rows; records with the same `cluster_id`
are equivalent), `xwalk cluster eval` scores the exported partition. It reads the
exports and the store, calls no model, and writes nothing in the run directory:

```console
$ xwalk cluster eval --run runs/food --gold food_gold.csv --out runs/food-eval.json
```

It reports pairwise and B-cubed precision, recall and F1 twice: with every record
outside the accepted clusters scored as a singleton (`as_singletons`, so sending
records to review cannot raise the score), and over the accepted records only
(`accepted_only`). It also counts outcomes and reports `gold_pairs_co_exposed`: how
many gold pairs the judge ever saw together in one prompt. When recall is low and
that share is low too, the search never put the records in front of the judge (try
`dense:` or a higher `scan_below`); when the share is high, the judge rejected what it
saw. It is a diagnostic, not a ceiling: merges and joins on a cluster's first members
link records no prompt showed together. Details: [`cluster eval`](../reference/cli.md#cluster-eval).

## Inspecting a run and explaining one record

Both commands read the run read-only and make no model call:

```console
$ xwalk cluster inspect --run runs/food
$ xwalk cluster explain --run runs/food c3          # add --full for the stored prompts
```

`inspect` shows the exported outcome counts, cluster sizes, merges, decisions by kind,
usage over every invocation (a lower bound when an invocation never finished), and the
selected versus the last state revision. When refinement stopped at `max_iterations`
or `oscillation`, the exports hold the state with the fewest unresolved records, which
can be earlier than the last: decisions made after it (a failed reverification, a
member moved) are kept in the store but are not in the exports, and `inspect` says so
(`selected_not_last`). `explain` shows one record's exported membership, the decisions
of the step that placed it (with the members each prompt actually listed), the merge
ancestry of its cluster, and any later change that went against it or its cluster.
Details: [`cluster inspect`](../reference/cli.md#cluster-inspect) and
[`cluster explain`](../reference/cli.md#cluster-explain).

Cluster ids are `K` + a hash of the run fingerprint and the record that started the
cluster. They are stable for a run and change when anything in its identity changes.

Exit codes: `0` complete with nothing to review; `1` complete with `needs_review` records;
`2` invalid job, duplicate ids, missing credential or extra; `3` aborted (fatal provider
error or `--max-calls`), records that failed, or a run directory that belongs to another
run; `130` interrupted (resume continues).

## Warnings about a run's limits

`xwalk cluster` (at the end of every invocation) and `xwalk cluster inspect` report what
bounded this run, each at most once. They describe the run; none of them changes a
decision.

| Warning | Reported when | What it means |
|---|---|---|
| `retrieval_bounded` | the largest number of live clusters at any step is greater than `pool.scan_below`, and there is no `dense:` head | above `scan_below`, expansion pages show only retrieved clusters, so a record meets only clusters that share a word with it: synonyms with no word in common may never have been compared. `dense:` or a larger `scan_below` widens the search; neither makes it exhaustive. Not reported for a format-1 run (its manifest does not record the retrieval heads) |
| `evidence_bounded` | an exported cluster has more than `pool.member_evidence` members | a record is verified against the first `member_evidence` members listed, not every member, so members that joined later may never have been compared with each other |
| `resume_determinism` | the run took more than one invocation, or the last one was aborted, interrupted or never finished | a resumed run reproduces an uninterrupted one only if the model answers identical prompts identically; with a sampling model it can differ |
| `selected_not_last` | refinement stopped at `max_iterations` or `oscillation` and the exports hold an earlier state revision than the last | changes after that revision (for instance a member removed by a failed reverification) are in the store and `decisions.jsonl`, not in the exports; `xwalk cluster explain` lists them per record |

## Limits of this first version

- Equivalence only, one level, clusters created in the run. No taxonomy seeding,
  broader/narrower roll-up, target cluster counts or cluster names.
- Sequential processing: one model call at a time.
- With lexical retrieval only, synonyms that share no word meet only through the
  expansion scan (`scan_below`) or a dense encoder (`dense:`). Above `scan_below`
  clusters, such synonyms can end up as separate clusters.
- A different processing order can change the result when the evidence is
  contradictory (A~B, B~C, but not A~C): whichever end arrives first keeps the middle and
  the other goes to review. `order: label` makes the order independent of the input file.
- `review apply` does not read clustering runs yet. The matching-run commands
  (`inspect`, `explain`, ...) refuse a clustering run and point to `xwalk cluster
  inspect`, `xwalk cluster explain` and `xwalk cluster eval`.

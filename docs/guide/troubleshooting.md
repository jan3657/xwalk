# Troubleshooting

Symptoms, causes, and the diagnostic that distinguishes them. Most of these are visible
in the evaluation report or the attempt trace before they are visible in your output.

## Reading the trace

Almost every question below is answered by looking at what actually happened to one
record. The full trace is in `results.jsonl`, or straight off the result object:

```python
for attempt in result.attempts:
    print(f"attempt {attempt.index}: query={attempt.query!r}")
    print(f"  candidates      : {attempt.candidate_count} (truncated {attempt.candidates_truncated})")
    print(f"  keys issued     : {list(attempt.issued_keys)[:5]}")
    print(f"  model answered  : {attempt.raw_selection!r}")
    print(f"  resolved to     : {attempt.chosen_id} via {attempt.resolution}")
    print(f"  score           : {attempt.primary_score}")
    print(f"  verifier        : {attempt.verifier_decision}")
    print(f"  finish_reason   : {attempt.finish_reason}")
    print(f"  error           : {attempt.error}")
    print(f"  dropped         : {attempt.dropped_proposals}")
```

`resolution`, `finish_reason` and `dropped_proposals` are the three fields people forget
exist, and they answer most of the confusing cases.

---

## Nothing matches; everything is unmatched or in review

**Check the `doc` template first.** A target record whose `doc` renders empty can never
be retrieved by anything.

```python
empty = [r.id for r in targets if not templates.render_doc(r)]
print(f"{len(empty)} of {len(targets)} targets render an empty doc: {empty[:5]}")
```

`BM25Retriever` also counts these for you at build time as `retriever.empty_doc_count`.

The usual cause is a field name that does not exist. Templates render a missing field as
an empty string rather than raising, so `{{ name }}` against a collection whose column is
`label` produces empty documents silently, and every record becomes unretrievable.

```python
print(templates.render_doc(targets[0]))   # should not be empty or blank
print(templates.render_query(sources[0]))
```

## Retrieval finds the right record, the model picks something else

That is the `misjudged` bucket, and it is a prompt problem. Check the decomposition to be
sure:

```bash
xwalk eval --run run/ --gold gold.csv
```

If the report agrees, the levers are the rubric bands and `hard_rules` — particularly a
hard rule that names the distinction being missed. See [prompt
optimisation](prompt-optimisation.md); the `SELECTOR` role mines exactly these cases.

## Retrieval finds it but the model never sees it

This is the `truncated` bucket: a retriever surfaced the gold record, but the selector
budget cut it before the prompt was assembled. The fix is one integer:

```python
SelectorPolicy(max_candidates=60)
```

`attempt.candidates_truncated` shows how many were dropped on a given attempt. Note that
raising `max_candidates` changes the run fingerprint, so it is a full re-run.

Retrieval depth (`limit` on each retriever) and selector budget are different knobs.
Raising depth without raising the budget just gives the budget more to discard.

## Confidence scores look arbitrary

Look for the calibration warning in the report. If it fires, the scorer's confidences do
not separate correct from incorrect matches, and no threshold you pick will help — you
would be tuning against noise.

The fix is the rubric. Bands need to be distinguishable *by the model*: if it cannot
reliably tell your 0.9 case from your 0.6 case, collapse them or give each an example
that makes the difference concrete.

## `unresolved_output` on many records

The model is answering with something that is not a candidate key. Look at
`attempt.raw_selection` for a few.

| What you see | Cause |
|---|---|
| A target id like `"CHEBI:17234"` | The model ignored the key instruction. Usually a custom skeleton that dropped it — run `validate_contract`. |
| Prose around a key: `"I think C01"` | The model is not honouring JSON-only output. Check that `profile` matches your provider, and consider a stronger model. |
| Truncated JSON, with `finish_reason == "length"` | The response hit the token cap. Raise `max_tokens` on the stage or the client. This is the one case that looks like a bad model but is not. |
| A key like `C05` that was not issued | The model reused a key from a previous, wider attempt. Keys are valid for one attempt only; this is expected occasionally and is correctly refused. |

`finish_reason` is the field that distinguishes the third row from the second, and
without it they are indistinguishable.

## Everything is `failed`

`failed` means infrastructure, not data. Check `attempt.error`:

- `selector: ...` or `scorer: ...` — the provider. Auth failures raise `LLMFatalError` and
  abort the record immediately rather than retrying.
- `retriever_failure` — every retriever raised or timed out. Check the index exists and
  `retriever_timeout` (default `60.0` seconds) is long enough.

A partial retriever failure is not fatal: the matcher degrades to the survivors and
records a note in `attempt.error`.

## Resume re-runs everything

The run fingerprint changed. It covers templates, prompts and slots, the target snapshot,
every retriever's fingerprint, retrieval depth and the fusion constant, model and
provider identity, generation parameters, policy thresholds, the selector budget, and the
library version.

Compare the manifests:

```bash
python -c "
import json
for d in ('run-a', 'run-b'):
    m = json.load(open(f'{d}/manifest.json'))
    print(d, m['run_fingerprint'], m.get('target_fingerprint'))
"
```

Common surprises: upgrading xwalk (the version is included on purpose — a changed
pipeline should not silently reuse old results), rebuilding an index, or adding a single
target record, which changes the store fingerprint for all of them.

## Resume *should* have re-run and did not

The more dangerous direction. `run_fingerprint` is a string you hand to `Matcher`, and it
is never checked against the matcher you actually built. If you hardcoded it, or built it
before changing a component, resume will happily serve stale results.

Always build it with `build_run_fingerprint`, passing the same objects you pass to
`Matcher`. Also pass `retriever_limit` and `rrf_k` to both if you override them — the
library does not cross-check those either.

## Two source records matched the same target

Reported, not resolved:

```python
print(report.duplicate_targets())   # {"CHEBI:17234": ["s1", "s2"]}
```

Both records stay `matched`. Whether that is an error depends entirely on your data —
"glucose" and "dextrose" both mapping to one ChEBI term is correct, two different
suppliers mapping to one SKU may not be. xwalk cannot know which, so it surfaces the
conflict and does nothing.

## `MissingExtra` on import

```
owl_source requires the optional dependency 'rdflib', which is part of xwalk[ontology].

    pip install 'xwalk[ontology]'
```

Do what it says. If you have already installed the extra and still see this, read the
`underlying import error` line at the bottom — the message carries the real error, so a
broken CUDA install inside an installed `torch` is not disguised as a missing package.

Note that sources are lazy generators, so `MissingExtra` fires on first iteration rather
than at the call that created it.

## Tantivy will not install

There is no fallback BM25 engine, and that is deliberate: substituting one would change
ranking, and ranking is part of a run's identity. A silent swap would quietly invalidate
reproducibility while appearing to work.

See [platforms](../platforms.md) for the supported matrix and the explicit alternatives.

## The CLI returned 1 and the run looks fine

It is fine. `xwalk match` returns `1` when the run completed but the review bucket is
non-empty — that is "something needs your attention", not failure.

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | completed, but something needs attention |
| `2` | usage error |
| `3` | runtime failure |

A `set -e` script needs to handle `1` explicitly:

```bash
xwalk match --job job.yaml --out run/ || [ $? -eq 1 ]
```

## A cache hit reports zero tokens

Correct. `CachingLLM` stores only the response text, so a hit reports zero calls and
zero tokens (it is counted in `Usage.cache_hits` instead) — a hit spends nothing, and
counting it would turn a cost report into a replayed estimate.

The other side of that: a hit also loses `finish_reason` (it becomes `"cached"`) and the
`structured` flag. Do not build logic on those fields downstream of a cache. And with
`temperature > 0` the cache freezes one sample forever, which is usually not what you
want from a nondeterministic run.

## Optimisation reports no improvement

Check `report.stopped_because`:

- `no failures for role X on prompt-train` — nothing to learn from. Either the prompt is
  already good on that partition, or the role's failure class is empty. Check the
  decomposition and pick the role it points at.
- `patience N exhausted` — rounds stopped improving validation.

Also check that `matcher_factory` actually uses the `PromptSet` it is passed. A factory
that ignores its argument re-runs the original slots every round and reports numbers that
measure nothing while looking entirely healthy.

And check the role reaches the prompt you think it does: `rubric` only reaches the
scorer, `disambiguation_steps` only the selector. Optimising a selector by rewriting the
rubric changes nothing the selector sees.

## Still stuck

Open an issue at <https://github.com/jan3657/xwalk/issues> with the relevant
`results.jsonl` line and the `manifest.json` from the run. The attempt trace usually
contains the answer.

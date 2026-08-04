# Drafting and optimising prompts

Everything domain-specific the model sees lives in one `slots.yaml`. This guide covers
writing a first version, improving it against gold labels, and the protocol that keeps
the improvement honest.

## You never write prompt text

The four prompts are assembled from skeletons that ship with xwalk plus your slots. The
skeleton owns the machine-readable contract — the section headings, the instruction that
the model must answer with a candidate key rather than an identifier, the JSON output
shape. Your slots own the domain content.

This split is why a model can be trusted to author slots. A bad slots file produces a
weak rubric; it cannot produce a prompt whose output will not parse, because it has no
access to the part that determines that.

Every candidate slots file — hand-written, drafted, or optimised — is run through
`validate_contract` before it is written anywhere.

## Drafting a first version

```bash
xwalk prompts draft \
  --job job.yaml \
  --describe "matching chemical entity mentions in clinical text to ChEBI terms" \
  --out slots.yaml
```

This shows a model your description plus the first eight source and eight target records
from your job, and asks for slots. If the job already points at a slots file, that is
passed in as a starting point to refine and you get a field-level diff of what changed.

From Python:

```python
from xwalk.prompts.author import draft_slots, write_slots

draft = await draft_slots(
    llm,
    description="matching chemical entity mentions in clinical text to ChEBI terms",
    source_samples=sources[:8],
    target_samples=targets[:8],
)
for warning in draft.warnings:
    print(f"warning: {warning}")
write_slots(draft.slots, "slots.yaml")
```

Warnings tell you what was mechanically repaired — a rubric score clamped into range, or
rows reordered into descending score. Read them: a model that produced an out-of-order
rubric may have produced a confused one.

**Treat the draft as a first draft.** It is a reasonable starting shape, not a finished
artefact. The parts that reliably need a human are the `hard_rules` — the distinctions
that are load-bearing in your domain and invisible from eight sample records. In
chemistry that is things like "a salt is a distinct entity from its parent compound" and
"an anomer is distinct from the unqualified parent". No model infers those from a
sample; you know them and it does not.

## What good slots look like

```yaml
entity_noun: chemical entity mention
target_noun: ChEBI ontology term
domain_brief: biomedical chemistry nomenclature

rubric:
  - score: 1.0
    name: Certain
    when: exact case-insensitive match to the candidate label or one of its synonyms
    example: garlic -> Garlic
  - score: 0.9
    name: High
    when: a normalized form, plural, or common abbreviation of the candidate
    example: ASA -> acetylsalicylic acid
  - score: 0.6
    name: Plausible
    when: the source names a specific instance of the candidate class
    example: Fuji apple -> apple
  - score: 0.4
    name: Speculative
    when: related only by broad category
    example: sugar -> carbohydrate

hard_rules:
  - A salt, ester, or hydrate is a distinct entity from its parent compound
  - >-
    An anomer (alpha-D-glucose, beta-D-glucose) is a distinct term from the unqualified
    parent (glucose). Choose the anomer only when the mention itself specifies it.
  - >-
    If no candidate names the same chemical species, abstain. A plausible-looking
    neighbour is worse than no match, because a wrong mapping is silently wrong.

disambiguation_steps: >-
  Read the surrounding context before choosing. "Sugar" in a clinical blood-glucose
  sentence is glucose; in a cooking sentence it is sucrose.
```

Three things make a rubric work.

**Bands must be distinguishable by the model, not just by you.** If it cannot reliably
tell 0.9 from 0.6, your confidences carry no signal and the calibration warning will fire
on your next evaluation.

**Every band needs an example.** "A normalized form or common abbreviation" is abstract;
`ASA -> acetylsalicylic acid` is not.

**An abstention rule belongs in `hard_rules`.** Without one, models reach for the nearest
neighbour, and a plausible wrong mapping is far more expensive than an abstention because
nobody reviews it.

Which slots reach which prompt is not uniform, and it is worth knowing before you
optimise: `rubric` reaches only the scorer, `disambiguation_steps` only the selector, and
`hard_rules` reaches the selector, scorer, and verifier but not the rewriter. Rewriting
the rubric to fix a *selector* problem changes nothing the selector sees. The full table
is in the [prompts reference](../reference/prompts.md).

## Optimising against gold labels

```python
from xwalk.evaluate import PromptRole
from xwalk.prompts.optimize import OptimizeConfig, optimize_prompt

report = await optimize_prompt(
    matcher_factory=build_matcher,        # (PromptSet) -> Matcher
    source_records=records,
    gold=gold,
    initial=load_slots("slots.yaml"),
    optimiser_llm=strong_model,
    config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4, max_calls=5_000),
    work_dir="opt/",
)
print(report.stopped_because)
print(report.test_report.accepted_precision)
```

```bash
xwalk prompts optimize --job job.yaml --gold gold.csv --out opt/ --rounds 4
```

Each round runs the current best slots over the prompt-train records, selects the
failures that are informative for the chosen role, shows them to the optimising model,
gets back new slots, validates them, and measures them on validation.

`matcher_factory` receives a `PromptSet` and **must use it**. A factory that ignores its
argument re-runs the original slots every round, and the optimiser will report scores
that measure nothing at all while looking entirely healthy.

### The three partitions

| Partition | Default share | Job |
|---|---|---|
| prompt-train | 50% | Supplies the failures shown to the optimising model. |
| validation | 25% | Chooses which round to keep and when to stop. |
| test | 25% | Scored exactly once, at the very end. |

Test is never reported per round. Reporting it each round would make it a second
validation set — you would start choosing rounds by it, and the number you finally quote
would be the maximum over rounds rather than an estimate of anything. That is the same
mistake as tuning on test, one level up.

Assignment is a salted hash of the source id, so it is stable: adding labelled records
later never reshuffles the existing split.

### Choosing the role

Which failures are worth showing depends on which prompt you are improving, and this is
not cosmetic.

| Role | Failures it learns from |
|---|---|
| `SELECTOR` | Only *misjudged* cases — the gold record was on screen and something else was chosen. A case where the gold record was never retrieved teaches a selector nothing, because it never saw the right answer. |
| `SCORER` | Confident wrong accepts (**including** when the gold record was never retrieved — rejecting a slate containing nothing correct is exactly the scorer's job), needless abstentions, and correct matches suppressed below threshold. |
| `REWRITER` | Cases where the first attempt's candidates did not contain a gold id, whether or not a later attempt recovered it. |
| `DOC_TEMPLATE` | Only *never retrieved* cases. |

Let the [failure decomposition](evaluation.md) pick the role for you. If the report says
misjudged, optimise the selector. If it says never retrieved, no amount of selector
optimisation will help — that is a `doc` template problem.

### Budget

```python
from xwalk.prompts.optimize import estimate_calls

print(estimate_calls(config, n_prompt_train=200, n_validation=100, n_test=100))
```

`max_calls` is checked before anything is spent and raises if the estimate exceeds it. It
is a pre-flight estimate assuming every round runs, not a runtime cap — early stopping
means actual usage is usually lower, and nothing re-checks mid-run.

### Why it stopped

`report.stopped_because` is one of exactly three things:

| Value | Meaning |
|---|---|
| `completed N rounds` | Ran to the round limit. |
| `no failures for role X on prompt-train` | Nothing to learn from. Either the prompt is already good on this partition, or you picked a role whose failure class is empty — check the decomposition. |
| `patience N exhausted` | N consecutive rounds failed to improve validation. |

A round whose output fails to validate is not fatal: it is recorded as a warning, counts
against patience, and the run continues with the previous best slots.

### What lands in the work directory

```
opt/
  round_01/slots.yaml      round_01/validation.json
  round_02/slots.yaml      round_02/validation.json
  best/slots.yaml
  report.json
```

Every round's candidate is kept, so you can read what the optimiser actually proposed
rather than only what it scored. A round that produced unusable slots writes no
directory.

## After optimising

Copy `opt/best/slots.yaml` into place and re-run. The prompts feed the run fingerprint,
so this is a full re-run — the optimiser's own numbers were measured on partitions, not
on your whole collection.

Then evaluate normally. If accepted precision moved on test roughly as much as it moved
on validation, the gain is real. If validation improved and test did not, you overfit the
validation set across rounds, and the honest move is to reduce `rounds` or `patience`
next time.

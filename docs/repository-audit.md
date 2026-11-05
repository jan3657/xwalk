# Repository audit and UI extraction review

Reviewed 2026-11-05, starting at commit `bdd0d5916dfbbb0afc188a0deef971dc57c110bc`.

## Coverage and evidence

This review inspected the Python package boundaries, UI imports, CLI entry points,
packaging configuration and assertions, CI and release workflows, deployment files,
and clustering warning/export behavior. It also ran the existing base test suite.
This is a bounded engineering review, not a claim that every defect has been found.

Before extraction, these commands passed in the existing local virtual environment:

```sh
.venv/bin/ruff check src tests examples scripts benchmarks ner-site
.venv/bin/pytest -q -m 'not integration and not dense and not ontology and not sql and not mcp'
```

The test result was **2,809 passed, 18 skipped, 32 deselected**, in 71.15 seconds.
No pre-existing failing tests were verified by these checks. This baseline does not
validate paid provider calls, the deselected optional integrations, every supported
OS/Python combination, browser behavior, or live deployment infrastructure. Post-split
validation is separate from this baseline and must run in both repositories.

## Extraction requirements found

These are dependencies identified in the original tree; their original paths are
recorded here so the migration remains reviewable after files move.

| Area | Finding | Required adjustment |
| --- | --- | --- |
| Backend dependency | `src/xwalk/llm/chatgpt.py` imported `RESOURCE` and `ChatGPTError` from `xwalk.ui.chatgpt`. | Give shared authentication definitions a backend home; prevent a core import from loading the UI package. |
| CLI | `src/xwalk/cli/main.py` implemented the `ui` command and imported the server. | Give the separate application its own entry point and update command documentation. |
| Tests | `tests/test_cluster_warnings.py` used the UI's `lexical_cluster_model` as a regression fixture. | Preserve the backend regression with an independent fixture. |
| Test ownership | `test_review_scale.py` and `test_chatgpt.py` included UI dependencies despite their filenames. | Move application tests and preserve backend adapter coverage explicitly. |
| Application tooling | `build_ontology_library.py`, `build_ui_ontologies.py`, and `bench_review_step.py` depended on UI code or resources. | Move them with the application and repair package/resource paths. |
| Packaging | `test_packaging.py` asserted bundled UI assets and ontology files. | Replace these assertions with checks for the new package boundary; validate the application wheel separately. |
| Configuration | pytest and mypy paths included `ner-site`; CI lint paths and its web build job assumed a monorepo. | Give each repository independently runnable checks. |
| Deployment | The root Dockerfile launched `xwalk ui` and built the ontology library. | Move and adapt deployment assets with the application. |
| Frontend build | The React build wrote into `src/xwalk/ui/static/app`. | Point build output and generated-asset checks at the application's new Python package. |

Extraction acceptance should include clean installs of both packages, backend imports
without the UI installed, application startup against the declared backend version,
frontend build/type checks, and separate test runs. Local editable installs alone
can hide missing package data and undeclared dependencies.

## Existing backend limitations requiring product decisions

The following behavior is documented and covered by regression tests. Changing it
would change matching semantics; it should not be silently bundled into a repository
move.

1. **Bounded clustering evidence can admit contradictory members.**
   `test_regression_hidden_contradictory_members_behind_the_evidence_bound` in
   `tests/test_cluster_warnings.py` demonstrates later members that the same judge
   rejects pairwise. `src/xwalk/cluster/diagnostics.py` reports `evidence_bounded` when
   accepted clusters exceed the member evidence limit. Follow-up: decide whether
   pairwise consistency is a supported guarantee; if so, design additional verification
   and measure its model-call cost before changing the contract.
2. **Exports can retain membership rejected in a later revision.**
   `test_regression_an_exported_membership_a_later_decision_rejected` demonstrates
   this when refinement stops at its iteration limit. `src/xwalk/cluster/exports.py`
   deliberately selects the converged revision or the revision with the fewest
   unresolved sources, breaking ties toward the earliest revision.
   `selected_not_last` makes the distinction visible. Follow-up: decide whether later
   adverse evidence must override revision selection, then update the contract,
   export logic, and regression expectations together.
3. **Lexical candidate retrieval does not guarantee synonym recall.**
   Above `pool.scan_below`, lexical-only clustering may never compare concepts with
   no shared words. The existing `retrieval_bounded` warning describes this limit.
   Follow-up: evaluate recall on representative domain data before choosing dense
   retrieval or larger candidate limits; neither makes retrieval exhaustive.
4. **Resume equivalence depends on deterministic model answers.**
   The existing `resume_determinism` warning states that identical prompts must receive
   identical answers to reproduce an uninterrupted run. Follow-up: treat this as an
   explicit reproducibility constraint in evaluations and provider configuration.

These are follow-up design/evaluation items, not newly verified implementation bugs.
No additional backend defect was established in this review.

## Completed extraction validation

Core: 2,307 tests passed, 40 skipped, 8 deselected (`not integration`).
After installing the missing local build tools, all 29 packaging tests passed.
Ruff lint/format and mypy passed. The core wheel excludes `xwalk/ui/`.
The separate UI repository passed 495 Python tests and 42 frontend tests,
its production frontend build, Ruff lint/format, and Python wheel build.
Both repositories remain uncommitted and unpublished; UI installation currently
requires the matching sibling backend checkout as documented in its README.

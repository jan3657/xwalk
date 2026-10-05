# Task 07: Documentation and release candidate

Protected target envelope: $20. Dependencies: integrated supported features. Drafts may start earlier, but examples must be verified against the final candidate.

## Outcome

A new user can understand xwalk, install it, run a useful example, inspect the result, and know its limits.

## Work

1. Lead the README with the concrete source-to-target or canonicalization problem, a tiny input/output example, and the shortest working route.
2. Include an offline bundled example and a clearly separate real-provider route. State which needs credentials and application inference spending.
3. Show the short high-level Python API and a small CLI workflow. Move the long constructor tour into advanced documentation while preserving useful explanations.
4. Explain matched, unmatched, review-required, failed, deferred, or other actual statuses precisely. Document raw versus reviewed exports and current versus historical views.
5. Explain matching cardinality, clustering relation, uncertainty, resumability, provider settings, and practical data limits.
6. Include measured benchmark evidence only where task 05 produced it. Document comparison methodology and honest limitations.
7. Add a compact architecture guide explaining responsibilities and invariants. Provide a contributor map from common changes to relevant modules and tests.
8. Reconcile outdated or contradictory documentation, including index replacement/append behavior and inference-cost descriptions. Summarize superseded design plans so agents need not ingest them all.
9. Verify installation extras, wheel contents, supported Python/platform gates, changelog, versioning, and release instructions. Check the recorded publishing failure against current setup.
10. Prepare a release candidate and reviewable release notes. Publishing and merging are final user actions unless separately authorized.

## Acceptance

- The README quickstart runs from a clean installation of the built wheel, using exactly the documented commands.
- Optional dependencies remain optional.
- No feature is advertised as supported if its required gates are still pending.
- Links and example paths resolve.
- CI/release gates validate the intended commit and built artifact. Publication status is checked separately.
- The package contains needed templates, type markers, and other runtime resources.
- The final README answers what it does, who it helps, how to start, what it produces, how to review it, and when to use a simpler tool.

Optimize for successful first use and maintainability, not a longer README.


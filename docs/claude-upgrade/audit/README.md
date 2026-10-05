# Audit evidence and limitations

Audit date: 5 October 2026.
Audited source commit: f737099f23ce0c6720963759e13e95d21d8bdf89.
Declared package version: 0.1.1.

## What was executed

diagnose_audited_source.py loads selected functions/classes from a compatible local xwalk checkout using Python AST transformations. Internal xwalk imports are removed and symbols are assembled into a controlled harness. External providers, retrieval, and templates are replaced with deterministic test doubles. The actual SQLite ledger and serialization functions are used for the changed-source resume probe.

This is a focused diagnostic. It is not the normal installed package, the complete pytest suite, a live provider test, or a semantic benchmark. observed.json is the output obtained against the audited snapshot.

To run it from a compatible checkout after copying the handoff into the repository:

```bash
python docs/claude-upgrade/audit/diagnose_audited_source.py .
```

The script prints observations. Its exit code does not assert that the project is correct. Future refactoring may make this harness incompatible. Port the relevant cases into ordinary package-level regression tests during the implementation work instead of maintaining the AST harness as production testing infrastructure.

## Observed cases

1. NaN confidence becomes 1.0 and the fixture is accepted.
2. A fixture with five stage calls reports four, omitting rewrite usage.
3. Verifier retry exhaustion escapes the source matcher.
4. A fatal rewriter error also escapes its stage boundary.
5. Blank lines inside the selected candidate split its detail into the other-candidates block.
6. Disabling stored candidate traces changes a fixture from matched to unmatched.
7. A changed source with ID s1 produces two current-export rows after resume.
8. A failed concurrent batch can close the ledger before another task attempts a write.

The verifier and rewriter observations establish current exception behavior. Whether a particular fatal error should terminate a whole run is a product contract decision. The issue to fix is inconsistent boundaries combined with unsafe batch cleanup, not a requirement to swallow every exception.

## Source-inspection follow-ups

These are hypotheses or directly visible code behaviors that still need normal runtime tests:

| Area | Finding to verify | Proposed check |
|---|---|---|
| CLI outcomes | Failure counts do not determine the match command's exit code | Script a failed record and inspect exit code and JSON |
| Configuration | Unknown job fields are ignored and some values are insufficiently constrained | Reject a misspelled field and an invalid selector |
| Generation settings | Stage requests can override job/client defaults | Capture the actual outgoing provider request |
| Dense identity | Query/document prefixes, normalization, and revision are not fully represented in compatibility checks | Change each semantic setting and verify explicit invalidation |
| Index reuse | Job construction rebuilds retrievers during execution/resume | Reopen a compatible persisted index with an encoder-call counter |
| Exact BM25 boost | Query normalization and indexed exact-form normalization differ for punctuation | Test identifiers such as IL-2 with Tantivy installed |
| Input loading | Some CLI paths load the same target collection twice and load all sources before limiting | Instrument source reads and peak memory |
| Dense memory | Python float lists coexist with array/index representations | Profile a declared corpus before changing storage |
| Concurrent cache misses | Identical simultaneous requests can all reach the provider | Use a barrier to create simultaneous misses |
| Batch scheduling | Chunk barriers can leave capacity idle behind a slow item | Compare a fixed mixed-latency workload |
| Release | Build checks succeeded but the inspected publish job failed | Verify current release availability and publisher setup separately |

## Evidence references

See RESEARCH_SOURCES.md in the parent directory for pinned source links, the successful recorded CI run, and the failed recorded publish workflow.

No repository implementation changes, full-suite test results, real-model quality results, or published release are claimed by this handoff.


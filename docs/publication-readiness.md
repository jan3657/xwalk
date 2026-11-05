# Publication readiness

Checked 5 November 2026. This checklist separates release blockers from later polish.
It does not certify production accuracy or security.

## Before publishing a release candidate

1. **Use the selected distribution name: `xwalk-match`.** Packaging, self-referencing
   extras, installed-version lookup, and artifact checks use this name. Imports and
   the CLI remain `xwalk`. The PyPI JSON endpoint returned 404 on 5 November 2026;
   availability is not a reservation. Do not install the unrelated PyPI `xwalk` package
   in the same environment. Downstream Git dependencies must use a revision containing
   the rename, not an older commit with `name = "xwalk"`.
2. **Configure the new PyPI Trusted Publisher.** Match the repository, workflow filename,
   and `pypi` environment exactly. The owner must do the PyPI account setup; a successful
   GitHub build does not prove it exists. See [releasing](releasing.md) and the
   [PyPI instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
3. **Cut release notes for the actual tree.** Reconcile the long Unreleased section with
   the `0.2.0rc1` version and state which APIs changed. Explain the separate UI install,
   supported Python versions, experimental clustering, and migration from older ledgers.
   Do not treat a GitHub release as proof of publication on PyPI.
4. **Verify the exact artifacts.** Run CI on the intended release commit; build a wheel
   and sdist, run `twine check`, install both in separate clean environments, run
   `pip check`, and execute the README quickstart. Exercise requested extras separately.
   Keep the lightweight base-install check and tag/version match check.
5. **Review data and provider disclosure.** Describe which source/target fields are sent
   to a configured provider and what prompts/output the local ledger retains. Confirm
   example and benchmark data licensing, attribution, and provenance. Do not ship keys,
   private inputs, local run output, or developer-specific paths.
6. **Make quality claims traceable.** Link to recorded configurations, revisions, labels,
   and limitations. Keep synthetic smoke results separate from real-model measurements.
   Any new paid validation needs an explicit budget and a representative labeled sample.

## Useful before wider sharing

- Give a new user only the README and a clean environment; record where they get stuck.
- Add a concise bug-report template requesting versions, a minimal redacted job, the
  command, and expected/actual behavior. Do not ask users to upload raw ledgers or keys.
- Decide a private vulnerability-reporting channel and response expectations before
  adding a security policy. Do not promise an SLA nobody can maintain.
- Add a citation file once the preferred authorship, title, and release identifier are
  agreed; do not invent a DOI or imply that a pilot result is a published paper.
- Consider generated, versioned documentation after the content is stable. The existing
  Markdown guides and tested quickstart are sufficient for an initial candidate.
- Declare the supported Python/platform matrix from CI evidence. Add newer Python
  versions only after testing their native dependencies and extras.
- Review dependency-update automation and branch protections with the maintainer's
  workflow in mind. Avoid checks that nobody can investigate or keep green.

## Evidence already available

The UI extraction PR passed the Linux/macOS/Windows Python 3.10–3.12 matrix, optional
extras, ARM BM25 smoke, lint, type checking, and package build. The browser application
has independent CI in `xwalk-ui`. The README quickstart is executed against the built
wheel by CI. Existing [benchmarks](benchmarks.md) include small real-model pilots;
they do not establish broad accuracy or performance guarantees.

For the bounded code review and known clustering limitations, see
[repository audit](repository-audit.md).

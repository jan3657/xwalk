# Releasing xwalk

**Status (checked 2026-10-05).** xwalk is **not on PyPI**: `GET
https://pypi.org/pypi/xwalk/json` returns 404. The only publish attempt, the
`v0.1.1` release run ([run 30898407571](https://github.com/jan3657/xwalk/actions/runs/30898407571)),
built and verified the artefacts and then failed in the upload step with
`invalid-publisher: valid token, but no corresponding publisher`. Its OIDC claims were
repository `jan3657/xwalk`, workflow `release.yml`, environment `pypi`, ref
`refs/tags/v0.1.1` — exactly what the workflow still sends — so the cause is that no
(pending) Trusted Publisher was registered on PyPI, not the workflow. Step 3 below fixes
it; it can only be done by the owner, and nothing in the repository can verify it.
The current candidate is `0.2.0rc1`.

## One-time setup

These steps need a GitHub session and a PyPI account, and can only be done by the
project owner.

### 1. The GitHub repository

Done: <https://github.com/jan3657/xwalk>, created 2026-08-04 with `main` as the default
branch. For reference, the equivalent from a fresh clone is:

```bash
gh repo create jan3657/xwalk --public --source=. --remote=origin --push
# or create it in the web UI, then:
git remote add origin https://github.com/jan3657/xwalk.git
git push -u origin main
```

### 2. The `pypi` environment

In **Settings → Environments**, create an environment named `pypi`. Add yourself as a
required reviewer if you want a manual gate between the tag and the upload — the release
workflow already targets this environment, so protecting it takes effect immediately.

### 3. PyPI Trusted Publishing

Trusted Publishing exchanges a GitHub OIDC token for a short-lived upload credential.
Nothing is stored: there is no API token in the repository, in CI secrets, or on a
developer machine, and therefore none to leak.

For a project that does not exist on PyPI yet, register a **pending** publisher at
<https://pypi.org/manage/account/publishing/>:

| Field | Value |
|---|---|
| PyPI Project Name | `xwalk` |
| Owner | `jan3657` |
| Repository name | `xwalk` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

All five values must match exactly: the workflow *file name* (not its display name
"Release") and the environment name are part of the token's claims. The first successful
publish converts the pending publisher into a real one. A pending publisher does not
reserve the name; if someone else registers `xwalk` first it is void.

> **The name `xwalk` was checked against the PyPI index on 2026-07-28 and again on 2026-10-05 and is
> available** (`GET https://pypi.org/pypi/xwalk/json` → 404). It is not reserved by that
> check, so re-check before registering the publisher. If it has since been claimed,
> change `[project] name` in `pyproject.toml` and the "PyPI Project Name" above —
> `xwalk-match` and `llm-xwalk` were also free. The import package stays `xwalk` either
> way; the distribution name and the import name are independent.

## Cutting a release

Release candidates use PEP 440 pre-release versions: `0.2.0rc1`, tagged `v0.2.0rc1`
(no hyphen, or the tag check below fails). pip ignores pre-releases unless asked, so a
candidate on PyPI is installed with `pip install --pre xwalk` or `xwalk==0.2.0rc1`, and
`pip install xwalk` keeps resolving to the last final release. After the candidate is
accepted, release the same tree as `0.2.0` (changelog section and version only).

1. **Update the changelog.** Move everything under `## [Unreleased]` into a new dated
   version section with release notes a reviewer can read on their own (why upgrade,
   what is new, what is experimental, breaking changes, known limitations).
2. **Set the version** in `pyproject.toml`. It is the only place a version number
   appears; `xwalk.__version__` reads it back from installed metadata, and
   `tests/test_packaging.py` checks the changelog has a section for it.
3. **Run every gate.** The release workflow re-runs them, but finding out locally is
   cheaper than finding out from a failed tag.

   ```bash
   python -m pytest -q -m "not integration"
   python -m ruff check src tests examples scripts benchmarks
   python -m ruff format --check src tests examples scripts benchmarks
   python -m mypy
   python -m build && python -m twine check dist/*
   python -m venv /tmp/xwalk-fresh && /tmp/xwalk-fresh/bin/pip install dist/*.whl
   python scripts/check_readme_quickstart.py --bin /tmp/xwalk-fresh/bin
   ```

   The last command runs the README quickstart, command for command, against the clean
   install of the wheel you just built, in an empty directory outside the repository.

4. **Commit, tag, push.**

   ```bash
   git commit -am "release: 0.2.0rc1"
   git tag -a v0.2.0rc1 -m "xwalk 0.2.0rc1"
   git push origin main --follow-tags
   ```

The tag push triggers `.github/workflows/release.yml`. Every job checks out the tagged
commit itself:

1. `gates` runs ruff, mypy and the test suite on that exact commit (a tag can point at
   a commit CI never ran) and records its SHA in the run summary.
2. `build` builds the sdist and wheel, runs `twine check`, refuses a tag that does not
   equal the packaged version, checks the wheel ships the prompt skeletons, the bundled
   quickstart, `py.typed` and the licence (and no tests), records the SHA-256 of each
   artefact, installs the wheel into a clean venv (no heavy extras pulled, `pip check`),
   runs the README quickstart against it, and uploads `dist/` as an artefact.
3. `publish` runs only after `build`, in the `pypi` environment, and uploads exactly the
   artefacts `build` verified; it does not rebuild.

Publication is a separate fact from a green build: after the run, confirm with
`pip index versions xwalk --pre` or `https://pypi.org/project/xwalk/`.

### If the tag push does not start a run

A tag pushed within seconds of the branch that first registered the workflows can be
dropped — GitHub had not finished registering `release.yml` when the tag event arrived.
The workflow also accepts `workflow_dispatch`, so re-running it needs no new tag:

```bash
gh workflow run release.yml --ref v0.2.0rc1
```

The tag-matches-version check reads `GITHUB_REF_NAME`, which is the tag name under a
dispatch on a tag ref, so the check still holds.

### If the publish step fails with `invalid-publisher`

The build verified; only the upload was refused. It means PyPI has no trusted publisher
matching this repository — register it as in **One-time setup** above, then re-run just
the failed job from the run page (or dispatch the workflow again). Nothing needs
rebuilding and the version is not burned, because nothing was uploaded.

## When a release is wrong

**Yank it. Do not delete and re-upload.** PyPI refuses a filename it has seen before, so
deleting `xwalk-0.2.0rc1-py3-none-any.whl` does not free the name — it destroys your ability
to ever publish that version again.

```bash
# In the PyPI web UI: Manage → Releases → Options → Yank
```

A yanked release stays installable by exact pin (so anything already depending on it
keeps working) but is skipped by resolvers. Then fix the problem and release the next
version (`0.2.0rc2`, or `0.2.1` after a final release).

## Versioning

Pre-1.0, minor versions may contain breaking changes; this is stated in `CHANGELOG.md`
rather than left implicit. From 1.0.0 the project follows semver, and the invariants
listed in `CONTRIBUTING.md` become compatibility promises rather than design intentions.

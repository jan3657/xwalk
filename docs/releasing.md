# Releasing xwalk

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

The first successful publish converts the pending publisher into a real one.

> **The name `xwalk` was checked against the PyPI index on 2026-07-28 and is
> available** (`GET https://pypi.org/pypi/xwalk/json` → 404). It is not reserved by that
> check, so re-check before registering the publisher. If it has since been claimed,
> change `[project] name` in `pyproject.toml` and the "PyPI Project Name" above —
> `xwalk-match` and `llm-xwalk` were also free. The import package stays `xwalk` either
> way; the distribution name and the import name are independent.

## Cutting a release

1. **Update the changelog.** Move everything under `## [Unreleased]` into a new dated
   version section.
2. **Set the version** in `pyproject.toml`. It is the only place a version number
   appears; `xwalk.__version__` reads it back from installed metadata.
3. **Run every gate.** The release workflow re-runs them, but finding out locally is
   cheaper than finding out from a failed tag.

   ```bash
   python -m pytest -q -m "not integration"
   python -m ruff check src tests examples scripts
   python -m ruff format --check src tests examples scripts
   python -m mypy
   python -m build && python -m twine check dist/*
   ```

4. **Commit, tag, push.**

   ```bash
   git commit -am "release: 0.1.1"
   git tag -a v0.1.1 -m "xwalk 0.1.1"
   git push origin main --follow-tags
   ```

The tag push triggers `.github/workflows/release.yml`, which builds, verifies the tag
matches the packaged version, checks the wheel's contents, installs it into a clean
environment, and only then publishes.

### If the tag push does not start a run

A tag pushed within seconds of the branch that first registered the workflows can be
dropped — GitHub had not finished registering `release.yml` when the tag event arrived.
The workflow also accepts `workflow_dispatch`, so re-running it needs no new tag:

```bash
gh workflow run release.yml --ref v0.1.1
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
deleting `xwalk-0.1.0-py3-none-any.whl` does not free the name — it destroys your ability
to ever publish that version again.

```bash
# In the PyPI web UI: Manage → Releases → Options → Yank
```

A yanked release stays installable by exact pin (so anything already depending on it
keeps working) but is skipped by resolvers. Then fix the problem and release `0.1.1`.

## Versioning

Pre-1.0, minor versions may contain breaking changes; this is stated in `CHANGELOG.md`
rather than left implicit. From 1.0.0 the project follows semver, and the invariants
listed in `CONTRIBUTING.md` become compatibility promises rather than design intentions.

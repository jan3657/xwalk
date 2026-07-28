# Releasing xwalk

## One-time setup

These steps need a GitHub session and a PyPI account, and can only be done by the
project owner.

### 1. The GitHub repository

```bash
gh repo create jan3657/xwalk --public --source=. --remote=origin --push
# or create it in the web UI, then:
git remote add origin git@github.com:jan3657/xwalk.git
git push -u origin master
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

> **The name `xwalk` is not yet claimed and has not been checked.** If it is taken,
> change `[project] name` in `pyproject.toml` and the "PyPI Project Name" above. The
> import package stays `xwalk` either way — the distribution name and the import name
> are independent.

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
   git commit -am "release: 0.1.0"
   git tag -a v0.1.0 -m "xwalk 0.1.0"
   git push origin master --follow-tags
   ```

The tag push triggers `.github/workflows/release.yml`, which builds, verifies the tag
matches the packaged version, checks the wheel's contents, installs it into a clean
environment, and only then publishes.

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

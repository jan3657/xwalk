# Releasing xwalk-match

The distribution is **`xwalk-match`**. The import package and command remain `xwalk`.
The PyPI name `xwalk` belongs to an unrelated project: never publish this repository
under that name or use it in index installation instructions.

As of 5 November 2026, `https://pypi.org/pypi/xwalk-match/json` returns 404. This
checks availability; it does not reserve the name or prove Trusted Publishing is set up.
The current candidate is `0.2.0rc1`. No `xwalk-match` artifact has been published by
this preparation pass.

## One-time account setup

The GitHub repository is `jan3657/xwalk`; its `pypi` environment exists. Before a
release, the owner should review that environment's protection settings and decide
whether to require approval of uploads.

In the intended PyPI account, register a [pending Trusted Publisher](https://pypi.org/manage/account/publishing/):

| Field | Value |
|---|---|
| PyPI Project Name | `xwalk-match` |
| Owner | `jan3657` |
| Repository name | `xwalk` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Follow [PyPI's setup instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
The workflow filename and environment must match exactly. The first successful upload
creates the project. A pending publisher does not reserve its name. GitHub OIDC provides
a short-lived upload credential; no long-lived PyPI API token is required.

The old `v0.1.1` upload failed with `invalid-publisher`; a GitHub release is not evidence
of a successful PyPI upload. Registering a publisher for the old name does not authorize
publishing `xwalk-match`.

## Prepare the candidate

1. Finish the [publication checklist](publication-readiness.md), including data licensing
   and provider-data disclosure. Reconcile the Unreleased notes with the actual release
   tree and document the distribution rename and separate UI installation.
2. Set the version in `pyproject.toml` and add the corresponding changelog section.
   Installed `xwalk.__version__` reads `importlib.metadata.version("xwalk-match")`.
3. Run the gates and build from the exact intended release commit:

   ```bash
   python -m pytest -q -m "not integration"
   python -m ruff check src tests examples scripts benchmarks
   python -m ruff format --check src tests examples scripts benchmarks
   python -m mypy
   python -m pip install build twine
   python -m build
   python -m twine check dist/*
   ```

   Use a clean output directory so stale artifacts cannot be uploaded. The distribution
   name is normalized in filenames: `xwalk_match-0.2.0rc1-py3-none-any.whl` and
   `xwalk_match-0.2.0rc1.tar.gz`. Wheel imports still live under `xwalk/`.
4. Install the wheel in a fresh environment, run `pip check`, `xwalk --version`, and
   `python scripts/check_readme_quickstart.py --bin <fresh-venv>/bin`. Repeat the install
   and quickstart from the sdist in another environment. On Windows, use the venv's
   `Scripts` directory. Test requested extras separately; the base install must not
   pull Torch or other optional providers.
5. Review CI on that commit, then commit, tag and push the intended candidate:

   ```bash
   git tag -a v0.2.0rc1 -m "xwalk-match 0.2.0rc1"
   git push origin v0.2.0rc1
   ```

   Do not move an existing release tag or reuse a published version. Choose the next
   candidate instead. The publication preparation itself must not create a release tag.

## Release workflow

`.github/workflows/release.yml` checks the tag and distribution name, runs the gates,
then builds the wheel and sdist. It verifies their metadata and package contents,
checks that the tag matches the version, installs the wheel in a clean environment,
runs the README quickstart, and records artifact hashes. The publish job uploads those
same artifacts through the `pypi` environment; it does not rebuild them.

Manual retries must target a version tag:

```bash
gh workflow run release.yml --ref v0.2.0rc1
```

A dispatch on a branch is rejected. If upload fails with `invalid-publisher`, verify
all five registration fields above, then retry the failed job. A failed upload does not
by itself mean the version was published; inspect PyPI before choosing the next step.

## Verify publication

After a successful upload, inspect `https://pypi.org/project/xwalk-match/` and install
in a fresh environment:

```bash
python -m pip install 'xwalk-match==0.2.0rc1'
python -m pip check
xwalk --version
python -c "import xwalk; print(xwalk.__version__)"
```

`pip install --pre xwalk-match` opts into release candidates. After a final release,
`pip install xwalk-match` normally selects the latest final version. Check that the
README renders correctly on PyPI and links resolve. Update Git-only installation
instructions once the package is actually available, and announce the tested version
with its limitations and migration notes.

Existing environments installed from the old `xwalk` Git distribution should be
recreated. The unrelated PyPI distribution and this project share an import name;
co-installation is not supported. The browser application must depend on `xwalk-match`
at a revision containing this rename, or on the published version once available.

## When a release is wrong

**Yank it. Do not delete and re-upload.** PyPI will not reuse an uploaded filename.
Yanked versions remain available to exact pins, but normal resolution skips them.
Fix the problem and publish a new candidate (`0.2.0rc2`) or patch release.

Before 1.0, minor versions may contain breaking changes; record them in the changelog.
From 1.0, follow semantic versioning and document the supported compatibility contract.

# xwalk Phase 4 — Release Implementation Plan

**Goal:** Everything between a working library and `pip install xwalk` working for a
stranger: a licence, a changelog, a contributor entry point, a release workflow that
publishes to PyPI via Trusted Publishing, and a 0.1.0 tag.

**Prerequisite:** Phases 1–3 complete. 738 tests green; wheel and sdist build and pass
`twine check`; a fresh-venv install of the wheel imports the public API and runs the
console script.

**Version:** first upload is **0.1.0**. Pre-1.0 is an honest statement about maturity —
the library has run end-to-end against a live provider on 20 records, not against
production volume — and it leaves room to move the API before committing to semver
compatibility guarantees.

## Global constraints

- **Nothing in this phase changes library behaviour.** If a test needs editing to make a
  release step pass, the release step is wrong.
- **No credential is ever written to a file in the repository.** Trusted Publishing (OIDC)
  means there is no API token to leak; the fallback token path is documented but not used.
- **The release workflow is triggered by a tag, never by a push to a branch.** A publish
  that can fire from a merge is a publish that fires by accident.
- **A version number appears in exactly one place.** `pyproject.toml` is the source;
  `xwalk.__version__` reads it from installed metadata.
- **All four gates pass at the end of every task.** Commit at the end of every task.

---

## Task 1: Licence, changelog, and contributor entry point

**Files:** `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `pyproject.toml`

The `license = { text = "MIT" }` declaration currently has no `LICENSE` file behind it,
which is a licensing claim with nothing to back it. PyPI also renders the changelog link
already declared in `[project.urls]`, so that file must exist before the URL is truthful.

- [x] **Step 1:** Write `LICENSE` — MIT, copyright the author, current year.
- [x] **Step 2:** Write `CHANGELOG.md` in Keep-a-Changelog form with a `0.1.0` section
      describing what the first release contains, honestly scoped.
- [x] **Step 3:** Write `CONTRIBUTING.md`: how to set up a dev environment, the four
      gates, the TDD expectation, and how to run the optional-extra suites.
- [x] **Step 4:** Switch `pyproject.toml` to `license = "MIT"` with
      `license-files = ["LICENSE"]` (PEP 639), and confirm the wheel ships the licence.
- [x] **Step 5:** Extend `tests/test_packaging.py` to assert the licence file exists, is
      referenced, and reaches the wheel; assert the changelog documents the declared
      version. Run the gates and commit.

---

## Task 2: Single-source the version

**Files:** `src/xwalk/__init__.py`, `pyproject.toml`, `tests/test_packaging.py`

`__version__` is currently a literal that must be kept in step with `pyproject.toml` by
hand. Two sources of truth for a version number is how a release ends up mislabelled.

- [x] **Step 1:** Write the failing test — `xwalk.__version__` equals the installed
      distribution metadata version.
- [x] **Step 2:** Read the version from `importlib.metadata.version("xwalk")`, falling
      back to a literal only when the package is not installed (a source checkout).
- [x] **Step 3:** Run the gates and commit.

---

## Task 3: The release workflow

**Files:** `.github/workflows/release.yml`, `docs/releasing.md`

Trusted Publishing exchanges a GitHub OIDC token for a short-lived PyPI credential. No
secret is stored anywhere. It requires a one-time configuration on PyPI that only the
project owner can perform, so this task writes and validates the workflow; firing it is
gated on that setup.

- [x] **Step 1:** Write `.github/workflows/release.yml`: triggered on `push: tags: v*`,
      builds, runs `twine check`, re-runs the test suite against the built wheel, then
      publishes with `pypa/gh-action-pypi-publish` using `id-token: write`.
- [x] **Step 2:** Gate the publish job on a GitHub Environment so the owner can require
      manual approval before anything leaves the machine.
- [x] **Step 3:** Write `docs/releasing.md`: the exact PyPI Trusted Publishing settings,
      the tag-and-push procedure, and what to do about a bad release (yank, never
      delete-and-reupload — PyPI will not accept the same filename twice).
- [x] **Step 4:** Validate the workflow YAML parses and its job graph is what it claims.
      Run the gates and commit.

---

## Task 4: Cut 0.1.0

**Files:** `pyproject.toml`, `CHANGELOG.md`

- [x] **Step 1:** Set `version = "0.1.0"` (dropping `.dev0`).
- [x] **Step 2:** Date the changelog entry.
- [x] **Step 3:** Rebuild; confirm `twine check` passes and the artefacts are named
      `xwalk-0.1.0`.
- [x] **Step 4:** Fresh-venv install of the final wheel; run the console script and the
      public-API smoke test.
- [x] **Step 5:** Run every gate, commit, and create an annotated `v0.1.0` tag.

---

## Task 5: Publish — blocked on external setup

These steps require credentials and permissions that do not exist on this machine and
cannot be created from it. They are listed so the remaining work is unambiguous.

- [ ] Create the GitHub repository `jan3657/xwalk` (needs a GitHub session; `gh` is not
      installed here).
- [ ] `git remote add origin` and push `master` plus the `v0.1.0` tag.
- [ ] Confirm CI is green on every matrix row — including the aarch64 and Windows rows,
      which have never executed.
- [ ] Register the Trusted Publisher on PyPI: project `xwalk`, owner `jan3657`,
      repository `xwalk`, workflow `release.yml`, environment `pypi`.
- [ ] Push the tag (or re-run the release workflow) to publish.
- [ ] Verify `pip install xwalk` from PyPI in a clean environment.

**The name `xwalk` is available on PyPI**, checked against the index on 2026-07-28
(`GET /pypi/xwalk/json` → 404). Availability is not a reservation, so re-check at
registration time. `xwalk-match` and `llm-xwalk` were also free as fallbacks; a change
would touch `[project] name` only, since the import package stays `xwalk` regardless.

---

## Definition of done for Phase 4 — VERIFIED 2026-07-28

- [x] `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/releasing.md` all exist, and
      `tests/test_packaging.py` asserts each is real rather than merely present.
- [x] `xwalk.__version__` reads `importlib.metadata.version("xwalk")`; a test asserts it
      equals the installed distribution version.
- [x] `python -m build && twine check dist/*` — both `xwalk-0.1.0-py3-none-any.whl` and
      `xwalk-0.1.0.tar.gz` PASSED.
- [x] The wheel ships `LICENSE` (in `dist-info/licenses/`), `py.typed`, and all four
      `.j2` skeletons.
- [x] Fresh venv, wheel only: `xwalk --version` → `0.1.0`; the public API imports;
      `validate_contract` runs; an end-to-end BM25 build-and-search returns the right
      record; none of torch, numpy, rdflib, sqlalchemy, faiss, litellm is present. The
      **sdist** was installed into a second clean venv and also works.
- [x] `.github/workflows/release.yml` fires only on `push: tags: [v*]` (plus manual
      dispatch), publishes only after the build job, and uses `id-token: write` with no
      token anywhere. Tests assert all of this by parsing the YAML.
- [x] Annotated `v0.1.0` tag exists locally at `a355287`.
- [x] All four gates green: 748 passed / 1 skipped / 1 deselected, ruff, ruff format,
      mypy --strict.

Remaining work is Task 5 only, and every item in it needs credentials or permissions
that do not exist on this machine.

# Contributing to xwalk

## Setup

```bash
git clone https://github.com/jan3657/xwalk
cd xwalk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The optional extras are genuinely optional; install them only for the suites you intend
to run:

```bash
pip install -e ".[dev,ontology]"   # OWL source tests
pip install -e ".[dev,sql]"        # SQL source tests
pip install -e ".[dev,dense]"      # the one test that loads a real sentence-transformer
```

## The four gates

All four must pass before every commit. Not at the end of a branch — at the end of every
commit, because a commit that fails a gate is a commit nobody can bisect through.

```bash
python -m pytest -q -m "not integration"
python -m ruff check src tests examples scripts
python -m ruff format --check src tests examples scripts
python -m mypy
```

`mypy` runs `--strict` over `src/xwalk`, `scripts`, and `examples`. Optional extras are
in an `ignore_missing_imports` override: typing the call sites is what matters, and
chasing stubs for a package the base install must not pull would make the gate depend on
which extras happen to be installed.

## Tests

Write the failing test first, run it, and confirm it fails **for the stated reason**
before writing the implementation. A test that has never been seen to fail is a test that
pins nothing — the review of this codebase's own plans found several assertions that
passed for the wrong reason, and every one of them was found by running it.

Markers:

| Marker | What it needs |
|---|---|
| `integration` | a real LLM provider and an API key |
| `dense` | `xwalk[dense]` |
| `ontology` | `xwalk[ontology]` |
| `sql` | `xwalk[sql]` |

`python -m pytest -q` on a base install must stay green: every optional-dependency test
skips cleanly when its extra is absent.

### Running against a live provider

Copy `.env.example` to `.env` and fill in `XWALK_TEST_API_KEY`, `XWALK_TEST_BASE_URL`,
and `XWALK_TEST_MODEL`. Any OpenAI-compatible endpoint works. `.env` is gitignored and
must stay that way.

```bash
python scripts/check_provider.py    # confirms the endpoint answers before spending
python -m pytest -q -m integration
```

## What the design will not accept

Some invariants are not negotiable, because giving them up is what the library exists to
avoid:

- **A malformed model answer must never resolve to a real target record.** This is why
  candidate keys are opaque and resolution is exact-only. Any change that lets a model's
  raw text become an identifier is wrong regardless of how convenient it is.
- **No silent retriever substitution.** Ranking is part of a run's identity; the retriever
  fingerprint feeds the run fingerprint. If Tantivy is unavailable, the user chooses an
  alternative explicitly. See [docs/platforms.md](docs/platforms.md).
- **The base install stays small.** Every new dependency lives behind an extra, imported
  lazily, with a `MissingExtra` error naming the extra and the install command.
- **Evaluation calls nothing.** `xwalk.evaluate` reads the ledger. It must remain free,
  repeatable, and runnable on a machine with no credentials.
- **The test partition is evaluated once.** Reporting it per optimisation round makes it a
  second validation set.

## Commits

One logical change per commit, with test and implementation together. The message should
say *why*, not restate the diff — the diff is already in the commit.

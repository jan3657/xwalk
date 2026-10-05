# Guidance for Claude sessions in xwalk

- Contributor rules live in `CONTRIBUTING.md`; follow its four gates before every commit:
  `python -m pytest -q -m "not integration"`, `ruff check src tests examples scripts`,
  `ruff format --check src tests examples scripts`, `mypy`.
- The 0.2 upgrade is planned in `docs/claude-upgrade/`. Read `START_HERE.md`, your task
  file, and `CONTRACTS.md` (the binding decisions). Do not load `docs/superpowers/plans/`
  wholesale; they are historical and partly contradicted by `CONTRACTS.md`.
- Use FakeLLM and deterministic fixtures. Never call paid LLM APIs or download large
  model weights unless explicitly configured with a separate budget.
- Never claim test, benchmark or provider results you did not execute. Unknown token
  usage is "unknown", not zero.
- Keep design invariants: opaque candidate keys with exact-only resolution, no silent
  retriever substitution, a light base install, the ledger as source of truth.
- Update `docs/claude-upgrade/STATUS.md` at the end of each task. Balances are
  "unreported" unless the user supplies them.

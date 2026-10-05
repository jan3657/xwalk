# Guidance for Claude implementation sessions

This is suggested project guidance. Read it alongside the repository's existing instructions. Do not replace later instructions or unrelated work.

## Goal

Implement the current assigned task toward a coherent xwalk release. Keep the existing matching SDK usable. Flat equivalence clustering is the main new feature. The audit and competitor analysis are starting evidence, not an instruction to execute every historical plan.

## Work style

- Read the assigned brief, relevant source and tests, and only the documentation needed for that task.
- Do not load all docs/superpowers/plans into context. They contain large historical plans and known conflicts.
- Reproduce reported defects against the actual checkout before fixing them.
- Use a separate working branch. Preserve unrelated changes.
- You may fix directly related bugs and simplify touched code without asking for routine implementation choices.
- Record larger discoveries as follow-up tasks rather than broadening the current change.
- Propose at most three extra features, each with a concrete use case and acceptance check. Implement one only if it fits the approved task scope and remaining allocation.
- Use meaningful behavioral tests. Do not inflate test counts or rewrite tests simply to bless changed behavior.
- Keep functions, names, types, and control flow readable. Do not optimize for line count.
- Reuse existing protocols, stages, FakeLLM, review semantics, and evaluation definitions.
- Preserve evidence and historical results while making current views explicit.

## Resources and costs

- Use offline deterministic fixtures for normal development.
- Do not call paid application APIs or download large model weights unless that specific work and its separate budget are explicitly configured.
- A prompt asking you to stay under a dollar amount is not a hard Claude spending limit. Report your work in bounded checkpoints. The user measures the actual promotional balance.
- Application call limits must include retries, rewrites, concurrent dispatch, and other real upstream calls.
- Unknown token usage must be labelled unknown, not reported as zero.

## Evidence

Do not claim that code compiles, tests pass, a provider works, a speedup exists, or a benchmark is state of the art without the corresponding execution evidence. Distinguish mocked tests from real model evaluation.

## Completion

Return the source revision, changed behavior, tests and commands run, relevant benchmark outputs, unresolved risks, and the next recommended task. Update docs/claude-upgrade/STATUS.md. Prepare reviewable commits or a pull request. Leave release publication and merging as explicit final actions for the user.

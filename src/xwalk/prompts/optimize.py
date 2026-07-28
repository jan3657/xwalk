"""Label-driven prompt optimisation over three partitions.

  prompt-train : failures shown to the optimising model
  validation   : chooses the retained round and the stopping point
  test         : evaluated ONCE, after the final prompt is selected

Reporting test per round would turn it into a second validation set -- the same mistake
one level up -- so this module never evaluates it inside the loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from xwalk.evaluate.failures import FailureCase, PromptRole, render_failure, select_failures
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import EvalReport, evaluate_results
from xwalk.evaluate.partition import Partition, Partitioner, partition_of
from xwalk.llm.base import LLMClient, LLMRequest
from xwalk.llm.parsing import parse_json_object
from xwalk.matcher import Matcher
from xwalk.prompts.author import SLOTS_SCHEMA, write_slots
from xwalk.prompts.contract import PromptSet, PromptSlots, validate_contract
from xwalk.records import MatchResult, Record, Usage

_SYSTEM = (
    "You improve the domain vocabulary of a record-matching system by revising its "
    "slots. You return JSON only. No prose, no code fences."
)

_ROLE_BRIEF: Mapping[PromptRole, str] = {
    PromptRole.SELECTOR: (
        "These are cases where the correct target record WAS shown to the model and the "
        "model picked a different one. Revise the slots so the distinction is stated "
        "explicitly -- usually in `hard_rules` or `disambiguation_steps`."
    ),
    PromptRole.SCORER: (
        "These are confidence-calibration failures: confident wrong accepts, and needless "
        "abstentions or low scores on correct matches. Revise the `rubric` so its bands "
        "separate these cases, and add `hard_rules` where a band is being misapplied."
    ),
    PromptRole.REWRITER: (
        "These are cases where the first search query did not surface the correct record. "
        "Revise `domain_brief` and `disambiguation_steps` to describe how names vary in "
        "this domain -- abbreviations, formal names, qualifiers worth dropping."
    ),
    PromptRole.DOC_TEMPLATE: (
        "These are cases where the correct record was never retrieved at all. Revise "
        "`domain_brief` to describe which target fields carry the searchable names, so a "
        "human can fix the indexing template."
    ),
}


@dataclass(frozen=True)
class OptimizeConfig:
    role: PromptRole = PromptRole.SELECTOR
    rounds: int = 4
    patience: int = 2
    failures_per_round: int = 12
    max_calls: int | None = None
    objective: str = "accepted_precision"
    max_tokens: int = 2048


_DEFAULT_PARTITIONER = Partitioner()


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    slots: PromptSlots
    validation: EvalReport
    improved: bool
    warnings: tuple[str, ...]
    usage: Usage

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "slots": json.loads(self.slots.model_dump_json()),
            "validation": self.validation.as_dict(),
            "improved": self.improved,
            "warnings": list(self.warnings),
        }


_DEFAULT_CONFIG = OptimizeConfig()


@dataclass(frozen=True)
class OptimizeReport:
    best_slots: PromptSlots
    baseline: EvalReport
    rounds: tuple[RoundResult, ...]
    test_report: EvalReport | None
    total_usage: Usage
    stopped_because: str
    partition_sizes: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_slots": json.loads(self.best_slots.model_dump_json()),
            "baseline_validation": self.baseline.as_dict(),
            "rounds": [r.as_dict() for r in self.rounds],
            "test": None if self.test_report is None else self.test_report.as_dict(),
            "stopped_because": self.stopped_because,
            "partition_sizes": dict(self.partition_sizes),
            "usage": {
                "prompt_tokens": self.total_usage.prompt_tokens,
                "completion_tokens": self.total_usage.completion_tokens,
                "calls": self.total_usage.calls,
            },
        }


def estimate_calls(
    config: OptimizeConfig,
    n_prompt_train: int,
    n_validation: int,
    n_test: int,
    *,
    calls_per_record: int = 2,
) -> int:
    """Printed before spending anything. Test is counted once, not once per round."""
    per_round = (n_prompt_train + n_validation) * calls_per_record + 1  # +1 optimiser call
    baseline = n_validation * calls_per_record  # the baseline runs validation only
    return baseline + config.rounds * per_round + n_test * calls_per_record


def _objective(report: EvalReport, name: str) -> float:
    value = getattr(report, name, None)
    return -1.0 if value is None else float(value)


async def _run(matcher: Matcher, records: Sequence[Record]) -> tuple[list[MatchResult], Usage]:
    results = [await matcher.match(record) for record in records]
    return results, sum((r.usage for r in results), Usage.zero())


def _render_optimiser_prompt(
    slots: PromptSlots,
    role: PromptRole,
    failures: Sequence[FailureCase],
) -> str:
    cases = "\n\n".join(render_failure(case) for case in failures)
    return "\n".join(
        [
            _ROLE_BRIEF[role],
            "",
            "## Current slots",
            json.dumps(json.loads(slots.model_dump_json()), indent=2, ensure_ascii=False),
            "",
            f"## Failing cases ({len(failures)})",
            cases,
            "",
            "Return the complete revised slots object. Keep every field; change only what "
            "the failures justify. Rubric scores must remain strictly decreasing and within "
            "[0, 1].",
        ]
    )


async def optimize_prompt(
    *,
    matcher_factory: Callable[[PromptSet], Matcher],
    source_records: Sequence[Record],
    gold: GoldSet,
    initial: PromptSlots,
    optimiser_llm: LLMClient,
    work_dir: str | Path,
    partitioner: Partitioner = _DEFAULT_PARTITIONER,
    partition_overrides: Mapping[str, Partition] | None = None,
    config: OptimizeConfig = _DEFAULT_CONFIG,
    progress: Callable[[str], None] | None = None,
) -> OptimizeReport:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    labelled = [r for r in source_records if r.id in gold]
    by_partition: dict[Partition, list[Record]] = {p: [] for p in Partition}
    for record in labelled:
        by_partition[partition_of(record.id, partitioner, partition_overrides)].append(record)

    train = by_partition[Partition.PROMPT_TRAIN]
    validation = by_partition[Partition.VALIDATION]
    test = by_partition[Partition.TEST]
    sizes = {p.value: len(by_partition[p]) for p in Partition}
    say(f"partitions: {sizes}")

    estimated = estimate_calls(config, len(train), len(validation), len(test))
    say(f"estimated LLM calls: ~{estimated}")
    if config.max_calls is not None and estimated > config.max_calls:
        raise ValueError(
            f"estimated {estimated} calls exceeds max_calls={config.max_calls}; "
            f"reduce rounds, shrink the labelled set, or raise the budget"
        )

    total_usage = Usage.zero()

    # --- baseline, on validation only ---
    best_slots = initial
    baseline_matcher = matcher_factory(PromptSet.from_slots(best_slots))
    baseline_results, usage = await _run(baseline_matcher, validation)
    total_usage = total_usage + usage
    baseline = evaluate_results(baseline_results, gold)
    best_score = _objective(baseline, config.objective)
    say(f"baseline {config.objective}: {best_score:.3f}")

    rounds: list[RoundResult] = []
    stopped_because = f"completed {config.rounds} rounds"
    since_improvement = 0
    source_fields = {r.id: {k: str(v) for k, v in r.fields.items()} for r in train}

    for round_index in range(1, config.rounds + 1):
        # failures come from prompt-train, run under the CURRENT best slots
        train_matcher = matcher_factory(PromptSet.from_slots(best_slots))
        train_results, usage = await _run(train_matcher, train)
        total_usage = total_usage + usage

        failures = [
            replace(case, source_fields=source_fields.get(case.source_id, {}))
            for case in select_failures(
                train_results, gold, config.role, limit=config.failures_per_round
            )
        ]

        if not failures:
            stopped_because = f"no failures for role {config.role.value} on prompt-train"
            say(stopped_because)
            break

        response = await optimiser_llm.complete(
            LLMRequest(
                system=_SYSTEM,
                user=_render_optimiser_prompt(best_slots, config.role, failures),
                schema=SLOTS_SCHEMA,
                schema_name="slots",
                max_tokens=config.max_tokens,
            )
        )
        total_usage = total_usage + response.usage

        warnings: list[str] = []
        try:
            payload = parse_json_object(response.text)
            candidate_slots = PromptSlots.model_validate(payload)
            validate_contract(PromptSet.from_slots(candidate_slots))
        except Exception as exc:  # a bad round must not end the run
            warnings.append(f"round {round_index} produced unusable slots: {exc}")
            say(warnings[-1])
            rounds.append(
                RoundResult(
                    round_index=round_index,
                    slots=best_slots,
                    validation=baseline,
                    improved=False,
                    warnings=tuple(warnings),
                    usage=response.usage,
                )
            )
            since_improvement += 1
            if since_improvement >= config.patience:
                stopped_because = f"patience {config.patience} exhausted"
                say(stopped_because)
                break
            continue

        candidate_matcher = matcher_factory(PromptSet.from_slots(candidate_slots))
        candidate_results, usage = await _run(candidate_matcher, validation)
        total_usage = total_usage + usage
        candidate_report = evaluate_results(candidate_results, gold)
        candidate_score = _objective(candidate_report, config.objective)

        improved = candidate_score > best_score
        say(
            f"round {round_index}: validation {config.objective} "
            f"{candidate_score:.3f} ({'kept' if improved else 'discarded'})"
        )

        round_dir = work_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        write_slots(candidate_slots, round_dir / "slots.yaml")
        (round_dir / "validation.json").write_text(
            json.dumps(candidate_report.as_dict(), indent=2), encoding="utf-8"
        )

        rounds.append(
            RoundResult(
                round_index=round_index,
                slots=candidate_slots,
                validation=candidate_report,
                improved=improved,
                warnings=tuple(warnings),
                usage=response.usage,
            )
        )

        if improved:
            best_slots = candidate_slots
            best_score = candidate_score
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                stopped_because = f"patience {config.patience} exhausted"
                say(stopped_because)
                break

    # --- test, once, after the final prompt is chosen ---
    say("evaluating the selected prompt on the test partition (once)")
    test_matcher = matcher_factory(PromptSet.from_slots(best_slots))
    test_results, usage = await _run(test_matcher, test)
    total_usage = total_usage + usage
    test_report = evaluate_results(test_results, gold)

    best_dir = work_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    write_slots(best_slots, best_dir / "slots.yaml")

    report = OptimizeReport(
        best_slots=best_slots,
        baseline=baseline,
        rounds=tuple(rounds),
        test_report=test_report,
        total_usage=total_usage,
        stopped_because=stopped_because,
        partition_sizes=sizes,
    )
    (work_dir / "report.json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report

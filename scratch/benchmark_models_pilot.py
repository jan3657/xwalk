#!/usr/bin/env python
"""Run a comparative test of candidate models on real Ref_zivila pilot records."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from xwalk.config import load_job
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.matcher import Matcher

MODELS = [
    # (id, profile, display_name, cost_tier)
    ("nex-agi/nex-n2.5-mini:free", "unknown", "Nex N2.5 Mini", "100% Free"),
    ("qwen/qwen3-next-80b-a3b-instruct", "unknown", "Qwen 3 Next 80B", "$0.09/M prompt"),
    ("deepseek/deepseek-v4-flash", "openai", "DeepSeek v4 Flash", "$0.30/M prompt"),
]


async def run_benchmark():
    job_path = Path("examples/ref_zivila/jobs/foodon/job.yaml")
    job = load_job(job_path)

    api_key = os.environ.get("XWALK_TEST_API_KEY")
    base_url = os.environ.get("XWALK_TEST_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        print("XWALK_TEST_API_KEY unset!")
        return 1

    print("Building target store and retriever from existing data...")
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = job.build_store()
    retrievers = job.build_retrievers(targets, templates, Path("runs/ref_zivila/foodon/index"))

    # Load 3 test records from source_pilot
    source_records = list(job.build_source_records())[:3]
    print(f"\nEvaluating on {len(source_records)} pilot records:")
    for r in source_records:
        print(f"  - {r.id}: {r.fields.get('mention_en')} (slo: {r.fields.get('name_slo')})")

    results = {}

    for model_id, profile, display_name, cost in MODELS:
        print(f"\n{'='*60}\nTesting {display_name} ({model_id})\nProfile: {profile}, Cost: {cost}\n{'='*60}")
        llm = OpenAICompatClient(
            base_url=base_url,
            model=model_id,
            api_key=api_key,
            profile=profile,
            timeout=30.0,
            max_retries=1,
        )

        matcher = job.build_matcher(store=store, retrievers=retrievers, llm=llm)
        model_results = []
        t_start = time.perf_counter()

        for record in source_records:
            t0 = time.perf_counter()
            try:
                res = await matcher.match(record)
                dt = time.perf_counter() - t0
                target_label = ""
                if res.matched_id and res.matched_id in store._records:
                    target_label = store._records[res.matched_id].fields.get("label", "")

                print(
                    f"  [{record.fields.get('mention_en')}] -> {res.status.value.upper()} "
                    f"id={res.matched_id} ({target_label!r}) "
                    f"conf={res.confidence} in {dt:.2f}s"
                )
                print(f"     Reason: {res.reason.value} | Explanation: {res.explanation[:120] if res.explanation else ''}")
                model_results.append({
                    "record_id": record.id,
                    "food": record.fields.get("mention_en"),
                    "status": res.status.value,
                    "matched_id": res.matched_id or "",
                    "label": target_label,
                    "confidence": res.confidence,
                    "reason": res.reason.value,
                    "explanation": res.explanation,
                    "elapsed": dt,
                    "tokens": f"{res.usage.prompt_tokens}+{res.usage.completion_tokens}",
                })
            except Exception as e:
                dt = time.perf_counter() - t0
                err_str = str(e).split("\n")[0][:100]
                print(f"  [{record.fields.get('mention_en')}] -> ERROR in {dt:.2f}s: {type(e).__name__}: {err_str}")
                model_results.append({
                    "record_id": record.id,
                    "food": record.fields.get("mention_en"),
                    "status": "ERROR",
                    "matched_id": "",
                    "label": "",
                    "confidence": None,
                    "reason": "error",
                    "explanation": f"{type(e).__name__}: {err_str}",
                    "elapsed": dt,
                    "tokens": "0+0",
                })

        total_time = time.perf_counter() - t_start
        await llm.aclose()
        results[display_name] = {
            "model_id": model_id,
            "cost": cost,
            "total_time": total_time,
            "items": model_results,
        }

    print("\n\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    for name, data in results.items():
        print(f"\nModel: {name} ({data['model_id']}) | Cost: {data['cost']} | Total Time: {data['total_time']:.2f}s")
        for item in data["items"]:
            print(
                f"  * {item['food']:<25} | status: {item['status']:<12} | "
                f"ID: {item['matched_id']:<16} | {item['label']:<28} | "
                f"conf: {str(item['confidence']):<5} | time: {item['elapsed']:.2f}s"
            )


if __name__ == "__main__":
    asyncio.run(run_benchmark())

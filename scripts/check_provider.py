#!/usr/bin/env python
"""Check that the configured live provider actually answers.

    source ./_env.sh && python scripts/check_provider.py

Exercises the same client the integration test uses, so a pass here means
`pytest -m integration` will pass too. Never prints the API key.
"""

from __future__ import annotations

import asyncio
import os
import sys

from xwalk.llm.base import LLMError, LLMRequest
from xwalk.llm.openai_compat import CAPABILITY_PROFILES, OpenAICompatClient
from xwalk.llm.parsing import parse_json_object

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "additionalProperties": False,
}


def _mask(secret: str) -> str:
    """Enough to tell two keys apart, not enough to use one."""
    return f"{secret[:6]}…{secret[-4:]} ({len(secret)} chars)" if len(secret) > 12 else "<short>"


async def main() -> int:
    key = os.environ.get("XWALK_TEST_API_KEY")
    base = os.environ.get("XWALK_TEST_BASE_URL")
    model = os.environ.get("XWALK_TEST_MODEL")
    profile = os.environ.get("XWALK_TEST_PROFILE", "openai")

    missing = [
        name
        for name, value in (
            ("XWALK_TEST_API_KEY", key),
            ("XWALK_TEST_BASE_URL", base),
            ("XWALK_TEST_MODEL", model),
        )
        if not value
    ]
    if missing:
        print(f"not configured: {', '.join(missing)} unset")
        print("fill them in .env, then `source ./_env.sh`")
        print("(the integration test skips in this state, which is green, not broken)")
        return 2
    assert key and base and model

    if profile not in CAPABILITY_PROFILES:
        print(f"unknown XWALK_TEST_PROFILE {profile!r}; pick one of {sorted(CAPABILITY_PROFILES)}")
        return 2

    print(f"endpoint : {base}")
    print(f"model    : {model}")
    print(f"profile  : {profile}")
    print(f"key      : {_mask(key)}")

    client = OpenAICompatClient(
        base_url=base, model=model, api_key=key, profile=profile, max_retries=1
    )
    try:
        response = await client.complete(
            LLMRequest(
                system="You return JSON only. No prose, no code fences.",
                user='Return exactly {"a": "x"}',
                schema=SCHEMA,
                max_tokens=64,
            )
        )
    except LLMError as exc:
        print(f"\nFAILED  {type(exc).__name__}: {exc}")
        return 1
    finally:
        await client.aclose()

    try:
        payload = parse_json_object(response.text)
    except LLMError as exc:
        print(f"\nreachable, but the reply did not parse: {exc}")
        print(f"raw: {response.text[:200]!r}")
        return 1

    print(
        f"\nOK  structured={response.structured}  "
        f"tokens={response.usage.prompt_tokens}+{response.usage.completion_tokens}  "
        f"parsed={payload}"
    )
    if payload.get("a") != "x":
        print("note: model answered but ignored the instruction; the transport still works")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

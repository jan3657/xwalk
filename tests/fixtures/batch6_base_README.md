# Batch 6 routing baseline

`batch6_base_requests.json` and `batch6_base_wire.json` were captured before source
edits from base commit `8e4a447c99c81a05f0f6005d36ae3901c3708eb3`.
`tests/test_bounded_routing.py` supplies the P3/P6/P8 and A#8 scripted scenarios
(adapted from the advisor probes). P4 is intentionally excluded: its default behavior
is the bug fixed in this batch.

The first fixture records complete FakeLLM requests and decision observations. The
second records exact UTF-8 HTTP request bodies from OpenAICompatClient using
httpx.MockTransport, plus the complete run fingerprint components. No network or
paid calls are involved. Tests compare bodies as strings, including whitespace and
key order, without JSON normalization.

To reproduce, archive the base commit into a temporary checkout, copy
`tests/test_bounded_routing.py` there, and run with that checkout's `src` on PYTHONPATH.
For each key in `base_cases()`, run `capture_wire(key)` and serialize the resulting
mapping with `json.dumps(..., indent=2) + "\n"`. For the FakeLLM fixture, construct
`FakeLLM(replies)`, `ScriptedRetriever(queries)`, run `build(llm, retriever).match(SOURCE)`
and serialize `observations(result, llm, retriever)` for each case. Keep all new
routing flags at their defaults. Regenerate only from the named base commit.

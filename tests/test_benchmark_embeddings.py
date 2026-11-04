"""The embedding bake-off (`benchmarks/embeddings.py`), offline.

A mocked OpenRouter answers the price list and `/embeddings` with letter-histogram
vectors, so the harness runs end to end on the in-repo TSV pilots without a key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from benchmarks import embeddings as bake
from xwalk.retrieval.dense import encoder_identity
from xwalk.retrieval.embeddings_api import OpenAIEmbeddingsEncoder

GOOD = bake.Preset("good", "vendor/good", query_prefix="q: ", doc_prefix="d: ")
OTHER = bake.Preset("other", "vendor/other", query_input_type="search_query")
BROKEN = bake.Preset("broken", "vendor/missing")
NLM = bake.DATASETS["nlm_gene"]
NCBI = bake.DATASETS["ncbi_disease"]


class OpenRouter:
    def __init__(self, price: str = "0.00000002") -> None:
        self.price = price
        self.posts: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            data = [
                {"id": m, "pricing": {"prompt": self.price}}
                for m in ("vendor/good", "vendor/other", "vendor/missing")
            ]
            return httpx.Response(200, json={"data": data})
        body = json.loads(request.content)
        self.posts.append(body)
        if body["model"] == "vendor/missing" or body.get("input_type") == "refused":
            return httpx.Response(404, json={"error": {"message": "No endpoints found"}})
        vectors = []
        for i, text in enumerate(body["input"]):
            vector = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            vectors.append({"index": i, "embedding": vector})
        usage = {"prompt_tokens": 10 * len(body["input"]), "total_tokens": 0, "cost": 0.001}
        return httpx.Response(200, json={"data": vectors, "usage": usage})


@pytest.fixture
def gateway(monkeypatch) -> OpenRouter:
    monkeypatch.setenv("XWALK_TEST_BAKE_KEY", "sk-bake")
    return OpenRouter()


def _options(tmp_path: Path, gateway: OpenRouter, *presets: bake.Preset, **kw: Any):
    settings: dict[str, Any] = {
        "presets": list(presets),
        "datasets": [NLM],
        "out": tmp_path / "out",
        "api_key_env": "XWALK_TEST_BAKE_KEY",
        "transport": httpx.MockTransport(gateway),
        "log": lambda line: None,
    }
    settings.update(kw)
    return bake.Options(**settings)


def test_rank_metrics_counts_the_first_gold_hit():
    ranked = [["a", "g", "b"], ["x", "y"], ["g"]]
    wanted = [frozenset({"g"}), frozenset({"g"}), frozenset({"g", "h"})]
    metrics = bake.rank_metrics(ranked, wanted, (1, 2))
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@2"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((0.5 + 0 + 1) / 3)


def test_the_bake_off_scores_each_preset_and_the_bm25_baseline(tmp_path, gateway):
    rows = bake.run(_options(tmp_path, gateway, GOOD, OTHER))
    by_method = {r.method: r for r in rows}
    assert set(by_method) == {"bm25", "good", "other"}
    for row in rows:
        assert row.error is None
        assert row.alone is not None and 0.0 <= row.alone["mrr"] <= 1.0
        assert row.labelled == 50
    assert by_method["bm25"].fused is None
    assert by_method["good"].fused is not None
    assert by_method["good"].dimension == 26
    assert by_method["good"].prompt_tokens and by_method["good"].cost is not None


def test_each_side_gets_its_own_prefix_and_input_type(tmp_path, gateway):
    bake.run(_options(tmp_path, gateway, GOOD, OTHER))
    good = [b for b in gateway.posts if b["model"] == "vendor/good"]
    assert all(t.startswith(("q: ", "d: ")) for b in good for t in b["input"])
    other = [b for b in gateway.posts if b["model"] == "vendor/other"]
    assert {b.get("input_type") for b in other} == {None, "search_query"}


def test_queries_are_embedded_in_batches_not_one_call_each(tmp_path, gateway):
    bake.run(_options(tmp_path, gateway, GOOD))
    queries = [b for b in gateway.posts if b["input"] and b["input"][0].startswith("q: ")]
    assert sum(len(b["input"]) for b in queries) <= 50
    assert len(queries) == 1


def test_a_rerun_reuses_the_cached_index(tmp_path, gateway):
    bake.run(_options(tmp_path, gateway, GOOD))
    first = len(gateway.posts)
    rows = bake.run(_options(tmp_path, gateway, GOOD))
    good = next(r for r in rows if r.method == "good")
    assert good.index_action == "opened"
    second = gateway.posts[first:]
    # the dimension probe and the batched queries; no document is embedded again
    assert not any(
        t.startswith("d: ") and t != "d: dimension probe" for b in second for t in b["input"]
    )


def test_a_failing_preset_is_recorded_and_the_run_goes_on(tmp_path, gateway):
    rows = bake.run(_options(tmp_path, gateway, BROKEN, GOOD))
    broken = [r for r in rows if r.method == "broken"]
    assert broken and all("No endpoints found" in (r.error or "") for r in broken)
    assert next(r for r in rows if r.method == "good").error is None


def test_a_preset_failing_after_its_index_is_built_is_recorded_per_dataset(tmp_path, gateway):
    """The probe and the documents succeed; the query embedding is refused."""
    half = bake.Preset("half", "vendor/good", query_input_type="refused")
    rows = bake.run(_options(tmp_path, gateway, half, GOOD, datasets=[NLM, NCBI]))
    failed = [r for r in rows if r.method == "half"]
    assert len(failed) == 2 and all("No endpoints found" in (r.error or "") for r in failed)
    assert all(r.alone is None for r in failed)
    assert all(r.error is None for r in rows if r.method == "good")


def test_a_preset_can_index_one_vector_per_name(tmp_path, gateway):
    named = bake.Preset("named", "vendor/good", name_fields=("label", "synonyms"))
    rows = bake.run(_options(tmp_path, gateway, named))
    row = next(r for r in rows if r.method == "named")
    assert row.error is None and row.alone is not None
    meta = json.loads(
        (tmp_path / "out" / "index" / "nlm_gene" / "named" / "xwalk_meta.json").read_text()
    )
    assert meta["components"]["name_fields"] == ["label", "synonyms"]
    assert meta["vector_count"] > meta["doc_count"]


def test_the_cost_guard_refuses_an_expensive_run(tmp_path, monkeypatch):
    monkeypatch.setenv("XWALK_TEST_BAKE_KEY", "sk-bake")
    pricey = OpenRouter(price="0.01")
    with pytest.raises(SystemExit, match="exceeds --max-cost"):
        bake.run(_options(tmp_path, pricey, GOOD, max_cost=0.01))
    assert pricey.posts == []


def test_a_missing_key_stops_before_any_call(tmp_path, gateway, monkeypatch):
    monkeypatch.delenv("XWALK_TEST_BAKE_KEY")
    with pytest.raises(SystemExit, match="XWALK_TEST_BAKE_KEY"):
        bake.run(_options(tmp_path, gateway, GOOD))
    assert gateway.posts == []


def test_the_report_and_raw_results_are_written(tmp_path, gateway):
    options = _options(tmp_path, gateway, GOOD, BROKEN, datasets=[NLM, NCBI])
    results, report = bake.write(bake.run(options), options)
    data = json.loads(results.read_text(encoding="utf-8"))
    assert {r["dataset"] for r in data["rows"]} == {"nlm_gene", "ncbi_disease"}
    assert data["meta"]["presets"]["good"]["model"] == "vendor/good"
    text = report.read_text(encoding="utf-8")
    assert "## nlm_gene" in text and "## ncbi_disease" in text
    assert "`vendor/good`" in text and "No endpoints found" in text
    assert "sk-bake" not in text and "sk-bake" not in results.read_text(encoding="utf-8")


def test_the_query_prefetch_keeps_the_encoder_identity(gateway):
    inner = OpenAIEmbeddingsEncoder(
        "https://x.example/v1",
        "m",
        dimensions=26,
        query_prefix="q: ",
        transport=httpx.MockTransport(gateway),
    )
    wrapped = bake.PrefetchedQueries(inner)
    assert encoder_identity(wrapped) == encoder_identity(inner)
    wrapped.prefetch(["glucose", "glucose", "car"])
    assert wrapped.encode(["car", "glucose"], is_query=True) == inner.encode(
        ["car", "glucose"], is_query=True
    )


def test_the_throttle_spaces_requests_evenly():
    now = [0.0]
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    inner = httpx.MockTransport(lambda request: httpx.Response(200))
    throttle = bake.ThrottledTransport(30, inner, clock=lambda: now[0], sleep=sleep)
    with httpx.Client(transport=throttle) as client:
        for _ in range(3):
            client.get("https://x.example/")
    assert waits == [2.0, 2.0]


def test_free_presets_are_throttled():
    free = [p for p in bake.PRESETS.values() if p.model.endswith(":free")]
    assert free and all(p.requests_per_minute for p in free)


def test_every_preset_is_well_formed():
    for name, preset in bake.PRESETS.items():
        assert name == preset.name
        assert preset.model and (preset.local or "/" in preset.model)


def test_the_cli_rejects_an_unknown_preset(capsys):
    with pytest.raises(SystemExit):
        bake.main(["--model", "nope", "--dry-run"])
    assert "unknown preset" in capsys.readouterr().err

"""OAuth trust boundaries and subscription inference, with no live account or spend."""

import base64
import hashlib
import json
import os
import threading
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from xwalk.config import LLMSpec
from xwalk.llm.base import LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.budget import CallBudget, CallLimitExceeded
from xwalk.llm.chatgpt import ChatGPTClient
from xwalk.ui import make_server
from xwalk.ui.api import ApiError, Workspace, dispatch
from xwalk.ui.chatgpt import ISSUER, PLAN_SCOPE, ChatGPTAuth, ChatGPTError, host_id
from xwalk.ui.projects import llm_block

pytestmark = pytest.mark.chatgpt
jwt = pytest.importorskip("jwt")
rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")


@pytest.fixture
def oauth(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    public.update(kid="test-key", alg="RS256", use="sig")
    issued = "oaiapp_test"
    ctx = {"claims": {}, "scope": PLAN_SCOPE + " resource.invoke", "requests": []}

    def handle(request):
        ctx["requests"].append(request)
        if request.url.path == "/.well-known/jwks.json":
            return httpx.Response(200, json={"keys": [public]})
        if request.url.path == "/api/accounts/oauth/token":
            form = parse_qs(request.content.decode())
            assert form["client_id"] == [issued]
            assert "client_secret" not in form
            claims = {
                "iss": ISSUER,
                "aud": issued,
                "sub": "user-1",
                "exp": time.time() + 300,
                "nonce": ctx["nonce"],
                "email": "person@example.com",
                **ctx["claims"],
            }
            return httpx.Response(
                200,
                json={
                    "access_token": "access-secret",
                    "refresh_token": "refresh-secret",
                    "token_type": "Bearer",
                    "scope": ctx["scope"],
                    "expires_in": 3600,
                    "id_token": jwt.encode(
                        claims, key, algorithm="RS256", headers={"kid": "test-key"}
                    ),
                },
            )
        if request.url.path == "/v1/models":
            assert request.headers["authorization"] == "Bearer access-secret"
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"slug": "allowed", "display_name": "Allowed model", "visibility": "list"},
                        {"slug": "hidden", "display_name": "Hidden model", "visibility": "hidden"},
                    ]
                },
            )
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={"revocation_endpoint": ISSUER + "/revoke"})
        if request.url.path == "/revoke":
            assert parse_qs(request.content.decode())["token"] == ["refresh-secret"]
            return httpx.Response(200)
        raise AssertionError(request.url)

    auth = ChatGPTAuth(
        "http://127.0.0.1:8080/auth/callback",
        "urn:uuid:test",
        transport=httpx.MockTransport(handle),
    )

    def start(account=None):
        params = parse_qs(urlsplit(auth.begin(account)).query)
        ctx["nonce"] = params["nonce"][0]
        return params, {"state": params["state"][0], "code": "test-code", "client_id": issued}

    return auth, ctx, start


def test_pkce_and_dynamic_registration(oauth):
    auth, _, start = oauth
    params, callback = start()
    assert params["client_id"] == ["dynamic_agent_client"]
    assert params["agent_name_hint"] == ["xwalk"]
    assert params["redirect_uri"] == ["http://127.0.0.1:8080/auth/callback"]
    verifier = auth._pending["verifier"]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    assert params["code_challenge"] == [challenge]
    auth.finish(callback)
    assert auth.status()["plan_enabled"]
    assert auth.access_token() == "access-secret"
    assert "secret" not in json.dumps(auth.status())
    assert auth.models() == [{"slug": "allowed", "display_name": "Allowed model"}]


def test_state_validation_and_replay(oauth):
    auth, ctx, start = oauth
    _, callback = start()
    with pytest.raises(ChatGPTError, match="state"):
        auth.finish({**callback, "state": "attacker"})
    assert not ctx["requests"]
    auth.finish(callback)
    with pytest.raises(ChatGPTError, match="state"):
        auth.finish(callback)


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "https://attacker.invalid"},
        {"aud": "another-client"},
        {"exp": 1},
        {"nonce": "wrong"},
        {"sub": None},
    ],
)
def test_invalid_identity_never_connects(oauth, claims):
    auth, ctx, start = oauth
    ctx["claims"] = claims
    _, callback = start()
    with pytest.raises(ChatGPTError):
        auth.finish(callback)
    assert not auth.status()["connected"]


def test_declined_consent_and_expired_transaction(oauth):
    auth, ctx, start = oauth
    _, callback = start()
    with pytest.raises(ChatGPTError, match="declined"):
        auth.finish({"state": callback["state"], "error": "access_denied"})
    assert not ctx["requests"]
    _, callback = start()
    auth._pending["created"] = 1
    with pytest.raises(ChatGPTError, match="expired"):
        auth.finish(callback)


def test_identity_alone_cannot_use_plan(oauth):
    auth, ctx, start = oauth
    ctx["scope"] = "openid profile email"
    _, callback = start()
    auth.finish(callback)
    assert auth.status()["connected"]
    assert not auth.status()["plan_enabled"]
    with pytest.raises(ChatGPTError, match="Enable"):
        auth.access_token()
    params, _ = start(auth.status()["active"])
    assert params["prompt"] == ["consent"]


def test_returning_registration_and_identity_binding(oauth):
    auth, ctx, start = oauth
    _, callback = start()
    auth.finish(callback)
    params, callback = start(auth.status()["active"])
    assert params["client_id"] == ["oaiapp_test"]
    assert "agent_name_hint" not in params
    assert params["id_token_hint"]
    with pytest.raises(ChatGPTError, match="different"):
        auth.finish({**callback, "client_id": "oaiapp_attacker"})
    _, callback = start(auth.status()["active"])
    ctx["claims"] = {"sub": "user-2"}
    with pytest.raises(ChatGPTError, match="identity did not match"):
        auth.finish(callback)
    assert auth.status()["connected"]  # original credentials survive


def test_persistence_is_owner_only_and_restores_account(oauth, tmp_path):
    auth, _, start = oauth
    auth.storage = tmp_path / "protected" / "accounts.json"
    _, callback = start()
    auth.finish(callback)
    if os.name != "nt":
        assert auth.storage.stat().st_mode & 0o777 == 0o600
    reopened = ChatGPTAuth(auth.redirect_uri, auth.host, storage=auth.storage)
    assert reopened.status()["active"] == "oaiapp_test"
    assert reopened.access_token() == "access-secret"
    assert host_id(tmp_path / "config") == host_id(tmp_path / "config")


def test_refresh_and_sign_out_preserve_registration(oauth):
    auth, ctx, start = oauth
    _, callback = start()
    auth.finish(callback)
    auth._profiles["oaiapp_test"]["expires_at"] = 1
    assert auth.access_token() == "access-secret"
    form = parse_qs(ctx["requests"][-1].content.decode())
    assert form["grant_type"] == ["refresh_token"]
    assert "scope" not in form
    assert auth.sign_out()
    assert not auth.status()["connected"]
    params, _ = start("oaiapp_test")
    assert params["client_id"] == ["oaiapp_test"]
    assert "id_token_hint" not in params
    with pytest.raises(ChatGPTError):
        auth.access_token()


def sse(*events):
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events)


def completed(text='{"key":"C01"}'):
    return {
        "type": "response.completed",
        "response": {
            "model": "allowed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 11, "output_tokens": 3},
        },
    }


async def test_responses_contract_completion_and_usage():
    def handle(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer secret"
        assert json.loads(request.content) == {
            "model": "allowed",
            "instructions": "system instructions",
            "input": [{"role": "user", "content": "user text"}],
            "store": False,
            "stream": True,
        }
        return httpx.Response(200, text=sse(completed()))

    client = ChatGPTClient("allowed", lambda: "secret", transport=httpx.MockTransport(handle))
    response = await client.complete(
        LLMRequest(
            system="system instructions",
            user="user text",
            temperature=1,
            max_tokens=5,
            seed=9,
            extra={"store": True},
        )
    )
    assert response.text == '{"key":"C01"}'
    assert (
        response.usage.calls,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    ) == (1, 11, 3)
    assert client.fingerprint == ChatGPTClient("allowed", lambda: "rotated").fingerprint


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "response.output_text.delta", "delta": "partial"}],
        [
            {
                "type": "response.failed",
                "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
            }
        ],
        [{"type": "response.incomplete"}],
        [{"type": "error"}],
        [],
    ],
)
async def test_partial_and_failed_streams_are_never_successful(events):
    client = ChatGPTClient(
        "allowed",
        lambda: "secret",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=sse(*events))),
    )
    with pytest.raises(LLMFatalError) as exc:
        await client.complete(LLMRequest(system="", user=""))
    assert exc.value.usage.calls == 1
    if events and events[0]["type"] == "response.failed":
        assert "subscription_sharing_usage_limit_exceeded" in str(exc.value)


@pytest.mark.parametrize(
    "status,error",
    [
        (401, LLMFatalError),
        (403, LLMFatalError),
        (429, LLMRetryableError),
        (503, LLMRetryableError),
    ],
)
async def test_http_admission_errors(status, error):
    client = ChatGPTClient(
        "allowed",
        lambda: "secret",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(status, json={"detail": "admission refused"})
        ),
    )
    with pytest.raises(error) as exc:
        await client.complete(LLMRequest(system="", user=""))
    assert str(status) in str(exc.value)
    assert exc.value.usage.calls == 1


async def test_call_budget_stops_requests_before_dispatch():
    seen = []

    def handle(r):
        seen.append(r)
        return httpx.Response(200, text=sse(completed()))

    client = ChatGPTClient("allowed", lambda: "secret", transport=httpx.MockTransport(handle))
    client.attach_call_budget(CallBudget(1))
    await client.complete(LLMRequest(system="", user=""))
    with pytest.raises(CallLimitExceeded):
        await client.complete(LLMRequest(system="", user=""))
    assert len(seen) == 1


@pytest.mark.parametrize("use_done", [False, True])
async def test_streamed_text_is_retained_when_completed_envelope_has_no_output(use_done):
    events = [
        {"type": "response.output_text.delta", "delta": '{"chosen_key":'},
        {"type": "response.output_text.delta", "delta": '"C01"}'},
    ]
    if use_done:
        events.append({"type": "response.output_text.done", "text": '{"chosen_key":"C01"}'})
    terminal = completed()
    terminal["response"]["output"] = []
    events.append(terminal)
    client = ChatGPTClient(
        "allowed",
        lambda: "secret",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=sse(*events))),
    )
    response = await client.complete(LLMRequest(system="", user=""))
    assert response.text == '{"chosen_key":"C01"}'
    assert response.usage.completion_tokens == 3


async def test_empty_completed_response_is_a_provider_error_with_usage():
    terminal = completed("")
    client = ChatGPTClient(
        "allowed",
        lambda: "secret",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=sse(terminal))),
    )
    with pytest.raises(LLMFatalError, match="without any answer text") as exc:
        await client.complete(LLMRequest(system="", user=""))
    assert exc.value.usage.completion_tokens == 3


def test_login_returns_to_the_requesting_page_and_rejects_external_redirects(oauth):
    auth, ctx, _ = oauth
    params = parse_qs(urlsplit(auth.begin(return_to="/#/quick?with=chatgpt")).query)
    ctx["nonce"] = params["nonce"][0]
    assert (
        auth.finish({"state": params["state"][0], "code": "test", "client_id": "oaiapp_test"})
        == "/#/quick?with=chatgpt"
    )
    with pytest.raises(ChatGPTError, match="return page"):
        auth.begin(return_to="https://attacker.invalid")


def test_chatgpt_jobs_are_explicit_and_contain_no_credentials():
    block = llm_block({"preset": "chatgpt", "model": "allowed"})
    assert block == {"kind": "chatgpt", "model": "allowed"}
    LLMSpec(**block)
    with pytest.raises(ValueError, match="official"):
        LLMSpec(kind="chatgpt", model="allowed", base_url="https://attacker.invalid")
    with pytest.raises(ValueError, match="api_key_env"):
        LLMSpec(kind="chatgpt", model="allowed", api_key_env="MY_KEY")


def test_hosted_and_offline_servers_cannot_enable_loopback_login(tmp_path):
    with pytest.raises(ValueError, match="127.0.0.1"):
        make_server(str(tmp_path), host="0.0.0.0", chatgpt_login=True)
    with pytest.raises(ValueError, match="offline"):
        make_server(str(tmp_path), offline_only=True, chatgpt_login=True)
    ws = Workspace(tmp_path)
    assert dispatch(ws, "GET", "/api/chatgpt", {}) == {"available": False}
    with pytest.raises(ApiError):
        dispatch(ws, "POST", "/api/chatgpt", {"action": "sign_in"})


def test_public_sessions_and_callback_state_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    server = make_server(str(tmp_path), port=0, public=True, chatgpt_login=True, token="test-token")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with httpx.Client(base_url=server.url) as a, httpx.Client(base_url=server.url) as b:
            a.get("/")
            b.get("/")
            headers = {"X-Xwalk-Token": "test-token"}
            begin = a.post("/api/chatgpt", headers=headers, json={"action": "sign_in"}).json()
            state = parse_qs(urlsplit(begin["url"]).query)["state"][0]
            assert (
                b.get(
                    "/auth/callback", params={"state": state, "error": "access_denied"}
                ).status_code
                == 400
            )
            assert (
                "declined"
                in a.get("/auth/callback", params={"state": state, "error": "access_denied"}).text
            )
            assert not b.get("/api/chatgpt", headers=headers).json()["connected"]
            assert a.get("/api/chatgpt").status_code == 403
            assert not list(tmp_path.rglob("accounts.json"))
    finally:
        server.shutdown()
        server.server_close()


def test_chatgpt_project_runs_the_pipeline_without_environment_keys(oauth, tmp_path, monkeypatch):
    import xwalk.llm.chatgpt as adapter

    auth, _, start = oauth
    _, callback = start()
    auth.finish(callback)
    ws = Workspace(tmp_path, public=True)
    ws.chatgpt_auth = auth
    original = adapter.ChatGPTClient
    seen = []

    def handle(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer access-secret"
        reply = json.dumps(
            {"chosen_key": "C01", "confidence_score": 1, "decision": "support", "queries": []}
        )
        return httpx.Response(200, text=sse(completed(reply)))

    monkeypatch.setattr(
        adapter,
        "ChatGPTClient",
        lambda model, token: original(model, token, transport=httpx.MockTransport(handle)),
    )
    started = dispatch(
        ws,
        "POST",
        "/api/quickmap",
        {
            "terms": "cheddar",
            "target": {"libraries": ["foodon-sample"]},
            "mode": "endpoint",
            "max_calls": 5,
            "model": {"preset": "chatgpt", "model": "allowed"},
        },
    )
    task = ws.tasks.wait(started["task"]["id"])
    assert task.result["status"] == "ok", task.result
    assert seen
    assert task.result["counts"]["matched"] == 1
    # Credentials must not leak into project files or persisted run artifacts.
    for path in tmp_path.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            assert b"access-secret" not in raw
            assert b"refresh-secret" not in raw


def test_unavailable_model_is_rejected_before_inference(oauth, tmp_path):
    auth, _, start = oauth
    _, callback = start()
    auth.finish(callback)
    ws = Workspace(tmp_path)
    ws.chatgpt_auth = auth
    project = dispatch(
        ws,
        "POST",
        "/api/projects",
        {
            "source": {"text": "cheddar"},
            "target": {"libraries": ["foodon-sample"]},
            "model": {"preset": "chatgpt", "model": "not-available"},
        },
    )
    with pytest.raises(ApiError, match="available"):
        ws.start_task(
            {"job": project["job_path"], "out": "run", "model": "endpoint", "max_calls": 2}
        )


def test_tampered_signature_is_rejected(oauth):
    auth, ctx, start = oauth
    _, callback = start()
    original = auth.transport.handler
    attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def handle(request):
        response = original(request)
        if request.url.path == "/api/accounts/oauth/token":
            body = response.json()
            claims = {
                "iss": ISSUER,
                "aud": "oaiapp_test",
                "sub": "attacker",
                "exp": time.time() + 300,
                "nonce": ctx["nonce"],
            }
            body["id_token"] = jwt.encode(
                claims, attacker, algorithm="RS256", headers={"kid": "test-key"}
            )
            return httpx.Response(200, json=body)
        return response

    auth.transport = httpx.MockTransport(handle)
    with pytest.raises(ChatGPTError, match="validate"):
        auth.finish(callback)
    assert not auth.status()["connected"]


def test_failed_revocation_clears_local_credentials(oauth):
    auth, _, start = oauth
    _, callback = start()
    auth.finish(callback)
    original = auth.transport.handler

    def handle(request):
        if request.url.path == "/revoke":
            return httpx.Response(503)
        return original(request)

    auth.transport = httpx.MockTransport(handle)
    assert not auth.sign_out()
    assert not auth.status()["connected"]
    assert "refresh-secret" not in json.dumps(auth._profiles)

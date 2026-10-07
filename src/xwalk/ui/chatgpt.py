"""Official Sign in with ChatGPT for a loopback explorer.

Access and refresh tokens never enter the page, job files, environment, or run manifests.
Public-mode browser sessions keep them in memory; personal local mode can persist
them outside the workspace, in an owner-only configuration directory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from xwalk._extras import require

ISSUER = "https://auth.openai.com"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"


class ChatGPTError(Exception):
    """A safe diagnostic that contains no credentials or authorization codes."""


def _save(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def host_id(directory: Path) -> str:
    """Persist this runtime's host identifier, with no account credentials."""
    path = directory / "host.json"
    if path.exists():
        return str(json.loads(path.read_text())["host_id"])
    value = f"urn:uuid:{uuid.uuid4()}"
    _save(path, {"host_id": value})
    return value


class ChatGPTAuth:
    def __init__(
        self,
        redirect_uri: str,
        host: str,
        *,
        storage: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        require("chatgpt", "jwt", purpose="Sign in with ChatGPT")
        self.redirect_uri = redirect_uri
        self.host = host
        self.storage = storage
        self.transport = transport
        self._lock = threading.RLock()
        self._pending: dict[str, Any] | None = None
        self._profiles: dict[str, dict[str, Any]] = {}
        self._active: str | None = None
        if storage and storage.exists():
            saved = json.loads(storage.read_text())
            self._profiles = saved.get("profiles", {})
            self._active = saved.get("active")

    def _persist(self) -> None:
        if self.storage:
            _save(self.storage, {"profiles": self._profiles, "active": self._active})

    def status(self) -> dict[str, Any]:
        with self._lock:
            profile = self._profiles.get(self._active or "", {})
            return {
                "available": True,
                "active": self._active,
                "connected": bool(profile.get("access_token")),
                "plan_enabled": bool(profile.get("access_token"))
                and PLAN_SCOPE in profile.get("scopes", []),
                "accounts": [
                    {
                        "id": key,
                        "label": (p.get("email") or "ChatGPT account")
                        + (f" · connection {i}" if len(self._profiles) > 1 else ""),
                        "connected": bool(p.get("access_token")),
                    }
                    for i, (key, p) in enumerate(self._profiles.items(), 1)
                ],
                "persistent": self.storage is not None,
                "manage_usage": "https://chatgpt.com/settings/usage",
            }

    def begin(self, account: str | None = None, *, return_to: str = "/#/quick") -> str:
        with self._lock:
            if not return_to.startswith("/#/") or return_to.split("?", 1)[0] not in (
                "/#/quick",
                "/#/new",
                "/#/settings",
                "/#/",
            ):
                raise ChatGPTError("Invalid sign-in return page.")
            if account and account not in self._profiles:
                raise ChatGPTError("Unknown saved ChatGPT account.")
            profile = self._profiles.get(account or "", {})
            verifier = secrets.token_urlsafe(48)
            state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            self._pending = {
                "state": state,
                "nonce": nonce,
                "verifier": verifier,
                "account": account,
                "created": time.time(),
                "return_to": return_to,
            }
            params = {
                "client_id": profile.get("client_id", "dynamic_agent_client"),
                "ext_agent_host_id": self.host,
                "response_type": "code",
                "redirect_uri": self.redirect_uri,
                "scope": SCOPES,
                "resource": RESOURCE,
                "state": state,
                "nonce": nonce,
                "code_challenge_method": "S256",
                "code_challenge": base64.urlsafe_b64encode(
                    hashlib.sha256(verifier.encode()).digest()
                )
                .decode()
                .rstrip("="),
            }
            if profile:
                if profile.get("access_token") and PLAN_SCOPE not in profile.get("scopes", []):
                    # The user explicitly clicked to enable previously declined plan use.
                    params["prompt"] = "consent"
                if profile.get("id_token"):
                    params["id_token_hint"] = profile["id_token"]
                if profile.get("email"):
                    params["login_hint"] = profile["email"]
            else:
                params["agent_name_hint"] = "xwalk"
            return AUTHORIZE + "?" + urlencode(params)

    def _token(self, client: httpx.Client, data: Mapping[str, str]) -> dict[str, Any]:
        response = client.post(TOKEN, data=data)
        if response.status_code != 200:
            raise ChatGPTError(
                f"ChatGPT token exchange failed (HTTP {response.status_code}); sign in again."
            )
        result: dict[str, Any] = response.json()
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("access_token"), str)
            or not result.get("access_token")
            or result.get("token_type", "").lower() != "bearer"
        ):
            raise ChatGPTError("ChatGPT returned incomplete credentials; sign in again.")
        return result

    def finish(self, params: Mapping[str, str]) -> str:
        import jwt

        with self._lock:
            pending = self._pending
            if not pending or not secrets.compare_digest(params.get("state", ""), pending["state"]):
                raise ChatGPTError("Invalid sign-in state; start sign-in again in Settings.")
            self._pending = None  # one use, even on a failed exchange
            if time.time() - pending["created"] > 600:
                raise ChatGPTError("Sign-in expired; start again in Settings.")
            if params.get("error"):
                raise ChatGPTError("ChatGPT sign-in was declined or could not complete.")
            profile = self._profiles.get(pending["account"] or "", {})
            issued = params.get("client_id") or profile.get("client_id")
            if not issued or issued == "dynamic_agent_client" or not params.get("code"):
                raise ChatGPTError("ChatGPT registration did not return a client ID and code.")
            if profile and issued != profile["client_id"]:
                raise ChatGPTError("ChatGPT returned a different account registration.")
            if not profile:
                # Retain registration if the code expires; it is not an authenticated account yet.
                self._profiles.setdefault(issued, {"client_id": issued})
                self._persist()
            try:
                with httpx.Client(timeout=30, transport=self.transport) as client:
                    tokens = self._token(
                        client,
                        {
                            "grant_type": "authorization_code",
                            "client_id": issued,
                            "code": params["code"],
                            "code_verifier": pending["verifier"],
                            "redirect_uri": self.redirect_uri,
                            "resource": RESOURCE,
                        },
                    )
                    jwks = client.get(ISSUER + "/.well-known/jwks.json")
                    jwks.raise_for_status()
                    raw_id = tokens.get("id_token", "")
                    header = jwt.get_unverified_header(raw_id)
                    if header.get("alg") != "RS256":
                        raise ChatGPTError("ChatGPT identity token uses an unsupported signature.")
                    keys = jwks.json()["keys"]
                    key = next(k for k in keys if k.get("kid") == header.get("kid"))
                    identity = jwt.decode(
                        raw_id,
                        jwt.PyJWK.from_dict(key).key,
                        algorithms=["RS256"],
                        issuer=ISSUER,
                        audience=issued,
                        options={"require": ["iss", "aud", "exp", "sub", "nonce"]},
                    )
                    if not secrets.compare_digest(identity["nonce"], pending["nonce"]):
                        raise ChatGPTError("ChatGPT identity nonce did not match.")
                    if profile.get("subject") and identity["sub"] != profile["subject"]:
                        raise ChatGPTError("ChatGPT identity did not match the saved account.")
                    self._profiles[issued] = {
                        "client_id": issued,
                        "subject": identity["sub"],
                        "email": identity.get("email", ""),
                        "id_token": raw_id,
                        "access_token": tokens["access_token"],
                        "refresh_token": tokens.get("refresh_token"),
                        "scopes": tokens.get("scope", "").split(),
                        "expires_at": time.time() + float(tokens.get("expires_in", 3600)),
                    }
                    self._active = issued
                    self._persist()
                    return str(pending.get("return_to", "/#/quick"))
            except (
                httpx.HTTPError,
                jwt.PyJWTError,
                ValueError,
                KeyError,
                StopIteration,
                TypeError,
            ) as exc:
                raise ChatGPTError(
                    "Could not validate ChatGPT sign-in; start again in Settings."
                ) from exc

    def select(self, account: str) -> None:
        with self._lock:
            if account not in self._profiles or not self._profiles[account].get("subject"):
                raise ChatGPTError("Sign in to this ChatGPT account before selecting it.")
            self._active = account
            self._persist()

    def access_token(self, account: str | None = None) -> str:
        with self._lock:
            profile = self._profiles.get(account or self._active or "", {})
            if not profile.get("access_token") or PLAN_SCOPE not in profile.get("scopes", []):
                raise ChatGPTError("Enable ChatGPT plan usage in Settings before running with it.")
            if time.time() >= profile["expires_at"] - 60:
                if not profile.get("refresh_token"):
                    raise ChatGPTError("ChatGPT connection expired; sign in again.")
                try:
                    with httpx.Client(timeout=30, transport=self.transport) as client:
                        tokens = self._token(
                            client,
                            {
                                "grant_type": "refresh_token",
                                "client_id": profile["client_id"],
                                "refresh_token": profile["refresh_token"],
                                "resource": RESOURCE,
                            },
                        )
                    profile.update(
                        {
                            "access_token": tokens["access_token"],
                            "refresh_token": tokens.get("refresh_token", profile["refresh_token"]),
                            "id_token": tokens.get("id_token", profile.get("id_token")),
                            "scopes": tokens.get("scope", " ".join(profile["scopes"])).split(),
                            "expires_at": time.time() + float(tokens.get("expires_in", 3600)),
                        }
                    )
                    self._persist()
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    raise ChatGPTError(
                        "ChatGPT connection could not refresh; sign in again."
                    ) from exc
            if PLAN_SCOPE not in profile["scopes"]:
                raise ChatGPTError("ChatGPT plan permission is no longer enabled.")
            return str(profile["access_token"])

    def models(self, account: str | None = None) -> list[dict[str, str]]:
        token = self.access_token(account)
        try:
            with httpx.Client(timeout=30, transport=self.transport) as client:
                response = client.get(
                    RESOURCE + "/models", headers={"Authorization": f"Bearer {token}"}
                )
                response.raise_for_status()
                return [
                    {"slug": m["slug"], "display_name": m["display_name"]}
                    for m in response.json()["models"]
                    if m.get("visibility") == "list"
                ]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ChatGPTError("Could not load models for this ChatGPT account.") from exc

    def sign_out(self) -> bool:
        """Clear credentials; return whether remote revocation was confirmed."""
        with self._lock:
            self._pending = None
            profile = self._profiles.get(self._active or "", {})
            confirmed = not profile.get("refresh_token")
            if profile.get("refresh_token"):
                try:
                    with httpx.Client(timeout=30, transport=self.transport) as client:
                        discovery = client.get(ISSUER + "/.well-known/openid-configuration")
                        discovery.raise_for_status()
                        endpoint = discovery.json()["revocation_endpoint"]
                        url = httpx.URL(endpoint)
                        if url.scheme != "https" or url.host != "auth.openai.com":
                            raise ChatGPTError("Unexpected ChatGPT revocation endpoint.")
                        response = client.post(
                            endpoint,
                            data={
                                "token": profile["refresh_token"],
                                "token_type_hint": "refresh_token",
                                "client_id": profile["client_id"],
                            },
                        )
                        confirmed = response.status_code == 200
                except (httpx.HTTPError, ValueError, KeyError, ChatGPTError):
                    confirmed = False
            for key in ("access_token", "refresh_token", "id_token", "expires_at", "scopes"):
                profile.pop(key, None)
            self._persist()
            return confirmed

"""Outbound HTTP to an address checked once and then pinned.

Checking that a host resolves to public addresses and then letting the HTTP client
resolve it again is a time-of-check/time-of-use gap: a name with a short TTL (DNS
rebinding) can answer a public address for the check and `169.254.169.254` or a private
address for the connection. Here the name is resolved once, every address is checked,
and the connection is made to one of those addresses. TLS still verifies the
certificate against the host name (httpcore sends the URL's host as SNI, whatever
address the socket was opened to), and the `Host` header is the URL's.

The transports refuse to connect to any other host, so a redirect, were one ever
followed, could not leave the pinned name either.
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
from collections.abc import Iterable, Sequence
from typing import Any

import httpcore
import httpx

# The backends that open the real sockets (module attributes so tests can replace them).
async_inner_backend: Any = httpcore.AnyIOBackend
sync_inner_backend: Any = httpcore.SyncBackend


def wire_host(url: str) -> str:
    """The host as it goes on the wire (IDNA-encoded, lower case), or ''."""
    try:
        return httpx.URL(url).raw_host.decode("ascii").lower()
    except (httpx.InvalidURL, UnicodeDecodeError):
        return ""


def resolve(host: str, port: int) -> list[str]:
    """Every address `host` resolves to, in resolver order. Raises `OSError`."""
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def is_public(address: str) -> bool:
    """Whether `address` is a globally routable unicast address."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _refuse(host: str, pinned: str) -> httpcore.ConnectError:
    return httpcore.ConnectError(f"refusing to connect to {host}: only {pinned} is allowed")


class PinnedAsyncBackend(httpcore.AsyncNetworkBackend):
    """Opens connections for `host` to `addresses` only, never by name."""

    def __init__(self, host: str, addresses: Sequence[str], inner: Any = None) -> None:
        if not addresses:
            raise ValueError("no addresses to pin")
        self.host = host.lower()
        self.addresses = tuple(addresses)
        self._inner = inner if inner is not None else async_inner_backend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if host.lower() != self.host:
            raise _refuse(host, self.host)
        error: Exception | None = None
        for address in self.addresses:
            try:
                stream: httpcore.AsyncNetworkStream = await self._inner.connect_tcp(
                    address, port, timeout, local_address, socket_options
                )
                return stream
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                error = exc
        assert error is not None
        raise error

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> httpcore.AsyncNetworkStream:
        raise _refuse(path, self.host)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class PinnedSyncBackend(httpcore.NetworkBackend):
    """The blocking twin of `PinnedAsyncBackend`."""

    def __init__(self, host: str, addresses: Sequence[str], inner: Any = None) -> None:
        if not addresses:
            raise ValueError("no addresses to pin")
        self.host = host.lower()
        self.addresses = tuple(addresses)
        self._inner = inner if inner is not None else sync_inner_backend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        if host.lower() != self.host:
            raise _refuse(host, self.host)
        error: Exception | None = None
        for address in self.addresses:
            try:
                stream: httpcore.NetworkStream = self._inner.connect_tcp(
                    address, port, timeout, local_address, socket_options
                )
                return stream
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                error = exc
        assert error is not None
        raise error

    def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> httpcore.NetworkStream:
        raise _refuse(path, self.host)

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0)


class PinnedAsyncTransport(httpx.AsyncHTTPTransport):
    """httpx's transport, its connection pool opening sockets through a pinned backend."""

    def __init__(
        self, host: str, addresses: Sequence[str], *, ssl_context: ssl.SSLContext | None = None
    ) -> None:
        super().__init__()
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context or httpx.create_ssl_context(),
            max_connections=_LIMITS.max_connections,
            max_keepalive_connections=_LIMITS.max_keepalive_connections,
            keepalive_expiry=_LIMITS.keepalive_expiry,
            network_backend=PinnedAsyncBackend(host, addresses),
        )


class PinnedTransport(httpx.HTTPTransport):
    """The blocking twin of `PinnedAsyncTransport`."""

    def __init__(
        self, host: str, addresses: Sequence[str], *, ssl_context: ssl.SSLContext | None = None
    ) -> None:
        super().__init__()
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl_context or httpx.create_ssl_context(),
            max_connections=_LIMITS.max_connections,
            max_keepalive_connections=_LIMITS.max_keepalive_connections,
            keepalive_expiry=_LIMITS.keepalive_expiry,
            network_backend=PinnedSyncBackend(host, addresses),
        )


__all__ = [
    "PinnedAsyncBackend",
    "PinnedAsyncTransport",
    "PinnedSyncBackend",
    "PinnedTransport",
    "is_public",
    "resolve",
    "wire_host",
]

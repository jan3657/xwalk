"""`xwalk.ui.egress`: connections pinned to checked addresses keep the URL's host for
TLS (SNI, certificate) and the `Host` header; library downloads check every redirect.
Local servers only: names resolve through a fake resolver, sockets go to 127.0.0.1."""

from __future__ import annotations

import datetime
import http.server
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpcore
import httpx
import pytest

# Required (the dev extra installs it): without it the TLS, SNI and redirect tests
# below would not run at all.
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from xwalk.ui.egress import PinnedTransport, is_public, wire_host
from xwalk.ui.files import FileProblem
from xwalk.ui.library import download

NAME = "pinned.test"
TOY_OBO = b"format-version: 1.2\n\n[Term]\nid: TOY:0001\nname: Seizure\n"


@pytest.fixture
def tls_server(tmp_path: Path) -> Iterator[tuple[int, Path, dict[str, Any]]]:
    """An HTTPS server on 127.0.0.1 with a self-signed certificate for NAME only."""
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, NAME)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(NAME)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_file, key_file = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    seen: dict[str, Any] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen["host"] = self.headers.get("Host")
            port = self.server.server_address[1]
            redirects = {
                "/hop-private": f"https://metadata.test:{port}/latest.obo",
                "/hop-http": f"http://{NAME}:{port}/toy.obo",
                "/hop-ok": "/toy.obo",
                "/loop": "/loop",
                # What purl.obolibrary.org answers for ChEBI (https -> http).
                "/obo/chebi.obo": "http://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo",
            }
            if self.path in redirects:
                self.send_response(302)
                self.send_header("Location", redirects[self.path])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = TOY_OBO if self.path == "/toy.obo" else b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)

    def remember_sni(sock: Any, server_name: str | None, ctx: Any) -> None:
        seen["sni"] = server_name

    context.sni_callback = remember_sni
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1], cert_file, seen
    finally:
        server.shutdown()
        server.server_close()


def test_a_pinned_https_request_keeps_the_host_name(
    tls_server: tuple[int, Path, dict[str, Any]],
) -> None:
    port, cert_file, seen = tls_server
    trust = ssl.create_default_context(cafile=str(cert_file))
    transport = PinnedTransport(NAME, ["127.0.0.1"], ssl_context=trust)
    with httpx.Client(transport=transport, trust_env=False) as client:
        response = client.get(f"https://{NAME}:{port}/")  # NAME resolves nowhere
    assert response.status_code == 200 and response.text == "ok"
    assert seen == {"sni": NAME, "host": f"{NAME}:{port}"}


def test_the_certificate_is_still_checked_against_the_name(
    tls_server: tuple[int, Path, dict[str, Any]],
) -> None:
    port, cert_file, _ = tls_server
    trust = ssl.create_default_context(cafile=str(cert_file))
    transport = PinnedTransport("other.test", ["127.0.0.1"], ssl_context=trust)
    with (
        httpx.Client(transport=transport, trust_env=False) as client,
        pytest.raises(httpx.ConnectError),
    ):
        client.get(f"https://other.test:{port}/")  # certificate is for NAME


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.215.14", True),
        ("2606:2800:21f:cb07:6820:80da:af6b:8b2c", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("169.254.169.254", False),
        ("::1", False),
        ("fe80::1%en0", False),
        ("::ffff:10.0.0.1", False),
        ("224.0.0.1", False),
        ("100.64.0.1", False),
    ],
)
def test_public_addresses(address: str, public: bool) -> None:
    assert is_public(address) is public


def test_wire_host() -> None:
    assert wire_host("https://API.Example.com:8443/v1") == "api.example.com"
    assert wire_host("https://bücher.example/v1") == "xn--bcher-kva.example"
    assert wire_host("https:///nohost") == ""


# --- library downloads ------------------------------------------------------------------


@pytest.fixture
def mirror(
    tls_server: tuple[int, Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> tuple[int, list[str], list[str]]:
    """NAME resolves to a public address and metadata.test to the metadata address;
    every socket actually goes to the local TLS server, which the client trusts."""
    import socket

    port, cert_file, _ = tls_server
    answers = {NAME: "93.184.215.14", "metadata.test": "169.254.169.254"}
    lookups: list[str] = []
    connects: list[str] = []
    real = socket.getaddrinfo

    def resolve(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        if host not in answers:  # the socket to 127.0.0.1 itself
            return real(host, *args, **kwargs)
        lookups.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (answers[host], port))]

    original = httpcore.SyncBackend.connect_tcp

    def to_local(self: Any, host: str, port: int, *args: Any, **kwargs: Any) -> Any:
        connects.append(host)
        return original(self, "127.0.0.1", port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", to_local)
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_file))
    return port, lookups, connects


def test_a_download_goes_to_the_checked_address(
    mirror: tuple[int, list[str], list[str]], tmp_path: Path
) -> None:
    port, lookups, connects = mirror
    path = download(f"https://{NAME}:{port}/hop-ok", tmp_path / "toy.obo")
    assert path.read_bytes() == TOY_OBO
    assert set(connects) == {"93.184.215.14"} and set(lookups) == {NAME}


def test_each_redirect_is_checked_again(
    mirror: tuple[int, list[str], list[str]], tmp_path: Path
) -> None:
    port, lookups, connects = mirror
    with pytest.raises(FileProblem, match="non-public"):
        download(f"https://{NAME}:{port}/hop-private", tmp_path / "a.obo")
    assert "metadata.test" in lookups and set(connects) == {"93.184.215.14"}
    with pytest.raises(FileProblem, match="https"):
        download(f"https://{NAME}:{port}/hop-http", tmp_path / "b.obo")
    with pytest.raises(FileProblem, match="redirects"):
        download(f"https://{NAME}:{port}/loop", tmp_path / "c.obo")
    assert not list(tmp_path.glob("*.obo*"))


def test_a_purl_that_redirects_to_http_is_refused(
    mirror: tuple[int, list[str], list[str]], tmp_path: Path
) -> None:
    """The ChEBI PURL's redirect: refused before anything is fetched from the http
    target, which is why the catalog names ChEBI by its https URL at EBI."""
    port, lookups, connects = mirror
    with pytest.raises(FileProblem, match="only https"):
        download(f"https://{NAME}:{port}/obo/chebi.obo", tmp_path / "chebi.obo")
    assert "ftp.ebi.ac.uk" not in lookups and set(connects) == {"93.184.215.14"}

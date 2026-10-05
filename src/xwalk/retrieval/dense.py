"""Dense retrieval.

There is deliberately no "second dense model" concept. `Matcher` takes a list of
retrievers; running two encoders means constructing two DenseRetrievers. The paper repo
hardcoded SapBERT as the second dense model, which is meaningless outside biomedicine --
here it is a line in the user's config.

The `Encoder` protocol means an API embedding service can be plugged in with no torch
installed at all.
"""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from xwalk._extras import require
from xwalk.fingerprint import hash_record, hash_value
from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import IndexMismatchError, SearchRequest, component_differences
from xwalk.templates import TemplateSet

_META_FILE = "xwalk_meta.json"
_VECTOR_FILE = "vectors.f32"
_IDS_FILE = "record_ids.json"
INDEX_FORMAT_VERSION = 1


@runtime_checkable
class Encoder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        """Return one unit-normalised vector per text."""


class SentenceTransformerEncoder:
    """The default encoder. Requires xwalk[dense]."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        normalize: bool = True,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
        revision: str | None = None,
    ) -> None:
        st = require("dense", "sentence_transformers", purpose="SentenceTransformerEncoder")
        extra = {} if revision is None else {"revision": revision}
        self._model = st.SentenceTransformer(model_name, device=device, **extra)
        self._model_name = model_name
        self._revision = revision
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def settings(self) -> dict[str, Any]:
        """Everything besides the name that changes what a vector means.

        Device and batch size are absent: they change speed, not vectors. A revision
        that was not pinned is recorded as "unknown" rather than guessed.
        """
        max_seq_length = getattr(self._model, "max_seq_length", None)
        return {
            "revision": self._revision or "unknown",
            "normalize": self._normalize,
            "query_prefix": self._query_prefix,
            "doc_prefix": self._doc_prefix,
            "max_seq_length": None if max_seq_length is None else int(max_seq_length),
        }

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        prefix = self._query_prefix if is_query else self._doc_prefix
        prepared = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        vectors = self._model.encode(
            prepared,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(v) for v in row] for row in vectors]


def encoder_identity(encoder: Encoder) -> dict[str, Any]:
    """Name, dimension and every setting that changes what a vector means.

    An encoder may expose a `settings` mapping (`SentenceTransformerEncoder` does:
    revision, normalize, prefixes, max sequence length). Reading the dimension may load
    the model; it never encodes anything.
    """
    settings = getattr(encoder, "settings", None) or {}
    return {"name": encoder.name, "dimension": encoder.dimension, **dict(settings)}


def index_components(
    records: Iterable[Record], templates: TemplateSet, encoder: Encoder
) -> dict[str, Any]:
    """The identity of a dense index, computed without encoding anything."""
    digests = sorted(hash_record(record) for record in records)
    return {
        "engine": "dense",
        "format_version": INDEX_FORMAT_VERSION,
        "doc_template": templates.doc,
        "record_count": len(digests),
        "records": hash_value(digests),
        "encoder": encoder_identity(encoder),
    }


def read_components(index_dir: str | Path) -> dict[str, Any] | None:
    """The components stored with a built index; `None` for an index built before 0.2."""
    meta = json.loads((Path(index_dir) / _META_FILE).read_text(encoding="utf-8"))
    components = meta.get("components")
    return dict(components) if isinstance(components, dict) else None


def _write_vectors(path: Path, vectors: Sequence[Sequence[float]]) -> None:
    with path.open("wb") as handle:
        for vector in vectors:
            handle.write(struct.pack(f"<{len(vector)}f", *vector))


def _read_vectors(path: Path, dimension: int) -> list[list[float]]:
    raw = path.read_bytes()
    size = dimension * 4
    return [
        list(struct.unpack(f"<{dimension}f", raw[i : i + size])) for i in range(0, len(raw), size)
    ]


class DenseRetriever:
    def __init__(
        self,
        *,
        vectors: Sequence[Sequence[float]],
        record_ids: Sequence[str],
        encoder: Encoder,
        fingerprint: str,
        name: str,
        default_limit: int = 20,
    ) -> None:
        self._vectors = [list(v) for v in vectors]
        self._record_ids = list(record_ids)
        self._encoder = encoder
        self._fingerprint = fingerprint
        self._name = name
        self._default_limit = default_limit
        self._faiss_index = self._try_build_faiss()

    # --- protocol ---------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def encoder_identity(self) -> dict[str, Any]:
        """The live encoder's identity, for the run fingerprint.

        Read from the encoder in use, not from the index meta file: reopening an index
        with a different query prefix changes every query vector, so it must change the
        run identity too. An encoder may expose a `settings` mapping for this.
        """
        return encoder_identity(self._encoder)

    @property
    def default_limit(self) -> int:
        """Retrieval depth belongs to the retriever, not the matcher.

        A dense index and a BM25 index have no reason to share a `k`, and the matcher
        reads this on every attempt -- a retriever without it breaks the loop.
        """
        return self._default_limit

    # --- construction -----------------------------------------------------------

    @classmethod
    def build(
        cls,
        records: Iterable[Record],
        templates: TemplateSet,
        index_dir: str | Path,
        encoder: Encoder,
        *,
        name: str | None = None,
        default_limit: int = 20,
        batch: int = 256,
    ) -> DenseRetriever:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        records = list(records)
        record_ids = [record.id for record in records]
        texts = [templates.render_doc(record) for record in records]

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            vectors.extend(encoder.encode(texts[start : start + batch], is_query=False))

        for vector in vectors:
            if len(vector) != encoder.dimension:
                raise ValueError(
                    f"encoder {encoder.name!r} declares dimension {encoder.dimension} "
                    f"but produced a vector of length {len(vector)}"
                )

        components = index_components(records, templates, encoder)
        fingerprint = hash_value(components)

        _write_vectors(index_dir / _VECTOR_FILE, vectors)
        (index_dir / _IDS_FILE).write_text(json.dumps(record_ids), encoding="utf-8")
        (index_dir / _META_FILE).write_text(
            json.dumps(
                {
                    "engine": "dense",
                    "name": name or f"dense:{encoder.name}",
                    "encoder": encoder.name,
                    "dimension": encoder.dimension,
                    "default_limit": default_limit,
                    "fingerprint": fingerprint,
                    "components": components,
                    "doc_count": len(record_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls(
            vectors=vectors,
            record_ids=record_ids,
            encoder=encoder,
            fingerprint=fingerprint,
            name=name or f"dense:{encoder.name}",
            default_limit=default_limit,
        )

    @classmethod
    def open(
        cls,
        index_dir: str | Path,
        encoder: Encoder,
        *,
        name: str | None = None,
        default_limit: int | None = None,
        expected: Mapping[str, Any] | None = None,
    ) -> DenseRetriever:
        """Reopen a built index without encoding anything.

        The encoder's name and dimension must match the index. With `expected` (from
        `index_components`) every stored component is compared -- encoder revision,
        normalize, prefixes and max sequence length included -- and a difference raises
        `IndexMismatchError`. `xwalk match` and `xwalk index` always pass `expected`.
        """
        index_dir = Path(index_dir)
        meta_path = index_dir / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{index_dir} is not an xwalk dense index (no {_META_FILE}); build it first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if expected is not None:
            differences = component_differences(meta.get("components"), expected)
            if differences:
                raise IndexMismatchError(index_dir, differences)
        if meta["encoder"] != encoder.name:
            raise ValueError(
                f"index was built with encoder {meta['encoder']!r} but {encoder.name!r} "
                f"was supplied; a vector index is meaningless to a different encoder"
            )
        if meta["dimension"] != encoder.dimension:
            raise ValueError(
                f"index dimension {meta['dimension']} != encoder dimension {encoder.dimension}"
            )
        return cls(
            vectors=_read_vectors(index_dir / _VECTOR_FILE, meta["dimension"]),
            record_ids=json.loads((index_dir / _IDS_FILE).read_text(encoding="utf-8")),
            encoder=encoder,
            fingerprint=meta["fingerprint"],
            name=name or meta["name"],
            default_limit=(
                default_limit if default_limit is not None else int(meta.get("default_limit", 20))
            ),
        )

    # --- search -----------------------------------------------------------------

    def _try_build_faiss(self) -> Any | None:
        """FAISS if available; otherwise exact search over the stored vectors.

        This is a *speed* fallback with identical semantics on normalised vectors
        (IndexFlatIP is exact), not a silent ranking change.
        """
        if not self._vectors:
            return None
        try:
            import faiss
            import numpy as np
        except ImportError:
            return None
        index = faiss.IndexFlatIP(len(self._vectors[0]))
        index.add(np.asarray(self._vectors, dtype="float32"))
        return index

    def _search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        if not text.strip() or not self._vectors:
            return []
        query = self._encoder.encode([text], is_query=True)[0]

        if self._faiss_index is not None:
            import numpy as np

            scores, indices = self._faiss_index.search(
                np.asarray([query], dtype="float32"), min(limit, len(self._vectors))
            )
            pairs = [
                (float(s), int(i)) for s, i in zip(scores[0], indices[0], strict=True) if i >= 0
            ]
        else:
            pairs = sorted(
                (
                    (sum(a * b for a, b in zip(query, vector, strict=True)), i)
                    for i, vector in enumerate(self._vectors)
                ),
                key=lambda p: (-p[0], p[1]),
            )[:limit]

        return [
            RetrievalHit(
                record_id=self._record_ids[index],
                retriever=self._name,
                raw_score=score,
                rank=rank,
            )
            for rank, (score, index) in enumerate(pairs, start=1)
        ]

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return await asyncio.to_thread(self._search_sync, request.text, request.limit)

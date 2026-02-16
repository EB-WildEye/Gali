"""
Gali - GaliVectorStore
========================
OOP wrapper around LanceDB providing a clean interface for:
  • Table creation with a strict PyArrow schema
  • Document insertion (embedding + metadata) with batching
  • Vector similarity search with optional metadata filtering

Design decisions:
  • **Singleton DB connection** — ``_get_connection()`` caches the
    ``lancedb.DBConnection`` at class level to avoid file-lock issues.
  • **Dependency Injection** — the embedder is injected, never
    hard-coded, making the store testable with mock embedders.
  • **Batch embedding** — large chunk lists are embedded in batches
    of ``_EMBED_BATCH_SIZE`` to keep memory bounded.
  • **Zero Any types** — all dict schemas use explicit union types
    (``str | int | float | list[float]``) instead of ``Any``.

Usage:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from gali.src.database.vector_store import GaliVectorStore

    embedder = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY.get_secret_value())
    store = GaliVectorStore(embedder)
    store.add_documents(texts=[...], metadatas=[...])
    results = store.search("query text", limit=5)
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import lancedb
import pyarrow as pa

from gali.config.settings import settings
from gali.src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Type Aliases ──────────────────────────────────────────────────────
DocumentMetadata = dict[str, str | int]
DocumentRecord = dict[str, str | int | float | list[float]]
SearchResult = dict[str, str | int | float | list[float]]


# ── Embedder Protocol ─────────────────────────────────────────────────

@runtime_checkable
class Embedder(Protocol):
    """Structural type for any LangChain-compatible embedding model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# ── LanceDB Table Schema ──────────────────────────────────────────────
GALI_SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32())),
    pa.field("text", pa.utf8()),
    pa.field("source_file", pa.utf8()),
    pa.field("protocol_type", pa.utf8()),
    pa.field("department", pa.utf8()),
    pa.field("chunk_index", pa.int32()),
])

# ── Constants ──────────────────────────────────────────────────────────
_EMBED_BATCH_SIZE = 64
_DB_LOCK = threading.Lock()
_db_connection_cache: dict[str, lancedb.DBConnection] = {}


def _get_connection(db_path: str) -> lancedb.DBConnection:
    """
    Return a **singleton** ``lancedb.DBConnection`` for *db_path*.

    Thread-safe via ``_DB_LOCK``.  Re-uses an existing connection
    for the same path, avoiding file-lock contention when multiple
    ``GaliVectorStore`` instances share the same DB directory.
    """
    if db_path not in _db_connection_cache:
        with _DB_LOCK:
            if db_path not in _db_connection_cache:
                logger.info("Opening new LanceDB connection: %s", db_path)
                _db_connection_cache[db_path] = lancedb.connect(db_path)
    return _db_connection_cache[db_path]


class GaliVectorStore:
    """
    High-level abstraction over a LanceDB vector table.

    Parameters
    ----------
    embedder : Embedder
        Any object satisfying the ``Embedder`` protocol (must expose
        ``embed_documents`` and ``embed_query``).
    db_path
        Override the database directory.  Defaults to ``settings.LANCEDB_PATH``.
    table_name
        Override the table name.  Defaults to ``settings.LANCEDB_TABLE_NAME``.
    """

    __slots__ = ("embedder", "_db_path", "_table_name", "db", "table")

    def __init__(self, embedder: Embedder, db_path: str | None = None, table_name: str | None = None) -> None:
        self.embedder: Embedder = embedder
        self._db_path: str = str(db_path or settings.LANCEDB_PATH)
        self._table_name: str = table_name or settings.LANCEDB_TABLE_NAME
        self.db: lancedb.DBConnection | None = None
        self.table: lancedb.table.Table | None = None
        self._connect()


    def _connect(self) -> None:
        """Open (or re-use) the LanceDB connection and initialise the table."""
        try:
            self.db = _get_connection(self._db_path)
            existing = self.db.table_names()

            if self._table_name in existing:
                self.table = self.db.open_table(self._table_name)
                logger.info("Opened existing table '%s' (%d rows).", self._table_name, self.table.count_rows())
            else:
                self.table = self.db.create_table(self._table_name, schema=GALI_SCHEMA)
                logger.info("Created new table '%s'.", self._table_name)

        except OSError as exc:
            logger.error("LanceDB filesystem error at %s: %s", self._db_path, exc)
            raise
        except Exception:
            logger.exception("Unexpected error connecting to LanceDB.")
            raise


    def add_documents(self, texts: list[str], metadatas: list[DocumentMetadata]) -> int:
        """
        Embed a batch of text chunks and persist them with metadata.

        Embedding is done in batches of ``_EMBED_BATCH_SIZE`` to limit
        peak memory usage during large ingestions.

        Parameters
        ----------
        texts
            List of plain-text chunks to embed and store.
        metadatas
            Parallel list of dicts (``source_file``, ``protocol_type``,
            ``department``, ``chunk_index``).

        Returns
        -------
        int
            Number of rows successfully added.

        Raises
        ------
        ValueError
            If ``texts`` and ``metadatas`` have mismatched lengths.
        RuntimeError
            If the table has not been initialised.
        """
        if len(texts) != len(metadatas):
            raise ValueError(f"Length mismatch: {len(texts)} texts vs {len(metadatas)} metadatas.")
        if self.table is None:
            raise RuntimeError("Vector table is not initialised. Call _connect() first.")

        logger.info("Embedding %d chunks in batches of %d …", len(texts), _EMBED_BATCH_SIZE)

        # ── Batched embedding ──────────────────────────────────────────
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            try:
                vectors = self.embedder.embed_documents(batch)
                all_vectors.extend(vectors)
            except Exception as exc:
                logger.error("Embedding batch %d–%d failed: %s", i, i + len(batch) - 1, exc)
                raise

        # ── Build records ──────────────────────────────────────────────
        records: list[DocumentRecord] = [
            {"vector": vec, "text": txt, "source_file": meta.get("source_file", "unknown"), "protocol_type": meta.get("protocol_type", "unknown"), "department": meta.get("department", "unknown"), "chunk_index": meta.get("chunk_index", 0)}
            for txt, vec, meta in zip(texts, all_vectors, metadatas)
        ]

        try:
            self.table.add(records)
        except OSError as exc:
            logger.error("Failed to write records to LanceDB: %s", exc)
            raise

        row_count = self.table.count_rows()
        logger.info("Added %d chunks. Table '%s' now has %d total rows.", len(records), self._table_name, row_count)
        return len(records)


    def search(self, query_text: str, limit: int = 5, filter_dict: dict[str, str] | None = None) -> list[SearchResult]:
        """
        Perform a vector similarity search with optional metadata filters.

        Parameters
        ----------
        query_text
            Natural-language query to embed and search.
        limit
            Maximum results (default 5).
        filter_dict
            Optional WHERE clause filters, e.g.
            ``{"protocol_type": "induced"}``.

        Returns
        -------
        list[SearchResult]
            Matched rows with a ``_distance`` score.
        """
        if self.table is None:
            raise RuntimeError("Vector table is not initialised. Call _connect() first.")

        try:
            query_vector = self.embedder.embed_query(query_text)
        except Exception as exc:
            logger.error("Failed to embed query: %s", exc)
            raise

        query = self.table.search(query_vector).limit(limit)

        if filter_dict:
            clauses = [f"{k} = '{v}'" for k, v in filter_dict.items()]
            where_str = " AND ".join(clauses)
            query = query.where(where_str)
            logger.info("Searching with filter: %s", where_str)
        else:
            logger.info("Searching without filters (limit=%d).", limit)

        results: list[SearchResult] = query.to_list()
        logger.info("Search returned %d results.", len(results))
        return results


    def count(self) -> int:
        """Return the total number of rows in the table."""
        if self.table is None:
            return 0
        return self.table.count_rows()


    def drop_table(self) -> None:
        """Drop the vector table (useful for testing / re-ingestion)."""
        if self.db is None:
            logger.warning("No database connection; nothing to drop.")
            return
        try:
            self.db.drop_table(self._table_name)
            self.table = None
            logger.info("Dropped table '%s'.", self._table_name)
        except ValueError:
            logger.warning("Table '%s' does not exist — nothing to drop.", self._table_name)
        except OSError as exc:
            logger.error("Filesystem error dropping table '%s': %s", self._table_name, exc)
            raise


    def __repr__(self) -> str:
        return f"GaliVectorStore(db='{self._db_path}', table='{self._table_name}', rows={self.count()})"

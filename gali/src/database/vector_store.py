"""
Gali - GaliVectorStore
========================
OOP wrapper around LanceDB providing a clean interface for:
  • Table creation with a strict schema
  • Document insertion (embedding + metadata)
  • Vector similarity search with optional metadata filtering

All database paths and table names are pulled from the centralised
``settings`` singleton so there is a single source of truth.

Usage:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from gali.src.database.vector_store import GaliVectorStore

    embedder = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
    store    = GaliVectorStore(embedder)
    store.add_documents(texts=[...], metadatas=[...])
    results  = store.search("query text", limit=5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import lancedb
import pyarrow as pa
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from gali.config.settings import settings
from gali.src.utils.logger import get_logger

logger = get_logger(__name__)

# ── LanceDB Table Schema ──────────────────────────────────────────────────
# Defined once at module level so it can be reused by both the table
# creation logic and any future migration / validation utilities.
GALI_SCHEMA = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float32())),
        pa.field("text", pa.utf8()),
        pa.field("source_file", pa.utf8()),
        pa.field("protocol_type", pa.utf8()),   # "induced" | "missed"
        pa.field("department", pa.utf8()),
        pa.field("chunk_index", pa.int32()),
    ]
)


class GaliVectorStore:
    """
    High-level abstraction over a LanceDB vector table.

    Parameters
    ----------
    embedder
        Any object that exposes an ``embed_documents(texts)`` method
        returning a list of float vectors (e.g. LangChain embeddings).
    db_path : str | None
        Override the database directory.  Defaults to ``settings.LANCEDB_PATH``.
    table_name : str | None
        Override the table name.  Defaults to ``settings.LANCEDB_TABLE_NAME``.
    """

    # ── Constructor ───────────────────────────────────────────────────
    def __init__(self, embedder: Any, db_path: Optional[str] = None, table_name: Optional[str] = None) -> None:
        
        self.embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=settings.GOOGLE_API_KEY)
        self._db_path = str(db_path or settings.LANCEDB_PATH)
        self._table_name = table_name or settings.LANCEDB_TABLE_NAME

        self.db: lancedb.DBConnection | None = None
        self.table: lancedb.table.Table | None = None

        self._connect()


    # ── Private helpers ────────────────────────────────────────────────
    def _connect(self) -> None:
        """Open (or create) the LanceDB database and initialise the table."""
        try:
            logger.info("Connecting to LanceDB at: %s", self._db_path)
            self.db = lancedb.connect(self._db_path)

            existing_tables = self.db.table_names()

            if self._table_name in existing_tables:
                self.table = self.db.open_table(self._table_name)
                logger.info(
                    "Opened existing table '%s' (%d rows).",
                    self._table_name,
                    self.table.count_rows(),
                )
            else:
                self.table = self.db.create_table(
                    self._table_name,
                    schema=GALI_SCHEMA,
                )
                logger.info("Created new table '%s'.", self._table_name)

        except Exception:
            logger.exception("Failed to connect to LanceDB.")
            raise


    # ── Public API ─────────────────────────────────────────────────────
    def add_documents(self,texts: List[str],metadatas: List[Dict[str, Any]],) -> int:
        """
        Embed a batch of text chunks and persist them with metadata.

        Parameters
        ----------
        texts
            List of plain-text chunks to embed and store.
        metadatas
            Parallel list of dicts, each containing at minimum:
            ``source_file``, ``protocol_type``, ``department``, ``chunk_index``.

        Returns
        -------
        int
            The number of rows successfully added.

        Raises
        ------
        ValueError
            If ``texts`` and ``metadatas`` have mismatched lengths.
        RuntimeError
            If the table has not been initialised.
        """
        if len(texts) != len(metadatas):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts vs {len(metadatas)} metadatas."
            )
        if self.table is None:
            raise RuntimeError("Vector table is not initialised. Call _connect() first.")

        try:
            logger.info("Embedding %d chunks …", len(texts))
            vectors = self.embedder.embed_documents(texts)

            records = []
            for text, vector, meta in zip(texts, vectors, metadatas):
                records.append(
                    {
                        "vector": vector,
                        "text": text,
                        "source_file": meta.get("source_file", "unknown"),
                        "protocol_type": meta.get("protocol_type", "unknown"),
                        "department": meta.get("department", "unknown"),
                        "chunk_index": meta.get("chunk_index", 0),
                    }
                )

            self.table.add(records)
            row_count = self.table.count_rows()
            logger.info(
                "Added %d chunks. Table '%s' now has %d total rows.",
                len(records),
                self._table_name,
                row_count,
            )
            return len(records)

        except Exception:
            logger.exception("Failed to add documents to LanceDB.")
            raise


    def search(self, query_text: str, limit: int = 5, filter_dict: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search, optionally filtered by metadata.

        Parameters
        ----------
        query_text
            The natural-language query to embed and search for.
        limit
            Maximum number of results to return (default 5).
        filter_dict
            Optional key/value pairs used to build a SQL WHERE clause.
            Example: ``{"protocol_type": "induced", "department": "ER"}``
            produces ``protocol_type = 'induced' AND department = 'ER'``.

        Returns
        -------
        list[dict]
            Each dict contains the matched row's fields plus a ``_distance``
            score (lower is more similar).
        """
        if self.table is None:
            raise RuntimeError("Vector table is not initialised. Call _connect() first.")

        try:
            # Embed the query using the same embedder
            query_vector = self.embedder.embed_query(query_text)

            # Build the LanceDB search query
            query = self.table.search(query_vector).limit(limit)

            # Apply optional metadata filters
            if filter_dict:
                where_clauses = [f"{key} = '{value}'" for key, value in filter_dict.items()]
                where_str = " AND ".join(where_clauses)
                query = query.where(where_str)
                logger.info("Searching with filter: %s", where_str)
            else:
                logger.info("Searching without filters (limit=%d).", limit)

            results = query.to_list()

            logger.info("Search returned %d results.", len(results))
            return results

        except Exception:
            logger.exception("Vector search failed.")
            raise


    # ── Utility ────────────────────────────────────────────────────────
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
        except Exception:
            logger.exception("Failed to drop table '%s'.", self._table_name)
            raise


    def __repr__(self) -> str:
        row_count = self.count()
        return (
            f"GaliVectorStore("
            f"db='{self._db_path}', "
            f"table='{self._table_name}', "
            f"rows={row_count})"
        )

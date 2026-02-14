"""
src/database/vector_store.py — Vector Store Operations (LanceDB)

Responsibility:
    Manages all interactions with the LanceDB vector database:
      - Initialize / connect to a local LanceDB instance.
      - Insert document embeddings (used by the ingestion pipeline).
      - Perform vector similarity search (top-K retrieval for RAG queries).

    This is the only file that imports or references LanceDB directly.
    All other modules access the vector store through this interface.

    The embedder (Google Generative AI Embeddings) is injected via the
    constructor — this module does not import or configure it directly.

Architecture Context:
    Replaces the S3-backed LanceDB layer from the original serverless
    architecture. Data is stored locally in .lancedb/ (git-ignored).
    Each cloned instance gets its own database — zero shared state.

Related Files:
    - src/core/rag_engine.py  → Calls search() during RAG flow
    - scripts/setup_db.py     → Calls add_documents() during ingestion
    - data/raw/               → Source documents for ingestion
"""

import lancedb
from pathlib import Path


class GaliVectorStore:
    """
    Single-tenant vector store backed by a local LanceDB instance.

    The embedder is injected at construction time. It must implement:
      - embed_documents(texts: list[str]) -> list[list[float]]
      - embed_query(text: str) -> list[float]
    (LangChain embedding objects satisfy this interface.)

    Usage:
        store = GaliVectorStore(db_path=".lancedb", embedder=my_embedder)
        store.add_documents(texts=["..."], metadatas=[{"source": "file.txt"}])
        results = store.search("my question", top_k=3)
    """

    def __init__(
        self,
        db_path: str = ".lancedb",
        table_name: str = "document_vectors",
        embedder=None,
    ):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.embedder = embedder
        self._table = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_table(self):
        """Lazily open the table if it exists, cache the handle."""
        if self._table is None and self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)
        return self._table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> int:
        """
        Embed text chunks and insert them into the vector store.

        Args:
            texts:     List of text chunks to embed and store.
            metadatas: Parallel list of metadata dicts. Each dict should
                       contain at least {"source": "<filename>"}.

        Returns:
            Number of records inserted.
        """
        vectors = self.embedder.embed_documents(texts)

        data = []
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            record = {
                "text": text,
                "vector": vec,
                "source": metadatas[i].get("source", "") if metadatas else "",
                "chunk_index": metadatas[i].get("chunk_index", i) if metadatas else i,
            }
            data.append(record)

        existing = self._open_table()
        if existing is not None:
            existing.add(data)
        else:
            self._table = self.db.create_table(self.table_name, data)

        return len(data)

    def search(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Embed a query string and perform vector similarity search.

        Args:
            query_text: Natural-language query.
            top_k:      Number of results to return.

        Returns:
            List of dicts with keys: text, source, chunk_index, _distance.
        """
        table = self._open_table()
        if table is None:
            raise ValueError(
                f"Table '{self.table_name}' does not exist. "
                f"Run the ingestion pipeline first (scripts/setup_db.py)."
            )

        query_vector = self.embedder.embed_query(query_text)
        results = table.search(query_vector).limit(top_k).to_list()
        return results

    def table_exists(self) -> bool:
        return self.table_name in self.db.table_names()

    def count(self) -> int:
        table = self._open_table()
        return table.count_rows() if table else 0

    def drop_table(self) -> None:
        if self.table_exists():
            self.db.drop_table(self.table_name)
            self._table = None

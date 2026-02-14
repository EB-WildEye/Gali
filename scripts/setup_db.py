"""
scripts/setup_db.py — Document Ingestion Pipeline (Offline)

Responsibility:
    Reads .txt documents from data/raw/, chunks them, generates embeddings
    via Google Gemini, and stores them in the local LanceDB instance.

    Each chunk retains its source filename as metadata, enabling the RAG
    system to distinguish between protocols during retrieval.

    Run:  python scripts/setup_db.py

Related Files:
    - data/raw/                    → Input documents
    - src/database/vector_store.py → Stores the embeddings
    - .env                         → GOOGLE_API_KEY
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import src.*
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.database.vector_store import GaliVectorStore

# ---------------------------------------------------------------------------
# Settings (no config.py — kept inline and minimal)
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = "models/embedding-001"
LANCEDB_PATH = str(_PROJECT_ROOT / ".lancedb")
TABLE_NAME = "document_vectors"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DATA_RAW = _PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def read_text_file(file_path: Path) -> str:
    """Read a text file with UTF-8 encoding."""
    return file_path.read_text(encoding="utf-8")


def chunk_text(text: str, source: str) -> tuple[list[str], list[dict]]:
    """
    Split text into chunks. Returns parallel lists of texts and metadatas.
    Each metadata dict carries the source filename.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    metadatas = [
        {"source": source, "chunk_index": i}
        for i in range(len(chunks))
    ]
    return chunks, metadatas


def create_embedder():
    """Create Google Gemini embedder."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file."
        )
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def main():
    # -- Discover files --
    files = sorted(
        f for f in DATA_RAW.iterdir()
        if f.suffix.lower() == ".txt" and f.is_file()
    )

    if not files:
        print(f"No .txt files found in {DATA_RAW}")
        return

    print(f"Found {len(files)} file(s) in {DATA_RAW}:")
    for f in files:
        print(f"  - {f.name}")
    print()

    # -- Initialise embedder and vector store --
    embedder = create_embedder()
    store = GaliVectorStore(
        db_path=LANCEDB_PATH,
        table_name=TABLE_NAME,
        embedder=embedder,
    )

    # -- Clean slate --
    if store.table_exists():
        print(f"Table '{TABLE_NAME}' exists — dropping for clean re-ingestion.\n")
        store.drop_table()

    # -- Ingest each file --
    for file_path in files:
        print(f"  Loading: {file_path.name}")
        text = read_text_file(file_path)

        chunks, metadatas = chunk_text(text, source=file_path.name)
        print(f"  Chunks:  {len(chunks)}")

        count = store.add_documents(texts=chunks, metadatas=metadatas)
        print(f"  Stored:  {count} record(s)")
        print()

    # -- Summary --
    print("=" * 50)
    print(f"Ingestion complete!")
    print(f"  Table:      {TABLE_NAME}")
    print(f"  Total rows: {store.count()}")
    print(f"  DB path:    {LANCEDB_PATH}")


if __name__ == "__main__":
    main()

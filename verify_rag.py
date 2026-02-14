"""
verify_rag.py — RAG Retrieval Verification (Temporary)

Searches LanceDB for a query and displays the results with source metadata.
Used to verify that the RAG can distinguish between medical protocols.

Run:  python verify_rag.py
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.database.vector_store import GaliVectorStore

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = "models/embedding-001"
LANCEDB_PATH = str(_PROJECT_ROOT / ".lancedb")
TABLE_NAME = "document_vectors"


def main():
    # -- Init --
    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    store = GaliVectorStore(
        db_path=LANCEDB_PATH,
        table_name=TABLE_NAME,
        embedder=embedder,
    )

    if not store.table_exists():
        print("Table does not exist. Run 'python scripts/setup_db.py' first.")
        return

    print(f"Table '{TABLE_NAME}' has {store.count()} rows.\n")

    # -- Query --
    query = "What is the dosage for Mifepristone?"
    print(f"Query: {query}")
    print("=" * 60)

    results = store.search(query, top_k=5)

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"  Source:    {result.get('source', 'N/A')}")
        print(f"  Distance:  {result.get('_distance', 'N/A'):.4f}")
        print(f"  Chunk #:   {result.get('chunk_index', 'N/A')}")
        print(f"  Text:")
        print(f"    {result.get('text', '')}")

    # -- Analysis --
    print("\n" + "=" * 60)
    print("SOURCE ISOLATION CHECK:")
    sources = set(r.get("source", "") for r in results)
    for src in sorted(sources):
        src_results = [r for r in results if r.get("source") == src]
        print(f"\n  [{src}]")
        for r in src_results:
            # Extract the Mifepristone dosage line from the text
            for line in r["text"].split("\n"):
                if "מפיג'ין" in line:
                    print(f"    → {line.strip()}")


if __name__ == "__main__":
    main()

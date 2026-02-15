"""
Gali - Database Setup & Ingestion Script
==========================================
CLI entry point that orchestrates:
    1. Initialise ``GaliVectorStore`` (optionally drop existing table).
    2. Run the ``IngestionPipeline``.
    3. Print a structured execution summary.

Usage:
    python -m gali.scripts.setup_db              # Normal ingestion
    python -m gali.scripts.setup_db --drop        # Drop table first, then ingest
    python -m gali.scripts.setup_db --drop-only   # Drop table and exit
"""

from __future__ import annotations

import argparse
import sys
import time

# Ensure the project root (parent of `gali/`) is on sys.path so that
# absolute imports work when running the script directly.
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gali.config.settings import settings
from gali.src.core.ingestor import IngestionPipeline
from gali.src.database.vector_store import GaliVectorStore
from gali.src.utils.logger import get_logger

logger = get_logger(__name__)


# ── CLI Argument Parsing ───────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="setup_db",
        description="Gali — Initialise the vector database and run document ingestion.",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        default=False,
        help="Drop the existing LanceDB table before ingesting.",
    )
    parser.add_argument(
        "--drop-only",
        action="store_true",
        default=False,
        help="Drop the existing LanceDB table and exit (no ingestion).",
    )
    return parser.parse_args()


# ── Main Orchestration ─────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    t_start = time.perf_counter()

    _print_header()

    # ── 1. Initialise embedder ─────────────────────────────────────────
    logger.info("Initialising embedding model: %s", settings.EMBEDDING_MODEL)
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embedder = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )
    except Exception:
        logger.exception("Failed to initialise embedding model.")
        sys.exit(1)

    # ── 2. Initialise GaliVectorStore ──────────────────────────────────
    logger.info("Connecting to LanceDB at: %s", settings.LANCEDB_PATH)
    store = GaliVectorStore(embedder=embedder)

    if args.drop or args.drop_only:
        logger.warning("Dropping table '%s' as requested.", settings.LANCEDB_TABLE_NAME)
        store.drop_table()

        if args.drop_only:
            logger.info("--drop-only: Table dropped. Exiting.")
            _print_footer(0, 0, 0, time.perf_counter() - t_start)
            return

        # Re-initialise store after drop so a fresh table is created
        store = GaliVectorStore(embedder=embedder)

    logger.info(
        "VectorStore ready — table '%s' (%d existing rows).",
        settings.LANCEDB_TABLE_NAME,
        store.count(),
    )

    # ── 3. Run IngestionPipeline ───────────────────────────────────────
    pipeline = IngestionPipeline(
        vector_store=store,
        embedder=embedder,
    )
    summary = pipeline.run()

    # ── 4. Print execution summary ─────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    _print_footer(
        summary["total_files"],
        summary["total_chunks"],
        summary["files_skipped"],
        elapsed,
    )


# ── Pretty-print helpers ──────────────────────────────────────────────

def _print_header() -> None:
    print()
    print("=" * 60)
    print("  GALI — Vector Database Setup & Ingestion")
    print("=" * 60)
    print(f"  Environment  : {settings.ENV}")
    print(f"  Embedding    : {settings.EMBEDDING_MODEL}")
    print(f"  LanceDB path : {settings.LANCEDB_PATH}")
    print(f"  Source dir    : {settings.DATA_RAW_DIR}")
    print(f"  Chunk size    : {settings.CHUNK_SIZE} chars")
    print("=" * 60)
    print()


def _print_footer(total_files: int, total_chunks: int, skipped: int, elapsed: float) -> None:
    print()
    print("=" * 60)
    print("  EXECUTION SUMMARY")
    print("-" * 60)
    print(f"  Total files scanned  : {total_files}")
    print(f"  Files ingested       : {total_files - skipped}")
    print(f"  Files skipped (cache): {skipped}")
    print(f"  Total chunks stored  : {total_chunks}")
    print(f"  Total time elapsed   : {elapsed:.2f}s")
    print("=" * 60)
    print()


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

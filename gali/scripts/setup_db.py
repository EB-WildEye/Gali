"""
Gali - Database Setup & Ingestion Script
==========================================
CLI entry point that orchestrates:
    1. Validate that ``GOOGLE_API_KEY`` is set (fail-fast).
    2. Initialise ``GaliVectorStore`` (optionally drop existing table).
    3. Run the ``IngestionPipeline``.
    4. Print a structured execution summary with timing breakdown.

Flags:
    --drop       Drop the LanceDB table before ingesting (cache preserved).
    --purge      Drop table AND clear the hash cache (full re-ingestion).
    --drop-only  Drop the table and exit immediately (no ingestion).

Observability:
    The script times every initialization phase independently —
    settings load, embedder init, LanceDB connection — so the
    final summary separates **Startup Time** from **Processing Time**.

Usage:
    python -m gali.scripts.setup_db              # Normal ingestion
    python -m gali.scripts.setup_db --drop        # Drop table, re-ingest (skip cached)
    python -m gali.scripts.setup_db --purge       # Drop table + cache, full re-ingest
    python -m gali.scripts.setup_db --drop-only   # Drop table and exit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Ensure project root is importable when run directly ────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── CLI Argument Parsing ───────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="setup_db", description="Gali — Initialise the vector database and run document ingestion.")
    parser.add_argument("--drop", action="store_true", default=False, help="Drop the LanceDB table before ingesting (hash cache preserved).")
    parser.add_argument("--purge", action="store_true", default=False, help="Drop the LanceDB table AND clear the hash cache (full clean re-ingestion).")
    parser.add_argument("--drop-only", action="store_true", default=False, help="Drop the LanceDB table and exit (no ingestion).")
    return parser.parse_args()


# ── Main Orchestration ─────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    t_start = time.perf_counter()

    # ── 0. Load settings + .env (timed) ────────────────────────────────
    t_settings = time.perf_counter()
    try:
        from gali.config.settings import settings
    except Exception as exc:
        print("\n[FATAL] Configuration error — check your .env file:\n")
        print(f"  {exc}")
        print()
        sys.exit(1)
    settings_ms = (time.perf_counter() - t_settings) * 1000

    # Now that settings is loaded, we can safely import the logger
    from gali.src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Settings loaded in %.1fms", settings_ms)

    _print_header(settings)

    # ── 1. Initialise embedder (timed) ─────────────────────────────────
    t_embedder = time.perf_counter()
    logger.info("Initialising embedding model: %s", settings.EMBEDDING_MODEL)
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embedder = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY.get_secret_value())
    except ImportError:
        logger.error("langchain-google-genai is not installed.")
        sys.exit(1)
    except Exception:
        logger.exception("Failed to initialise embedding model.")
        sys.exit(1)
    embedder_ms = (time.perf_counter() - t_embedder) * 1000
    logger.info("Embedder initialised in %.1fms", embedder_ms)

    # ── 2. Initialise GaliVectorStore (timed) ──────────────────────────
    from gali.src.database.vector_store import GaliVectorStore

    t_lancedb = time.perf_counter()
    logger.info("Connecting to LanceDB at: %s", settings.LANCEDB_PATH)
    store = GaliVectorStore(embedder=embedder)
    lancedb_ms = (time.perf_counter() - t_lancedb) * 1000
    logger.info("LanceDB connection established in %.1fms", lancedb_ms)

    if args.drop or args.purge or args.drop_only:
        logger.warning("Dropping table '%s' as requested.", settings.LANCEDB_TABLE_NAME)
        store.drop_table()

        # --purge also clears the hash cache for a full clean re-ingestion
        if args.purge:
            cache_path = settings.DATA_PROCESSED_DIR / "ingestion_hashes.json"
            if cache_path.exists():
                cache_path.unlink()
                logger.warning("Hash cache deleted: %s", cache_path)
            else:
                logger.info("No hash cache to clear.")

        if args.drop_only:
            logger.info("--drop-only: Table dropped. Exiting.")
            startup_ms = settings_ms + embedder_ms + lancedb_ms
            _print_footer(0, 0, 0, time.perf_counter() - t_start, settings_ms, embedder_ms, lancedb_ms, startup_ms)
            return

        # Re-initialise store so a fresh table is created
        store = GaliVectorStore(embedder=embedder)

    logger.info("VectorStore ready — table '%s' (%d existing rows).", settings.LANCEDB_TABLE_NAME, store.count())

    # ── Startup timing complete ────────────────────────────────────────
    startup_ms = settings_ms + embedder_ms + lancedb_ms
    logger.info("Total startup time: %.1fms (settings: %.1fms, embedder: %.1fms, lancedb: %.1fms)", startup_ms, settings_ms, embedder_ms, lancedb_ms)

    # ── 3. Run IngestionPipeline ───────────────────────────────────────
    from gali.src.core.ingestor import IngestionPipeline

    pipeline = IngestionPipeline(vector_store=store, embedder=embedder)
    summary = pipeline.run()

    # ── 4. Print execution summary ─────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    _print_footer(summary["total_files"], summary["total_chunks"], summary["files_skipped"], elapsed, settings_ms, embedder_ms, lancedb_ms, startup_ms)


# ── Pretty-print helpers ──────────────────────────────────────────────

def _print_header(settings: object) -> None:
    api_key_val = settings.GOOGLE_API_KEY.get_secret_value()  # type: ignore[attr-defined]
    masked = f"****{api_key_val[-4:]}" if len(api_key_val) > 4 else "****"

    mongo_uri_val = settings.MONGO_URI.get_secret_value()  # type: ignore[attr-defined]
    mongo_masked = mongo_uri_val.split("@")[-1] if "@" in mongo_uri_val else mongo_uri_val

    print()
    print("=" * 60)
    print("  GALI — Vector Database Setup & Ingestion")
    print("=" * 60)
    print(f"  Environment  : {settings.ENV}")                   # type: ignore[attr-defined]
    print(f"  Embedding    : {settings.EMBEDDING_MODEL}")       # type: ignore[attr-defined]
    print(f"  LanceDB path : {settings.LANCEDB_PATH}")          # type: ignore[attr-defined]
    print(f"  MongoDB      : {mongo_masked} (db: {settings.MONGO_DB_NAME})")  # type: ignore[attr-defined]
    print(f"  Source dir   : {settings.DATA_RAW_DIR}")           # type: ignore[attr-defined]
    print(f"  Chunk size   : {settings.CHUNK_SIZE} chars")      # type: ignore[attr-defined]
    print(f"  Workers      : {settings.MAX_WORKERS}")           # type: ignore[attr-defined]
    print(f"  API Key      : {masked}")
    print("=" * 60)
    print()


def _print_footer(total_files: int, total_chunks: int, skipped: int, elapsed: float, settings_ms: float, embedder_ms: float, lancedb_ms: float, startup_ms: float) -> None:
    processing_s = elapsed - (startup_ms / 1000)

    print()
    print("=" * 60)
    print("  EXECUTION SUMMARY")
    print("-" * 60)
    print(f"  Total files scanned  : {total_files}")
    print(f"  Files ingested       : {total_files - skipped}")
    print(f"  Files skipped (cache): {skipped}")
    print(f"  Total chunks stored  : {total_chunks}")
    print("-" * 60)
    print("  TIMING BREAKDOWN")
    print("-" * 60)
    print(f"  Settings + .env load : {settings_ms:>8.1f}ms")
    print(f"  Embedder init        : {embedder_ms:>8.1f}ms")
    print(f"  LanceDB connection   : {lancedb_ms:>8.1f}ms")
    print(f"  Startup time (total) : {startup_ms:>8.1f}ms")
    print(f"  Processing time      : {processing_s:>8.2f}s")
    print(f"  Total elapsed        : {elapsed:>8.2f}s")
    print("=" * 60)
    print()


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

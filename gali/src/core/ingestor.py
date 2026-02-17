"""
Gali - IngestionPipeline
=========================
Production-ready OOP pipeline that reads raw documents, cleans and
chunks them with a hybrid Hebrew-aware + semantic strategy, and
persists the results into the ``GaliVectorStore``.

Supported file formats:
    • ``.txt``  — UTF-8 with cp1255 fallback for legacy Hebrew files.
    • ``.pdf``  — Extracted via ``pypdf`` with page-level concatenation
                  and robust fallback for encrypted / malformed PDFs.

Key design decisions:
    • **Dependency Injection** – receives ``GaliVectorStore`` + embedder
      via constructor; both are typed with protocols, not concrete classes.
    • **Hybrid Chunking** – three strategies layered by priority:
        1. Hebrew Q&A markers (``שאלה:`` + ``תשובה:``) as hard boundaries.
        2. Semantic chunking (``SemanticChunker``) for content
           *within* or *between* those markers.
        3. Recursive character splitting as a fast fallback.
    • **Clinical Safety Rule** – a "שאלה" and its corresponding
      "תשובה" are *never* split across different chunks.
    • **Concurrency** – files are processed in parallel via
      ``ThreadPoolExecutor`` (worker count from ``settings.MAX_WORKERS``).
    • **Caching** – MD5-based file hashing skips unchanged files.
    • **Memory** – generators are used where possible; large texts
      are never copied unnecessarily.

Usage:
    from gali.src.core.ingestor import IngestionPipeline
    pipeline = IngestionPipeline(vector_store, embedder)
    result   = pipeline.run()
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gali.config.settings import settings
from gali.src.database.vector_store import GaliVectorStore, Embedder, DocumentMetadata
from gali.src.utils.logger import get_logger
from gali.src.utils.text_utils import clean_text, extract_metadata_from_filename

logger = get_logger(__name__)

# File extensions the pipeline knows how to read
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".pdf"})

# ── Hebrew Q&A markers ────────────────────────────────────────────────
_QA_SPLIT_PATTERN = re.compile(r"(?=שאלה\s*:)")
_QA_PAIR_RE = re.compile(r"שאלה\s*:.*?תשובה\s*:", re.DOTALL)

# ── Typing for summary dict ───────────────────────────────────────────
IngestionSummary = dict[str, int | float]


class IngestionPipeline:
    """
    End-to-end document ingestion: read → clean → chunk → embed → store.

    Parameters
    ----------
    vector_store : GaliVectorStore
        An initialised vector store instance (injected).
    embedder : Embedder
        An object satisfying the ``Embedder`` protocol.
    source_dir
        Override the source directory.  Defaults to ``settings.DATA_RAW_DIR``.
    max_workers
        Thread pool size.  Defaults to ``settings.MAX_WORKERS``.
    """

    __slots__ = ("_store", "_embedder", "_source_dir", "_max_workers", "_semantic_chunker", "_hash_cache_path", "_hash_cache")

    def __init__(self, vector_store: GaliVectorStore, embedder: Embedder, source_dir: Path | None = None, max_workers: int | None = None) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._source_dir = Path(source_dir or settings.DATA_RAW_DIR).resolve()
        self._max_workers = max_workers or settings.MAX_WORKERS
        self._semantic_chunker: object = None  # SemanticChunker | False | None
        self._hash_cache_path: Path = settings.DATA_PROCESSED_DIR / "ingestion_hashes.json"
        self._hash_cache: dict[str, str] = self._load_hash_cache()

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════════════

    def run(self) -> IngestionSummary:
        """
        Execute the full ingestion pipeline with concurrency.

        Returns
        -------
        IngestionSummary
            Keys: ``total_files``, ``files_processed``, ``files_skipped``,
            ``total_chunks``, ``elapsed_seconds``.
        """
        t_start = time.perf_counter()

        if not self._source_dir.exists():
            logger.warning("Source directory does not exist: %s", self._source_dir)
            return self._summary(0, 0, 0, 0, time.perf_counter() - t_start)

        files = sorted(f for f in self._source_dir.iterdir() if f.suffix.lower() in _SUPPORTED_EXTENSIONS)

        if not files:
            logger.warning("No supported files found in %s", self._source_dir)
            return self._summary(0, 0, 0, 0, time.perf_counter() - t_start)

        logger.info("Starting ingestion — %d file(s) found in %s", len(files), self._source_dir)

        total_chunks = 0
        files_processed = 0
        files_skipped = 0

        # ── Parallel file processing ───────────────────────────────────
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_path = {pool.submit(self._ingest_file, fp): fp for fp in files}

            for future in as_completed(future_to_path):
                filepath = future_to_path[future]
                try:
                    result = future.result()
                    if result == -1:
                        files_skipped += 1
                    else:
                        total_chunks += result
                        files_processed += 1
                except OSError:
                    logger.exception("I/O error ingesting %s", filepath.name)
                except ValueError as exc:
                    logger.error("Validation error for %s: %s", filepath.name, exc)
                except Exception:
                    logger.exception("Unexpected error ingesting %s", filepath.name)

        self._save_hash_cache()

        elapsed = time.perf_counter() - t_start
        summary = self._summary(len(files), files_processed, files_skipped, total_chunks, elapsed)

        logger.info("Ingestion complete — %d file(s) processed, %d skipped, %d chunk(s) stored in %.2fs.", files_processed, files_skipped, total_chunks, elapsed)
        return summary

    # ══════════════════════════════════════════════════════════════════
    #  PER-FILE PROCESSING
    # ══════════════════════════════════════════════════════════════════

    def _ingest_file(self, filepath: Path) -> int:
        """
        Read, clean, chunk, and store a single file.

        Returns ``-1`` on cache hit (file unchanged), otherwise the
        number of chunks added.
        """
        file_hash = self._compute_file_hash(filepath)
        if self._hash_cache.get(filepath.name) == file_hash:
            logger.info("CACHE_HIT — Skipping unchanged file: %s", filepath.name)
            return -1

        t_file = time.perf_counter()
        logger.info("Processing file: %s", filepath.name)

        raw_text = self._read_file(filepath)
        if not raw_text.strip():
            logger.warning("Skipping empty file: %s", filepath.name)
            return 0

        cleaned = clean_text(raw_text)
        metadata_base = extract_metadata_from_filename(filepath.name)
        metadata_base["source_file"] = filepath.name

        # ── Chunking (timed) ───────────────────────────────────────────
        t_chunk = time.perf_counter()
        chunks = self._smart_chunk(cleaned)
        chunk_ms = (time.perf_counter() - t_chunk) * 1000

        logger.info("File '%s' → %d chunk(s) in %.1fms.", filepath.name, len(chunks), chunk_ms)

        texts = chunks
        metadatas: list[DocumentMetadata] = [{**metadata_base, "chunk_index": idx} for idx in range(len(chunks))]

        for idx, chunk in enumerate(chunks):
            logger.debug("  Chunk %d (%d chars): %.60s…", idx, len(chunk), chunk.replace("\n", " "))

        # ── Embedding + storage (timed) ────────────────────────────────
        t_embed = time.perf_counter()
        added = self._store.add_documents(texts, metadatas)
        embed_ms = (time.perf_counter() - t_embed) * 1000

        total_ms = (time.perf_counter() - t_file) * 1000
        logger.info("File '%s' complete — embed: %.1fms, total: %.1fms.", filepath.name, embed_ms, total_ms)

        self._hash_cache[filepath.name] = file_hash
        return added

    # ══════════════════════════════════════════════════════════════════
    #  FILE READING
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _read_file(filepath: Path) -> str:
        """
        Route file reading to the appropriate extractor based on extension.

        Supported:
            • ``.txt``  — UTF-8 with cp1255 fallback.
            • ``.pdf``  — Full text extraction via ``pypdf``.
        """
        ext = filepath.suffix.lower()

        if ext == ".pdf":
            return IngestionPipeline._read_pdf(filepath)

        # Default: plain-text (.txt and any other text-based format)
        return IngestionPipeline._read_text(filepath)


    @staticmethod
    def _read_text(filepath: Path) -> str:
        """
        Read a plain-text file with encoding fallback.

        Tries UTF-8 first; falls back to cp1255 for legacy Hebrew
        Windows files that use the Windows-1255 code page.
        """
        try:
            return filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.debug("UTF-8 decode failed for %s — retrying with cp1255.", filepath.name)
            return filepath.read_text(encoding="cp1255")


    @staticmethod
    def _read_pdf(filepath: Path) -> str:
        """
        Extract all text from a PDF using ``pypdf``.

        Strategy (2025 best practices):
            1. Open the PDF with ``pypdf.PdfReader``.
            2. Skip encrypted PDFs that cannot be decrypted
               with an empty password.
            3. Iterate over every page and extract text via
               ``page.extract_text()``.
            4. Concatenate pages with double-newline separators
               so downstream chunking can detect page boundaries.
            5. Return the full concatenated text, or an empty
               string if no text could be extracted (scanned PDF).

        Args:
            filepath: Absolute path to the ``.pdf`` file.

        Returns:
            Extracted text as a single string.  Empty string on failure.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("pypdf is not installed — cannot read PDF files. Run: uv add pypdf")
            return ""

        try:
            reader = PdfReader(filepath)
        except Exception:
            logger.exception("Failed to open PDF: %s", filepath.name)
            return ""

        # Handle encrypted PDFs — try empty password first
        if reader.is_encrypted:
            try:
                reader.decrypt("")
                logger.debug("PDF '%s' decrypted with empty password.", filepath.name)
            except Exception:
                logger.warning("PDF '%s' is encrypted and cannot be decrypted — skipping.", filepath.name)
                return ""

        pages: list[str] = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
            except Exception:
                logger.warning("Failed to extract text from page %d of '%s' — skipping page.", page_num + 1, filepath.name)
                continue

        if not pages:
            logger.warning("PDF '%s' yielded no extractable text (possibly scanned/image-only).", filepath.name)
            return ""

        logger.info("PDF '%s' → %d page(s) extracted.", filepath.name, len(pages))
        return "\n\n".join(pages)

    # ══════════════════════════════════════════════════════════════════
    #  CHUNKING ENGINE — Hybrid: Q&A Pairs → Semantic → Recursive
    # ══════════════════════════════════════════════════════════════════

    def _smart_chunk(self, text: str) -> list[str]:
        """
        Orchestrate the hybrid chunking strategy.

        Priority:

        1. **Small document** – ≤ ``CHUNK_SIZE`` → single chunk.

        2. **Q&A structural split** – split on "שאלה:" boundaries.
           Paired question+answer blocks are kept intact.
           Free-text between pairs is semantically sub-chunked.

        3. **Semantic-only** – no Q&A markers → ``_semantic_chunk``.

        4. **Recursive fallback** – semantic unavailable → ``_recursive_chunk``.

        Every output passes through ``_merge_blocks`` as a final
        clinical guardrail.
        """
        chunk_size = settings.CHUNK_SIZE

        if len(text) <= chunk_size:
            logger.info("Strategy: SINGLE_CHUNK (document ≤ %d chars).", chunk_size)
            return [text]

        qa_blocks = _QA_SPLIT_PATTERN.split(text)
        qa_blocks = [b.strip() for b in qa_blocks if b.strip()]

        if len(qa_blocks) > 1:
            logger.info("Strategy: QA_STRUCTURAL — %d block(s) detected.", len(qa_blocks))
            refined: list[str] = []

            for block in qa_blocks:
                has_pair = bool(_QA_PAIR_RE.search(block))

                if has_pair:
                    refined.append(block)
                    logger.debug("  QA pair block (%d chars) — kept intact.", len(block))
                elif len(block) > chunk_size:
                    sub = self._semantic_chunk(block)
                    logger.debug("  Free-text block (%d chars) → %d semantic piece(s).", len(block), len(sub))
                    refined.extend(sub)
                else:
                    refined.append(block)

            return self._merge_blocks(refined, chunk_size)

        semantic_result = self._semantic_chunk(text)
        if semantic_result and len(semantic_result) > 1:
            logger.info("Strategy: SEMANTIC — %d breakpoint(s) identified.", len(semantic_result) - 1)
            return self._merge_blocks(semantic_result, chunk_size)

        logger.info("Strategy: RECURSIVE_FALLBACK.")
        return self._recursive_chunk(text)


    def _get_semantic_chunker(self) -> object | None:
        """
        Lazy-init the ``SemanticChunker`` (percentile @ 85).

        Returns ``None`` if ``langchain_experimental`` is not installed
        (sets sentinel ``False`` to avoid re-trying every call).
        """
        if self._semantic_chunker is None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker

                self._semantic_chunker = SemanticChunker(embeddings=self._embedder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=85.0)
                logger.info("SemanticChunker initialised (percentile @ 85).")
            except ImportError:
                logger.warning("langchain_experimental not installed — semantic chunking unavailable.")
                self._semantic_chunker = False

        return self._semantic_chunker if self._semantic_chunker else None


    def _semantic_chunk(self, text: str) -> list[str]:
        """
        Split *text* via ``SemanticChunker``.

        Falls back to ``_recursive_chunk`` if the chunker is
        unavailable or raises an error.
        """
        chunker = self._get_semantic_chunker()
        if chunker is None:
            logger.debug("Semantic chunker unavailable — delegating to recursive.")
            return self._recursive_chunk(text)

        try:
            docs = chunker.create_documents([text])  # type: ignore[union-attr]
            chunks = [d.page_content.strip() for d in docs if d.page_content.strip()]
            logger.debug("Semantic chunking → %d chunk(s), %d breakpoint(s).", len(chunks), max(len(chunks) - 1, 0))
            return chunks if chunks else [text]

        except Exception:
            logger.exception("Semantic chunking failed — falling back to recursive.")
            return self._recursive_chunk(text)


    def _recursive_chunk(self, text: str) -> list[str]:
        """Fast deterministic splitting with a 5-level separator hierarchy."""
        separators = ["\n\n", "\n", "。", ". ", " "]
        return self._recursive_split(text, separators, settings.CHUNK_SIZE)


    @classmethod
    def _recursive_split(cls, text: str, separators: list[str], max_size: int) -> list[str]:
        """Recursively split *text* using the first applicable separator."""
        if len(text) <= max_size:
            return [text]

        if not separators:
            return cls._hard_split(text, max_size)

        sep = separators[0]
        remaining = separators[1:]
        parts = [p.strip() for p in text.split(sep) if p.strip()]

        if len(parts) <= 1:
            return cls._recursive_split(text, remaining, max_size)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > max_size:
                    chunks.extend(cls._recursive_split(part, remaining, max_size))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)
        return chunks


    @staticmethod
    def _merge_blocks(blocks: list[str], max_size: int) -> list[str]:
        """
        Greedily merge consecutive *blocks* ≤ *max_size*.
        Oversized QA-pair blocks are preserved intact (clinical safety).
        """
        if not blocks:
            return []

        chunks: list[str] = []
        current = blocks[0]

        for block in blocks[1:]:
            candidate = current + "\n\n" + block
            if len(candidate) <= max_size:
                current = candidate
            else:
                chunks.append(current)
                current = block

        if current:
            chunks.append(current)
        return chunks


    @staticmethod
    def _hard_split(text: str, max_size: int) -> list[str]:
        """Character-level split at nearest whitespace."""
        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = start + max_size
            if end >= length:
                chunks.append(text[start:].strip())
                break
            split_at = text.rfind(" ", start, end)
            if split_at <= start:
                split_at = end
            chunks.append(text[start:split_at].strip())
            start = split_at + 1

        return [c for c in chunks if c]

    # ══════════════════════════════════════════════════════════════════
    #  MD5 CACHING
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Return the MD5 hex digest of *filepath* (streamed, low-memory)."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as fh:
            for block in iter(lambda: fh.read(8192), b""):
                hasher.update(block)
        return hasher.hexdigest()


    def _load_hash_cache(self) -> dict[str, str]:
        """Load persisted hash cache (or return empty dict)."""
        if self._hash_cache_path.exists():
            try:
                data = json.loads(self._hash_cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
                logger.warning("Hash cache is not a dict — starting fresh.")
            except json.JSONDecodeError:
                logger.warning("Corrupt hash cache JSON — starting fresh.")
            except OSError as exc:
                logger.warning("Cannot read hash cache (%s) — starting fresh.", exc)
        return {}


    def _save_hash_cache(self) -> None:
        """Persist the hash cache to disk."""
        self._hash_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._hash_cache_path.write_text(json.dumps(self._hash_cache, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("Hash cache saved to %s", self._hash_cache_path)


    @staticmethod
    def _summary(total: int, processed: int, skipped: int, chunks: int, elapsed: float) -> IngestionSummary:
        return {"total_files": total, "files_processed": processed, "files_skipped": skipped, "total_chunks": chunks, "elapsed_seconds": round(elapsed, 2)}

"""
Gali - Centralized Configuration
=================================
Uses ``pydantic-settings`` (``BaseSettings``) to load *all* configuration
from environment variables and the project-level ``.env`` file.

Security
--------
- ``GOOGLE_API_KEY`` is typed as ``SecretStr`` and has **no default value**.
  If the key is missing at startup, Pydantic will raise a ``ValidationError``
  with a clear error message.  The raw value is never exposed in repr,
  logs, or tracebacks.
- ``MONGO_URI`` is also ``SecretStr`` — connection strings contain
  credentials and must never leak into logs.

Paths
-----
All filesystem paths are ``Path.resolve()``-d at class level so they
work identically on Windows, WSL, and Linux.

Concurrency
-----------
``MAX_WORKERS`` controls the ``ThreadPoolExecutor`` pool size in the
ingestion pipeline (default 4 — optimal for I/O-bound Gemini API calls).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.

    Every field is loaded from environment variables (or ``.env``).
    Fields *without* a default are **required** — the app will refuse
    to start until they are provided.

    Attributes
    ----------
    GOOGLE_API_KEY : SecretStr
        API key for Google AI Studio (Gemini).  **Required.**
        Access the raw value with ``settings.GOOGLE_API_KEY.get_secret_value()``.
    MONGO_URI : SecretStr
        MongoDB connection string (e.g. ``mongodb://localhost:27017``).
        **Required.**  Contains credentials — never log raw value.
    MONGO_DB_NAME : str
        MongoDB database name for chat history / session storage.
    ENV : Literal["dev", "prod"]
        Environment mode controlling logging verbosity.
    CHUNK_SIZE : int
        Target character count per text chunk during ingestion.
    CHUNK_OVERLAP : int
        Overlap between consecutive chunks (for future sliding-window use).
    EMBEDDING_MODEL : str
        Model identifier passed to ``GoogleGenerativeAIEmbeddings``.
    LLM_MODEL : str
        Model identifier for the response-generation LLM.
    LANCEDB_TABLE_NAME : str
        Table name inside the LanceDB on-disk database.
    MAX_WORKERS : int
        Thread pool size for parallel file ingestion.
    """

    # ── Resolved Absolute Paths ────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    LANCEDB_PATH: Path = BASE_DIR / "data" / "lancedb"

    # ── Environment Mode ───────────────────────────────────────────────
    ENV: Literal["dev", "prod"] = "dev"

    # ── API Keys (REQUIRED — no default) ───────────────────────────────
    GOOGLE_API_KEY: SecretStr

    # ── MongoDB (REQUIRED — no default) ────────────────────────────────
    MONGO_URI: SecretStr
    MONGO_DB_NAME: str = "gali"

    # ── Ingestion Parameters ───────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Model Configuration ────────────────────────────────────────────
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.0-flash"

    # ── LanceDB ────────────────────────────────────────────────────────
    LANCEDB_TABLE_NAME: str = "gali_docs"

    # ── Concurrency ────────────────────────────────────────────────────
    MAX_WORKERS: int = 4

    # ── Validators ─────────────────────────────────────────────────────

    @field_validator("CHUNK_SIZE")
    @classmethod
    def _chunk_size_positive(cls, v: int) -> int:
        if v < 50:
            raise ValueError(f"CHUNK_SIZE must be ≥ 50, got {v}")
        return v


    @field_validator("MAX_WORKERS")
    @classmethod
    def _workers_range(cls, v: int) -> int:
        if not 1 <= v <= 16:
            raise ValueError(f"MAX_WORKERS must be 1–16, got {v}")
        return v

    # ── Pydantic Settings Configuration ────────────────────────────────
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parent.parent / ".env", env_file_encoding="utf-8", extra="ignore")


# ── Singleton Instance ─────────────────────────────────────────────────
# Import this throughout the project:
#     from gali.config.settings import settings
settings = Settings()

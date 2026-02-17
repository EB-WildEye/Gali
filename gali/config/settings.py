"""
Gali - Centralized Configuration
=================================
Uses ``pydantic-settings`` (``BaseSettings``) to load *all* configuration
from environment variables and the project-level ``.env`` file.

Security
--------
- ``GOOGLE_API_KEY`` is typed as ``SecretStr`` — never exposed in logs.
- ``MONGO_URI`` is also ``SecretStr`` — connection strings contain
  credentials and must never leak.

RAG Engine
----------
``LLM_TEMPERATURE`` controls Gemini response creativity (0.0–1.0).
``SESSION_HISTORY_LIMIT`` caps the chat messages sent to the LLM.
``SEARCH_RESULTS_LIMIT`` controls vector results entering the prompt.
``RELEVANCE_THRESHOLD`` filters out low-quality search results.
``TOKEN_LIMIT`` triggers auto-summarization of long conversations.

Paths
-----
All filesystem paths are ``Path.resolve()``-d at class level so they
work identically on Windows, WSL, and Linux.
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

    # ── MongoDB (REQUIRED — no default for URI) ────────────────────────
    MONGO_URI: SecretStr
    MONGO_DB_NAME: str = "gali"

    # ── Ingestion Parameters ───────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Model Configuration ────────────────────────────────────────────
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.0-pro"
    LLM_TEMPERATURE: float = 0.3

    # ── RAG Engine ─────────────────────────────────────────────────────
    SESSION_HISTORY_LIMIT: int = 10
    SEARCH_RESULTS_LIMIT: int = 5
    RELEVANCE_THRESHOLD: float = 1.5
    TOKEN_LIMIT: int = 2000

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


    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def _temperature_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"LLM_TEMPERATURE must be 0.0–1.0, got {v}")
        return v


    @field_validator("TOKEN_LIMIT")
    @classmethod
    def _token_limit_range(cls, v: int) -> int:
        if v < 500:
            raise ValueError(f"TOKEN_LIMIT must be ≥ 500, got {v}")
        return v

    # ── Pydantic Settings Configuration ────────────────────────────────
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parent.parent / ".env", env_file_encoding="utf-8", extra="ignore")


# ── Singleton Instance ─────────────────────────────────────────────────
# Import this throughout the project:
#     from gali.config.settings import settings
settings = Settings()

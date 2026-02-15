"""
Gali - Centralized Configuration
=================================
Uses pydantic-settings (BaseSettings) to load environment variables
and define absolute paths for all critical directories.

All paths are computed relative to BASE_DIR (the `gali/` package root),
ensuring portability across environments.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables and .env file.

    Attributes:
        GOOGLE_API_KEY: API key for Google AI Studio (Gemini).
        CHUNK_SIZE: Number of characters per text chunk during ingestion.
        CHUNK_OVERLAP: Number of overlapping characters between consecutive chunks.
        EMBEDDING_MODEL: The embedding model identifier used for vectorisation.
        LLM_MODEL: The LLM model identifier used for response generation.
        LANCEDB_TABLE_NAME: Name of the table inside the LanceDB database.
    """

    # ── Resolved Absolute Paths ────────────────────────────────────────
    # BASE_DIR points to the `gali/` package root (parent of `config/`).
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Directory containing raw input documents (.txt, .pdf, etc.)
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"

    # Directory for ingestion logs and processing metadata
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

    # Path to the LanceDB on-disk database directory
    LANCEDB_PATH: Path = BASE_DIR / "data" / "lancedb"

    # ── Environment Mode ───────────────────────────────────────────────
    # "dev" = verbose logging (DEBUG), "prod" = quiet logging (WARNING)
    ENV: str = "dev"

    # ── API Keys ───────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = "your_google_api_key_here"

    # ── Ingestion Parameters ───────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Model Configuration ────────────────────────────────────────────
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.0-flash"

    # ── LanceDB ────────────────────────────────────────────────────────
    LANCEDB_TABLE_NAME: str = "gali_docs"

    # ── Pydantic Settings Configuration ────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ── Singleton Instance ─────────────────────────────────────────────────
# Import this throughout the project:  from gali.config.settings import settings
settings = Settings()

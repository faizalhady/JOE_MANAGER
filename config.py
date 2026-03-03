"""
Configuration module - loads all settings from .env file.
Single source of truth for database, embedding, and chunking settings.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ─── Security Validation ───
REQUIRED_DB_VARS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_vars = [var for var in REQUIRED_DB_VARS if not os.getenv(var)]

if missing_vars:
    print(f"CRITICAL ERROR: Missing database configuration in .env file.")
    print(f"Missing variables: {', '.join(missing_vars)}")
    print("Please check your .env file and connection string. The application cannot start.")
    sys.exit(1)

# ─── Database ───
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# ─── Hardware Auto-Detection ───
# Automatically use GPU if available, fall back to CPU.
# Can be overridden in .env with EMBEDDING_DEVICE=cpu or EMBEDDING_DEVICE=cuda
def _detect_device() -> str:
    """Detect the best available compute device."""
    env_device = os.getenv("EMBEDDING_DEVICE", "").strip().lower()

    # If explicitly set in .env, respect it
    if env_device in ("cuda", "cpu"):
        return env_device

    # Auto-detect: try CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


EMBEDDING_DEVICE = _detect_device()

# ─── Embedding Model ───
# Default: BAAI/bge-small-en-v1.5 (384 dims) — optimized for CPU search on legacy hardware
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

# BGE models require a query prefix for optimal retrieval performance.
# This is applied ONLY when embedding search queries, NOT during ingestion.
# Set to empty string for non-BGE models (e.g. all-mpnet-base-v2).
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")

# ─── Chunking ───
# Target ~450 tokens per chunk. BGE max is 512 tokens.
# 1800 chars ÷ ~4 chars/token = ~450 tokens, leaving room for context header.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ─── Ingestion ───
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── Supported file types ───
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
    ".txt": "text",
    ".pptx": "pptx",
    ".msg": "email",
    ".eml": "email",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
}


def get_db_connection_string():
    """Returns a PostgreSQL connection string."""
    c = DB_CONFIG
    return f"postgresql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['dbname']}"


def print_config():
    """Print current configuration for debugging."""
    print("=" * 50)
    print("VECTOR DB MANAGER - CONFIGURATION")
    print("=" * 50)
    print(f"  Database:      {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    print(f"  Model:         {EMBEDDING_MODEL}")
    print(f"  Device:        {EMBEDDING_DEVICE}")
    print(f"  Dimensions:    {EMBEDDING_DIMENSIONS}")
    print(f"  Query Prefix:  {'Yes' if EMBEDDING_QUERY_PREFIX else 'None'}")
    print(f"  Chunk Size:    {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
    print(f"  Batch Size:    {BATCH_SIZE}")
    print("=" * 50)

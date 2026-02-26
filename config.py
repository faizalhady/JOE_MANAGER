"""
Configuration module - loads all settings from .env file.
Single source of truth for database, embedding, and chunking settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


# ─── Database ───
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "joe"),
    "user": os.getenv("DB_USER", "joe_pg"),
    "password": os.getenv("DB_PASSWORD", "joe123"),
}

# ─── Embedding Model ───
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")  # "cuda" or "cpu"
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# ─── Chunking ───
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ─── Ingestion ───
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── Supported file types ───
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
    ".txt": "text",
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
    print(f"  Database:    {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    print(f"  Model:       {EMBEDDING_MODEL}")
    print(f"  Device:      {EMBEDDING_DEVICE}")
    print(f"  Dimensions:  {EMBEDDING_DIMENSIONS}")
    print(f"  Chunk Size:  {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
    print(f"  Batch Size:  {BATCH_SIZE}")
    print("=" * 50)

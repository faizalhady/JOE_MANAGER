"""
Ingestion Pipeline
Orchestrates the complete flow: File → Extract → Chunk → Embed → Store

This is the core engine. It connects all the pieces:
  1. Processors (extract text from files)
  2. Chunker (split into right-sized pieces)
  3. Embedder (convert text to vectors)
  4. DatabaseManager (store in pgvector)
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from db.connection import DatabaseManager
from embeddings.embedder import Embedder
from processors import get_processor
from chunking.chunker import Chunker
from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Main pipeline that processes files and stores them as vectors.

    device_override    : 'cuda' | 'cpu' | None
                         None  → use EMBEDDING_DEVICE from .env (default behaviour)
                         'cuda' → force GPU  (pass --cuda on CLI)
                         'cpu'  → force CPU  (pass --cpu  on CLI)

    batch_size_override: int | None
                         None → use BATCH_SIZE from .env
                         int  → override batch size for this run

    Each file gets its own fresh DB connection so a dropped connection on one
    file never cascades to the next.

    Usage:
        # Use .env settings (default)
        pipeline = IngestionPipeline()

        # Force CUDA for fast local ingestion
        pipeline = IngestionPipeline(device_override='cuda', batch_size_override=32)

        # Force CPU for server with no GPU
        pipeline = IngestionPipeline(device_override='cpu', batch_size_override=64)
    """

    def __init__(self, device_override: str = None, batch_size_override: int = None):
        self.device_override = device_override
        self.batch_size_override = batch_size_override
        self.embedder = Embedder(device=device_override)  # None = use config default
        self.chunker = Chunker()
        self._initialized = False

    def initialize(self):
        """Load embedding model and verify DB connectivity."""
        logger.info("Initializing ingestion pipeline...")

        device_label = self.device_override or "from .env"
        batch_label = self.batch_size_override or "from .env"
        logger.info(f"Device: {device_label}  |  Batch size: {batch_label}")

        # Verify DB is reachable and schema is ready — short-lived connection
        with DatabaseManager() as db:
            db.init_schema()
        logger.info("Database schema verified.")

        # Load embedding model onto the selected device
        self.embedder.load()
        logger.info(f"Embedding model ready on: {self.embedder.device}")

        self._initialized = True
        logger.info("Pipeline ready.")

    def shutdown(self):
        """Clean up resources."""
        self._initialized = False
        logger.info("Pipeline shut down.")

    def ingest_file(self, file_path: str) -> Optional[int]:
        """
        Process a single file through the full pipeline.
        Opens and closes a fresh DB connection per file — prevents
        'connection already closed' on long batch runs.

        Returns:
            source_id if successful, None if skipped (already completed)
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Preserve UNC network paths (\\server\share)
        p = Path(file_path)
        if str(file_path).startswith('\\\\') or str(file_path).startswith('//'):
            file_path = str(p)
        else:
            file_path = str(p.resolve())

        file_name = Path(file_path).name
        file_ext = Path(file_path).suffix.lower()

        logger.info(f"{'='*60}")
        logger.info(f"INGESTING: {file_name}")
        logger.info(f"{'='*60}")

        # ─── Step 0: Validate ───
        if file_ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {file_ext}. Skipping {file_name}")
            return None

        file_type = SUPPORTED_EXTENSIONS[file_ext]
        file_size = Path(file_path).stat().st_size

        # ─── Step 1: Hash (duplicate detection) ───
        file_hash = self._compute_hash(file_path)
        logger.info(f"File hash: {file_hash[:16]}...")

        # ─── Steps 2–6: Fresh DB connection per file ───
        with DatabaseManager() as db:

            # ─── Step 2: Register ───
            source_id = db.register_source(
                file_name=file_name,
                file_path=file_path,
                file_type=file_type,
                file_hash=file_hash,
                file_size_bytes=file_size,
            )

            if source_id is None:
                logger.info(f"Skipped: {file_name} (already completed)")
                return None

            try:
                # ─── Step 3: Extract ───
                logger.info(f"[1/4] Extracting text from {file_name}...")
                processor = get_processor(file_path)
                sections = processor.extract(file_path)
                logger.info(f"       Extracted {len(sections)} sections")

                if not sections:
                    db.update_source_status(source_id, "completed", total_chunks=0)
                    logger.warning(f"No content extracted from {file_name}")
                    return source_id

                # ─── Step 4: Chunk ───
                logger.info(f"[2/4] Chunking into pieces...")
                chunks = self.chunker.chunk_sections(sections)
                logger.info(f"       Created {len(chunks)} chunks")

                # ─── Step 5: Embed ───
                logger.info(f"[3/4] Embedding {len(chunks)} chunks on {self.embedder.device}...")
                texts = [c["content"] for c in chunks]
                embeddings = self.embedder.embed_batch(
                    texts,
                    batch_size=self.batch_size_override  # None = use config default
                )
                logger.info(f"       Generated {len(embeddings)} vectors")

                # ─── Step 6: Store ───
                logger.info(f"[4/4] Storing in database...")
                db.insert_chunks(
                    source_id=source_id,
                    chunks=chunks,
                    embeddings=embeddings,
                    embedding_model=self.embedder.get_model_name(),
                )

                db.update_source_status(source_id, "completed", total_chunks=len(chunks))
                logger.info(f"SUCCESS: {file_name} → {len(chunks)} chunks stored")
                return source_id

            except Exception as e:
                logger.error(f"FAILED: {file_name} → {e}")
                try:
                    db.update_source_status(source_id, "failed", error_message=str(e))
                except Exception:
                    pass
                raise

    def ingest_directory(self, dir_path: str, recursive: bool = True) -> dict:
        """
        Process all supported files in a directory.

        Returns:
            Summary dict with counts of processed, skipped, and failed files.
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Path not found: {dir_path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        files = []
        pattern = "**/*" if recursive else "*"
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(dir_path.glob(f"{pattern}{ext}"))

        files = sorted(set(files))
        logger.info(f"Found {len(files)} supported files in {dir_path}")

        results = {"processed": 0, "skipped": 0, "failed": 0, "files": []}

        for i, file_path in enumerate(files, 1):
            logger.info(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            try:
                source_id = self.ingest_file(str(file_path))
                if source_id is not None:
                    results["processed"] += 1
                    results["files"].append({"file": file_path.name, "status": "processed"})
                else:
                    results["skipped"] += 1
                    results["files"].append({"file": file_path.name, "status": "skipped"})
            except Exception as e:
                results["failed"] += 1
                results["files"].append({"file": file_path.name, "status": "failed", "error": str(e)})
                logger.error(f"Failed to process {file_path.name}: {e}")
                # Always continue — one failure never stops the batch

        logger.info(f"\nIngestion complete: {results['processed']} processed, "
                    f"{results['skipped']} skipped, {results['failed']} failed")
        return results

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file for duplicate detection."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

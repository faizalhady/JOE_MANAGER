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
from typing import List, Optional

from db.connection import DatabaseManager
from embeddings.embedder import Embedder
from processors import get_processor
from chunking.chunker import Chunker
from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Main pipeline that processes files and stores them as vectors.
    
    Usage:
        pipeline = IngestionPipeline()
        pipeline.initialize()
        
        # Single file
        pipeline.ingest_file("path/to/document.pdf")
        
        # Entire directory
        pipeline.ingest_directory("path/to/documents/")
        
        pipeline.shutdown()
    """

    def __init__(self):
        self.db = DatabaseManager()
        self.embedder = Embedder()
        self.chunker = Chunker()
        self._initialized = False

    def initialize(self):
        """Connect to DB, load embedding model, initialize schema."""
        logger.info("Initializing ingestion pipeline...")

        # Connect to database
        self.db.connect()

        # Create tables if they don't exist
        self.db.init_schema()

        # Load embedding model onto GPU
        self.embedder.load()

        self._initialized = True
        logger.info("Pipeline ready.")

    def shutdown(self):
        """Clean up resources."""
        self.db.disconnect()
        self._initialized = False
        logger.info("Pipeline shut down.")

    def ingest_file(self, file_path: str) -> Optional[int]:
        """
        Process a single file through the full pipeline.
        
        Returns:
            source_id if successful, None if skipped (duplicate)
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        file_path = str(Path(file_path).resolve())
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

        # ─── Step 1: Hash the file (detect duplicates) ───
        file_hash = self._compute_hash(file_path)
        logger.info(f"File hash: {file_hash[:16]}...")

        # ─── Step 2: Register in database ───
        source_id = self.db.register_source(
            file_name=file_name,
            file_path=file_path,
            file_type=file_type,
            file_hash=file_hash,
            file_size_bytes=file_size,
        )

        if source_id is None:
            logger.info(f"Skipped: {file_name} (already ingested, same content)")
            return None

        try:
            # ─── Step 3: Extract text ───
            logger.info(f"[1/4] Extracting text from {file_name}...")
            processor = get_processor(file_path)
            sections = processor.extract(file_path)
            logger.info(f"       Extracted {len(sections)} sections")

            if not sections:
                self.db.update_source_status(source_id, "completed", total_chunks=0)
                logger.warning(f"No content extracted from {file_name}")
                return source_id

            # ─── Step 4: Chunk ───
            logger.info(f"[2/4] Chunking into pieces...")
            chunks = self.chunker.chunk_sections(sections)
            logger.info(f"       Created {len(chunks)} chunks")

            # ─── Step 5: Embed ───
            logger.info(f"[3/4] Embedding {len(chunks)} chunks...")
            texts = [c["content"] for c in chunks]
            embeddings = self.embedder.embed_batch(texts)
            logger.info(f"       Generated {len(embeddings)} vectors")

            # ─── Step 6: Store ───
            logger.info(f"[4/4] Storing in database...")
            self.db.insert_chunks(
                source_id=source_id,
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=self.embedder.get_model_name(),
            )

            # ─── Done ───
            self.db.update_source_status(source_id, "completed", total_chunks=len(chunks))
            logger.info(f"SUCCESS: {file_name} → {len(chunks)} chunks stored")
            return source_id

        except Exception as e:
            logger.error(f"FAILED: {file_name} → {e}")
            self.db.update_source_status(source_id, "failed", error_message=str(e))
            raise

    def ingest_directory(self, dir_path: str, recursive: bool = True) -> dict:
        """
        Process all supported files in a directory.
        
        Returns:
            Summary dict with counts of processed, skipped, and failed files.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Find all supported files
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

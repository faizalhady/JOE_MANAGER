"""
Database connection handler.
Manages PostgreSQL connections and provides schema initialization.
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from pathlib import Path
from config import DB_CONFIG

import logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles all PostgreSQL + pgvector operations."""

    def __init__(self):
        self.conn = None
        self.config = DB_CONFIG

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                dbname=self.config["dbname"],
                user=self.config["user"],
                password=self.config["password"],
            )
            self.conn.autocommit = False
            logger.info(f"Connected to PostgreSQL at {self.config['host']}:{self.config['port']}/{self.config['dbname']}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def disconnect(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed.")

    def init_schema(self):
        """Run schema.sql to create tables and indexes."""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        try:
            with self.conn.cursor() as cur:
                cur.execute(schema_sql)
            self.conn.commit()
            logger.info("Database schema initialized successfully.")
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Schema initialization failed: {e}")
            raise

    # ─── Source Document Operations ───

    def register_source(self, file_name, file_path, file_type, file_hash, file_size_bytes):
        """Register a new source document. Returns source_id or None if duplicate."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO source_documents (file_name, file_path, file_type, file_hash, file_size_bytes, status)
                    VALUES (%s, %s, %s, %s, %s, 'processing')
                    ON CONFLICT (file_hash) DO NOTHING
                    RETURNING id
                """, (file_name, file_path, file_type, file_hash, file_size_bytes))

                result = cur.fetchone()
                self.conn.commit()

                if result:
                    logger.info(f"Registered source: {file_name} (id={result[0]})")
                    return result[0]
                else:
                    logger.warning(f"Skipped duplicate: {file_name} (same content already ingested)")
                    return None

        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to register source {file_name}: {e}")
            raise

    def update_source_status(self, source_id, status, total_chunks=None, error_message=None):
        """Update the status of a source document after processing."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE source_documents
                    SET status = %s,
                        total_chunks = COALESCE(%s, total_chunks),
                        error_message = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, total_chunks, error_message, source_id))
            self.conn.commit()
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to update source {source_id}: {e}")
            raise

    # ─── Chunk Operations ───

    def insert_chunks(self, source_id, chunks, embeddings, embedding_model):
        """
        Batch insert chunks with their embeddings.
        
        Args:
            source_id: ID from source_documents table
            chunks: list of dicts with keys: content, chunk_index, chunk_type, metadata
            embeddings: list of vectors (list of floats), same length as chunks
            embedding_model: string name of the model used
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        try:
            with self.conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    cur.execute("""
                        INSERT INTO document_chunks 
                            (source_id, content, embedding, chunk_index, chunk_type, metadata, embedding_model)
                        VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
                    """, (
                        source_id,
                        chunk["content"],
                        str(embedding),
                        chunk["chunk_index"],
                        chunk.get("chunk_type", "text"),
                        Json(chunk.get("metadata", {})),
                        embedding_model,
                    ))

            self.conn.commit()
            logger.info(f"Inserted {len(chunks)} chunks for source_id={source_id}")

        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to insert chunks for source {source_id}: {e}")
            raise

    def search_similar(self, query_vector, top_k=5, embedding_model=None):
        """
        Find the most similar chunks to a query vector.
        This is what the copilot API will call.
        
        Args:
            query_vector: list of floats (the embedded question)
            top_k: number of results to return
            embedding_model: optional filter to only search chunks from a specific model
            
        Returns:
            list of dicts with content, source_file, similarity, metadata
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                model_filter = ""
                params = [str(query_vector), str(query_vector), top_k]

                if embedding_model:
                    model_filter = "AND dc.embedding_model = %s"
                    params = [str(query_vector), str(query_vector), embedding_model, top_k]

                cur.execute(f"""
                    SELECT
                        dc.content,
                        dc.chunk_type,
                        dc.metadata,
                        dc.embedding_model,
                        sd.file_name AS source_file,
                        sd.file_type AS source_type,
                        1 - (dc.embedding <=> %s::vector) AS similarity
                    FROM document_chunks dc
                    JOIN source_documents sd ON sd.id = dc.source_id
                    WHERE sd.status = 'completed'
                    {model_filter}
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s
                """, params)

                results = cur.fetchall()
                return [dict(r) for r in results]

        except psycopg2.Error as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    # ─── Management Operations ───

    def get_ingestion_summary(self):
        """Get overview of all ingested documents."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM ingestion_summary")
                return [dict(r) for r in cur.fetchall()]
        except psycopg2.Error as e:
            logger.error(f"Failed to get summary: {e}")
            raise

    def delete_source(self, source_id):
        """Delete a source document and all its chunks (CASCADE)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM source_documents WHERE id = %s RETURNING file_name", (source_id,))
                result = cur.fetchone()
            self.conn.commit()
            if result:
                logger.info(f"Deleted source: {result[0]} (id={source_id})")
                return result[0]
            return None
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to delete source {source_id}: {e}")
            raise

    def purge_all(self):
        """Delete ALL data. Use with caution."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks")
                cur.execute("DELETE FROM source_documents")
            self.conn.commit()
            logger.warning("All data purged from vector database.")
        except psycopg2.Error as e:
            self.conn.rollback()
            raise

    def get_stats(self):
        """Get database statistics."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM source_documents) AS total_sources,
                        (SELECT COUNT(*) FROM source_documents WHERE status = 'completed') AS completed_sources,
                        (SELECT COUNT(*) FROM source_documents WHERE status = 'failed') AS failed_sources,
                        (SELECT COUNT(*) FROM document_chunks) AS total_chunks,
                        (SELECT COUNT(DISTINCT embedding_model) FROM document_chunks) AS embedding_models_used
                """)
                return dict(cur.fetchone())
        except psycopg2.Error as e:
            logger.error(f"Failed to get stats: {e}")
            raise

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

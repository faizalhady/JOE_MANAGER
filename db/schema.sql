-- ============================================================
-- VECTOR DB MANAGER - PostgreSQL + pgvector Schema
-- ============================================================
-- Run this once to set up your database.
-- Requires: PostgreSQL 13+ with pgvector extension installed.
--
-- To install pgvector on your server:
--   1. Download from https://github.com/pgvector/pgvector
--   2. cd pgvector && make && make install
--   3. Then run this script.
-- ============================================================
-- Model: BAAI/bge-small-en-v1.5  →  384 dimensions
-- ============================================================

-- Enable pgvector extension globally in the public schema
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- Create the dedicated schema for our RAG data
CREATE SCHEMA IF NOT EXISTS rag;

-- Tell PostgreSQL to create all subsequent tables and views inside the 'rag' schema
SET search_path TO rag, public;

-- ─── SOURCE DOCUMENTS TABLE ───
CREATE TABLE IF NOT EXISTS source_documents (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    file_size_bytes BIGINT,
    total_chunks INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_file_hash UNIQUE (file_hash)
);

-- ─── DOCUMENT CHUNKS TABLE ───
-- 384 dimensions for BAAI/bge-small-en-v1.5
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source_documents (id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(384),
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(50) DEFAULT 'text',
    metadata JSONB DEFAULT '{}',
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'BAAI/bge-small-en-v1.5',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ─── INDEXES ───
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_source_id
    ON document_chunks (source_id);

CREATE INDEX IF NOT EXISTS idx_chunks_model
    ON document_chunks (embedding_model);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON document_chunks USING gin (metadata);

CREATE INDEX IF NOT EXISTS idx_source_file_hash
    ON source_documents (file_hash);

CREATE INDEX IF NOT EXISTS idx_source_status
    ON source_documents (status);

-- ─── HELPER VIEW ───
CREATE OR REPLACE VIEW rag.ingestion_summary AS
SELECT
    sd.file_name,
    sd.file_type,
    sd.status,
    sd.total_chunks,
    sd.file_size_bytes,
    sd.created_at,
    COUNT(dc.id) AS actual_chunks,
    sd.error_message
FROM
    rag.source_documents sd
    LEFT JOIN rag.document_chunks dc ON dc.source_id = sd.id
GROUP BY
    sd.id
ORDER BY
    sd.created_at DESC;

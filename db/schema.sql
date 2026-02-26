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
-- Enable pgvector extension globally in the public schema
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- Create the dedicated schema for our RAG data
CREATE SCHEMA IF NOT EXISTS rag;

-- Tell PostgreSQL to create all subsequent tables and views inside the 'rag' schema
-- We include 'public' so it can still find the vector data type
SET
    search_path TO rag,
    public;

-- ─── SOURCE DOCUMENTS TABLE ───
-- Tracks every file that has been ingested.
-- One row per file. Used to avoid re-processing and for audit trail.
CREATE TABLE IF NOT EXISTS source_documents (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(500) NOT NULL, -- original filename
    file_path VARCHAR(1000) NOT NULL, -- full path at time of ingestion
    file_type VARCHAR(20) NOT NULL, -- pdf, docx, xlsx, csv, text
    file_hash VARCHAR(64) NOT NULL, -- SHA-256 hash for change detection
    file_size_bytes BIGINT, -- file size for reference
    total_chunks INTEGER DEFAULT 0, -- how many chunks were created
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    error_message TEXT, -- if status = failed, why
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Prevent duplicate ingestion of same file content
    CONSTRAINT uq_file_hash UNIQUE (file_hash)
);

-- ─── DOCUMENT CHUNKS TABLE ───
-- The core table. Each row is one chunk of text + its vector embedding.
-- This is what gets searched during copilot queries.
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source_documents (id) ON DELETE CASCADE,
    content TEXT NOT NULL, -- the actual text chunk (LLM reads this)
    embedding vector (768), -- the vector (similarity search uses this)
    -- 768 dimensions for all-mpnet-base-v2
    chunk_index INTEGER NOT NULL, -- position in the original document
    chunk_type VARCHAR(50) DEFAULT 'text', -- text, table, header, row_group, etc.
    -- ─── Metadata (for source attribution + filtering) ───
    metadata JSONB DEFAULT '{}', -- flexible: page number, sheet name,
    -- section heading, row range, etc.
    -- ─── Embedding tracking ───
    embedding_model VARCHAR(100) NOT NULL -- which model created this vector
    DEFAULT 'all-mpnet-base-v2', -- CRITICAL for model swap compatibility
    -- ─── Timestamps ───
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ─── INDEXES ───
-- For small datasets (< 10,000 chunks), use HNSW instead - works on empty tables
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Speed up filtering by source document
CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON document_chunks (source_id);

-- Speed up filtering by embedding model (for future model swaps)
CREATE INDEX IF NOT EXISTS idx_chunks_model ON document_chunks (embedding_model);

-- Speed up metadata queries (GIN index for JSONB)
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON document_chunks USING gin (metadata);

-- Speed up source document lookups
CREATE INDEX IF NOT EXISTS idx_source_file_hash ON source_documents (file_hash);

CREATE INDEX IF NOT EXISTS idx_source_status ON source_documents (status);

-- ─── HELPER VIEW ───
-- Quick overview of what's in the database
CREATE
OR REPLACE VIEW ingestion_summary AS
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
    source_documents sd
    LEFT JOIN document_chunks dc ON dc.source_id = sd.id
GROUP BY
    sd.id
ORDER BY
    sd.created_at DESC;
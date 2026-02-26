"""
Chunking Engine
Takes extracted sections from processors and splits them into chunks
suitable for embedding. Handles overlap to prevent context loss at boundaries.
"""

import logging
from typing import List, Dict, Any
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class Chunker:
    """
    Splits text sections into chunks of consistent size.
    
    Why chunk?
    - Embedding models have input limits (~512 tokens for mpnet)
    - Smaller chunks = more precise retrieval
    - Overlap ensures no sentence is lost at a boundary
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    def chunk_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes sections from a processor and returns properly sized chunks.
        
        Small sections (already under chunk_size) pass through unchanged.
        Large sections get split with overlap.
        
        Args:
            sections: list of dicts from a processor's extract() method
            
        Returns:
            list of chunk dicts ready for embedding, each with:
            {
                "content": str,
                "chunk_index": int,
                "chunk_type": str,
                "metadata": dict,
            }
        """
        chunks = []
        global_index = 0

        for section in sections:
            content = section["content"]
            metadata = section.get("metadata", {})
            chunk_type = section.get("chunk_type", "text")

            # If the section is small enough, keep it as-is
            if len(content) <= self.chunk_size:
                chunks.append({
                    "content": content,
                    "chunk_index": global_index,
                    "chunk_type": chunk_type,
                    "metadata": metadata,
                })
                global_index += 1
                continue

            # Section is too large — split it
            sub_chunks = self._split_text(content)

            for i, sub_chunk in enumerate(sub_chunks):
                chunk_metadata = {
                    **metadata,
                    "sub_chunk": i,
                    "total_sub_chunks": len(sub_chunks),
                }

                chunks.append({
                    "content": sub_chunk,
                    "chunk_index": global_index,
                    "chunk_type": chunk_type,
                    "metadata": chunk_metadata,
                })
                global_index += 1

        logger.info(
            f"Chunking complete: {len(sections)} sections → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks, trying to break at sentence boundaries.
        """
        # Try to split at sentence boundaries first
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap from end of previous chunk
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Fallback: if sentence splitting didn't work (no punctuation),
        # do hard character splits
        if not chunks:
            chunks = self._hard_split(text)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple approach)."""
        import re

        # Split on sentence-ending punctuation followed by space or newline
        parts = re.split(r'(?<=[.!?])\s+', text)
        # Re-add spacing
        sentences = [p + " " for p in parts if p.strip()]
        return sentences

    def _get_overlap(self, text: str) -> str:
        """Get the last N characters as overlap for the next chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]

    def _hard_split(self, text: str) -> List[str]:
        """Last resort: split by character count with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        return [c for c in chunks if c]

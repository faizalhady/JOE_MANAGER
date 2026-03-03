"""
Chunking Engine
Takes extracted sections from processors and splits them into chunks
suitable for embedding. Handles overlap to prevent context loss at boundaries.

Design for BGE-small-en-v1.5 (max 512 tokens):
- 1 token ≈ 4 characters on average
- 512 tokens ≈ 2048 characters theoretical max
- We target ~1800 chars per chunk to stay safely under the limit
- Each chunk gets the source section heading prepended for context
- Overlap is sentence-based, not character-based, for cleaner boundaries
"""

import re
import logging
from typing import List, Dict, Any
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# ─── Sentence Tokenizer Setup ───
# Try NLTK's trained tokenizer (handles abbreviations, decimals, etc.)
# Falls back to regex if NLTK is not installed.
_nltk_tokenizer = None
try:
    import nltk
    try:
        _nltk_tokenizer = nltk.data.load('tokenizers/punkt_tab/english.pickle')
    except LookupError:
        # Download on first use
        nltk.download('punkt_tab', quiet=True)
        try:
            _nltk_tokenizer = nltk.data.load('tokenizers/punkt_tab/english.pickle')
        except LookupError:
            # Try older punkt as fallback
            nltk.download('punkt', quiet=True)
            _nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    if _nltk_tokenizer:
        print("[Chunker] Sentence splitter: NLTK punkt (trained, high accuracy)")
except ImportError:
    pass

if _nltk_tokenizer is None:
    print("[Chunker] Sentence splitter: regex fallback (install nltk for better accuracy)")


class Chunker:
    """
    Splits text sections into chunks optimized for embedding models.
    
    Key principles:
    - Sentence-aware splitting: never cuts mid-sentence
    - Context headers: prepends source/heading info to every chunk
    - Smart overlap: overlaps by full sentences, not raw characters
    - Size-aware: respects model token limits (512 tokens for BGE)
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    def chunk_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes sections from a processor and returns properly sized chunks.
        
        Small sections (already under chunk_size) pass through unchanged.
        Large sections get split with sentence-aware overlap.
        
        Each chunk gets a context header prepended from its metadata
        (source file, heading, sheet name, page number, etc.)
        so that even in isolation, the chunk makes sense.
        """
        chunks = []
        global_index = 0

        for section in sections:
            content = section["content"]
            metadata = section.get("metadata", {})
            chunk_type = section.get("chunk_type", "text")

            # Build a context header from metadata
            context_header = self._build_context_header(metadata, chunk_type)
            header_len = len(context_header)

            # Available space for actual content (reserve room for header)
            available_size = self.chunk_size - header_len

            if available_size < 100:
                # Header is too long, skip prepending
                context_header = ""
                available_size = self.chunk_size

            # If the section is small enough, keep it as-is
            if len(content) <= available_size:
                final_content = f"{context_header}{content}" if context_header else content
                chunks.append({
                    "content": final_content,
                    "chunk_index": global_index,
                    "chunk_type": chunk_type,
                    "metadata": metadata,
                })
                global_index += 1
                continue

            # Section is too large — split it with sentence-aware overlap
            sub_chunks = self._split_text(content, available_size)

            for i, sub_chunk in enumerate(sub_chunks):
                final_content = f"{context_header}{sub_chunk}" if context_header else sub_chunk

                chunk_metadata = {
                    **metadata,
                    "sub_chunk": i + 1,
                    "total_sub_chunks": len(sub_chunks),
                }

                chunks.append({
                    "content": final_content,
                    "chunk_index": global_index,
                    "chunk_type": chunk_type,
                    "metadata": chunk_metadata,
                })
                global_index += 1

        logger.info(
            f"Chunking complete: {len(sections)} sections → {len(chunks)} chunks "
            f"(target_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _build_context_header(self, metadata: dict, chunk_type: str) -> str:
        """
        Build a short context header from metadata so each chunk
        carries enough context to be understood in isolation.
        
        Examples:
          "[Source: report.pdf | Page 3]\n"
          "[Source: data.xlsx | Sheet: Revenue | Rows 12-21]\n"
          "[Source: email.msg | Subject: Q3 Budget | From: john@co.com]\n"
        """
        parts = []

        source = metadata.get("source", "")
        if source:
            parts.append(f"Source: {source}")

        # Page info (PDFs)
        page = metadata.get("page")
        if page:
            parts.append(f"Page {page}")

        # Sheet info (Excel)
        sheet = metadata.get("sheet")
        if sheet:
            parts.append(f"Sheet: {sheet}")

        # Row range (Excel/CSV)
        row_range = metadata.get("row_range")
        if row_range:
            parts.append(f"Rows {row_range}")

        # Slide info (PPTX)
        slide = metadata.get("slide")
        if slide:
            parts.append(f"Slide {slide}")

        # Heading info (DOCX)
        heading = metadata.get("heading")
        if heading and heading != "Document Start":
            parts.append(f"Section: {heading}")

        # Email info
        subject = metadata.get("subject")
        if subject:
            parts.append(f"Subject: {subject}")

        sender = metadata.get("sender")
        if sender:
            parts.append(f"From: {sender}")

        if not parts:
            return ""

        return f"[{' | '.join(parts)}]\n"

    def _split_text(self, text: str, max_size: int) -> List[str]:
        """
        Split text into overlapping chunks using sentence-aware boundaries.
        
        Strategy:
        1. Split into sentences
        2. Accumulate sentences until chunk is full
        3. Overlap by carrying the last N chars worth of sentences into the next chunk
        """
        sentences = self._split_into_sentences(text)

        # If no sentence boundaries found, fall back to hard split
        if len(sentences) <= 1 and len(text) > max_size:
            return self._hard_split(text, max_size)

        chunks = []
        current_sentences = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            # If a single sentence exceeds max_size, hard-split it
            if sent_len > max_size:
                # First, save what we have
                if current_sentences:
                    chunks.append("".join(current_sentences).strip())
                    current_sentences = []
                    current_len = 0

                # Hard-split the oversized sentence
                hard_parts = self._hard_split(sentence, max_size)
                chunks.extend(hard_parts)
                continue

            # If adding this sentence would exceed max_size, finalize current chunk
            if current_len + sent_len > max_size and current_sentences:
                chunk_text = "".join(current_sentences).strip()
                chunks.append(chunk_text)

                # Sentence-based overlap: carry over trailing sentences
                overlap_sentences = self._get_overlap_sentences(
                    current_sentences, self.chunk_overlap
                )
                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_len += sent_len

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = "".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses NLTK's trained punkt tokenizer when available — it correctly
        handles abbreviations (Dr., Mr., U.S.A.), decimals (3.14),
        and other tricky punctuation that regex gets wrong.
        
        Falls back to regex if NLTK is not installed.
        """
        # Strategy 1: NLTK trained tokenizer (best quality)
        if _nltk_tokenizer is not None:
            try:
                sentences = _nltk_tokenizer.tokenize(text)
                if len(sentences) > 1:
                    return [s.strip() + " " for s in sentences if s.strip()]
            except Exception:
                pass  # Fall through to regex

        # Strategy 2: Regex — split on sentence-ending punctuation
        parts = re.split(
            r'(?<=[.!?])\s+(?=[A-Z0-9"\'])',
            text
        )

        if len(parts) <= 1:
            parts = re.split(r'(?<=[.!?])\s+', text)

        # Strategy 3: Newlines as last resort
        if len(parts) <= 1:
            parts = text.split("\n")

        sentences = [p.strip() + " " for p in parts if p.strip()]
        return sentences

    def _get_overlap_sentences(self, sentences: List[str], target_overlap: int) -> List[str]:
        """
        Get trailing sentences that fit within the overlap character budget.
        Returns whole sentences — no mid-sentence cuts.
        """
        overlap = []
        total_len = 0

        for sentence in reversed(sentences):
            if total_len + len(sentence) > target_overlap and overlap:
                break
            overlap.insert(0, sentence)
            total_len += len(sentence)

        return overlap

    def _hard_split(self, text: str, max_size: int = None) -> List[str]:
        """
        Last resort: split by character count with overlap.
        Tries to break at word boundaries.
        """
        size = max_size or self.chunk_size
        chunks = []
        start = 0

        while start < len(text):
            end = start + size

            # Try to break at a word boundary
            if end < len(text):
                # Look backwards for a space
                last_space = text.rfind(" ", start + (size // 2), end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move forward with overlap
            start = end - self.chunk_overlap
            if start <= (end - size):
                start = end  # Prevent infinite loop

        return chunks

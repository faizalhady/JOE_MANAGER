"""
Legacy .doc (Word 97-2003) Document Processor
Uses antiword or textract to extract text from binary .doc files.
Falls back to a raw binary text extraction if neither is available.
"""

import logging
import subprocess
import re
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class DOCProcessor(BaseProcessor):

    supported_extensions = [".doc"]

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from legacy .doc files."""
        self.validate_file(file_path)

        file_name = Path(file_path).name
        text = None

        # Strategy 1: Try antiword (best quality, common on Linux)
        text = self._try_antiword(file_path)

        # Strategy 2: Try textract
        if text is None:
            text = self._try_textract(file_path)

        # Strategy 3: Raw binary extraction (last resort)
        if text is None:
            text = self._raw_extract(file_path)

        if not text or not text.strip():
            logger.warning(f"Could not extract text from {file_name}")
            return []

        # Split by double newlines for natural sections
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        sections = []
        for i, para in enumerate(paragraphs):
            sections.append({
                "content": para,
                "metadata": {
                    "source": file_name,
                    "paragraph_index": i,
                },
                "chunk_type": "text",
            })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections

    def _try_antiword(self, file_path: str) -> str:
        """Try extracting with antiword CLI tool."""
        try:
            result = subprocess.run(
                ["antiword", file_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info("Extracted .doc using antiword")
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def _try_textract(self, file_path: str) -> str:
        """Try extracting with textract library."""
        try:
            import textract
            text = textract.process(file_path).decode("utf-8", errors="replace")
            if text.strip():
                logger.info("Extracted .doc using textract")
                return text
        except (ImportError, Exception) as e:
            logger.debug(f"textract not available or failed: {e}")
        return None

    def _raw_extract(self, file_path: str) -> str:
        """
        Last resort: read the binary .doc file and extract readable ASCII text.
        Not great quality but better than nothing.
        """
        try:
            with open(file_path, "rb") as f:
                raw = f.read()

            # Extract printable ASCII sequences of 20+ characters
            text_chunks = re.findall(rb'[\x20-\x7e]{20,}', raw)
            if text_chunks:
                text = "\n".join(chunk.decode("ascii", errors="replace") for chunk in text_chunks)
                logger.info("Extracted .doc using raw binary extraction (fallback)")
                return text
        except Exception as e:
            logger.error(f"Raw extraction failed: {e}")

        return None

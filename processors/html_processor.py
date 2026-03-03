"""
HTML Document Processor
Extracts readable text from HTML/HTM files, stripping tags and scripts.
"""

import re
import logging
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class HTMLProcessor(BaseProcessor):

    supported_extensions = [".html", ".htm"]

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text content from HTML files."""
        self.validate_file(file_path)

        file_name = Path(file_path).name

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()

        if not raw_html.strip():
            return []

        # Try BeautifulSoup first, fall back to regex stripping
        text = self._try_beautifulsoup(raw_html)
        if text is None:
            text = self._regex_strip(raw_html)

        if not text or not text.strip():
            return []

        # Extract title if present
        title = self._extract_title(raw_html)

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # If no paragraph breaks, split on single newlines with some content
        if len(paragraphs) <= 1 and len(text) > 1000:
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]

        # Still just one big block? Keep as single section
        if not paragraphs:
            paragraphs = [text.strip()]

        sections = []
        for i, para in enumerate(paragraphs):
            metadata = {
                "source": file_name,
                "paragraph_index": i,
            }
            if title:
                metadata["title"] = title

            sections.append({
                "content": para,
                "metadata": metadata,
                "chunk_type": "text",
            })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections

    def _try_beautifulsoup(self, html: str) -> str:
        """Try using BeautifulSoup for clean text extraction."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script, style, nav, footer elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n")

            # Collapse excessive whitespace
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)

            return text if text.strip() else None

        except ImportError:
            logger.debug("BeautifulSoup not available, using regex fallback")
            return None

    def _regex_strip(self, html: str) -> str:
        """Fallback: strip HTML tags using regex."""
        # Remove script and style blocks
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Remove tags
        text = re.sub(r'<[^>]+>', '\n', text)
        # Decode common HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&nbsp;", " ").replace("&#39;", "'")
        # Collapse whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        return text

    def _extract_title(self, html: str) -> str:
        """Extract the <title> tag content."""
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

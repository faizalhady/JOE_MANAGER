"""
PDF Document Processor
Extracts text from PDFs page by page, preserving page numbers as metadata.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    
    supported_extensions = [".pdf"]

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF, one section per page."""
        self.validate_file(file_path)
        
        # Try pdfplumber first (better for tables), fall back to PyPDF2
        try:
            return self._extract_with_pdfplumber(file_path)
        except ImportError:
            logger.info("pdfplumber not available, using PyPDF2")
            return self._extract_with_pypdf2(file_path)

    def _extract_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract using pdfplumber (better table extraction)."""
        import pdfplumber

        sections = []
        file_name = Path(file_path).name

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing PDF: {file_name} ({total_pages} pages)")

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                sections.append({
                    "content": text.strip(),
                    "metadata": {
                        "page": i + 1,
                        "total_pages": total_pages,
                        "source": file_name,
                    },
                    "chunk_type": "text",
                })

                # Also extract tables separately if present
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if not table:
                        continue
                    # Convert table to readable text
                    table_text = self._table_to_text(table)
                    if table_text.strip():
                        sections.append({
                            "content": table_text,
                            "metadata": {
                                "page": i + 1,
                                "total_pages": total_pages,
                                "source": file_name,
                                "table_index": t_idx,
                            },
                            "chunk_type": "table",
                        })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections

    def _extract_with_pypdf2(self, file_path: str) -> List[Dict[str, Any]]:
        """Fallback extraction using PyPDF2."""
        from PyPDF2 import PdfReader

        sections = []
        file_name = Path(file_path).name
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        logger.info(f"Processing PDF: {file_name} ({total_pages} pages)")

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            sections.append({
                "content": text.strip(),
                "metadata": {
                    "page": i + 1,
                    "total_pages": total_pages,
                    "source": file_name,
                },
                "chunk_type": "text",
            })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections

    def _table_to_text(self, table: list) -> str:
        """Convert a pdfplumber table (list of lists) to readable text."""
        if not table or not table[0]:
            return ""

        headers = table[0]
        rows = table[1:]
        lines = []

        for row in rows:
            pairs = []
            for header, value in zip(headers, row):
                if header and value:
                    pairs.append(f"{header}: {value}")
            if pairs:
                lines.append(". ".join(pairs))

        return "\n".join(lines)
